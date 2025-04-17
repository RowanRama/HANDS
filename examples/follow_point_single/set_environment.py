import gymnasium
from gymnasium import spaces

import numpy as np
from functools import partial
import copy
import sys

from post_processing import plot_video_with_sphere

from HANDS.TendonForces import TendonForces
from HANDS.env_helpers import add_finger
from elastica.external_forces import GravityForces

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *

# Define base simulator class
class SoftRobotSimulator(
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
    Damping,
):
    pass

class Environment(gymnasium.Env):
    """
    Custom environment for a single tendon-driven finger with 4 cardinal direction tendons.
    The finger is modeled as a Cosserat rod with tendon forces applied in the cardinal directions.
    The finger must reach or follow a target point in 3D space.
    
    Attributes
    ----------
    n_elem : int
        Number of elements in the Cosserat rod
    final_time : float
        Final simulation time
    time_step : float
        Simulation time-step
    num_steps_per_update : int
        Number of simulation steps before updating actions
    max_tension : float
        Maximum allowed tension in any tendon
    n_elements : int
        Total number of nodes in the rod system
    shearable_rod : object
        Cosserat rod object representing the finger
    tendon_forces : list
        List of TendonForces objects for each cardinal direction
    mode : int
        Target following mode:
        1: Fixed target position to be reached (default)
        2: Randomly placed fixed target position to be reached
        3: Moving target on fixed trajectory
        4: Randomly moving target
    target_position : numpy.ndarray
        Current target position in 3D space
    sphere : object
        Target sphere object
    """

    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}

    """
    FOUR modes: (specified by mode)
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached
    3. fixed trajectory to be followed
    4. random trajectory to be followed (no need to worry in current phase)
    """

    def __init__(
        self,
        n_elem=50,
        final_time=1.0,
        sim_dt=1.5e-5,
        num_steps_per_update=100,
        max_tension=10.0,
        num_vertebrae=10,
        vertebra_height=0.0105,
        vertebra_mass=0.002,
        mode=1,
        target_position=None,
        sphere_initial_velocity=0.1,
        gravity_enable = True,  # Enable gravity by default
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        *args,
        **kwargs,
    ):
        super(Environment, self).__init__()
        
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        
        self.final_time = final_time
        self.time_step = sim_dt
        self.total_steps = int(final_time / sim_dt)
        print("Total steps", self.total_steps)

        self.max_tension = max_tension
        self.num_vertebrae = num_vertebrae
        self.vertebra_height = vertebra_height
        self.vertebra_mass = vertebra_mass

        self.sphere_initial_velocity = sphere_initial_velocity
        self.gravity_enable = gravity_enable

        # Video speed
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # target position
        self.target_position = target_position

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        # Tendon Parameters
        self.directions = [
            np.array([1.0, 0.0, 0.0]),  # +X
            np.array([-1.0, 0.0, 0.0]), # -X
            np.array([0.0, 1.0, 0.0]),  # +Y
            np.array([0.0, -1.0, 0.0]), # -Y
        ]

        # Define action space (4 tension values)
        self.action_space = spaces.Box(
            low=0.0,
            high=max_tension,
            shape=(len(self.directions),),
            dtype=np.float64
        )

        self.obs_state_points = 10
        num_points = int(n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(n_elem + 1)[0::num_points])

        # 8: 4 points for velocity and 4 points for orientation
        # 11: 3 points for target position plus 8 for velocity and orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 11,),
            dtype=np.float64,
        )

        self.mode = mode # currently only have implemented mode 1

        self.time_tracker = np.float64(0.0)

        self.E = kwargs.get("E", 16.598637e6)

        self.NU = kwargs.get("NU", 0.1)

        self.max_rate_of_change_of_activation = kwargs.get(
            "max_rate_of_change_of_activation", np.infty
        )

        self.n_elem = n_elem

    def reset(self):
        """Reset the environment to initial state."""
        # Create simulator
        self.simulator = SoftRobotSimulator()

        tensions, rod = add_finger(self, self.simulator)
        self.shearable_rod = rod
        self.tensions = tensions
        
        if self.mode != 2:
            # fixed target position to reach
            target_position = self.target_position

        # initialize sphere
        self.sphere = Sphere(
            center=target_position,  # initialize target position of the ball
            base_radius=0.05,
            density=1000,
        )

        # Set rod and sphere directors to each other.
        self.sphere.director_collection[
            ..., 0
        ] = self.shearable_rod.director_collection[..., 0]
        self.simulator.append(self.sphere)

        # Call back function to collect arm data from simulation
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod
            """

            def __init__(
                self, step_skip: int, callback_params: dict,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        # Call back function to collect target sphere data from simulation
        class RigidSphereCallBack(CallBackBaseClass):
            """
            Call back function for target sphere
            """

            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )

                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

            self.post_processing_dict_sphere = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for cyclinder and collect data
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_sphere,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # set state
        state = self.get_state()

        # reset on_goal
        self.on_goal = 0
        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        # reset previous_action
        self.previous_action = None

        # After resetting the environment return state information
        return state

    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """
        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]  # x_info
        r_s_b = rod_state[1]  # y_info
        r_s_c = rod_state[2]  # z_info

        num_points = int(self.n_elem / self.obs_state_points)
        ## get full 3D state information
        rod_compact_state = np.concatenate(
            (
                r_s_a[0 : len(r_s_a) + 1 : num_points],
                r_s_b[0 : len(r_s_b) + 1 : num_points],
                r_s_c[0 : len(r_s_b) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])
        # rod_compact_velocity_dir = np.where(
        #     (rod_compact_velocity_norm != 0) or (np.isnan(rod_compact_velocity_norm).any()),
        #     rod_compact_velocity / rod_compact_velocity_norm,
        #     0.0,
        # )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        # sphere_compact_velocity_dir = np.where(
        #     (sphere_compact_velocity_norm != 0) or (np.isnan(sphere_compact_velocity_norm).any()),
        #     sphere_compact_velocity / sphere_compact_velocity_norm,
        #     0.0,
        # )
        if rod_compact_velocity_norm == 0 or np.isnan(rod_compact_velocity_norm):
            rod_compact_velocity_dir = np.zeros_like(rod_compact_velocity)
        else:
            rod_compact_velocity_dir = rod_compact_velocity / rod_compact_velocity_norm
            
        if sphere_compact_velocity_norm == 0 or np.isnan(sphere_compact_velocity_norm):
            sphere_compact_velocity_dir = np.zeros_like(sphere_compact_velocity)
        else:
            sphere_compact_velocity_dir = sphere_compact_velocity / sphere_compact_velocity_norm
        state = np.concatenate(
            (
                # rod information
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                # target information
                sphere_compact_state,
                sphere_compact_velocity_norm,
                sphere_compact_velocity_dir,
            )
        )

        return state

    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : numpy.ndarray
            Array of 4 tension values for each cardinal direction
            
        Returns
        -------
        state : numpy.ndarray
            Current state of the system
        reward : float
            Reward value
        done : bool
            Whether the episode is finished
        info : dict
            Additional information
        """
        action = np.clip(action, 0.0, self.max_tension)
        self.action = action
        
        # Update tendon forces
        # if the tension value is negative, clip to 0
        self.tensions[:] = self.action[:]
        
        
        # Simulate for num_steps_per_update steps
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        # # Update target position based on mode
        # if self.mode == 3:
        #     if self.current_step % (1.0 / (self.time_step * self.num_steps_per_update)) == 0:
        #         if self.dir_indicator == 1:
        #             self.sphere.velocity_collection[..., 0] = np.array([0.0, -self.sphere_initial_velocity, 0.0])
        #             self.dir_indicator = 2
        #         elif self.dir_indicator == 2:
        #             self.sphere.velocity_collection[..., 0] = np.array([-self.sphere_initial_velocity, 0.0, 0.0])
        #             self.dir_indicator = 3
        #         elif self.dir_indicator == 3:
        #             self.sphere.velocity_collection[..., 0] = np.array([0.0, self.sphere_initial_velocity, 0.0])
        #             self.dir_indicator = 4
        #         elif self.dir_indicator == 4:
        #             self.sphere.velocity_collection[..., 0] = np.array([self.sphere_initial_velocity, 0.0, 0.0])
        #             self.dir_indicator = 1

        # elif self.mode == 4:
        #     self.trajectory_iteration += 1
        #     if self.trajectory_iteration == 500:
        #         self._update_random_target_velocity()
        #         self.trajectory_iteration = 0
        
        self.current_step += 1

        # Get current state
        state = self.get_state()
        
        # print(self.shearable_rod.position_collection[..., -1])
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )

        # Reward engineering
        reward_dist = -np.square(dist).sum()

        reward = 1.0 * reward_dist
        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward = -1000
            state = self.get_state()
            done = True

        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
            self.on_goal += self.time_step
            reward += 0.5
        # for this specific case, check on_goal parameter
        if np.isclose(dist, 0.0, atol=0.05).all():
            self.on_goal += self.time_step
            reward += 1.5

        else:
            self.on_goal = 0

        if self.current_step >= self.total_learning_steps:
            done = True
            if reward > 0:
                print(
                    " Reward greater than 0! Reward: %0.3f, Distance: %0.3f "
                    % (reward, dist)
                )
            else:
                print(
                    " Finished simulation. Reward: %0.3f, Distance: %0.3f"
                    % (reward, dist)
                )
        """ Done is a boolean to reset the environment before episode is completed """

        self.previous_action = action

        return state, reward, done, {"time": self.time_tracker, "position": self.shearable_rod.position_collection.copy()}

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        """Post processing after simulation (not implemented)."""
        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                [self.post_processing_dict_sphere],
                video_name="3d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            if SAVE_DATA == True:
                import os

                save_folder = os.path.join(os.getcwd(), "data")
                os.makedirs(save_folder, exist_ok=True)

                # Transform nodal to elemental positions
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                np.savez(
                    os.path.join(save_folder, "arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                    position_sphere=np.array(
                        self.post_processing_dict_sphere["position"]
                    ),
                    radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
                )

                np.savez(
                    os.path.join(save_folder, "arm_activation.npz"),
                    torque_mag=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]
                    ),
                    torque_muscle=np.array(
                        self.torque_profile_list_for_muscle_in_normal_dir["torque"]
                    ),
                )
        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )


    def sampleAction(self):
        """
        Sample usable random actions are returned.

        Returns
        -------
        numpy.ndarray
            1D (4,) array containing data with 'float' type, in range [0, max_tension].
            Represents random tension values for each cardinal direction.
        """
        random_action = (np.random.rand(4)) * self.max_tension
        return random_action
