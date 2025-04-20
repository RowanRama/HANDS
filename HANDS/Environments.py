import gymnasium
from gymnasium import spaces

import numpy as np
from functools import partial
import copy
import sys

from HANDS.Finger import Finger
from HANDS.env_helpers import generate_circle_points_np
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
    Contact,
):
    pass

class SingleFinger(gymnasium.Env):
    """
    Class representing a single finger environment for the soft manipulator.
    """

    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            final_time=1.0,
            sim_dt=1.5e-5,
            num_steps_per_update=100,
            max_tension=10.0,
            target_position=np.array([0.5, 0.5, 0.5]),
            finger_position=np.array([0.0, 0.0, 0.0]),
            finger_controller=None,
            COLLECT_DATA_FOR_POSTPROCESSING=False,
            cylinder_enabled = False,
            cylin_params = {
                "length": 0.2,
                "direction": np.array([0.0, 1.0, 0.0]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "radius": 0.002,
                "start_pos": np.array([0.05, 0.0, 0.2]),
                "k": 1e4,
                "nu": 10,
                "density": 1000,
            },
            sphere_enabled = False,
            sphere_params = {
                "density": 1000,
                "radius": 0.08,
                "position": np.array([0.1, 0.1, 0.2]),
            },
            **kwargs
    ):
        """
        Initialize the Single_Finger environment.

        :param simulation: The simulation environment.
        :param kwargs: Additional parameters for the environment.
        """
        super(SingleFinger, self).__init__()

        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.time_step = sim_dt
        self.total_steps = int(final_time / sim_dt)
        print("Total steps", self.total_steps)

        self.target_position = target_position
        self.finger_position = finger_position
        self.max_tension = max_tension
        self.controller = finger_controller
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING
        self.kwargs = kwargs
        self.n_elem = kwargs.get("n_elem", 50)  # Number of elements in the finger

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        # Define action space (4 tension values)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.obs_state_points = 10
        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])

        # 8: 4 points for velocity and 4 points for orientation
        # 11: 3 points for target position plus 8 for velocity and orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 3,),
            dtype=np.float64,
        )

        self.time_tracker = np.float64(0.0)

        self.finger = Finger(
            self.finger_position,
            self.time_step,
            self.controller,
            **self.kwargs,
        )

        ## Adding a sphere to the simulation
        self.cylin_params = cylin_params
        self.cylinder_enabled = cylinder_enabled

        self.sphere_enabled = sphere_enabled
        self.sphere_params = sphere_params

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation of the environment.
        """
        self.simulator = SoftRobotSimulator()

        self.finger.reset(self.simulator)

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
        ] = self.finger.rod.director_collection[..., 0]
        self.simulator.append(self.sphere)

        if self.sphere_enabled:
            # Set the contact sphere parameters
            self.sphere2 = Sphere(
                center=self.sphere_params["position"],
                base_radius=self.sphere_params["radius"],
                density=self.sphere_params["density"],
            )
            self.simulator.append(self.sphere2)

            # Add contact forces
            self.simulator.detect_contact_between(self.finger.rod, self.sphere2).using(
                RodSphereContact,
                k = 1e4,
                nu = 10,
            )
            # Add constraints
            self.simulator.constrain(self.sphere2).using(
                GeneralConstraint,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                translational_constraint_selector=np.array([False, True, True]),
                rotational_constraint_selector=np.array([True, True, True]),
            )

        if self.cylinder_enabled:
            # Set the contact sphere parameters
            self.cylin = Cylinder(
                start = self.cylin_params["start_pos"],
                direction = self.cylin_params["direction"],
                normal = self.cylin_params["normal"],
                base_length=self.cylin_params["length"],
                base_radius=self.cylin_params["radius"],
                density=self.cylin_params["density"],
            )
            self.simulator.append(self.cylin)
            
            # Add contact forces
            self.simulator.detect_contact_between(self.finger.rod, self.cylin).using(
                RodCylinderContact,
                k = 1e4,
                nu = 10,
            )
            # Add damping
            # self.cylin.ring_rod_flag = False
            # print(f"element_mass: {self.cylin.element_mass}")
            # print(f"nodal_mass: {nodal_mass}")
            # self.simulator.dampen(self.cylin).using(
            #     AnalyticalLinearDamper,
            #     damping_constant=0.5,
            #     time_step = self.time_step
            # )
            # Add constraints
            self.simulator.constrain(self.cylin).using(
                GeneralConstraint,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                translational_constraint_selector=np.array([True, True, True]),
                rotational_constraint_selector=np.array([True, True, False]),
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
        Get the current state of the environment.

        :return: Current state of the environment.
        """
        finger_state = self.finger.get_state()

        return finger_state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : numpy.ndarray
            Array of target coordinates for the fingers.
            
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

        action = np.clip(action, -1.0, 1.0)
        self.action = action
        self.finger.update(action)

        # Simulate for num_steps_per_update steps
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        self.current_step += 1

        state = self.get_state()

        dist = np.linalg.norm(
            self.finger.rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )

        # Reward engineering
        reward_dist = -np.square(dist).sum()

        reward = 1.0 * reward_dist
        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # check for Nan rod position
        if self.finger.check_nan():
            print(" Nan detected, exiting simulation now")
            self.finger.rod.position_collection = np.zeros(
                self.finger.rod.position_collection.shape
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

        extra_info = {"time": self.time_tracker, "position": self.finger.rod.position_collection.copy()}

        if self.sphere_enabled:
            extra_info["sphere_position"] = self.sphere.position_collection.copy()
        if self.cylinder_enabled:
            extra_info["cylinder_position"] = self.cylin.position_collection.copy()
            extra_info["cylinder_director"] = self.cylin.director_collection.copy()
        

        return state, reward, done, extra_info

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

class MultipleFinger(gymnasium.Env):
    """
    Class representing a single finger environment for the soft manipulator.
    """

    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            mode = 1,
            final_time=1.0,
            sim_dt=1.5e-5,
            num_steps_per_update=100,
            max_tension=10.0,
            num_fingers = 2,
            finger_radius = 0.01,
            finger_controllers=None,
            COLLECT_DATA_FOR_POSTPROCESSING=False,
            cylinder_enabled = False,
            cylin_params = {
                "length": 0.2,
                "direction": np.array([0.0, 1.0, 0.0]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "radius": 0.002,
                "start_pos": np.array([0.0, 0.0, 0.8]),
                "k": 1e4,
                "nu": 10,
                "density": 1000,
            },
            sphere_enabled = False,
            sphere_params = {
                "density": 1000,
                "radius": 0.08,
                "position": np.array([0.0, 0.0, 0.2]),
            },
            **kwargs
    ):
        """
        Initialize the Single_Finger environment.

        :param simulation: The simulation environment.
        :param kwargs: Additional parameters for the environment.
        """
        super(MultipleFinger, self).__init__()

        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.time_step = sim_dt
        self.total_steps = int(final_time / sim_dt)
        print("Total steps", self.total_steps)


        self.max_tension = max_tension

        self.num_fingers = num_fingers
        self.controllers = finger_controllers
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING
        self.kwargs = kwargs
        self.n_elem = kwargs.get("n_elem", 50)  # Number of elements in the finger

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps", self.total_learning_steps)

        # Define action space (4 tension values)
        self.action_space = spaces.Box(low=-.3, high=.3, shape=(num_fingers,2), dtype=np.float32)

        self.obs_state_points = 2
        num_points = int(self.n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(self.n_elem + 1)[0::num_points])

        # 8: 4 points for velocity and 4 points for orientation
        # 11: 3 points for target position plus 8 for velocity and orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_fingers * 3,),
            dtype=np.float64,
        )

        self.time_tracker = np.float64(0.0)

        self.finger_points = generate_circle_points_np(finger_radius, self.num_fingers, 0.0, 0.0)

        if self.controllers is None:
            self.controllers = [None] * self.num_fingers
        elif isinstance(self.controllers, list) and len(self.controllers) == self.num_fingers:
            print("Controllers list is valid!")
        else:
            self.controllers = [None] * self.num_fingers
            print("Controllers list is invalid or not a list!")

        self.fingers = []
        for i in range(self.num_fingers):
            self.fingers.append(Finger(
                self.finger_points[i],
                self.time_step,
                self.controllers[i],
                **self.kwargs,
            ))

        ## Adding a sphere to the simulation
        self.cylin_params = cylin_params
        self.cylinder_enabled = cylinder_enabled

        self.sphere_enabled = sphere_enabled
        self.sphere_params = sphere_params

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation of the environment.
        """
        self.simulator = SoftRobotSimulator()

        for finger in self.fingers:
            finger.reset(self.simulator)

        if self.sphere_enabled:
            # Set the contact sphere parameters
            self.sphere = Sphere(
                center=self.sphere_params["position"],
                base_radius=self.sphere_params["radius"],
                density=self.sphere_params["density"],
            )
            self.simulator.append(self.sphere)

            # Add contact forces
            for finger in self.fingers:
                self.simulator.detect_contact_between(finger.rod, self.sphere).using(
                    RodSphereContact,
                    k = 1e4,
                    nu = 10,
                )

            # Add constraints
            self.simulator.constrain(self.sphere).using(
                GeneralConstraint,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                translational_constraint_selector=np.array([False, True, True]),
                rotational_constraint_selector=np.array([True, True, True]),
            )

        if self.cylinder_enabled:
            # Set the contact sphere parameters
            self.cylin = Cylinder(
                start = self.cylin_params["start_pos"],
                direction = self.cylin_params["direction"],
                normal = self.cylin_params["normal"],
                base_length=self.cylin_params["length"],
                base_radius=self.cylin_params["radius"],
                density=self.cylin_params["density"],
            )
            self.simulator.append(self.cylin)
            
            # Add contact forces
            for finger in self.fingers:
                self.simulator.detect_contact_between(finger.rod, self.cylin).using(
                    RodCylinderContact,
                    k = 1e4,
                    nu = 10,
                )

            # Add constraints
            self.simulator.constrain(self.cylin).using(
                GeneralConstraint,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                translational_constraint_selector=np.array([True, True, True]),
                rotational_constraint_selector=np.array([True, True, False]),
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

        # After resetting the environment return state information
        return state
    
    def get_state(self):
        """
        Get the current state of the environment.

        :return: Current state of the environment.
        """
        state = np.zeros((0,))
        for finger in self.fingers:
            finger_state = finger.get_state()
            state = np.concatenate((state, finger_state), axis=0)

        return state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : numpy.ndarray
            Array of target coordinates for the fingers.
            
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

        if len(action) != self.num_fingers:
            raise ValueError("Action length must match the number of fingers.")

        action = np.clip(action, -1.0, 1.0)
        self.action = action
        
        for i in range(self.num_fingers):
            self.fingers[i].update(action[i])

        # Simulate for num_steps_per_update steps
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        self.current_step += 1

        state = self.get_state()

        reward = 1.0 
        """ Done is a boolean to reset the environment before episode is completed """
        done = False

        # check for Nan rod position
        for finger in self.fingers:
            if finger.check_nan():
                print(" Nan detected, exiting simulation now")
                finger.rod.position_collection = np.zeros(
                    finger.rod.position_collection.shape
                )
                reward = -1000
                state = self.get_state()
                done = True

        
        # if self.current_step >= self.total_learning_steps:
        #     done = True
        #     if reward > 0:
        #         print(
        #             " Reward greater than 0! Reward: %0.3f, Distance: %0.3f "
        #             % (reward, dist)
        #         )
        #     else:
        #         print(
        #             " Finished simulation. Reward: %0.3f, Distance: %0.3f"
        #             % (reward, dist)
        #         )
        """ Done is a boolean to reset the environment before episode is completed """

        extra_info = {"time": self.time_tracker}

        if self.sphere_enabled:
            extra_info["sphere_position"] = self.sphere.position_collection.copy()
        if self.cylinder_enabled:
            extra_info["cylinder_position"] = self.cylin.position_collection.copy()
            extra_info["cylinder_director"] = self.cylin.director_collection.copy()
        

        return state, reward, done, extra_info

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

class HLControlEnv(MultipleFinger):
    """
    Class representing a high-level control environment for the soft manipulator.
    """

    def __init__(
            self, 
            reward_function, 
            done_function,
            convergence_steps=200,
            save_logs=True,
            **kwargs):
        super(HLControlEnv, self).__init__(**kwargs)
        self.reward_function = reward_function
        self.convergence_steps = convergence_steps
        self.step_count = 0
        self.save_logs = save_logs
        self.done_function = done_function
        self.dt_L = self.time_step * self.num_steps_per_update  # The effective time step for the tension function
        self.time_points = 4

        # need to update observation space to have points over time
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_fingers * self.time_points * 3,),
            dtype=np.float64,
        )

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation of the environment.
        """
        self.step_count = 0
        self.outputs = []  # Initialize outputs for each episode
        return super().reset()
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : numpy.ndarray
            Array of target coordinates for the fingers.
            
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
        state, reward, done, info = None, None, False, None
        intermediate = []
        state_cat = np.zeros((0,))

        time_steps = self.convergence_steps // self.time_points

        for i in range(1, self.convergence_steps+1):
            state, reward, done, info = super().step(action)
            
            if i % time_steps == 0:
                state_cat = np.concatenate((state_cat, state), axis=0)

            if self.save_logs:
                step_data = {
                    "step": (self.step_count * self.convergence_steps + i),
                    "action": action,
                    "state": state,
                    "reward": reward,
                    "done": done,
                    "time": (self.step_count * self.convergence_steps + i) * self.dt_L,
                    "num_fingers": self.num_fingers,
                }

                if self.cylinder_enabled:
                    step_data["cylinder_position"] = info["cylinder_position"]
                    step_data["cylinder_director"] = info["cylinder_director"]

                intermediate.append(step_data)
        
            if done:
                break
        
        self.step_count += 1

        # Check if the episode is done
        done = done or self.done_function(state, action, info)

        # Calculate the reward using the provided reward function
        reward = self.reward_function(state, action, info)

        info["data"] = intermediate
        
        return state_cat, reward, done, info

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

