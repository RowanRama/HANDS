import gymnasium
from gymnasium import spaces
import numpy as np

import copy
import sys

from HANDS.TendonForces import TendonForces
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

    Custom environment that follows OpenAI Gym interface. This environment, generates an
    arm (Cosserat rod) and target (rigid sphere). Target is moving throughout the simulation in a space, defined
    by the user. Controller has to select control points (stored in action) and input to step class method.
    Control points have to be in between [-1,1] and are used to generate a beta spline. This beta spline is scaled
    by the torque scaling factor (alpha or beta) and muscle torques acting along arm computed. Muscle torques bend
    or twist the arm and tracks the moving target.

    Attributes
    ----------
    dim : float
        Dimension of the problem.
        If dim=2.0 2D problem only muscle torques in normal direction is activated.
        If dim=2.5 or 3.0 3D problem muscle torques in normal and binormal direction are activated.
        If dim=3.5 3D problem muscle torques in normal, binormal and tangent direction are activated.
    n_elem : int
        Cosserat rod number of elements.
    final_time : float
        Final simulation time.
    time_step : float
        Simulation time-step.
    number_of_control_points : int
        Number of control points for beta-spline that generate muscle torques.
    alpha : float
        Muscle torque scaling factor for normal/binormal directions.
    beta : float
        Muscle torque scaling factor for tangent directions (generates twist).
    target_position :  numpy.ndarray
        1D (3,) array containing data with 'float' type.
        Initial target position, If mode is 2 or 4 target randomly placed.
    num_steps_per_update : int
        Number of Elastica simulation steps, before updating the actions by control algorithm.
    action : numpy.ndarray
        1D (n_torque_directions * number_of_control_points,) array containing data with 'float' type.
        Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
        is number of torque directions, this is controlled by the dim.
    action_space : spaces.Box
        1D (n_torque_direction * number_of_control_poinst,) array containing data with 'float' type in range [-1., 1.].
    obs_state_points : int
        Number of arm (Cosserat rod) points used for state information.
    number_of_points_on_cylinder : int
        Number of cylinder points used for state information.
    observation_space : spaces.Box
        1D ( total_number_of_states,) array containing data with 'float' type.
        State information of the systems are stored in this variable.
    mode : int
        There are 4 modes available.
        mode=1 fixed target position to be reached (default)
        mode=2 randomly placed fixed target position to be reached. Target position changes every reset call.
        mode=3 moving target on fixed trajectory.
        mode=4 randomly moving target.
    COLLECT_DATA_FOR_POSTPROCESSING : boolean
        If true data from simulation is collected for post-processing. If false post-processing making videos
        and storing data is not done.
    E : float
        Young's modulus of the arm (Cosserat rod).
    NU : float
        Dissipation constant of the arm (Cosserat rod).
    COLLECT_CONTROL_POINTS_DATA : boolean
        If true actions or selected control points by the controller are stored throughout the simulation.
    total_learning_steps : int
        Total number of steps, controller is called. Also represents how many times actions changed throughout the
        simulation.
    control_point_history_array : numpy.ndarray
         2D (total_learning_steps, number_of_control_points) array containing data with 'float' type.
         Stores the actions or control points selected by the controller.
    shearable_rod : object
        shearable_rod or arm is Cosserat Rod object.
    sphere : object
        Target sphere is rigid Sphere object.
    spline_points_func_array_normal_dir : list
        Contains the control points for generating spline muscle torques in normal direction.
    torque_profile_list_for_muscle_in_normal_dir : defaultdict(list)
        Records, muscle torques and control points in normal direction throughout the simulation.
    spline_points_func_array_binormal_dir : list
        Contains the control points for generating spline muscle torques in binormal direction.
    torque_profile_list_for_muscle_in_binormal_dir : defaultdict(list)
        Records, muscle torques and control points in binormal direction throughout the simulation.
    spline_points_func_array_tangent_dir : list
        Contains the control points for generating spline muscle torques in tangent direction.
    torque_profile_list_for_muscle_in_tangent_dir : defaultdict(list)
        Records, muscle torques and control points in tangent direction throughout the simulation.
    post_processing_dict_rod : defaultdict(list)
        Contains the data collected by rod callback class. It stores the time-history data of rod and only initialized
        if COLLECT_DATA_FOR_POSTPROCESSING=True.
    post_processing_dict_sphere : defaultdict(list)
        Contains the data collected by target sphere callback class. It stores the time-history data of rod and only
        initialized if COLLECT_DATA_FOR_POSTPROCESSING=True.
    step_skip : int
        Determines the data collection step for callback functions. Callback functions collect data every step_skip.
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
        self
    ):
        pass

    def reset(self):
        pass

    def sampleAction(self):
        """
        Sample usable random actions are returned.

        Returns
        -------
        numpy.ndarray
            1D (number_tendon_segments * number_of_tendons) array containing data with 'float' type, in range [0, max_tension].
        """
        random_action = (np.random.rand(self.number_tendon_segments * self.number_of_tendons)) * self.max_tension
        return random_action

    #TODO: Update state information to return the position of rod end effector, position of intermediate tendon endpoints and the position of the target sphere
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
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()  # 2
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array(
            [np.linalg.norm(sphere_compact_velocity)]
        )
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

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
        This method integrates the simulation number of steps given in num_steps_per_update, using the actions
        selected by the controller and returns state information, reward, and done boolean.

        Parameters
        ----------
        action :  numpy.ndarray
            1D (number_tendon_segments * number_of_tendons) array containing data with 'float' type.
            Action returns control points selected by control algorithm to the Elastica simulation. n_torque_directions
            is number of torque directions, this is controlled by the dim.

        Returns
        -------
        state : numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        reward : float
            Reward after the integration.
        done: boolean
            Stops, simulation or training if done is true. This means, simulation reached final time or NaN is
            detected in the simulation.

        """
        # action contains the control tensions for actuation in speific a specific direction in range [0,max_tension]
        action = np.clip(action, 0, self.max_tension)
        self.action = action

        # self.spline_points_func_array_normal_dir[:] = action[
        #     : self.number_of_control_points
        # ]
        # self.spline_points_func_array_binormal_dir[:] = action[
        #     self.number_of_control_points : 2 * self.number_of_control_points
        # ]
        # self.spline_points_func_array_twist_dir[:] = action[
        #     2 * self.number_of_control_points :
        # ]

        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )
            
        if self.mode == 3:
            ##### (+1, 0, 0) -> (0, -1, 0) -> (-1, 0, 0) -> (0, +1, 0) -> (+1, 0, 0) #####
            if (
                self.current_step
                % (1.0 / (self.h_time_step * self.num_steps_per_update))
                == 0
            ):
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        -self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [
                        -self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [
                        0.0,
                        +self.sphere_initial_velocity,
                        0.0,
                    ]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [
                        +self.sphere_initial_velocity,
                        0.0,
                        0.0,
                    ]
                    self.dir_indicator = 1
                else:
                    print("ERROR")

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                # print('changing direction')
                self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
                if self.dim == 2.0 or self.dim == 2.5:
                    self.rand_direction_2 = np.pi / 2.0
                elif self.dim == 3.0 or self.dim == 3.5:
                    self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

                self.v_x = (
                    self.target_v
                    * np.cos(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_y = (
                    self.target_v
                    * np.sin(self.rand_direction_1)
                    * np.sin(self.rand_direction_2)
                )
                self.v_z = self.target_v * np.cos(self.rand_direction_2)

                self.sphere.velocity_collection[..., 0] = [
                    self.v_x,
                    self.v_y,
                    self.v_z,
                ]
                self.trajectory_iteration = 0

        self.current_step += 1

        # observe current state: current as sensed signal
        state = self.get_state()

        # print(self.sphere.position_collection[..., 0])
        dist = np.linalg.norm(
            self.shearable_rod.position_collection[..., -1]
            - self.sphere.position_collection[..., 0]
        )

        """ Reward Engineering """
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

        return state, reward, done, {"ctime": self.time_tracker}

    def render(self, mode="human"):
        """
        This method does nothing, it is here for interfacing with OpenAI Gym.

        Parameters
        ----------
        mode

        Returns
        -------

        """
        return
    
    def post_processing(self):
        pass
