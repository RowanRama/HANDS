import gymnasium
from gymnasium import spaces

import numpy as np
from functools import partial
import copy
import sys

from HANDS.Finger import Finger
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

    def reset(self):
        """
        Reset the environment to its initial state.

        :return: Initial observation of the environment.
        """
        self.simulator = SoftRobotSimulator()

        self.finger = Finger(
            self.simulator,
            self.finger_position,
            self.kwargs,
            controller=self.controller,
        )

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
        ] = self.finger.rod.director_collection[..., 0]
        self.simulator.append(self.sphere)

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

        self.get_state()

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

        return state, reward, done, {"time": self.time_tracker, "position": self.finger.rod.position_collection.copy()}

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass
