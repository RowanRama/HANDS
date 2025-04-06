import gymnasium
from gymnasium import spaces
import numpy as np
from functools import partial
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
        final_time=10.0,
        time_step=1e-4,
        num_steps_per_update=100,
        max_tension=1.0,
        # rod_length=0.1,
        # rod_radius=0.001,
        youngs_modulus=16.598637e6,
        shear_modulus=7.216880e6,
        density=1000.0,
        num_vertebrae=10,
        vertebra_height=0.0105,
        vertebra_mass=0.002,
        mode=1,
        target_position=None,
        sphere_radius=0.005,
        sphere_density=1000.0,
        sphere_initial_velocity=0.1,
        gravity_enable = True,  # Enable gravity by default
    ):
        super(Environment, self).__init__()

        # Simulation parameters
        self.n_elem = n_elem
        self.final_time = final_time
        self.time_step = time_step
        self.num_steps_per_update = num_steps_per_update
        self.max_tension = max_tension
        self.n_elements = n_elem + 1
        self.mode = mode
        self.sphere_initial_velocity = sphere_initial_velocity
        self.StatefulStepper = PositionVerlet()

        # Create simulator
        self.simulator = SoftRobotSimulator()
        
        # Create rod
        base_length = 0.25  # rod base length
        # radius_tip = 0.05  # radius of the arm at the tip
        radius_base = 0.011/2  # radius of the arm at the base
        # radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)
        # print("radius along rod:", radius_along_rod)
        density = 1000
        # nu = 10  # dissipation coefficient
        # E = 1e7  # Young's Modulus
        # poisson_ratio = 0.5
        start = np.zeros((3,))
        direction =  np.array([0.0, 0.0, 1.0]) # Rod is along +Z direction
        normal = np.array([1.0, 0.0, 0.0]) # Normal should be a vector in the cross-section
        self.tendon_force_classes = []


        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start=start,
            direction = direction,# rod direction: pointing upwards
            normal = normal,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
            base_length=base_length,
            base_radius= radius_base,
            density=density
            )
        
        # Add rod to simulator
        self.simulator.append(self.shearable_rod)
        
        # Add gravity
        if gravity_enable:
            self.simulator.add_forcing_to(self.shearable_rod).using(
                GravityForces, acc_gravity=np.array([0.0, 0.0, -9.80665])
            )
        # else:
        #     self.simulator.add_forcing_to(self.shearable_rod).using(
        #         GravityForces, acc_gravity=np.array([0.0, 0.0, 0.0])
        #     )
        
        ## Damping for the rod
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=0.1,
            time_step = time_step
        )
        ## Constrain the rod at one end (fixed boundary condition)
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedBC,                  # Displacement BC being applied
            constrained_position_idx=(0,),  # Node number to apply BC
            constrained_director_idx=(0,)   # Element number to apply BC
        )
        # Create tendon forces for each cardinal direction
        self.tendon_forces = []
        directions = [
            np.array([1.0, 0.0, 0.0]),  # +X
            np.array([-1.0, 0.0, 0.0]), # -X
            np.array([0.0, 1.0, 0.0]),  # +Y
            np.array([0.0, -1.0, 0.0]), # -Y
        ] # These vectors should be orthogonal to the rod's direction
        
        # for direction in directions: #1 force for each direction
        #     tendon_force_class = TendonForces(
        #         vertebra_height=vertebra_height,
        #         num_vertebrae=num_vertebrae,
        #         first_vertebra_node=1,
        #         final_vertebra_node=n_elem,
        #         vertebra_mass=vertebra_mass,
        #         tensions=[0.0],  # Initial tension is 0
        #         vertebra_height_orientation=direction,
        #         n_elements=n_elem
        #     )
        for direction in directions: #1 force for each direction
            self.simulator.add_forcing_to(self.shearable_rod).using(
                TendonForces,vertebra_height=vertebra_height,
                num_vertebrae=num_vertebrae,
                first_vertebra_node=2,
                final_vertebra_node=n_elem-2,
                vertebra_mass=vertebra_mass,
                tensions=[0.0],  # Initial tension is 0
                vertebra_height_orientation=direction,
                n_elements=n_elem
            )
        print("added tendon forces")
            # print("tendon force is a", type(tendon_force_class))
            # self.simulator.add_forcing_to(self.shearable_rod).using(tendon_force_class)
            # self.tendon_force_classes.append(tendon_force_class)

        # Create target sphere
        self.sphere = Sphere(
            center=np.zeros(3),
            base_radius=sphere_radius,
            density=sphere_density,
        )
        self.simulator.append(self.sphere)

        # Set initial target position based on mode
        if mode == 1 and target_position is not None:
            self.target_position = target_position
            self.sphere.position_collection[..., 0] = target_position
        elif mode == 2:
            self.target_position = self._generate_random_target()
            self.sphere.position_collection[..., 0] = self.target_position
        elif mode == 3:
            self.target_position = np.array([0.1, 0.0, 0.0])  # Initial position for fixed trajectory
            self.sphere.position_collection[..., 0] = self.target_position
            self.sphere.velocity_collection[..., 0] = np.array([0.0, self.sphere_initial_velocity, 0.0])
            self.dir_indicator = 1
        elif mode == 4:
            self.target_position = self._generate_random_target()
            self.sphere.position_collection[..., 0] = self.target_position
            self._update_random_target_velocity()
            self.trajectory_iteration = 0

        # Define action space (4 tension values)
        self.action_space = spaces.Box(
            low=0.0,
            high=max_tension,
            shape=(4,),
            dtype=np.float32
        )

        # Define observation space (end effector position, orientation, and target position)
        # Position (3) + Orientation (3) + Target Position (3) = 9 values
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )

        # Initialize simulation
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        self.time_tracker = 0.0
        self.current_step = 0
        self.total_learning_steps = int(final_time / (time_step * num_steps_per_update))
        self.on_goal = 0

    def _generate_random_target(self):
        """Generate a random target position within reasonable bounds."""
        return np.array([
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(0.0, 0.2)
        ])

    def _update_random_target_velocity(self):
        """Update random target velocity for mode 4."""
        self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
        self.rand_direction_2 = np.pi * np.random.uniform(0, 2)
        
        self.v_x = self.sphere_initial_velocity * np.cos(self.rand_direction_1) * np.sin(self.rand_direction_2)
        self.v_y = self.sphere_initial_velocity * np.sin(self.rand_direction_1) * np.sin(self.rand_direction_2)
        self.v_z = self.sphere_initial_velocity * np.cos(self.rand_direction_2)
        
        self.sphere.velocity_collection[..., 0] = np.array([self.v_x, self.v_y, self.v_z])

    def reset(self):
        """Reset the environment to initial state."""
        # Reset rod to straight configuration
        self.shearable_rod.reset()
        
        # Reset time tracking
        self.time_tracker = 0.0
        self.current_step = 0
        self.on_goal = 0
        
        # Reset tendon forces
        for tendon in self.tendon_forces:
            tendon.tensions = [0.0]
        
        # Reset target based on mode
        if self.mode == 2:
            self.target_position = self._generate_random_target()
            self.sphere.position_collection[..., 0] = self.target_position
        elif self.mode == 3:
            self.target_position = np.array([0.1, 0.0, 0.0])
            self.sphere.position_collection[..., 0] = self.target_position
            self.sphere.velocity_collection[..., 0] = np.array([0.0, self.sphere_initial_velocity, 0.0])
            self.dir_indicator = 1
        elif self.mode == 4:
            self.target_position = self._generate_random_target()
            self.sphere.position_collection[..., 0] = self.target_position
            self._update_random_target_velocity()
            self.trajectory_iteration = 0
        
        return self.get_state()

    def get_state(self):
        """Get current state of the system."""
        # Get end effector position
        end_pos = self.shearable_rod.position_collection[:, -1]
        
        # Get end effector orientation (using the last director matrix)
        end_orientation = self.shearable_rod.director_collection[:, :, -1]
        
        # Get target position
        target_pos = self.sphere.position_collection[:, 0]
        
        # Combine position, orientation, and target into state
        state = np.concatenate([end_pos, end_orientation.flatten(), target_pos])
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
        # Clip actions to valid range
        action = np.clip(action, 0, self.max_tension)
        
        # Update tendon forces
        for i, tendon in enumerate(self.tendon_forces):
            tendon.tensions = [action[i]]
        
        # Simulate for num_steps_per_update steps
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        # Update target position based on mode
        if self.mode == 3:
            if self.current_step % (1.0 / (self.time_step * self.num_steps_per_update)) == 0:
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = np.array([0.0, -self.sphere_initial_velocity, 0.0])
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = np.array([-self.sphere_initial_velocity, 0.0, 0.0])
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = np.array([0.0, self.sphere_initial_velocity, 0.0])
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = np.array([self.sphere_initial_velocity, 0.0, 0.0])
                    self.dir_indicator = 1

        elif self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                self._update_random_target_velocity()
                self.trajectory_iteration = 0
        
        # Get current state
        state = self.get_state()
        
        # Calculate reward based on distance to target
        current_pos = self.shearable_rod.position_collection[:, -1]
        target_pos = self.sphere.position_collection[:, 0]
        dist = np.linalg.norm(current_pos - target_pos)
        
        # Reward engineering
        reward = -np.square(dist).sum()
        
        # Additional reward for being close to target
        if np.isclose(dist, 0.0, atol=0.05 * 2.0).all():
            self.on_goal += self.time_step
            reward += 0.5
        if np.isclose(dist, 0.0, atol=0.05).all():
            self.on_goal += self.time_step
            reward += 1.5
        else:
            self.on_goal = 0
        
        # Check if episode is done
        done = False
        if self.current_step >= self.total_learning_steps:
            done = True
            
        # Check for invalid values
        if _isnan_check(self.shearable_rod.position_collection):
            print("NaN detected, ending episode")
            reward = -1000
            done = True
            
        self.current_step += 1
        
        return state, reward, done, {"time": self.time_tracker, "position": self.shearable_rod.position_collection.copy()}

    def render(self, mode="human"):
        """Render the environment (not implemented)."""
        pass

    def post_processing(self):
        """Post processing after simulation (not implemented)."""
        pass

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
