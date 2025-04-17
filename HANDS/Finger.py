from numpy import np
from elastica._calculus import _isnan_check
from HANDS.env_helpers import add_finger
from HANDS.Controller import BaseController, PIDController

class Finger:
    """
    Class representing a finger of the soft manipulator.
    """

    def __init__(self, simulation, position, controller=None, **kwargs):
        """
        Initialize the Finger object.

        :param simulation: The simulation environment.
        :param position: Initial position of the finger.
        :param controller: The controller to be used for the finger.
        :param kwargs: Additional parameters for the finger.
        :param kwargs.n_elem: Number of elements in the finger (default: 50).
        :param kwargs.obs_state_points: Number of points for observation state (default: 10).
        :param kwargs.shear_mod: Shear modulus for the rod material (default: 7.216880e6).
        :param kwargs.E: Young's modulus for the rod material (default: 16.598637e6).
        :param kwargs.density: Density of the rod material (default: 1000).
        :param kwargs.NU: Damping coefficient (default: 0.1).
        :param kwargs.length: Length of the rod (default: 0.25).
        :param kwargs.radius: Radius of the rod (default: 0.0055).
        :param kwargs.gravity: Enable gravity (default: True).
        :param kwargs.time_step: Time step for the simulation (default: 1.5e-5).
        :param kwargs.vertebra_height: Height of the vertebrae (default: 0.025).
        :param kwargs.vertebra_mass: Mass of the vertebrae (default: 0.01).
        :param kwargs.num_vertebrae: Number of vertebrae (default: 4).
        """
        if controller is None:
            controller = PIDController(max_tension=20.0)
        if not issubclass(controller, BaseController):
            raise ValueError("Controller must be a subclass of BaseController.")
        
        self.controller = controller
        self.simulation = simulation
        self.init_position = position
        self.init_target_position = np.zeros(3)
        
        self.kwargs = kwargs

        self.n_elem = kwargs.get("n_elem", 50)  # Number of elements in the finger
        self.obs_state_points = kwargs.get("obs_state_points", 10)  # Number of points for observation state

    def reset(self):
        """
        Reset the finger to its initial state.
        """
        self.target_position = self.init_target_position
        tensions, rod = add_finger(self.simulation, self.init_position, self.kwargs)
        self.tensions = tensions
        self.rod = rod

    def get_state(self):
        """
        Get the current state of the finger.

        :return: Current backbone of the finger
        """
        rod_state = self.rod.position_collection
        rod_x = rod_state[0]
        rod_y = rod_state[1]
        rod_z = rod_state[2]

        num_points = int(self.n_elem / self.obs_state_points)
        ## get full 3D state information
        rod_compact_state = np.concatenate(
            (
                rod_x[0 : len(rod_x) + 1 : num_points],
                rod_y[0 : len(rod_y) + 1 : num_points],
                rod_z[0 : len(rod_z) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])

        rod_state = np.concatenate(
            (
                rod_compact_state,
                self.target_position
            )
        )

        return rod_state

    def update(self, target_position):
        """
        Set the joint angles for the finger.

        :param angles: List of joint angles.
        """
        self.target_position = target_position
        cur_position = self.rod.position_collection[-1]
        tensions = self.controller.get_tensions(target_position, cur_position, self.simulation.time_step)
        self.tensions[:] = tensions[:]

    def check_nan(self) -> bool:
        """
        Check if the finger has NaN values in its state.

        :return: True if NaN values are present, False otherwise.
        """
        return _isnan_check(self.rod.position_collection)