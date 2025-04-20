import numpy as np
from elastica import *
from HANDS.TendonForces import TendonForces
from elastica.external_forces import GravityForces

def add_finger(simulator, start, **kwargs):
    """
    Add a finger to the environment.
    :param environment: The environment to which the finger will be added.
    :param simulator: The simulator instance.
    :param start: Starting position of the finger.
    :param length: Length of the finger.
    :param radius: Radius of the finger.
    :param inflections: Number of inflections in the finger.
    """
    # Create a Cosserat rod representing the finger
    direction = np.array([0.0, 0.0, 1.0])  # rod direction: pointing upwards
    normal = np.array([1.0, 0.0, 0.0])
    binormal = np.cross(direction, normal)
    directions = [
        np.array([1.0, 0.0, 0.0]),  # +X
        np.array([-1.0, 0.0, 0.0]), # -X
        np.array([0.0, 1.0, 0.0]),  # +Y
        np.array([0.0, -1.0, 0.0]), # -Y
    ]

    shear_modulus = kwargs.get("shear_mod", 7.216880e6) # Shear modulus for the rod material
    youngs_modulus = kwargs.get("E", 16.598637e6)  # Young's modulus for the rod material
    density = kwargs.get("density", 1000)  # Density of the rod material
    nu = kwargs.get("NU", 0.1)  # Damping coefficient
    n_elem = kwargs.get("n_elem", 50)  # Number of elements in the rod
    length = kwargs.get("length", 0.25)  # Length of the rod
    radius = kwargs.get("radius", 0.0055)  # Radius of the rod
    gravity_enable = kwargs.get("gravity", True)  # Enable gravity
    time_step = kwargs.get("time_step", 1.5e-5)  # Time step for the simulation
    vertebra_height = kwargs.get("vertebra_height", 0.025)  # Height of the vertebrae
    vertebra_mass = kwargs.get("vertebra_mass", 0.01)  # Mass of the vertebrae
    num_vertebrae = kwargs.get("num_vertebrae", 4)  # Number of vertebrae

    # Create the Cosserat rod with specified parameters
    cosserat_rod = CosseratRod.straight_rod(
        n_elem,
        start,
        direction,
        normal,
        length,
        base_radius=radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )

    # Add the rod to the simulator
    simulator.append(cosserat_rod)

    # Add gravity
    if gravity_enable:
        simulator.add_forcing_to(cosserat_rod).using(
            GravityForces, acc_gravity=np.array([0.0, 0.0, -9.80665])
        )

    # Add boundary constraints as fixing one end
    simulator.constrain(cosserat_rod).using(
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
    )

    simulator.dampen(cosserat_rod).using(
        AnalyticalLinearDamper,
        damping_constant=nu,
        time_step = time_step
    )

    # Add muscle torques acting on the arm for actuation
    # TendonForces uses the tensions selected by RL to
    # generate torques along the arm.

    tensions = np.zeros(len(directions))

    for i, direction in enumerate(directions):
        simulator.add_forcing_to(cosserat_rod).using(
            TendonForces,
            vertebra_height=vertebra_height,
            num_vertebrae=num_vertebrae,
            first_vertebra_node=2,
            final_vertebra_node=n_elem-2,
            vertebra_mass=vertebra_mass,
            tendon_id=i,
            tension_func_array=tensions,
            vertebra_height_orientation=direction,
            n_elements=n_elem
        )

    return tensions, cosserat_rod

def generate_circle_points_np(r, N, cx=0, cy=0):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    z = angles*0
    return np.column_stack((x, y, z))

def reshape_state(state, num_fingers):
    elements_per_finger = state.shape[0] // num_fingers

    backbone_points = (elements_per_finger) // 3

    # Reshape and exclude targets
    reshaped = (
        state.reshape(num_fingers, elements_per_finger)  # Split into fingers
        [:, :]                                            # Remove target columns
        .reshape(num_fingers, 3, backbone_points)           # Final shape
        .transpose(0, 2, 1)                                 # Rearrange to (num_fingers, backbone_points, 3)
    )

    return reshaped