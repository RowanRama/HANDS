import numpy as np
from elastica import *
from HANDS.TendonForces import TendonForces
from elastica.external_forces import GravityForces

def add_finger(environment, simulator, start=np.zeros((3,)), length = 0.25, radius = 0.011/2, inflections = 1):
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

    shear_modulus = 7.216880e6  # Shear modulus for the rod material
    youngs_modulus = environment.E  # Young's modulus for the rod material
    density = 1000  # Density of the rod material
    nu = environment.NU  # Damping coefficient

    # Create the Cosserat rod with specified parameters
    cosserat_rod = CosseratRod.straight_rod(
        environment.n_elem,
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
    if environment.gravity_enable:
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
        time_step = environment.time_step
    )

    # Add muscle torques acting on the arm for actuation
    # TendonForces uses the tensions selected by RL to
    # generate torques along the arm.

    tensions = np.zeros(len(environment.directions))

    for i, direction in enumerate(environment.directions):
        simulator.add_forcing_to(cosserat_rod).using(
            TendonForces,
            vertebra_height=environment.vertebra_height,
            num_vertebrae=environment.num_vertebrae,
            first_vertebra_node=2,
            final_vertebra_node=environment.n_elem-2,
            vertebra_mass=environment.vertebra_mass,
            tendon_id=i,
            tension_func_array=tensions,
            vertebra_height_orientation=direction,
            n_elements=environment.n_elem
        )

    return tensions, cosserat_rod