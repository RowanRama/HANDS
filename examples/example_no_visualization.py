# Copied and modified from TendonForces project by gabotuzl under MIT License
"""
This example script is for a simulation made using TendonForces. The simulation is carried out, however the visualization of the results is not implemented. This is to showcase 
only the implementation of the TendonForces forcing module into the PyElastica simulation software framework.
"""

from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import GravityForces
from elastica.external_forces import TendonForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import sys
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
plt.close("all")

class CantileverRodSimulator(
    BaseSystemCollection,
    Constraints, # Enabled to use boundary conditions 'OneEndFixedBC'
    Forcing,     # Enabled to use forcing 'GravityForces'
    CallBacks,   # Enabled to use callback
    Damping,     # Enabled to use damping models on systems.
):
    pass


CantileverRod = CantileverRodSimulator()

# Simulation parameters
final_time = 2.0
time_step = 1.8e-5
total_steps = int(final_time / time_step)
rendering_fps = 30.0
step_skip = int(1.0 / (rendering_fps * time_step))

# Create rod
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 0.25
n_elements = 100
youngs_modulus=16.598637e6
shear_modulus=7.216880e6
base_radius=0.011/2
density=997.7 

dtmax = (base_length/n_elements) * np.sqrt(density / max(youngs_modulus, shear_modulus)) # Maximum size of time step
print("Maximum time_step magnitude: ",dtmax)

rod1 = CosseratRod.straight_rod(
    n_elements= n_elements,                       # Number of elements
    start=np.array([0.0, 0.0, 0.0]),              # Starting position of first node in rod
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # Normal vector of rod
    base_length=base_length,                      # Original length of rod (m)
    base_radius=base_radius,                      # Original radius of rod (m)
    density=density,                              # Density of rod (kg/m^3)
    youngs_modulus=youngs_modulus,                # Elastic Modulus (Pa)
    shear_modulus=shear_modulus,                  # Shear Modulus (Pa)
)

#Add rod to simulator
CantileverRod.append(rod1)

#Constrain rod
CantileverRod.constrain(rod1).using(
    OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)

CantileverRod.add_forcing_to(rod1).using(
    TendonForces,
    vertebra_height = 0.015,
    num_vertebrae = 6,
    first_vertebra_node = 2,
    final_vertebra_node = 98,
    vertebra_mass = 0.002,
    tension = 5.0,
    vertebra_height_orientation = np.array([0.0, -1.0, 0.0]), # Orientation in the local frame (X Y Z)
    n_elements = n_elements
)

CantileverRod.add_forcing_to(rod1).using(
    TendonForces,
    vertebra_height = 0.0075,
    num_vertebrae = 15,
    first_vertebra_node = 2,
    final_vertebra_node = 50,
    vertebra_mass = 0.002,
    tension = 10.0,
    vertebra_height_orientation = np.array([1.0, 0.0, 0.0]), # Orientation in the local frame (X Y Z)
    n_elements = n_elements
)

CantileverRod.add_forcing_to(rod1).using(
    TendonForces,
    vertebra_height = 0.0105,
    num_vertebrae = 10,
    first_vertebra_node = 2,
    final_vertebra_node = 30,
    vertebra_mass = 0.002,
    tension = 10.0,
    vertebra_height_orientation = np.array([0.0, 1.0, 0.0]), # Orientation in the local frame (X Y Z)
    n_elements = n_elements
)

gravity_magnitude = -9.80665 #Value in m^2/s for gravity in simulation
acc_gravity = np.zeros(3)
acc_gravity[2] = gravity_magnitude
CantileverRod.add_forcing_to(rod1).using(
    GravityForces,
    acc_gravity = acc_gravity
)


CantileverRod.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant=0.1,
    time_step = time_step
)

# MyCallBack class is derived from the base call back class.
class MyCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    # This function is called every time step
    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            #self.callback_params["time"].append(time)
            #self.callback_params["step"].append(current_step)
            self.callback_params["position" ].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            # self.callback_params["velocity" ].append(system.velocity_collection.copy())
            for i in range(len(system.position_collection)):
                # Iterate over each element of the row
                for j in range(len(system.position_collection[i])):
                    # Check if the current element is NaN
                    if np.isnan(system.position_collection[i][j]):
                        print("NAN VALUE ENCOUNTERED at position: ", (i, j), "AT TIME: ",time)
                        sys.exit()

            return
            


# Create dictionary to hold data from callback function
callback_data_rod1= defaultdict(list)

# Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
CantileverRod.collect_diagnostics(rod1).using(
    MyCallBack, step_skip=step_skip, callback_params=callback_data_rod1)

CantileverRod.finalize()


timestepper = PositionVerlet()
integrate(timestepper, CantileverRod, final_time, total_steps)
