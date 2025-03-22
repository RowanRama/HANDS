# Copied and modified from TendonForces project by gabotuzl under MIT License
"""
This script shows an example of a simulation carried out on a cantilever rod using the PyElastica simulation software, and also using the TendonForces forcing module to apply the effects of 
tendon actuation on the rod. This script also includes the use of matplotlib and moviepy to facilitate the visualization of the results output by PyElastica.
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
from HANDS.TendonForces import TendonForces
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
    vertebra_height_orientation = np.array([0.0, 0.0, 0.0]), # Orientation in the local frame (X Y Z)
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
            # self.callback_params["time"].append(time)
            # self.callback_params["step"].append(current_step)
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

position_data = callback_data_rod1["position"]
directors_data = callback_data_rod1["directors"]


# Creating a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# initialize counter
count=0

# method to get frames
def make_frame(t):

    global count

    # clear
    ax.clear()
     
    # Scatter plot
    ax.scatter(position_data[count][0], position_data[count][1], position_data[count][2])
    ax.axes.set_zlim3d(bottom=-base_length,top=base_length)
    ax.axes.set_ylim3d(bottom=-base_length,top=base_length)
    ax.axes.set_xlim(-base_length/2,base_length)

    # Labeling axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    calculated_x = position_data[count][0][-1]
    calculated_y = position_data[count][1][-1]
    calculated_z = position_data[count][2][-1]

    x_point = round(calculated_x,3)
    y_point = round(calculated_y,3)
    z_point = round(calculated_z,3)
    annotation_text = f'Tip Pos: ({x_point}, {y_point}, {z_point})'
    ax.text(x_point, y_point, z_point, annotation_text, color='red')

    original_directors = directors_data[count][...,0]
    
    num_vertebrae = 6
    first_vertebra_node = 2
    final_vertebra_node = 98
    vertebra_nodes = []
    vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
    for i in range(num_vertebrae):
        vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

    for node in vertebra_nodes:
        local_directors = directors_data[count][...,node]
        vertebra_pos = np.array([position_data[count][0][node], position_data[count][1][node], position_data[count][2][node]])
        vertebra_coord_syst_x = np.array([local_directors[0][0], local_directors[0][1], local_directors[0][2]])
        vertebra_coord_syst_y = np.array([local_directors[1][0], local_directors[1][1], local_directors[1][2]])
        vertebra_coord_syst_z = np.array([local_directors[2][0], local_directors[2][1], local_directors[2][2]])

        # Scale the arrows for better visibility
        if count > 1:
            scale = 0.1
        else:
            scale = 0.03

        # Plot the arrows using quiver
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_x[0], vertebra_coord_syst_x[1], vertebra_coord_syst_x[2],
            length=scale, color='r', normalize=True)
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_y[0], vertebra_coord_syst_y[1], vertebra_coord_syst_y[2],
            length=scale, color='g', normalize=True)
        ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
            vertebra_coord_syst_z[0], vertebra_coord_syst_z[1], vertebra_coord_syst_z[2],
            length=scale, color='b', normalize=True)


    # Update counter
    count=count+1
    # returning numpy imagedef make_frame(t):
    return plt.savefig(fig)
 
# creating animation
#clip = VideoClip(make_frame, duration = final_time)
 
# displaying animation with auto play and looping
#clip.write_videofile("Rod_Simulation.mp4", codec = "libx264", fps = rendering_fps)

original_matrix = np.array([[0, 0, 1],[0, -1, 0],[1, 0, 0]])

# Create a 3D plot
plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_title(f'\nArbitrary tendon configuration, \nE = {round(youngs_modulus * 1e-6,4)} MPa, G = {round(shear_modulus * 1e-6,4)} MPa', fontsize= 22)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Scatter plot
ax.scatter(position_data[-1][0], position_data[-1][1], position_data[-1][2])
ax.axes.set_zlim3d(bottom=-base_length,top=base_length)
ax.axes.set_ylim3d(bottom=-base_length,top=base_length)
ax.axes.set_xlim(-0.01,base_length)

# Labeling axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

calculated_x = position_data[-1][0][-1]
calculated_y = position_data[-1][1][-1]
calculated_z = position_data[-1][2][-1]

x_point = round(calculated_x,3)
y_point = round(calculated_y,3)
z_point = round(calculated_z,3)
annotation_text = f'Tip Pos: ({x_point}, {y_point}, {z_point})'
ax.text(x_point, y_point, z_point, annotation_text, color='red', fontsize= 15)

original_directors = directors_data[-1][...,0]

num_vertebrae = 6
first_vertebra_node = 2
final_vertebra_node = 98
vertebra_nodes = []
vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
for i in range(num_vertebrae):
    vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

for node in vertebra_nodes:
    local_directors = directors_data[-1][...,node]
    vertebra_pos = np.array([position_data[-1][0][node], position_data[-1][1][node], position_data[-1][2][node]])
    vertebra_coord_syst_x = np.array([0.0, 0.0, 1.0])
    vertebra_coord_syst_y = np.array([0.0, 1.0, 0.0])
    vertebra_coord_syst_z = np.array([0.0, 0.0, -1.0])
    vertebra_coord_syst_zz = np.array([0.0, -1.0, 0.0])

    # Scale the arrows for better visibility
    scale = 0.10

    #Plot the arrows using quiver
    ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
        vertebra_coord_syst_y[0], vertebra_coord_syst_y[1], vertebra_coord_syst_y[2],
        length=scale, color='b', normalize=True)


num_vertebrae = 15
first_vertebra_node = 2
final_vertebra_node = 50
vertebra_nodes = []
vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
for i in range(num_vertebrae):
    vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

for node in vertebra_nodes:
    local_directors = directors_data[-1][...,node]
    vertebra_pos = np.array([position_data[-1][0][node], position_data[-1][1][node], position_data[-1][2][node]])
    vertebra_coord_syst_x = np.array([0.0, 0.0, 1.0])
    vertebra_coord_syst_y = np.array([0.0, 1.0, 0.0])
    vertebra_coord_syst_z = np.array([0.0, 0.0, -1.0])
    vertebra_coord_syst_zz = np.array([0.0, -1.0, 0.0])

    # Scale the arrows for better visibility
    scale = 0.05

    # Plot the arrows using quiver
    ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
        vertebra_coord_syst_x[0], vertebra_coord_syst_x[1], vertebra_coord_syst_x[2],
        length=scale, color='r', normalize=True)

num_vertebrae = 10
first_vertebra_node = 2
final_vertebra_node = 30
vertebra_nodes = []
vertebra_increment = (final_vertebra_node - first_vertebra_node)/(num_vertebrae - 1)
for i in range(num_vertebrae):
    vertebra_nodes.append(round(i * vertebra_increment + first_vertebra_node))

for node in vertebra_nodes:
    local_directors = directors_data[-1][...,node]
    vertebra_pos = np.array([position_data[-1][0][node], position_data[-1][1][node], position_data[-1][2][node]])
    vertebra_coord_syst_x = np.array([0.0, 0.0, 1.0])
    vertebra_coord_syst_y = np.array([0.0, 1.0, 0.0])
    vertebra_coord_syst_z = np.array([0.0, 0.0, -1.0])
    vertebra_coord_syst_zz = np.array([0.0, -1.0, 0.0])

    # Scale the arrows for better visibility
    scale = 0.07

    ax.quiver(vertebra_pos[0], vertebra_pos[1], vertebra_pos[2],
        vertebra_coord_syst_zz[0], vertebra_coord_syst_zz[1], vertebra_coord_syst_zz[2],
        length=scale, color='g', normalize=True)

ax.view_init(elev=30, azim=20)
plt.show()
