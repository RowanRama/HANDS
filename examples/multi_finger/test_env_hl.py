from HANDS.Environments import HLControlEnv
from HANDS.env_helpers import reshape_state

# test_env.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# dT = 1.5e-5 # timestep has to be 1.5e-5 for the simulation to work
# minimum timestep can be calculated using this:
# dtmax = (base_length/n_elements) * np.sqrt(density / max(youngs_modulus, shear_modulus)) # Maximum size of time step
import pickle

# List of points to go in X-Y plane
points_to_go = np.array([
    #[0.1, 0.1],
    [0.2, 0.2, 0.0],
    [0.0, 0.2, 0.0],
    [-0.1, 0.2, 0.0],
    [-0.2, 0.2, 0.0],
    [-0.2, 0.1, 0.0],
    [-0.2, 0.0, 0.0],
    [-0.2, -0.1, 0.0],
    [-0.2, -0.2, 0.0],
    [-0.1, -0.2, 0.0],
    [0.0, -0.2, 0.0],
    [0.1, -0.2, 0.0],
    [0.2, -0.2, 0.0],
    [0.2, -0.1, 0.0],
    [0.2, 0.0, 0.0]])/2
# initial_point = np.array([0.1,0, 0,0.1,0, -0.1,0,0, -0.1,0,0])

            
l = 0.1 # Length of the cylinder
r = 0.005 # Radius of the cylinder
def gpc(theta, l = 0.8*l, r = 1.3*r):
    # restrict the input range to 0 to pi/2
    v1 = np.array([np.cos(theta), np.sin(theta), 0.0])*l/2
    v2 = np.array([np.sin(theta), -np.cos(theta), 0.0])*r
    pp1 = v1 + v2
    pp2 = v1 - v2
    pp3 = -v1 + v2
    pp4 = -v1 - v2
    return [pp1, pp2, pp3, pp4]
def gpc2(theta, l = 0.8*l, r = 1.3*r):
    # restrict the input range to 0 to -pi/2
    v1 = np.array([np.cos(theta), np.sin(theta), 0.0])*l/2
    v2 = np.array([np.sin(theta), -np.cos(theta), 0.0])*r
    pp1 = v1 + v2
    pp2 = v1 - v2
    pp3 = -v1 + v2
    pp4 = -v1 - v2
    return [pp2, pp3, pp4, pp1]
# initial_point = [
#     [np.array([0.1, 0.0, 0.0]),    np.array([0.03, 0.03, 0.0]),     np.array([-0.1, 0.0, 0.0]), np.array([0.0, -0.1, 0.0])],
#     [np.array([0.02, -0.02, 0.0]),     np.array([0.07, 0.0, 0.0]),      np.array([-0.1, -0.0, 0.0]), np.array([-0., -0.1, 0.0])],
#     [np.array([0.1, 0.0, 0.0]),     np.array([0.0, 0.1, 0.0]),      np.array([-0.1, 0.0, 0.0]), np.array([0., -0.1, 0.0])],
# ]
# for i in range(len(initial_point)):
#     for j in range(len(initial_point[i])):
#         if j>=2:
#             initial_point[i][j] = -np.array(initial_point[i][j-2]) # to make it symmetric
original_state =   [np.array([0.1, 0.0, 0.0]),    np.array([0.0, 0.1, 0.0]),     np.array([-0.1, 0.0, 0.0]), np.array([0.0, -0.1, 0.0])]
original_state =   [np.array([0.1, 0.0, 0.0])*0.5,    np.array([0.0, 0.1, 0.0])*0.5,     np.array([-0.1, 0.0, 0.0])*0.5, np.array([0.0, -0.1, 0.0])*0.5]
initial_point = [gpc(np.pi/4), gpc(0), gpc2(-np.pi/2), gpc(np.pi/2), gpc(0), original_state] # 4 points in the X-Y plane
initial_point = [gpc(np.pi/2), gpc(0), original_state, gpc2(0), gpc2(-np.pi/2), original_state] # 4 points in the X-Y plane
def point_fn(time, total_time):
    """
    Function to determine the point to go based on time.
    This function can be modified to implement different trajectories.
    """
    # Example: Move in a circular trajectory
    radius = 0.15
    angle = time * 2 * np.pi / total_time  # Complete one circle in 5 seconds
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.array([x, y, 0.0])  # Return the point in X-Y plane with Z=0.0

def reward_function(state, action, info, target_angle):
    return 0.0

def done_function(state, action, info):
    return False

def test_environment():
    num_fingers = 4
    total_time = 10  # Total time for the simulation
    time_step = 1.5e-5 # timestep has to be 1.5e-5 for the simulation to work
    steps_per_tension_update = 100  # Number of steps per tension update
    controller_steps_per_convergence = 200
    num_steps = int(total_time/(time_step*steps_per_tension_update*controller_steps_per_convergence)) # Number of steps in the simulation
    cylinder_enabled = True
    
    env = HLControlEnv(
        reward_function, 
        convergence_steps=controller_steps_per_convergence,
        final_time= total_time,
        num_fingers=num_fingers, 
        num_steps_per_update=steps_per_tension_update,
        finger_radius= 0.1, 
        gravity=False, 
        cylinder_enabled=cylinder_enabled,
        cylin_params = {
                "length": l,
                "direction": np.array([0.0, 1.0, 0.0]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "radius": r,
                "start_pos": np.array([0,-l/2, 0.2]),
                "k": 1e4,
                "nu": 1e4, # Cylinder's damping coefficient
                "density": 1000,
            },)
    
    state = env.reset() #initializes with params

    outputs = []  # Store the outputs for each step
    
    # Initialize the PID contaroller with gains

    
    for hl_step in tqdm(range(num_steps)):

        point_to_go = initial_point[hl_step % len(initial_point)]  # Get the point to go based on the current step
        # print("point_to_go", point_to_go)
        state, reward, done, _, additional_info = env.step(point_to_go)  # Take a step in the environment
    
        outputs.extend(additional_info["data"])
        if done:
            print("Episode finished")
            break
        
    # print(outputs)
    return outputs  # Return the collected outputs for plotting or further analysis

def plot_results(outputs):
    """
    Optional function to plot the results of the simulation.
    """
    
    steps = [data['step'] for data in outputs]
    rewards = [data['reward'] for data in outputs]
    actions = [(data['action']) for data in outputs]
    states = [np.array(data['state']) for data in outputs]
    num_fingers = outputs[0]['num_fingers']

    new_state = np.array([reshape_state(state, num_fingers) for state in states])
    
    # Plot the backbone of the Cosserat rod using points_bb for the first and final position in two subplots
    fig = plt.figure(figsize=(12, 6))

    base_length = 0.25 # Assuming the base length of the rod is 0.25 m

    # First position
    ax1 = fig.add_subplot(121, projection='3d')
    first_points_bb = new_state[0]  # Extract the position_collection for the first step
    for i in range(len(first_points_bb)):
        ax1.plot(first_points_bb[i, :, 0], first_points_bb[i, :, 1], first_points_bb[i, :, 2], marker='o', linestyle='-', alpha=0.7)
    ax1.set_title('Backbone of the Cosserat Rod (First Position)')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.axes.set_zlim3d(bottom=0,top=base_length)
    ax1.axes.set_ylim3d(bottom=-base_length,top=base_length)
    ax1.axes.set_xlim(-base_length,base_length)
    ax1.grid()

    # Final position
    ax2 = fig.add_subplot(122, projection='3d')
    final_points_bb = new_state[-1]  # Extract the position_collection for the first step
    for i in range(len(final_points_bb)):
        ax2.plot(final_points_bb[i, :, 0], final_points_bb[i, :, 1], final_points_bb[i, :, 2], marker='o', linestyle='-', alpha=0.7)
    ax2.set_title('Backbone of the Cosserat Rod (Final Position)')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position')
    ax2.axes.set_zlim3d(bottom=0,top=base_length)
    ax2.axes.set_ylim3d(bottom=-base_length,top=base_length)
    ax2.axes.set_xlim(-base_length,base_length)
    ax2.grid()

    plt.tight_layout()
    plt.show()
    
    # Plot for the entire backbone of the Cosserat rod over all steps
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # for data in outputs:
    #     points_bb = data['points_bb']  # Extract the position_collection
    #     ax.plot(points_bb[0, :], points_bb[1, :], points_bb[2, :], marker='o', linestyle='-', alpha=0.7)

    # ax.set_title('Backbone of the Cosserat Rod')
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_zlabel('Z Position')
    # ax.grid()

    # plt.show()
    
if __name__ == "__main__":
    outputs = test_environment()
    print("tested")
    # plot_results(outputs)
    # save outputs to pickle file
    
    with open('multi_hl.pkl', 'wb') as f:
        pickle.dump(outputs, f)
        