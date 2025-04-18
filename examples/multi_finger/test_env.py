from HANDS.Environments import MultipleFinger
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

def test_environment():
    num_fingers = 4
    total_time = 5  # Total time for the simulation
    env = MultipleFinger(final_time= total_time, num_fingers=num_fingers, finger_radius= 0.1, gravity=False)
    dT_L = env.time_step*env.num_steps_per_update # The effective time step for the tension function
    
    state = env.reset() #initializes with params

    num_steps = env.total_learning_steps
    outputs = []  # Store the outputs for each step
    
    # Initialize the PID controller with gains

    
    for step in tqdm(range(num_steps)):
        # action = np.random.uniform(0, env.max_tension, size=(4,))  # Random action
        # action = tension_function(step * dT_L, state)  # Use the tension function to get the action for this step
        point_to_go = [point_fn(step*dT_L, total_time)] * num_fingers
        state, reward, done, additional_info = env.step(point_to_go)  # Take a step in the environment
        
        # print(f"Step {step}: Reward = {reward}, State = {state}")
        step_data = {
            "step": step,
            "action": point_to_go,
            "state": state,
            "reward": reward,
            "done": done,
            "time": step * dT_L,
            "num_fingers": num_fingers,
        }
        outputs.append(step_data)
        if done:
            print("Episode finished")
            break
        # if step*dT_L > 1.95:
        #     print(state)
        
    #print(outputs)
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
    plot_results(outputs)
    # save outputs to pickle file
    
    with open('outputs2.pkl', 'wb') as f:
        pickle.dump(outputs, f)
        