from set_environment import Environment

# test_env.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
dT = 1e-4
def tension_function(t):
    """
    Example tension function that returns a tension value based on time.
    This can be replaced with any function that defines how the tendon tension changes over time.
    
    Returns a tension value for the four tendons (assuming 4 tendons in this example).
    """
    tension1 = 0.1*(1-np.cos(t))  # Tension for tendon 1
    tension2 = 0  # Tension for tendon 2
    tension3 = 0
    tension4 = 0
    return np.ones(4) * t # Return zero for all tendons if you want to keep them slack, otherwise return the tensions defined above.
    return np.array([tension1, tension2, tension3, tension4])  # Return tensions for all tendons
    
def test_environment():
    import numpy as np

    env = Environment(n_elem=50, mode=1, target_position=np.array([0.05, 0.05, 0.05]), time_step=dT, gravity_enable=False)
    #state = env.reset() #initializes with params

    num_steps = 100
    outputs = []  # Store the outputs for each step
    
    for step in tqdm(range(num_steps)):
        # action = np.random.uniform(0, env.max_tension, size=(4,))  # Random action
        action = tension_function(step * dT)  # Use the tension function to get the action for this step
        state, reward, done, _ = env.step(action)  # Take a step in the environment
        
        # print(f"Step {step}: Reward = {reward}, State = {state}")
        step_data = {
            "step": step,
            "action": action,
            "state": state,
            "reward": reward,
            "done": done,
        }
        outputs.append(step_data)
        if done:
            print("Episode finished")
            break
    return outputs  # Return the collected outputs for plotting or further analysis

def plot_results(outputs):
    """
    Optional function to plot the results of the simulation.
    """
    
    
    steps = [data['step'] for data in outputs]
    rewards = [data['reward'] for data in outputs]
    actions = [(data['action']) for data in outputs]
    states = [np.array(data['state']) for data in outputs]
    plt.figure(figsize=(10, 5))
    for i in range(4):  # Assuming there are 4 tendons
        tendon_tensions = [action[i] for action in actions]
        plt.plot(steps, tendon_tensions, marker='o', linestyle='-', label=f'Tendon {i+1}')
    plt.title('Tendon Tensions over Steps')
    plt.xlabel('Step')
    plt.ylabel('Tension')
    plt.legend()
    plt.grid()
    plt.show()
    # Extract the 3D positions from the states
    positions = [state[:3] for state in states]  # Assuming the first three entries are the 3D positions
    positions = np.array(positions)  # Convert to a NumPy array for easier manipulation

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', label='Trajectory')
    ax.set_title('3D Trajectory of the Object')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()
    ax.grid()
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])

    plt.show()
    
if __name__ == "__main__":
    outputs = test_environment()
    print("tested")
    plot_results(outputs)