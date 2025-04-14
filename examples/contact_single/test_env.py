from set_environment import Environment

# test_env.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# dT = 1.5e-5 # timestep has to be 1.5e-5 for the simulation to work
# minimum timestep can be calculated using this:
# dtmax = (base_length/n_elements) * np.sqrt(density / max(youngs_modulus, shear_modulus)) # Maximum size of time step
import pickle

class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits=(None, None)):
        """
        Initialize the PID controller.
        
        Parameters:
        - Kp: Proportional gain
        - Ki: Integral gain
        - Kd: Derivative gain
        - output_limits: Tuple (min, max) to limit the output value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits

        # Internal variables
        self._prev_error = 0
        self._integral = 0
        self._last_time = None

    def compute(self, error, current_time):
        """
        Compute the PID output based on the error.
        
        Parameters:
        - error: The difference between the desired and current value
        - current_time: The current time (used for calculating the derivative term)
        
        Returns:
        - PID output
        """
        dt = current_time - self._last_time if self._last_time is not None else 0

        # Proportional term
        P = self.Kp * error

        # Integral term
        self._integral += error * dt
        I = self.Ki * self._integral

        # Derivative term
        D = 0
        if dt > 0:
            D = self.Kd * (error - self._prev_error) / dt

        # Compute the output
        output = P + I + D

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        # Update internal state
        self._prev_error = error
        self._last_time = current_time

        return output



# List of points to go in X-Y plane
points_to_go = np.array([
    #[0.1, 0.1],
    [0.2, 0.2],
    [0.0, 0.2],
    [-0.1, 0.2],
    [-0.2, 0.2],
    [-0.2, 0.1],
    [-0.2, 0.0],
    [-0.2, -0.1],
    [-0.2, -0.2],
    [-0.1, -0.2],
    [0.0, -0.2],
    [0.1, -0.2],
    [0.2, -0.2],
    [0.2, -0.1],
    [0.2, 0.0]])/2

pid_x = PIDController(Kp=0., Ki=50.0, Kd=0., output_limits=(-20, 20))
pid_y = PIDController(Kp=0., Ki=50.0, Kd=0., output_limits=(-20, 20))

def tension_function(t, state):
    """
    Example tension function that returns a tension value based on time.
    This can be replaced with any function that defines how the tendon tension changes over time.
    
    Returns a tension value for the four tendons (assuming 4 tendons in this example).
    """
    tension1 = 15.0 * (1 - np.cos(t*0.2))  # Tension for tendon 1, now oscillates between 0 and 4
    tension2 = 0  # Tension for tendon 2
    tension3 = 0
    tension4 = 0
    # print(f"Tension at time {t}: {tension1}, {tension2}, {tension3}, {tension4}")
    
    tension1 = np.sin(t*np.pi) 
    tension2 = np.cos(t*np.pi/2) 
    tension3 = (1-np.sin(t*np.pi))  
    tension4 = (1-np.cos(t*np.pi/2)) 
    
    # Apply tensions to each tendon one by one. For 0 to 2 seconds, keep tension1 active, then switch to tension2, and so on.
    # Switching every 2 seconds
    
    # Controller to verify that negative tension does not move anything.
    # if t < 2:
    #     tension[0] = -2*np.sin(t*np.pi)  # Tension for tendon 1
    # elif t < 4:
    #     tension[1] = 2*np.sin(t*np.pi)  # Tension for tendon 2
    # elif t < 6:
    #     tension[2] = 2*np.sin(t*np.pi)
    # elif t < 8:
    #     tension[3] = 2*np.sin(t*np.pi)

    point_to_go = points_to_go[int(t//5.0) % len(points_to_go)] # Get the point to go based on time
    # # Controller v1.0 for 4 tendons
    # tx = point_to_go[0]*20
    # ty = point_to_go[1]*20
    # Observation: Open loop control moves it the correct direction, but the amplitude is not correct.

    # Controller v1.1 for 4 tendons - Closed loop using PI controller
    current_x = state[10]
    current_y = state[21]
    # Compute errors
    error_x = point_to_go[0] - current_x
    error_y = point_to_go[1] - current_y
    # Compute PID outputs for x and y directions
    tx = pid_x.compute(error_x, t)
    ty = pid_y.compute(error_y, t)
    # For Kp = 3, Ki = 50, Kd = 0, the response if fairly quick and the steady state error converges to zero in around 1.5 secs.
    
    tension = np.zeros(4)
    tension[0], tension[1] = (tx, 0) if tx > 0 else (0, -tx)
    tension[2], tension[3] = (ty, 0) if ty > 0 else (0, -ty)
    return tension  # Return tensions for all tendons



def test_environment():
    n_elem = 50
    # tendon_config = {
    # "directions": [
    #     np.array([np.cos(angle), np.sin(angle), 0.0])
    #     for angle in np.linspace(0, 2*np.pi, 6, endpoint=False)
    # ],
    # "final_vertebra_node": [n_elem-2, n_elem//2-1, n_elem-2, n_elem//2-1, n_elem-2, n_elem//2-1],
    # }

    env = Environment(n_elem=n_elem, mode=1, final_time= 3, target_position=np.array([0.5, 0.5, 0.5]), gravity_enable=False)
    dT_L = env.time_step*env.num_steps_per_update # The effective time step for the tension function
    
    state = env.reset() #initializes with params

    num_steps = env.total_learning_steps
    outputs = []  # Store the outputs for each step
    
    # Initialize the PID controller with gains

    
    for step in tqdm(range(num_steps)):
        # action = np.random.uniform(0, env.max_tension, size=(4,))  # Random action
        action = tension_function(step * dT_L, state)  # Use the tension function to get the action for this step
        state, reward, done, additional_info = env.step(action)  # Take a step in the environment
        
        # print(f"Step {step}: Reward = {reward}, State = {state}")
        step_data = {
            "step": step,
            "action": action,
            "state": state,
            "reward": reward,
            "done": done,
            "points_bb": additional_info["position"],
            "time": step * dT_L,
            "tensions": action
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
    
    # Print the size of points_bb
    if not outputs:
        print("No data to plot.")
        return
    if 'points_bb' in outputs[0]:
        print("points_bb found in outputs, size of points_bb:", len(outputs[0]['points_bb']))
    else:
        print("points_bb not found in outputs, cannot plot.")
        return
    steps = [data['step'] for data in outputs]
    rewards = [data['reward'] for data in outputs]
    actions = [(data['action']) for data in outputs]
    states = [np.array(data['state']) for data in outputs]
    # plt.figure(figsize=(10, 5))
    # for i in range(4):  # Assuming there are 4 tendons
    #     tendon_tensions = [action[i] for action in actions]
    #     plt.plot(steps, tendon_tensions, marker='o', linestyle='-', label=f'Tendon {i+1}')
    # plt.title('Tendon Tensions over Steps')
    # plt.xlabel('Step')
    # plt.ylabel('Tension')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # # Extract the 3D positions from the states
    # positions = [state[:3] for state in states]  # Assuming the first three entries are the 3D positions
    # positions = np.array(positions)  # Convert to a NumPy array for easier manipulation

    # # Create a 3D plot
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', label='Trajectory')
    # ax.set_title('3D Trajectory of the Object')
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_zlabel('Z Position')
    # ax.legend()
    # ax.grid()
    # # ax.set_xlim([-0.1, 0.1])
    # # ax.set_ylim([-0.1, 0.1])
    # # ax.set_zlim([-0.1, 0.1])

    # plt.show()
    
    # Plot the backbone of the Cosserat rod using points_bb for the first and final position in two subplots
    fig = plt.figure(figsize=(12, 6))

    base_length = 0.25 # Assuming the base length of the rod is 0.25 m

    # First position
    ax1 = fig.add_subplot(121, projection='3d')
    first_points_bb = outputs[0]['points_bb']  # Extract the position_collection for the first step
    ax1.plot(first_points_bb[0, :], first_points_bb[1, :], first_points_bb[2, :], marker='o', linestyle='-', alpha=0.7)
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
    final_points_bb = outputs[-1]['points_bb']  # Extract the position_collection for the final step
    ax2.plot(final_points_bb[0, :], final_points_bb[1, :], final_points_bb[2, :], marker='o', linestyle='-', alpha=0.7)
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
        