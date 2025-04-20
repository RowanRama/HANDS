from HANDS.Environments import HLControlEnv
from HANDS.env_helpers import reshape_state
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
# test_env.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# dT = 1.5e-5 # timestep has to be 1.5e-5 for the simulation to work
# minimum timestep can be calculated using this:
# dtmax = (base_length/n_elements) * np.sqrt(density / max(youngs_modulus, shear_modulus)) # Maximum size of time step
import pickle
import os

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

def reward_function(state, action, info, target_orientation):
    directors = info["cylinder_director"][2, ...]
   # print(f"Directors: {directors}")

    current_x = np.reshape(directors, (3,))  # Normal vector (z world, x cylinder)
    #project to XY (world) which is YZ (cylinder)
    current_angle = np.arctan2(current_x[1], current_x[0])  # Project to YZ plane

    print(f"Angle about Z in rad: {(current_angle)}")
    print(f"current_angle: {np.degrees(current_angle):.2f}°")
    print("target_angle: ", target_orientation)


    diff = current_angle - target_orientation
    reward = -((diff + np.pi) % (2*np.pi) - np.pi) ** 2



    print(f"Angle difference: {diff:.2f}")
    print(f"Reward: {reward:.2f}")
    return reward


def make_env():
    target_angle = 30  # Target orientation in world coordinates
    num_fingers = 4
    total_time = 5  # Total time for the simulation
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
        target_angle=30,
        cylinder_enabled=cylinder_enabled)
    return env
    

def done_function(state, action, info):
    return False


def get_model(env):

    log_dir = "./ppo_case1_batch16000/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=500000,     # <- this matches the best curve (green)
        batch_size=256,          # <- reasonable default
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        verbose=1,
        tensorboard_log=log_dir,
    )
    

    model.set_logger(new_logger)

    model.learn(total_timesteps=int(5000))
    model.save(os.path.join(log_dir, "sac"))
    return model

def test_environment():
    env = make_env()
    state = env.reset() #initializes with param
    model = get_model(env)

    outputs = []  # Store the outputs for each step


    # Evaluate rollout
    outputs = []
    obs, _ = env.reset()
    done = False
    step = 0
    dT_L = env.time_step * env.num_steps_per_update

    while not done:
        action, _states = model.predict(obs, deterministic=True) #action space should be 4, 2
        obs, reward, done, _, info = env.step(action)

        step_data = {
            "step": step,
            "action": action,
            "state": obs,
            "reward": reward,
            "points_bb": info["position"],
            "time": step * dT_L,
            "tensions": action,
            "sphere_position": env.sphere.position_collection.copy()
        }
        outputs.append(step_data)
        step += 1



            
        print(outputs)
        return outputs 

    # Save rollout
    with open(os.path.join(log_dir, "sac_hl.pkl"), "wb") as f:
        pickle.dump(outputs, f)







    # # Initialize the PID controller with gains 
    # for hl_step in tqdm(range(env.num_steps)):

    #     point_to_go = [points_to_go[hl_step%len(points_to_go)]] * env.num_fingers
        
    #     state, reward, done, additional_info = env.step(point_to_go)  # Take a step in the environment
    
    #     outputs.extend(additional_info["data"])
    #     if done:
    #         print("Episode finished")
    #         break
        
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
    # Setup logging
    log_dir = "./ppo_case1_batch16000/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    env = make_env()
    test_environment()

    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
    #     n_steps=16000,        # <- match paper
    #     batch_size=4000,      # <- must divide n_steps
    #     gae_lambda=0.95,
    #     gamma=0.99,
    #     ent_coef=0.01,
    #     learning_rate=3e-4,
    # )

    