from HANDS.Environments import MultipleFinger, HierarchicalFingerEnv
from HANDS.env_helpers import reshape_state

# test_env.py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# dT = 1.5e-5 # timestep has to be 1.5e-5 for the simulation to work
# minimum timestep can be calculated using this:
# dtmax = (base_length/n_elements) * np.sqrt(density / max(youngs_modulus, shear_modulus)) # Maximum size of time step
import pickle


'''
[ cos(θ)  -sin(θ)  0 ]
[ sin(θ)   cos(θ)  0 ]
[   0        0     1 ]

but in this cylinder's case z-axis is along worls Y, x-axis is along world Z. So it rotates along world z, which is cylinder's x-axis
project Z (world) to XY (world), 
'''

def compute_reward(info, env, target_angle=None):
    reward = 0.0
    
    if "cylinder_director" in info:
        directors = info["cylinder_director"][2, ...]
        print(f"Directors: {directors}")

        current_x = np.reshape(directors, (3,))  # Normal vector (z world, x cylinder)
        #project to XY (world) which is YZ (cylinder)
        current_angle = np.arctan2(current_x[1], current_x[0])  # Project to YZ plane

        print(f"Angle about Z in rad: {(current_angle)}")
        print(f"current_angle: {np.degrees(current_angle):.2f}°")
        angle_diff = current_angle - target_angle if target_angle is not None else 0.0
        cos_angle_diff = np.cos(current_angle)  # Cosine of the angle differe
        reward += 0.5 * cos_angle_diff  # Reward for aligning with the target angle
        return reward
      

    # # Add distance penalty for each finger
    # for i, finger in enumerate(env.base_env.fingers):
    #     tip = finger.rod.position_collection[:, -1]
    #     dist = np.linalg.norm(tip - env.current_goals[i])
    #     reward -= 0.5 * dist  # Distance penalty coefficient
        
    # return reward  # Make sure to return the reward!

# Create the env with proper reward function
env = HierarchicalFingerEnv(
    base_env=MultipleFinger(
        cylinder_enabled=True,
        cylin_params = {
                "length": 0.2,
                "direction": np.array([0.0, 1.0, 0.0]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "radius": 0.002,
                "start_pos": np.array([0.0, 0.0, 0.0]),
                "k": 1e4,
                "nu": 10,
                "density": 1000,
            }
    ),
    reward_func=lambda info: compute_reward(info, env)  # Pass both info and env
)

obs = env.reset()

def create_env():
    num_fingers = 4
    total_time = 5  # Total time for the simulation
    env = MultipleFinger(final_time= total_time, num_fingers=num_fingers, finger_radius= 0.1, gravity=False, cylinder_enabled=True)
    return env

def step_time():
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
        #print(additional_info)
        
        # print(f"Step {step}: Reward = {reward}, State = {state}")
        step_data = {
            "step": step,
            "action": point_to_go,
            "state": state,
            "reward": reward,
            "done": done,
            "time": step * dT_L,
            "num_fingers": num_fingers,
            "cylinder_position": additional_info["cylinder_position"],
            "cylinder_director": additional_info["cylinder_director"]
        }
        outputs.append(step_data)
        if done:
            print("Episode finished")
            break
        # if step*dT_L > 1.95:
        #     print(state)
        
    #print(outputs)
    return outputs 
