import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from set_environment import Environment
from stable_baselines3 import SAC

import pickle

def make_env():
    # env = Environment(
    #     final_time=10.0,                 # Episode length in seconds
    #     num_steps_per_update=7,          # Steps per action hold
    #     COLLECT_DATA_FOR_POSTPROCESSING=False,
    #     mode=4,                          # Case 1: Randomly moving target
    #     target_position=   [0.10040713, 0.0, 0.22228797]],
    #     target_v=0.5,                    
    #     gravity_enable=False,
    #     sim_dt=2.5e-4,
    #     n_elem=40,
    #     num_vertebrae=20,
    #     max_tension=5,
    #     NU=30,
    # )
    target_position = [0.10040713, 0.0, 0.22228797]
    env = Environment(n_elem=50, mode=1, final_time= 2
                      , target_position=target_position, gravity_enable=False)
  
    #return Monitor(env)  # Wrap with Monitor
    return env

if __name__ == "__main__":
    # Setup logging
    log_dir = "./ppo_case1_batch16000/"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    env = make_env()

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

    model = SAC(
        policy="MlpPolicy", #ffn as value & policy
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,     # <- this matches the best curve (green)
        batch_size=256,          # <- reasonable default
        train_freq=1, #train once per env step (?)
        gradient_steps=1, #How many samples to draw per gradient step
        learning_starts=1500, #How many steps of the model to collect transitions for before learning starts
        gamma=0.99,
        tau=0.005,
        verbose=1,
        tensorboard_log=log_dir,
    )
    #sample randomly from buffer, get corresponding reward and make the update accordingly
    

    model.set_logger(new_logger)

    model.learn(total_timesteps=int(40000)) #40k seems like a pretty good length
    model.save(os.path.join(log_dir, "ppo_case1"))

    # Evaluate rollout
    outputs = []
    obs, _ = env.reset()
    done = False
    step = 0
    dT_L = env.time_step * env.num_steps_per_update

    while not done:
        action, _states = model.predict(obs, deterministic=True)
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
        print("tip position:", info["position"][:, -1])
        if done:
            print("Episode finished")
            break
        outputs.append(step_data)
        step += 1

    # Save rollout
    with open(os.path.join(log_dir, "outputs.pkl"), "wb") as f:
        pickle.dump(outputs, f)