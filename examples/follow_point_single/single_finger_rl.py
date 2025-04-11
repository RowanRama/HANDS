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
    env = Environment(
        final_time=10.0,                 # Episode length in seconds
        num_steps_per_update=7,          # Steps per action hold
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        mode=4,                          # Case 1: Randomly moving target
        target_position=[0.5, 0.5, 0.5],
        target_v=0.5,                    
        gravity_enable=False,
        sim_dt=2.5e-4,
        n_elem=40,
        num_vertebrae=20,
        max_tension=5,
        NU=30,
    )
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

    model.learn(total_timesteps=int(1000000))
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
        outputs.append(step_data)
        step += 1

    # Save rollout
    with open(os.path.join(log_dir, "ppo_case1_rollout.pkl"), "wb") as f:
        pickle.dump(outputs, f)
