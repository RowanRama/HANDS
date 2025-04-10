import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from set_environment import Environment

def make_env():
    env = Environment(
        final_time=10.0,
        num_steps_per_update=7,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        mode=4,  # Case 1: Randomly moving target
        target_position=[0.5, 0.5, 0.5],
        target_v=0.5,
        gravity_enable=False,
        sim_dt=2e-4,
        n_elem=20,
        NU=30,
        E=1e7,
    )
    return env

if __name__ == "__main__":
    #monitor environment
    log_dir = "./ppo_case1/"
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()
    # env = Monitor(env, log_dir)
    env = Monitor(env)

    #only ppo for now
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,  
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=500_000)

    model.save("ppo_case1")

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

# Save the rollout
import pickle
with open("ppo_case1_rollout.pkl", "wb") as f:
    pickle.dump(outputs, f)
