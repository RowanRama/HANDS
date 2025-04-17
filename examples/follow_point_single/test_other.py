import numpy as np
import pickle
from set_environment import Environment
from stable_baselines3 import SAC

# Load your trained model
model = SAC.load("./ppo_case1_batch16000/ppo_case1")

# List of new test targets
test_targets = [
    [0.05, 0.0, 0.22],
    [0.1, 0.1, 0.22],
    [-0.1, 0.0, 0.22],
    [0.0, -0.1, 0.2],
    [0.0, 0.0, 0.25],  # top center
]

results = {}

for idx, target in enumerate(test_targets):
    print(f"\Testing on target {target}")
    
    env = Environment(
        n_elem=50,
        mode=1,
        final_time=2.0,
        target_position=np.array(target),
        gravity_enable=False,
    )

    obs, _ = env.reset()
    done = False
    step = 0
    dT_L = env.time_step * env.num_steps_per_update
    outputs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        outputs.append({
            "step": step,
            "time": step * dT_L,
            "action": action,
            "reward": reward,
            "tip_pos": info["position"][:, -1],
            "sphere_pos": env.sphere.position_collection.copy()
        })

        step += 1

    results[str(target)] = outputs

# Optional: save all results
with open("evaluation_outputs.pkl", "wb") as f:
    pickle.dump(results, f)
