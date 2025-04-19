import os
import numpy as np
import pickle
from stable_baselines3 import SAC
from set_environment import Environment  # your custom environment
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_gif import create_gif

def make_env():
    return Environment(
        n_elem=50,
        mode=2,  # Random target every reset
        final_time=2.0,
        gravity_enable=False
    )

def evaluate(model_path, log_dir, num_episodes=10):
    env = make_env()
    model = SAC.load(model_path, env=env)
    dT_L = env.time_step * env.num_steps_per_update

    all_outputs = []

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep+1} ===")
        obs, _ = env.reset()
        done = False
        step = 0
        outputs = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            step_data = {
                "step": step,
                "action": action,
                "state": obs,
                "reward": reward,
                "points_bb": info["position"],
                "time": step * dT_L,
                "tensions": action,
                "sphere_position": env.sphere.position_collection.copy(),
                "target_position": env.target_position.copy()
            }
            outputs.append(step_data)
            step += 1

        all_outputs.append(outputs)

    with open(os.path.join(log_dir, "all_outputs.pkl"), "wb") as f:
        pickle.dump(all_outputs, f)

    print(f"\n✅ Saved {num_episodes} evaluation episodes to all_outputs.pkl")

def plot_tension():
    with open("./sac/all_outputs.pkl", "rb") as f:
        outputs = pickle.load(f)
      # Extract the endpoint trajectory
    outputs = outputs[0]
    end_points = [data['points_bb'][:, -1] for data in outputs]  # Last point in the backbone
    end_points = np.array(end_points)  # Convert to NumPy array for easier manipulation

    # Extract time and tension values
    time_stamps = [data['time'] for data in outputs]
    tensions = [data['tensions'] for data in outputs]
    tensions = np.array(tensions)  # Convert to NumPy array

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the trajectory of the endpoint
    ax1 = axes[0]
    ax1.plot(end_points[:, 0], end_points[:, 1], label="Trajectory (XY Plane)", marker='o', linestyle='-')
    ax1.set_title("Trajectory of the Endpoint of the Backbone")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid()
    ax1.legend()

def plot():
   
    with open("./sac/all_outputs.pkl", "rb") as f:
        all_outputs = pickle.load(f)

    episode_results = []
    for episode in all_outputs:
        target = episode[-1]["target_position"]
        create_gif(episode, gif_path=f"{target}.gif")
        
        tip_final = episode[-1]["points_bb"][:, -1]
        
        distance = np.linalg.norm(tip_final - target)
        episode_results.append({"target": target, "final_tip": tip_final, "distance": distance})

    targets = np.array([ep["target"] for ep in episode_results])
    distances = np.array([ep["distance"] for ep in episode_results])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c=distances, cmap='viridis', s=100)
    ax.set_title("Performance by Target Position")
    ax.set_xlabel("Target X")
    ax.set_ylabel("Target Y")
    ax.set_zlabel("Target Z")
    plt.colorbar(sc, label="Final Distance to Target")
    plt.show()

if __name__ == "__main__":
    # evaluate(
    #     model_path="./sac/sac_case2",  # path to your saved model (no .zip needed)
    #     log_dir="./sac/",
    #     num_episodes=10
    # )
    # plot()
    plot_tension()