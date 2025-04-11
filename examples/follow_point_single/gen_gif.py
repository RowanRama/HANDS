''
# This script generates a GIF animation of the backbone points for each instant
# and plots the trajectory of the endpoint of the backbone and the tension values over time.
# It uses the outputs from a simulation stored in a pickle file.
# The GIF shows two views: a 3D view and a top-down view (-z axis).
''
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_outputs(file_path):
    """
    Load the outputs from the pickle file.
    """
    with open(file_path, 'rb') as f:
        outputs = pickle.load(f)
    return outputs

def create_gif(outputs, gif_path="backbone_animation_follow_sphere.gif"):
    """
    Create a GIF of the backbone points for each instant with two views.
    """
    simulation_duration = outputs[-1]['time']

    

    # Extract the backbone points for each step
    backbone_points = [data['points_bb'] for data in outputs]
    # Extract the time stamps for the frames
    time_stamps = [data['time'] for data in outputs]

    # Calculate the total number of frames for the GIF
    fps = 10  # Frames per second for the GIF
    total_frames = int(simulation_duration * fps)
    end_points = [data['points_bb'][:, -1] for data in outputs]  # Last point in the backbone
    end_points = np.array(end_points)

    # Sample frames to match the desired duration
    frame_indices = np.linspace(0, len(backbone_points) - 1, total_frames, dtype=int)
    backbone_points = [backbone_points[i] for i in frame_indices] #3xn matrix, xyz for each point
    if "sphere_position" not in outputs[0]:
        print("Warning: 'sphere_position' not found in outputs.")
        sphere_points = np.zeros_like(end_points)  # just use dummy placeholder
    else:
        sphere_points = [data['sphere_position'][:, 0] for data in outputs]  # Only node 0
        sphere_points = np.array(sphere_points)
    # sphere_positions = [data['sphere_position'][:, 0] for data in outputs] # sphere is a rigid body with one node at index 0.
    sphere_positions = [sphere_points[i] for i in frame_indices]
    time_stamps = [time_stamps[i] for i in frame_indices]

    #print("backbone POSITIONS", backbone_points)

    # Set up the figure and 3D axes
    fig = plt.figure(figsize=(12, 6))

    # First subplot: 3D view
    ax1 = fig.add_subplot(121, projection='3d')

    ax1.set_xlim([-0.9, 0.9])
    ax1.set_ylim([-0.9, 0.9])
    ax1.set_zlim([0, 0.9])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D View')

    # Second subplot: Top-down view (-z axis)
    ax2 = fig.add_subplot(122)
    ax2.set_xlim([-0.9, 0.9])
    ax2.set_ylim([-0.9, 0.9])
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Top-Down View (-Z Axis)')

    # Initialize the line objects
    line1, = ax1.plot([], [], [], marker='o', linestyle='-', alpha=0.7)
    line2, = ax2.plot([], [], marker='o', linestyle='-', alpha=0.7)
    sphere_plot = ax1.scatter([], [], [], color='red', s=50, label="Sphere")


    def update(frame):
        """
        Update function for the animation.
        """
        points = backbone_points[frame]

        # Update 3D view
        line1.set_data(points[0, :], points[1, :])
        line1.set_3d_properties(points[2, :])
        ax1.set_title(f"3D View - Time: {time_stamps[frame]:.2f} s")

        # Update top-down view
        line2.set_data(points[0, :], points[1, :])
        ax2.set_title(f"Top-Down View - Time: {time_stamps[frame]:.2f} s")

        #update sphere position
        sphere_pos = sphere_positions[frame]
        sphere_plot._offsets3d = ([sphere_pos[0]], [sphere_pos[1]], [sphere_pos[2]])


        return line1, line2, sphere_plot

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(backbone_points), interval=1000 / fps, blit=False)
  

    # Save the animation as a GIF
    ani.save(gif_path, writer='imagemagick', fps=fps)
    print(f"GIF saved to {gif_path}")

def plot_trajectory_and_tensions(outputs):
    """
    Plot the trajectory of the endpoint of the backbone and the target sphere,
    along with tension values over time.
    """
    # Extract the endpoint trajectory
    end_points = [data['points_bb'][:, -1] for data in outputs]  # Last point in the backbone
    end_points = np.array(end_points)
    if "sphere_position" not in outputs[0]:
        print("Warning: 'sphere_position' not found in outputs.")
        sphere_points = np.zeros_like(end_points)  # just use dummy placeholder
    else:
        sphere_points = [data['sphere_position'][:, 0] for data in outputs]  # Only node 0
        sphere_points = np.array(sphere_points)

    # Extract time and tension values
    time_stamps = [data['time'] for data in outputs]
    tensions = [data['tensions'] for data in outputs]
    tensions = np.array(tensions)

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the trajectory of the endpoint and sphere
    #print(sphere_points)
    for i, data in enumerate(outputs[:10]):
        print(f"Step {i}: tensions = {data['tensions']}, tip = {data['points_bb'][:, -1]}") #only moving in z
    

    ax1 = axes[0]
    ax1.plot(end_points[:, 0], end_points[:, 1], label="Rod Tip Trajectory", marker='o', linestyle='-')
    ax1.plot(sphere_points[:, 0], sphere_points[:, 1], label="Sphere Trajectory", marker='x', linestyle='--', alpha=0.7)
    ax1.set_title("XY Trajectory: Rod Tip vs Sphere")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid()
    ax1.legend()

    # Plot the tension values over time
    ax2 = axes[1]
    for i in range(tensions.shape[1]):  # Assuming 4 tendons
        ax2.plot(time_stamps, tensions[:, i], label=f"Tendon {i+1}")
    ax2.set_title("Tension Values Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tension")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()

    tip_positions = [data['points_bb'][:, -1] for data in outputs]
    tip_positions = np.array(tip_positions)
    plt.plot(tip_positions[:, 0], tip_positions[:, 1])  # XY trajectory
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rod Tip Trajectory (XY)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Load the outputs from the pickle file
    outputs = load_outputs("ppo_case1_rollout.pkl")

    # Create and save the GIF
    create_gif(outputs)
    plot_trajectory_and_tensions(outputs)