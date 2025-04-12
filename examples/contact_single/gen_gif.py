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

def create_gif(outputs, gif_path="backbone_animation.gif"):
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

    # Sample frames to match the desired duration
    frame_indices = np.linspace(0, len(backbone_points) - 1, total_frames, dtype=int)
    backbone_points = [backbone_points[i] for i in frame_indices]
    time_stamps = [time_stamps[i] for i in frame_indices]

    # Set up the figure and 3D axes
    fig = plt.figure(figsize=(12, 6))

    # First subplot: 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([-0.25, 0.25])
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_zlim([0, 0.25])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D View')

    # Second subplot: Top-down view (-z axis)
    ax2 = fig.add_subplot(122)
    ax2.set_xlim([-0.25, 0.25])
    ax2.set_ylim([-0.25, 0.25])
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Top-Down View (-Z Axis)')

    # Initialize the line objects
    line1, = ax1.plot([], [], [], marker='o', linestyle='-', alpha=0.7)
    line2, = ax2.plot([], [], marker='o', linestyle='-', alpha=0.7)

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

        return line1, line2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(backbone_points), interval=1000 / fps, blit=True)

    # Save the animation as a GIF
    ani.save(gif_path, writer='imagemagick', fps=fps)
    print(f"GIF saved to {gif_path}")

def plot_trajectory_and_tensions(outputs):
    """
    Plot the trajectory of the endpoint of the backbone and the tension values over time.
    """
    # Extract the endpoint trajectory
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

if __name__ == "__main__":
    # Load the outputs from the pickle file
    outputs = load_outputs("outputs2.pkl")

    # Create and save the GIF
    create_gif(outputs)
    plot_trajectory_and_tensions(outputs)