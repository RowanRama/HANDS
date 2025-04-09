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

def create_gif(outputs, gif_path="backbone_animation.gif", simulation_duration=2.0):
    """
    Create a GIF of the backbone points for each instant.
    """
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

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits (adjust based on your data)
    base_length = 0.25  # Assuming the base length of the rod is 0.25 m
    ax.set_xlim([-base_length, base_length])
    ax.set_ylim([-base_length, base_length])
    ax.set_zlim([0, base_length])

    # Labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Initialize the line object
    line, = ax.plot([], [], [], marker='o', linestyle='-', alpha=0.7)

    def update(frame):
        """
        Update function for the animation.
        """
        points = backbone_points[frame]
        line.set_data(points[0, :], points[1, :])
        line.set_3d_properties(points[2, :])
        ax.set_title(f"Time: {time_stamps[frame]:.2f} s")
        return line,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(backbone_points), interval=1000 / fps, blit=True)

    # Save the animation as a GIF
    ani.save(gif_path, writer='imagemagick', fps=fps)
    print(f"GIF saved to {gif_path}")

if __name__ == "__main__":
    # Load the outputs from the pickle file
    outputs = load_outputs("outputs.pkl")

    # Create and save the GIF
    create_gif(outputs, simulation_duration=2.0)  # Adjust simulation_duration as needed