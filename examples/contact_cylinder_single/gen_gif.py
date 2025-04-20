import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    # Extract cylinder parameters
    cylinder_center = [data["cylin_pos"] for data in outputs]
    cylinder_direction = [data["cylin_dir"][2, ...] for data in outputs]
    cylinder_radius = 0.02
    cylinder_length = 0.2

    # Calculate the total number of frames for the GIF
    fps = 30  # Frames per second for the GIF
    total_frames = int(simulation_duration * fps)

    # Sample frames to match the desired duration
    frame_indices = np.linspace(0, len(backbone_points) - 1, total_frames, dtype=int)
    backbone_points = [backbone_points[i] for i in frame_indices]
    time_stamps = [time_stamps[i] for i in frame_indices]
    cylinder_center = [cylinder_center[i] for i in frame_indices]
    cylinder_direction = [cylinder_direction[i] for i in frame_indices]

    # Set up the figure and 3D axes
    fig = plt.figure(figsize=(16, 12))

    # First 3D subplot: Perspective view
    ax1 = fig.add_subplot(221, projection='3d')  # Top-left
    ax1.set_xlim([-0.25, 0.25])
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_zlim([0, 0.25])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D View - Perspective')

    # Second 3D subplot: Side view
    ax2 = fig.add_subplot(222, projection='3d')  # Top-right
    ax2.set_xlim([-0.25, 0.25])
    ax2.set_ylim([-0.25, 0.25])
    ax2.set_zlim([0, 0.25])
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position')
    ax2.set_title('3D View - Side')

    # Third 3D subplot: Front view
    ax3 = fig.add_subplot(223, projection='3d')  # Bottom-left
    ax3.set_xlim([-0.25, 0.25])
    ax3.set_ylim([-0.25, 0.25])
    ax3.set_zlim([0, 0.25])
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_zlabel('Z Position')
    ax3.set_title('3D View - Front')

    # Fourth 3D subplot: Top view
    ax4 = fig.add_subplot(224, projection='3d')  # Bottom-right
    ax4.set_xlim([-0.25, 0.25])
    ax4.set_ylim([-0.25, 0.25])
    ax4.set_zlim([0, 0.25])
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('Z Position')
    ax4.set_title('3D View - Top')

    # Initialize the line objects for all subplots
    line1, = ax1.plot([], [], [], marker='o', linestyle='-', alpha=0.7)
    line2, = ax2.plot([], [], [], marker='o', linestyle='-', alpha=0.7)
    line3, = ax3.plot([], [], [], marker='o', linestyle='-', alpha=0.7)
    line4, = ax4.plot([], [], [], marker='o', linestyle='-', alpha=0.7)

    # Initialize the cylinder objects for all subplots
    cylinder_surface1 = None
    cylinder_surface2 = None
    cylinder_surface3 = None
    cylinder_surface4 = None

    def update(frame):
        """
        Update function for the animation.
        """
        nonlocal cylinder_surface1, cylinder_surface2, cylinder_surface3, cylinder_surface4

        points = backbone_points[frame]
        cyl_center = np.reshape(cylinder_center[frame], (3,))  # Ensure it's a 1D array
        cyl_dir = cylinder_direction[frame]

        # Update 3D view in the first subplot (Perspective)
        line1.set_data(points[0, :], points[1, :])
        line1.set_3d_properties(points[2, :])
        ax1.set_title(f"3D View - Perspective - Time: {time_stamps[frame]:.2f} s")

        if cylinder_surface1 is not None:
            cylinder_surface1.remove()

        # Create the cylinder for the first subplot
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(-cylinder_length / 2, cylinder_length / 2, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cylinder_radius * np.cos(theta_grid)
        y_grid = cylinder_radius * np.sin(theta_grid)

        # Rotate and translate the cylinder
        cyl_dir = np.array(cyl_dir).flatten()  # Flatten to ensure it's a 1D array
        cyl_dir = cyl_dir / np.linalg.norm(cyl_dir)  # Normalize direction

        # Compute the rotation matrix to align the z-axis with cyl_dir
        z_axis = np.array([0, 0, 1])  # Default z-axis
        rotation_vector = np.cross(z_axis, cyl_dir)  # Axis of rotation
        rotation_angle = np.arccos(np.dot(z_axis, cyl_dir))  # Angle of rotation
        if np.linalg.norm(rotation_vector) > 1e-6:  # Avoid division by zero
            rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
            rotation = R.from_rotvec(rotation_angle * rotation_vector)
            rotation_matrix = rotation.as_matrix()
        else:
            rotation_matrix = np.eye(3)  # No rotation needed if cyl_dir == z_axis

        # Apply the rotation and translation
        rotated_points = np.dot(rotation_matrix, np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
        x_grid_rotated = rotated_points[0, :].reshape(x_grid.shape) + cyl_center[0]
        y_grid_rotated = rotated_points[1, :].reshape(y_grid.shape) + cyl_center[1]
        z_grid_rotated = rotated_points[2, :].reshape(z_grid.shape) + cyl_center[2]

        cylinder_surface1 = ax1.plot_surface(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.3)

        # Update 3D view in the second subplot (Side)
        line2.set_data(points[0, :], points[1, :])
        line2.set_3d_properties(points[2, :])
        ax2.set_title(f"3D View - Side - Time: {time_stamps[frame]:.2f} s")

        if cylinder_surface2 is not None:
            cylinder_surface2.remove()

        cylinder_surface2 = ax2.plot_surface(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.3)

        # Update 3D view in the third subplot (Front)
        line3.set_data(points[0, :], points[1, :])
        line3.set_3d_properties(points[2, :])
        ax3.set_title(f"3D View - Front - Time: {time_stamps[frame]:.2f} s")

        if cylinder_surface3 is not None:
            cylinder_surface3.remove()

        cylinder_surface3 = ax3.plot_surface(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.3)

        # Update 3D view in the fourth subplot (Top)
        line4.set_data(points[0, :], points[1, :])
        line4.set_3d_properties(points[2, :])
        ax4.set_title(f"3D View - Top - Time: {time_stamps[frame]:.2f} s")

        if cylinder_surface4 is not None:
            cylinder_surface4.remove()

        cylinder_surface4 = ax4.plot_surface(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.3)

        # Set different view angles for the subplots
        ax1.view_init(elev=30, azim=45)  # Perspective view
        ax2.view_init(elev=0, azim=0)    # Side view
        ax3.view_init(elev=0, azim=90)   # Front view
        ax4.view_init(elev=90, azim=0)   # Top view

        return line1, line2, line3, line4

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(backbone_points), interval=1000 / fps, blit=False)

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