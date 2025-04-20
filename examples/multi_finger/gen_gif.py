import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from HANDS.env_helpers import reshape_state

def load_outputs(file_path):
    """
    Load the outputs from the pickle file.
    """
    with open(file_path, 'rb') as f:
        outputs = pickle.load(f)
    return outputs



def create_gif(outputs, gif_path="backbone_with_cylinder.gif"):
    """
    Create a GIF of the backbone points and the cylinder for each instant with two views.
    """
    simulation_duration = outputs[-1]['time']
    time_stamps = [data['time'] for data in outputs]
    num_fingers = outputs[0].get('num_fingers', 1)
    states = [reshape_state(np.array(data['state']), num_fingers) for data in outputs]

    cylinder_centers = [data.get("cylinder_position") for data in outputs]
    cylinder_directions = [data.get("cylinder_director")[2, ...] for data in outputs]
    cylinder_radius = 0.02
    cylinder_length = 0.2

    fps = 10
    total_frames = int(simulation_duration * fps)
    frame_indices = np.linspace(0, len(states) - 1, total_frames, dtype=int)

    states = [states[i] for i in frame_indices]
    time_stamps = [time_stamps[i] for i in frame_indices]
    cylinder_centers = [cylinder_centers[i] for i in frame_indices]
    cylinder_directions = [cylinder_directions[i] for i in frame_indices]

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim([-0.25, 0.25])
    ax1.set_ylim([-0.25, 0.25])
    ax1.set_zlim([0, 0.25])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View')

    ax2 = fig.add_subplot(122)
    ax2.set_xlim([-0.25, 0.25])
    ax2.set_ylim([-0.25, 0.25])
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-Down View (-Z Axis)')

    # Finger visuals
    lines_3d = [ax1.plot([], [], [], marker='o', linestyle='-')[0] for _ in range(num_fingers)]
    lines_2d = [ax2.plot([], [], marker='o', linestyle='-')[0] for _ in range(num_fingers)]
    cylinder_surface = None
    rectangle_patch = None

    def update(frame):
        nonlocal cylinder_surface, rectangle_patch

        state = states[frame]
        cyl_center = np.reshape(cylinder_centers[frame], (3,))
        cyl_dir = np.reshape(cylinder_directions[frame], (3,))
        cyl_dir = cyl_dir / np.linalg.norm(cyl_dir)

        z_axis = np.array([0, 0, 1])
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, cyl_dir), -1.0, 1.0))
        rotation_axis = np.cross(z_axis, cyl_dir)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation = R.from_rotvec(rotation_angle * rotation_axis)
            rotation_matrix = rotation.as_matrix()
        else:
            rotation_matrix = np.eye(3)

        # Clear cylinder surface if it exists
        if cylinder_surface:
            cylinder_surface.remove()
        if rectangle_patch:
            rectangle_patch.remove()

        # Plot finger states
        for i, rod in enumerate(state):
            lines_3d[i].set_data(rod[:, 0], rod[:, 1])
            lines_3d[i].set_3d_properties(rod[:, 2])
            lines_2d[i].set_data(rod[:, 0], rod[:, 1])

        ax1.set_title(f"3D View - Time: {time_stamps[frame]:.2f}s")
        ax2.set_title(f"Top-Down - Time: {time_stamps[frame]:.2f}s")

        # Create the cylinder surface in 3D
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(-cylinder_length / 2, cylinder_length / 2, 20)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cylinder_radius * np.cos(theta_grid)
        y_grid = cylinder_radius * np.sin(theta_grid)

        coords = np.dot(rotation_matrix, np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
        x = coords[0].reshape(x_grid.shape) + cyl_center[0]
        y = coords[1].reshape(y_grid.shape) + cyl_center[1]
        z = coords[2].reshape(z_grid.shape) + cyl_center[2]

        cylinder_surface = ax1.plot_surface(x, y, z, alpha=0.3, color='b')

        # 2D Rectangle in top-down view
        cyl_dir_xy = cyl_dir[:2]
        if np.linalg.norm(cyl_dir_xy) < 1e-6:
            cyl_dir_xy = np.array([1.0, 0.0])
        else:
            cyl_dir_xy = cyl_dir_xy / np.linalg.norm(cyl_dir_xy)

        perp_dir = np.array([-cyl_dir_xy[1], cyl_dir_xy[0]])
        half_l = cylinder_length / 2
        half_w = cylinder_radius
        c = cyl_center[:2]

        corners = [
            c + half_l * cyl_dir_xy + half_w * perp_dir,
            c + half_l * cyl_dir_xy - half_w * perp_dir,
            c - half_l * cyl_dir_xy - half_w * perp_dir,
            c - half_l * cyl_dir_xy + half_w * perp_dir,
            c + half_l * cyl_dir_xy + half_w * perp_dir,
        ]
        corners = np.array(corners)
        rectangle_patch, = ax2.plot(corners[:, 0], corners[:, 1], color='b', alpha=0.7)

        return lines_3d + lines_2d + [cylinder_surface, rectangle_patch]

    ani = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
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
    import sys
    
    # Use first command line argument as file name if provided, otherwise use default
    file_name = sys.argv[1] if len(sys.argv) > 1 else "outputs2.pkl"
    
    try:
        # Load the outputs from the pickle file
        outputs = load_outputs(file_name)
        print(f"Loading data from {file_name}")
    except FileNotFoundError:
        print(f"File {file_name} not found, falling back to outputs2.pkl")
        outputs = load_outputs("outputs2.pkl")

    # Create and save the GIF
    create_gif(outputs)
    # plot_trajectory_and_tensions(outputs)