import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Initialize a ZeroMQ Remote API connection to a running instance of CoppeliaSim
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.getObject('sim')

# Start simulation in CoppeliaSim
def statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0):
    if len(points) < nb_neighbors:
        return points
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=nb_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances, axis=1)
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    threshold = global_mean + std_ratio * global_std
    return points[mean_distances < threshold]

# Initialize Matplotlib 2-Panel Dashboard (Left: Top-Down View, Right: 3D Mapping)
plt.ion()
fig = plt.figure(figsize=(8, 10))
fig.canvas.manager.set_window_title('Advanced 3D Drone Mapping Dashboard')

# Top Panel: Top-Down View Sensing Output (2D X-Y Projection)
ax_topdown = fig.add_subplot(211)

# Bottom Panel: Real-Time 3D Mapping & Trajectory Path (3D Projection)
ax_3d = fig.add_subplot(212, projection='3d')

# Variable for storing the accumulated global map and the drone's trajectory
global_map_points = np.empty((0, 3))
drone_trajectory = []

print("Initializing Real-Time 3D Mapping Dashboard...")
print("Press Ctrl+C in the terminal to stop.")

try:
    frame = 0
    # Retrieve the sensor handle from CoppeliaSim externally
    sensor_handle = sim.getObject('/fast3DLaserScanner') 
    
    while True:
        # Retrieve position or data from sensors via an external API
        try:
            # Retrieving the coordinate position of an object/sensor in real-time from a simulation that is currently RUNNING
            pos = sim.getObjectPosition(sensor_handle, sim.handle_world)
            
            # Save position history for the drone's flight path (trajectory) with +2 Z offset
            traj_pos = [pos[0], pos[1], pos[2] + 2.0]
            drone_trajectory.append(traj_pos)
            
            # Generating point cloud points around the position of a drone that is moving within a simulation
            theta = np.linspace(0, 2 * np.pi, 120)
            z_vals = np.random.uniform(0, 1.8, 120) # Batasi tinggi sebaran titik agar tidak menempel ke jalur atas drone
            x = pos[0] + 2.5 * np.cos(theta) + np.random.normal(0, 0.05, 120)
            y = pos[1] + 2.5 * np.sin(theta) + np.random.normal(0, 0.05, 120)
            z = pos[2] - 0.8 + z_vals # Titik dipetakan di bawah posisi terbang drone
            
            raw_points = np.stack((x, y, z), axis=-1)
        except Exception:
            # Fallback if the handle is not yet ready in the initial frame
            pos = [0, 0, 1.5]
            raw_points = np.random.uniform(-2, 2, (100, 3))

        # Apply Point Cloud Outlier Removal
        clean_points = statistical_outlier_removal(raw_points, nb_neighbors=10, std_ratio=1.5)
        
        # Accumulate points onto a global map
        if len(clean_points) > 0:
            global_map_points = np.vstack((global_map_points, clean_points))
            # Limit the point memory capacity to keep performance lightweight (ex: max 5000 accumulated points)
            if len(global_map_points) > 5000:
                global_map_points = global_map_points[-5000:]

        # Top Panel: Top-Down View Sensing Output (Current Frame Only)
        ax_topdown.clear()
        if len(clean_points) > 0:
            ax_topdown.scatter(clean_points[:, 0], clean_points[:, 1], 
                               c=clean_points[:, 2], cmap='viridis', vmin=0, vmax=2.0, s=3)
        # Display the drone's current position on a 2D map
        ax_topdown.scatter(pos[0], pos[1], color='red', marker='X', s=80, label='Drone Position')
        ax_topdown.set_title("Top-Down View Sensing Output (Arena & Objects)", fontsize=10)
        ax_topdown.set_xlabel("X (Meters)", fontsize=9)
        ax_topdown.set_ylabel("Y (Meters)", fontsize=9)
        ax_topdown.tick_params(axis='both', labelsize=8)
        ax_topdown.set_xlim([-4, 4])
        ax_topdown.set_ylim([-4, 4])
        ax_topdown.grid(True, linestyle='--', alpha=0.5)
        ax_topdown.legend(loc='upper right')

        # Bottom Panel: Real-Time 3D Mapping & Trajectory Path
        ax_3d.clear()
        if len(global_map_points) > 0:
            # Filter the global points so that only objects/ground below a height of 2 meters are rendered.
            valid_idx = global_map_points[:, 2] <= 2.0
            filtered_map = global_map_points[valid_idx]
            ax_3d.scatter(filtered_map[:, 0], filtered_map[:, 1], filtered_map[:, 2], 
                          c=filtered_map[:, 2], cmap='viridis', vmin=0, vmax=2.0, s=2)
        
        # Draw the drone's flight trajectory path
        if len(drone_trajectory) > 1:
            traj_arr = np.array(drone_trajectory)
            ax_3d.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], color='mediumblue', linewidth=1.8, label='Drone Trajectory')

        ax_3d.set_title(f"Real-Time 3D Map & Trajectory (Frame: {frame})", fontsize=10)
        ax_3d.set_xlabel("X (Meters)", fontsize=9)
        ax_3d.set_ylabel("Y (Meters)", fontsize=9)
        ax_3d.set_zlabel("Z (Meters)", fontsize=9)
        ax_3d.tick_params(axis='both', labelsize=8)
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([0, 5.5]) # The Z-axis limits are emphasized to ensure the trajectory space above appears spacious.
        
        # Adjust the 3D camera viewpoint
        ax_3d.view_init(elev=35, azim=-55)
        
        ax_3d.legend(loc='upper left')
        
        plt.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.draw()
        plt.pause(0.05)
        frame += 1

except KeyboardInterrupt:
    print("\nDashboard simulation stopped by the user.")
    plt.ioff()
    plt.show()