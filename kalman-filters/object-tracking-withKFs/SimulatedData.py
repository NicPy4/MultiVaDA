import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import PillowWriter


# Simulation parameters
timesteps = 50  # Number of timesteps
dt = 0.04  # Time step for smoother motion
gravity = np.array([0.0, -9.81])  # Gravity vector in m/s^2


# Simulation area
x_len, y_len = 150, 150
spread_radius = 4  # Radius of spread for the "ball" effect

# Initialize the simulation
def initialize_simulation():
    position = np.array([0.0, 0.0])  # Initial position in meters
    velocity = np.array([15.0, 45.0])  # Initial velocity in m/s
    data_matrix = np.zeros((x_len, y_len))  # Area looked at by the camera/sensor
    return position, velocity, data_matrix


# Initialize Kalman filter variables in initialize_simulation
def initialize_kalman_filter():
    estimated_position = np.array([0.0, 0.0])  # Initial estimated position
    estimated_velocity = np.array([20.0, 55.0])  # Initial estimated velocity
    error_covariance = np.eye(4)  # State covariance matrix
    process_noise = 1e-2 * np.eye(4)  # Process noise covariance
    measurement_noise = 1e-1 * np.eye(2)  # Measurement noise covariance
    return estimated_position, estimated_velocity, error_covariance, process_noise, measurement_noise



# Update position and velocity based on kinematic equations
def update_position_velocity(position, velocity):
    position += velocity * dt + 0.5 * gravity * (dt ** 2)
    velocity += gravity * dt
    return position, velocity


# Apply Gaussian spread around the ball's position
def apply_gaussian_spread(data_matrix, position):
    x_idx, y_idx = int(position[0]), int(position[1])
    if 0 <= x_idx < x_len and 0 <= y_idx < y_len:
        for dx in range(-spread_radius, spread_radius + 1):
            for dy in range(-spread_radius, spread_radius + 1):
                nx, ny = x_idx + dx, y_idx + dy
                if 0 <= nx < x_len and 0 <= ny < y_len:
                    distance = np.sqrt(dx**2 + dy**2)
                    data_matrix[ny, nx] = np.exp(-0.5 * (distance / spread_radius) ** 2) * 0.7
    return data_matrix


# Add random Gaussian noise across the matrix
def add_noise(data_matrix):
    noise = np.random.normal(0, 0.15, data_matrix.shape)
    noisy_matrix = data_matrix + noise
    return np.clip(noisy_matrix, 0, 1)  # Ensure values stay within [0, 1]


# Define the update function for the animation
def update_frame(t, position, velocity, data_matrix, heatmap):
    # Reset matrix
    data_matrix.fill(0)
    
    # Update position and velocity
    position, velocity = update_position_velocity(position, velocity)

    # Apply Gaussian spread for the object
    data_matrix = apply_gaussian_spread(data_matrix, position)
    
    # Add noise to the data matrix
    noisy_matrix = add_noise(data_matrix)

    # Update heatmap with the noisy matrix
    heatmap.set_array(noisy_matrix)
    return position, velocity, heatmap


# Main function to set up the plot and run the animation
def run_animation():
    position, velocity, data_matrix = initialize_simulation()
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(data_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title("Simulated object")
    ax.set_xlim(0, x_len)
    ax.set_ylim(0, y_len)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    # Wrapping update_frame in a lambda to pass additional arguments
    ani = animation.FuncAnimation(
        fig,
        lambda t: update_frame(t, position, velocity, data_matrix, heatmap),
        frames=timesteps,
        interval=dt * 100
    )
    
    plt.show()




# Run the simulation
run_animation()
