# So here, blob detector is not always very precise. But Kalman filter keeps on track and "averages" the results from several blobs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from filterpy.kalman import KalmanFilter
import cv2

# Simulation parameters
timesteps = 50  # Number of timesteps
dt = 0.07  # Time step for smoother motion
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
                    data_matrix[ny, nx] = np.exp(-0.5 * (distance / spread_radius) ** 2) * 0.8
    return data_matrix

# Add random Gaussian noise across the matrix
def add_noise(data_matrix):
    noise = np.random.normal(0, 0.1, data_matrix.shape)
    noisy_matrix = data_matrix + noise
    return np.clip(noisy_matrix, 0, 1)  # Ensure values stay within [0, 1]

# Initialize the Kalman filter
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])  # Measurement function
    kf.R *= 20  # Measurement noise 
    kf.P *= 100  # Initial uncertainty
    kf.Q = np.array([[0.1, 0, 0, 0],
                     [0, 0.1, 0, 0],
                     [0, 0, 0.1, 0],
                     [0, 0, 0, 0.1]]) * 2  # Process noise
    return kf

# Initialize the blob detector
def initialize_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 2
    params.maxArea = 15
    params.filterByCircularity = True
    params.minCircularity = 0.6
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

# Detect the blob in the noisy matrix
def detect_blob(detector, noisy_matrix):
    # Convert matrix to 8-bit grayscale for OpenCV
    noisy_image = (noisy_matrix * 255).astype(np.uint8)
    keypoints = detector.detect(noisy_image)
    if keypoints:
        # Return the coordinates of the largest detected blob
        return keypoints[0].pt
    return None

# Define the update function for the animation
def update_frame(t, position, velocity, data_matrix, heatmap, kf, detector, detection_circle):
    # Reset matrix
    data_matrix.fill(0)
    
    # Update position and velocity
    position, velocity = update_position_velocity(position, velocity)
    
    # Apply Gaussian spread for the object
    data_matrix = apply_gaussian_spread(data_matrix, position)
    
    # Add noise to the data matrix
    noisy_matrix = add_noise(data_matrix)

    # Blob detection to find the object in the noisy matrix
    detected_position = detect_blob(detector, noisy_matrix)
    if detected_position:
        # Plot the detected position as a yellow hollow circle
        detection_circle.set_data([detected_position[0]], [detected_position[1]])
        
        # Update the Kalman filter using the detected position
        measurement = np.array([detected_position[0], detected_position[1]])
        kf.predict()
        kf.update(measurement)
    else:
        # If no detection, continue predicting without updating
        detection_circle.set_data([], [])
        kf.predict()

    # Print detected position
    if detected_position:
        print(f"Detected position: {detected_position}")
    else:
        print("No detection")

    # Extract predicted position from the Kalman filter for the red circle
    predicted_position = kf.x[[0, 2]]  # Predicted x and y positions

    # Update heatmap with the noisy matrix
    heatmap.set_array(noisy_matrix)
    return heatmap, detection_circle

# Main function to set up the plot and run the animation
def run_animation():
    position, velocity, data_matrix = initialize_simulation()
    
    # Initialize the Kalman filter
    kf = initialize_kalman_filter()
    kf.x[:2] = np.array([[position[0]], [velocity[0]]])  # Initial x and x-velocity
    kf.x[2:] = np.array([[position[1]], [velocity[1]]])  # Initial y and y-velocity

    # Initialize the blob detector
    detector = initialize_blob_detector()

    fig, ax = plt.subplots()
    heatmap = ax.imshow(data_matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title("Simulated Object Tracking with Kalman Filter")
    ax.set_xlim(0, x_len)
    ax.set_ylim(0, y_len)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    # Add circles for detection and tracking
    detection_circle, = ax.plot([], [], 'oy', markersize=15, markerfacecolor='none', markeredgewidth=1.5)  # Yellow hollow circle for detection

    # Wrapping update_frame in a lambda to pass additional arguments
    ani = animation.FuncAnimation(
        fig,
        lambda t: update_frame(t, position, velocity, data_matrix, heatmap, kf, detector, detection_circle),
        frames=timesteps,            # Ends after the specified number of frames
        interval=dt * 1000,           # Speed control, matching the simulation's dt (in milliseconds)
        blit=False                    # Set to False for compatibility testing
    )
        
    plt.show()

# Run the simulation
run_animation()
