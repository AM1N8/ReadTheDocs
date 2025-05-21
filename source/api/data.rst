.. CARLA Trajectory Prediction data API

#############################################
Data API
#############################################

The Data API provides utilities for loading, processing, and managing trajectory data from the CARLA simulator.

Data Loading
===========

.. py:function:: load_lidar_point_cloud(file_path)
   
   Load LiDAR point cloud from a PLY file.
   
   :param str file_path: Path to the PLY file
   :return: Numpy array of shape (N, 4) containing x, y, z coordinates and intensity
   :rtype: numpy.ndarray

   .. code-block:: python
   
       # Example usage
       point_cloud = load_lidar_point_cloud("lidar_1234567890.ply")
       print(f"Loaded {len(point_cloud)} points with shape {point_cloud.shape}")

.. py:function:: map_timestamps_to_lidar(ego_df, lidar_path)
   
   Map timestamps from ego vehicle data to corresponding LiDAR files.
   
   :param pandas.DataFrame ego_df: DataFrame containing ego vehicle data with 'timestamp' column
   :param str lidar_path: Path to directory containing LiDAR files
   :return: Dictionary mapping timestamps to LiDAR file paths
   :rtype: dict

.. py:function:: load_camera_image(file_path)
   
   Load camera image from file (PNG, JPEG).
   
   :param str file_path: Path to the image file
   :return: RGB image as numpy array of shape (H, W, 3)
   :rtype: numpy.ndarray

   .. code-block:: python
   
       # Example usage
       image = load_camera_image("rgb_1234567890.png")
       print(f"Loaded image with shape {image.shape}")

.. py:function:: map_timestamps_to_images(ego_df, image_path, camera_type='rgb')
   
   Map timestamps from ego vehicle data to corresponding camera images.
   
   :param pandas.DataFrame ego_df: DataFrame containing ego vehicle data with 'timestamp' column
   :param str image_path: Path to directory containing image files
   :param str camera_type: Type of camera images ('rgb', 'depth', 'semantic')
   :return: Dictionary mapping timestamps to image file paths
   :rtype: dict

Point Cloud Processing
=====================

.. py:function:: sample_point_cloud(points, n_points=1024)
   
   Sample a fixed number of points from a point cloud with improved sampling strategy.
   
   :param numpy.ndarray points: Input point cloud array of shape (N, D)
   :param int n_points: Number of points to sample (default: 1024)
   :return: Sampled point cloud of shape (n_points, D)
   :rtype: numpy.ndarray

   Key features:
   
   * Distance-based weighting prioritizes points closer to the vehicle
   * Handles point clouds with fewer points than requested
   * Ensures consistent output dimension regardless of input size

.. py:function:: process_point_cloud(points, max_points=1024)
   
   Process point cloud data with advanced normalization techniques.
   
   :param numpy.ndarray points: Input point cloud array
   :param int max_points: Maximum number of points (default: 1024)
   :return: Processed point cloud with normalized features
   :rtype: numpy.ndarray
   
   Processing steps:
   
   1. Center points using robust median centering
   2. Scale to unit sphere using 90th percentile normalization
   3. Normalize intensity values to [0,1] range

Vehicle Data Processing
=====================

.. py:function:: calculate_vehicle_dynamics(ego_df)
   
   Calculate enhanced vehicle dynamics features from raw ego vehicle data.
   
   :param pandas.DataFrame ego_df: DataFrame with basic vehicle data
   :return: DataFrame with additional dynamics features
   :rtype: pandas.DataFrame
   
   Computed features include:
   
   * Speed (magnitude of velocity)
   * Heading (direction of travel)
   * Acceleration components and magnitude
   * Jerk (rate of change of acceleration)
   * Heading change rate
   * Curvature
   * Movement state (binary indicator)

.. py:function:: calculate_relative_displacement(ego_df)
   
   Calculate relative displacements between consecutive frames.
   
   :param pandas.DataFrame ego_df: DataFrame with vehicle positions
   :return: DataFrame with added relative displacement columns
   :rtype: pandas.DataFrame

.. py:function:: filter_and_segment_data(ego_df, min_speed=0.5, min_segment_length=5)
   
   Filter stationary data and segment into meaningful driving sequences.
   
   :param pandas.DataFrame ego_df: DataFrame with vehicle dynamics
   :param float min_speed: Minimum speed threshold for "moving" state
   :param int min_segment_length: Minimum length of a valid segment
   :return: Filtered DataFrame with segment information
   :rtype: pandas.DataFrame

Image Processing
==============

.. py:function:: process_camera_image(image, target_size=(224, 224), normalize=True)
   
   Process camera image for model input with resizing and normalization.
   
   :param numpy.ndarray image: Input RGB image
   :param tuple target_size: Target image dimensions (height, width)
   :param bool normalize: Whether to normalize pixel values to [0,1]
   :return: Processed image
   :rtype: numpy.ndarray
   
   Processing steps:
   
   1. Resize to target dimensions
   2. Convert to float32 data type
   3. Normalize pixel values if requested
   4. Handle different input formats (RGB, grayscale, etc.)

.. py:function:: extract_road_features(semantic_image)
   
   Extract road features from semantic segmentation image.
   
   :param numpy.ndarray semantic_image: Semantic segmentation image
   :return: Binary road mask and road boundary features
   :rtype: tuple

.. py:function:: augment_camera_images(images, augmentation_strength=0.5)
   
   Apply data augmentation techniques to camera images.
   
   :param numpy.ndarray images: Array of images
   :param float augmentation_strength: Strength of augmentation (0.0 to 1.0)
   :return: Augmented images
   :rtype: numpy.ndarray
   
   Augmentation techniques:
   
   * Random brightness and contrast adjustment
   * Random color jitter
   * Random blur and noise addition
   * Random horizontal flip (for training only)

Map Data Processing
================

.. py:function:: load_carla_map(map_name)
   
   Load CARLA map data for a specific map.
   
   :param str map_name: Name of the CARLA map (e.g., 'Town01', 'Town10HD')
   :return: Map object with road network information
   :rtype: object

.. py:function:: get_map_image(map_name, pixels_per_meter=10, include_lanes=True)
   
   Generate a bird's-eye view image of a CARLA map.
   
   :param str map_name: Name of the CARLA map
   :param int pixels_per_meter: Resolution factor for the image
   :param bool include_lanes: Whether to include lane markings
   :return: RGB image of the map as numpy array
   :rtype: numpy.ndarray

   .. code-block:: python
   
       # Example usage
       map_img = get_map_image("Town05", pixels_per_meter=12)
       plt.figure(figsize=(10, 10))
       plt.imshow(map_img)
       plt.axis('off')
       plt.title("Town05 Map")
       plt.show()

.. py:function:: project_vehicle_to_map(vehicle_location, map_data, pixels_per_meter=10)
   
   Project vehicle location to map image coordinates.
   
   :param tuple vehicle_location: Vehicle location (x, y) in world coordinates
   :param object map_data: Map data object
   :param int pixels_per_meter: Resolution factor of the map image
   :return: (x, y) coordinates in the map image
   :rtype: tuple

.. py:function:: visualize_trajectory_on_map(trajectory, map_image, color=(255, 0, 0), thickness=2)
   
   Visualize vehicle trajectory on a map image.
   
   :param numpy.ndarray trajectory: Array of (x, y) positions in world coordinates
   :param numpy.ndarray map_image: Map image as numpy array
   :param tuple color: RGB color for trajectory visualization
   :param int thickness: Line thickness for visualization
   :return: Map image with trajectory overlay
   :rtype: numpy.ndarray

Sequence Preparation
==================

.. py:function:: prepare_relative_sequences(df_processed, timestamp_to_lidar, seq_length=10, prediction_horizon=5, min_speed=0.5)
   
   Prepare input sequences with relative displacement representation for model training.
   
   :param pandas.DataFrame df_processed: Processed vehicle data
   :param dict timestamp_to_lidar: Mapping from timestamps to LiDAR files
   :param int seq_length: Length of input sequence
   :param int prediction_horizon: Number of future steps to predict
   :param float min_speed: Minimum average speed threshold
   :return: tuple containing (input sequences, LiDAR data, target sequences, input scaler, target scaler, segment info)
   :rtype: tuple

   The returned data is structured as:
   
   * X_seq: Array of shape (N, seq_length, n_features)
   * X_lidar: Array of shape (N, 1024, 4)
   * y_seq: Array of shape (N, prediction_horizon, 2)
   * scaler_input: StandardScaler for input normalization
   * scaler_target: StandardScaler for target normalization
   * segment_info: List of dictionaries with segment metadata

.. py:function:: prepare_multimodal_sequences(df_processed, timestamp_to_lidar, timestamp_to_image, map_data, seq_length=10, prediction_horizon=5)
   
   Prepare multimodal input sequences combining vehicle dynamics, LiDAR, and camera data.
   
   :param pandas.DataFrame df_processed: Processed vehicle data
   :param dict timestamp_to_lidar: Mapping from timestamps to LiDAR files
   :param dict timestamp_to_image: Mapping from timestamps to camera image files
   :param object map_data: Map data object
   :param int seq_length: Length of input sequence
   :param int prediction_horizon: Number of future steps to predict
   :return: tuple containing multimodal data components
   :rtype: tuple
   
   The returned data includes:
   
   * Vehicle dynamics sequences
   * LiDAR point cloud sequences
   * Camera image sequences
   * Map image patches centered on the vehicle
   * Target trajectories

Data Collection
============

.. py:function:: setup_carla_client(host='localhost', port=2000, timeout=10.0)
   
   Setup connection to CARLA simulator.
   
   :param str host: Host address of CARLA server
   :param int port: Port number
   :param float timeout: Connection timeout in seconds
   :return: CARLA client instance
   :rtype: carla.Client

.. py:function:: configure_sensors(vehicle, output_path, lidar_freq=10, camera_freq=10)
   
   Configure and attach sensors to a vehicle for data collection.
   
   :param carla.Vehicle vehicle: Vehicle actor in simulation
   :param str output_path: Path to save sensor data
   :param int lidar_freq: LiDAR capture frequency in Hz
   :param int camera_freq: Camera capture frequency in Hz
   :return: Dictionary of attached sensors
   :rtype: dict
   
   Configured sensors include:
   
   * RGB camera (front-facing)
   * Depth camera (front-facing)
   * Semantic segmentation camera (front-facing)
   * LiDAR sensor (roof-mounted)
   * GNSS sensor for positioning

.. py:function:: collect_vehicle_data(client, map_name, duration=300, num_vehicles=50, save_path='./data')
   
   Collect comprehensive vehicle trajectory and sensor data in CARLA.
   
   :param carla.Client client: CARLA client instance
   :param str map_name: Name of the map to use
   :param int duration: Data collection duration in seconds
   :param int num_vehicles: Number of background vehicles
   :param str save_path: Path to save collected data
   :return: Path to saved data
   :rtype: str
   
   Data collection process:
   
   1. Load specified map and set weather conditions
   2. Spawn autopilot ego vehicle with sensors
   3. Spawn background traffic
   4. Record vehicle state, sensor data, and ground truth trajectories
   5. Process and organize collected data

.. py:function:: capture_map_images(client, map_names, resolution=4096, save_path='./maps')
   
   Capture high-resolution orthographic images of CARLA maps.
   
   :param carla.Client client: CARLA client instance
   :param list map_names: List of map names to capture
   :param int resolution: Image resolution
   :param str save_path: Path to save map images
   :return: Dictionary mapping map names to saved image paths
   :rtype: dict
   
   Captures various map representations:
   
   * RGB aerial view
   * Road network visualization
   * Semantic segmentation view
   * Lane marking visualization

Utility Functions
===============

.. py:function:: convert_relative_to_absolute(start_x, start_y, relative_displacements)
   
   Convert relative displacements to absolute positions.
   
   :param float start_x: Starting x position
   :param float start_y: Starting y position
   :param numpy.ndarray relative_displacements: Array of relative displacements
   :return: Array of absolute positions
   :rtype: numpy.ndarray

.. py:function:: calculate_steering_from_path(future_positions, wheelbase=2.7)
   
   Calculate steering angles from a predicted path using bicycle model.
   
   :param numpy.ndarray future_positions: Array of predicted positions
   :param float wheelbase: Vehicle wheelbase length in meters
   :return: Array of steering angles
   :rtype: list

.. py:function:: world_to_pixel(location, map_data)
   
   Convert world coordinates to pixel coordinates in map image.
   
   :param carla.Location location: World location
   :param object map_data: Map data object
   :return: (x, y) pixel coordinates
   :rtype: tuple

.. py:function:: pixel_to_world(pixel_x, pixel_y, map_data)
   
   Convert pixel coordinates in map image to world coordinates.
   
   :param int pixel_x: X pixel coordinate
   :param int pixel_y: Y pixel coordinate
   :param object map_data: Map data object
   :return: (x, y) world coordinates
   :rtype: tuple