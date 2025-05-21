.. CARLA Trajectory Prediction model API

#############################################
Model API
#############################################

The Model API provides components for building trajectory prediction models.

Model Architecture
================

.. py:function:: build_trajectory_prediction_model(seq_length, n_features, n_lidar_features, prediction_horizon)
   
   Build a trajectory prediction model using relative displacements.
   
   :param int seq_length: Length of input sequence
   :param int n_features: Number of features in the sequence input
   :param int n_lidar_features: Number of features in LiDAR point cloud
   :param int prediction_horizon: Number of future steps to predict
   :return: Compiled Keras model
   :rtype: tensorflow.keras.Model
   
   .. code-block:: python
   
       # Example usage
       model = build_trajectory_prediction_model(
           seq_length=10,
           n_features=15,
           n_lidar_features=4,
           prediction_horizon=10
       )
       model.summary()

   **Model Architecture Diagram**
   
   .. image:: ../_static/architecture.svg
      :alt: Model Architecture
      :width: 100%
   
   The model consists of two input branches:
   
   1. Sequence branch for processing vehicle state history
   2. LiDAR branch for processing environmental context
   
   These branches are fused and processed through fully connected layers to predict future trajectories.

Point Cloud Encoder
=================

.. py:function:: build_point_cloud_encoder(point_cloud_input)
   
   Build an enhanced PointNet-inspired network for point cloud processing.
   
   :param tensorflow.Tensor point_cloud_input: Input tensor for point cloud data
   :return: Encoded point cloud features
   :rtype: tensorflow.Tensor
   
   **Architecture Details**
   
   * Initial point-wise convolution (1x1) to transform features
   * Residual block for improved gradient flow
   * Hierarchical feature extraction (64 → 128 → 256 channels)
   * Global feature pooling with max and average operations
   * Dense layers for final feature embedding

Sequence Encoder
==============

.. py:function:: build_sequence_encoder(sequence_input)
   
   Build a sequence encoder with attention mechanism.
   
   :param tensorflow.Tensor sequence_input: Input tensor for sequence data
   :return: Encoded sequence features
   :rtype: tensorflow.Tensor
   
   **Architecture Details**
   
   * Bidirectional LSTM layers for temporal feature extraction
   * Self-attention mechanism to prioritize important time steps
   * Context vector creation through weighted averaging
   * Combination of context vector and last step features

Loss Functions
============

.. py:function:: weighted_displacement_loss(y_true, y_pred)
   
   Custom loss function for displacement prediction with temporal weighting.
   
   :param tensorflow.Tensor y_true: Ground truth displacements
   :param tensorflow.Tensor y_pred: Predicted displacements
   :return: Weighted loss value
   :rtype: tensorflow.Tensor
   
   **Loss Calculation**
   
   * Calculate Euclidean distance error at each time step
   * Apply temporal weights that prioritize earlier predictions
   * Normalize weights for consistent scaling
   * Average the weighted errors

Inference Functions
================

.. py:function:: predict_future_trajectory(model, current_sequence, current_lidar, scaler_input, scaler_target, start_x, start_y, start_heading)
   
   Predict a future trajectory given current state.
   
   :param tensorflow.keras.Model model: Trained trajectory prediction model
   :param numpy.ndarray current_sequence: Normalized sequence of recent vehicle dynamics
   :param numpy.ndarray current_lidar: Processed LiDAR point cloud data
   :param sklearn.preprocessing.StandardScaler scaler_input: Scaler used for input normalization
   :param sklearn.preprocessing.StandardScaler scaler_target: Scaler used for target normalization
   :param float start_x: Current vehicle x position
   :param float start_y: Current vehicle y position
   :param float start_heading: Current vehicle heading in radians
   :return: Tuple containing predicted path and steering angles
   :rtype: tuple
   
   .. code-block:: python
   
       # Example usage
       predicted_path, steering_angles = predict_future_trajectory(
           model, current_sequence, current_lidar, 
           scaler_input, scaler_target,
           start_x=10.0, start_y=5.0, start_heading=0.5
       )

Model Serialization
=================

The model can be saved and loaded using standard TensorFlow/Keras functions:

.. code-block:: python

    # Save model
    model.save('trajectory_prediction_model.h5')
    
    # Save preprocessing scalers
    import pickle
    with open('trajectory_scalers.pkl', 'wb') as f:
        pickle.dump({
            'input_scaler': scaler_input,
            'target_scaler': scaler_target
        }, f)
    
    # Load model
    from tensorflow import keras
    model = keras.models.load_model(
        'trajectory_prediction_model.h5', 
        custom_objects={'weighted_displacement_loss': weighted_displacement_loss}
    )
    
    # Load scalers
    with open('trajectory_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
        scaler_input = scalers['input_scaler']
        scaler_target = scalers['target_scaler']