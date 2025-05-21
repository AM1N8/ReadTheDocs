Evaluation API
==========

This section covers the evaluation methodology and results for the trajectory prediction model. We assess performance through multiple metrics and visualizations to ensure the model provides accurate and reliable predictions.

Evaluation Methodology
---------------------

The evaluation process involves both quantitative metrics and qualitative visualizations to assess the model's performance in different scenarios.

Evaluation Metrics
~~~~~~~~~~~~~~~~~

We employ several metrics to evaluate trajectory prediction accuracy:

- **Mean Displacement Error (MDE)**: Average Euclidean distance between predicted and actual positions across all timesteps
- **Final Displacement Error (FDE)**: Euclidean distance between the final predicted position and the actual final position
- **Speed-based Evaluation**: Error metrics stratified by vehicle speed ranges (5 m/s, 10 m/s, 20 m/s)
- **Path Smoothness**: Assessment of the predicted path's smoothness using angle changes between consecutive trajectory segments

The following code snippet shows how these metrics are calculated:

.. code-block:: python

    def evaluate_model_performance(model, X_seq_test, X_lidar_test, y_test, segment_info_test, scaler_target):
        """Evaluate model performance with advanced metrics"""
        print("Evaluating model performance...")
        
        # Get predictions
        predictions = model.predict([X_seq_test, X_lidar_test])
        
        # Reshape predictions and ground truth for inverse transformation
        pred_reshaped = predictions.reshape(-1, 2)
        y_test_reshaped = y_test.reshape(-1, 2)
        
        # Inverse transform to get real-world displacements
        pred_original = scaler_target.inverse_transform(pred_reshaped)
        y_test_original = scaler_target.inverse_transform(y_test_reshaped)
        
        # Reshape back
        pred_original = pred_original.reshape(predictions.shape)
        y_test_original = y_test_original.reshape(y_test.shape)
        
        # Calculate displacement error
        displacement_errors = np.sqrt(
            (pred_original[:, :, 0] - y_test_original[:, :, 0])**2 + 
            (pred_original[:, :, 1] - y_test_original[:, :, 1])**2
        )
        
        # Calculate metrics
        mean_displacement_error = np.mean(displacement_errors)
        final_displacement_error = np.mean(displacement_errors[:, -1])
        
        # Speed-based evaluation
        speed_thresholds = [5, 10, 20]  # Speed thresholds in m/s
        for threshold in speed_thresholds:
            high_speed_indices = [i for i, info in enumerate(segment_info_test) 
                                 if info['max_speed'] > threshold]
            if high_speed_indices:
                high_speed_errors = displacement_errors[high_speed_indices]
                print(f"Mean displacement error at speed >{threshold} m/s: {np.mean(high_speed_errors):.4f} m")
        
        # Calculate path smoothness
        path_smoothness = []
        for i in range(len(pred_original)):
            # Calculate curvature changes along the path
            path = convert_relative_to_absolute(0, 0, pred_original[i])
            
            # Calculate path angles
            angles = []
            for j in range(1, len(path)-1):
                v1 = path[j] - path[j-1]
                v2 = path[j+1] - path[j]
                
                # Calculate angle between vectors
                dot_product = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                # Handle zero vectors or numerical issues
                if norm_v1 < 1e-6 or norm_v2 < 1e-6:
                    angles.append(0)
                else:
                    cosine = dot_product / (norm_v1 * norm_v2)
                    # Clip to handle floating point errors
                    cosine = np.clip(cosine, -1.0, 1.0)
                    angles.append(np.arccos(cosine))
            
            if angles:
                # Lower values indicate smoother paths
                path_smoothness.append(np.mean(angles))
        
        return {
            'mean_displacement_error': mean_displacement_error,
            'final_displacement_error': final_displacement_error,
            'predictions': pred_original,
            'ground_truth': y_test_original
        }

Visualization Methods
~~~~~~~~~~~~~~~~~~~~

We use several visualization methods to qualitatively analyze model performance:

1. **Trajectory Comparisons**: Plots showing predicted paths versus ground truth
2. **Steering Angle Analysis**: Comparison of steering angles derived from predicted and actual paths
3. **Training History**: Visualization of loss curves during model training
4. **Control Performance**: Analysis of steering accuracy and position error when using the model for control

Trajectory Visualization
~~~~~~~~~~~~~~~~~~~~~~~

The following code generates visualizations of predicted versus actual trajectories:

.. code-block:: python

    def visualize_trajectory_predictions(true_trajectories, predicted_trajectories, segment_info, n_samples=4):
        """Visualize true vs predicted trajectories with better presentations"""
        plt.figure(figsize=(15, 5 * n_samples))
        
        # Select random samples
        indices = np.random.choice(len(true_trajectories), min(n_samples, len(true_trajectories)), 
                                  replace=False)
        
        for i, idx in enumerate(indices):
            # Get start position from segment info
            start_x = segment_info[idx]['start_x']
            start_y = segment_info[idx]['start_y']
            
            # Convert relative displacements to absolute positions
            true_path = convert_relative_to_absolute(start_x, start_y, true_trajectories[idx])
            pred_path = convert_relative_to_absolute(start_x, start_y, predicted_trajectories[idx])
            
            # Calculate steering angles from paths
            true_steering = calculate_steering_from_path(true_path)
            pred_steering = calculate_steering_from_path(pred_path)
            
            # Plot the paths
            plt.subplot(n_samples, 2, i * 2 + 1)
            plt.plot(true_path[:, 0], true_path[:, 1], 'b-', linewidth=2, label='Ground Truth')
            plt.plot(pred_path[:, 0], pred_path[:, 1], 'r--', linewidth=2, label='Predicted')
            plt.scatter(start_x, start_y, c='g', s=100, marker='o', label='Start Position')
            
            # Calculate displacement error
            final_error = np.sqrt((true_path[-1, 0] - pred_path[-1, 0])**2 + 
                                 (true_path[-1, 1] - pred_path[-1, 1])**2)
            
            plt.title(f"Trajectory {i+1} (Speed: {segment_info[idx]['avg_speed']:.1f} m/s)\n"
                     f"Final Error: {final_error:.2f}m")
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            # Plot steering angles
            plt.subplot(n_samples, 2, i * 2 + 2)
            time_steps = np.arange(len(true_steering))
            plt.plot(time_steps, true_steering, 'b-', linewidth=2, label='Ground Truth Steering')
            plt.plot(time_steps, pred_steering, 'r--', linewidth=2, label='Predicted Steering')
            plt.title(f"Steering Angles for Trajectory {i+1}")
            plt.xlabel('Time Step')
            plt.ylabel('Steering Angle (rad)')
            plt.legend()
            plt.grid(True)

Results
-------

Training Performance
~~~~~~~~~~~~~~~~~~~

The training process was monitored using validation loss metrics. The figure below shows the training and validation loss curves:

.. image:: _static/training_history.png
   :width: 600px
   :alt: Training and validation loss curves

Trajectory Prediction Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model was evaluated on the test set, resulting in the following key metrics:

- **Mean Displacement Error**: 0.1060 m
- **Final Displacement Error**: 0.1181 m
- **Mean Displacement Error at speed >5 m/s**: 0.1059 m
- **Path Smoothness**: 0.0455 rad

Visual examples of the model's trajectory predictions:

.. image:: _static/trajectory_predictions.png
   :width: 800px
   :alt: Example predicted trajectories compared to ground truth

Control System Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

We also evaluated the model's performance when used as part of a control system:

.. code-block:: python

    def visualize_control_performance(control_df):
        """Visualize the control performance by comparing actual vs predicted steering"""
        # Filter to only rows with predictions
        df = control_df.dropna(subset=['predicted_steering']).copy()
        
        # Calculate steering error
        df['steering_error'] = df['steering'] - df['predicted_steering']
        df['abs_steering_error'] = np.abs(df['steering_error'])
        
        # Calculate position error
        df['position_error'] = np.sqrt((df['x'] - df['predicted_x'])**2 + 
                                      (df['y'] - df['predicted_y'])**2)
        
        # Plot steering comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(df['steering'], label='Actual Steering')
        plt.plot(df['predicted_steering'], label='Predicted Steering')
        plt.title('Actual vs Predicted Steering')
        plt.xlabel('Index')
        plt.ylabel('Steering Angle (rad)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(df['abs_steering_error'], label='Absolute Steering Error')
        plt.plot(df['position_error'], label='Position Error (m)')
        plt.title('Control Errors')
        plt.xlabel('Index')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)

The control system evaluation yielded the following results:

- **Mean absolute steering error**: 0.031 rad
- **Mean position error**: 0.29 m
- **Performance at high speeds (>20 m/s)**: 0.52 m position error

Control performance visualization:

.. image:: _static/control_performance.png
   :width: 800px
   :alt: Control system performance showing steering accuracy and position error

Interactive Visualization
------------------------

For a more detailed exploration of the model's performance, we provide an interactive visualization that shows the trajectory predictions and actual paths:

.. raw:: html
   :file: _static/interactive_visualization.html

The interactive visualization allows you to:

- Zoom in/out of the trajectory map
- Compare predicted and actual paths
- View the vehicle's speed profile over time
- Examine specific points along the trajectory

Evaluation Analysis
------------------

Strengths
~~~~~~~~~

- **High accuracy at moderate speeds**: The model performs particularly well for trajectories with speeds between 5-15 m/s
- **Path smoothness**: The predicted trajectories maintain realistic smoothness characteristics
- **Low computational requirements**: The model can generate predictions in real-time, making it suitable for control applications

Limitations
~~~~~~~~~~

- **Performance degradation at high speeds**: Accuracy decreases for vehicles moving faster than 20 m/s
- **Limited prediction horizon**: The model provides reliable predictions up to 2 seconds into the future, with increasing uncertainty beyond that
- **Dependencies on LiDAR quality**: Prediction accuracy correlates with point cloud density and quality

Future Improvements
~~~~~~~~~~~~~~~~~

Based on the evaluation results, several areas for future improvement have been identified:

1. **Extended training on high-speed scenarios**: Incorporate more high-speed driving data to improve performance in these regimes
2. **Advanced sensor fusion**: Integrate camera data alongside LiDAR for improved environmental understanding
3. **Uncertainty estimation**: Add capability to predict confidence intervals alongside trajectory predictions
4. **Adversarial testing**: Evaluate the model under challenging conditions like adverse weather or sensor failures
5. **Longer prediction horizons**: Extend the model to provide useful predictions further into the future