.. CARLA Trajectory Prediction training API

#############################################
Training API
#############################################

The Training API provides utilities for training and fine-tuning trajectory prediction models.

Training Configuration
===================

.. py:function:: build_and_train_model(X_seq_train, X_lidar_train, y_train, seq_length, n_features, n_lidar_features, prediction_horizon, validation_split=0.1, batch_size=32, epochs=50)
   
   Build and train a trajectory prediction model with best practices.
   
   :param numpy.ndarray X_seq_train: Training sequence data
   :param numpy.ndarray X_lidar_train: Training LiDAR data
   :param numpy.ndarray y_train: Training target data
   :param int seq_length: Length of input sequence
   :param int n_features: Number of features in sequence data
   :param int n_lidar_features: Number of features in LiDAR data
   :param int prediction_horizon: Number of future steps to predict
   :param float validation_split: Fraction of data to use for validation
   :param int batch_size: Training batch size
   :param int epochs: Maximum number of training epochs
   :return: Trained model and training history
   :rtype: tuple
   
   .. code-block:: python
   
       # Example usage
       model, history = build_and_train_model(
           X_seq_train, X_lidar_train, y_train,
           seq_length=10, n_features=15, n_lidar_features=4, prediction_horizon=10
       )

.. py:function:: initialize_model_for_training(model, learning_rate=0.001)
   
   Initialize model for training with appropriate optimizer and loss.
   
   :param tensorflow.keras.Model model: Model to initialize
   :param float learning_rate: Initial learning rate
   :return: Compiled model
   :rtype: tensorflow.keras.Model

Training Callbacks
================

The framework provides several training callbacks for optimal training:

Early Stopping
------------

Prevents overfitting by monitoring validation loss and stopping when it stops improving.

.. code-block:: python

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

Learning Rate Scheduling
----------------------

Adapts the learning rate during training to improve convergence.

.. code-block:: python

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

Model Checkpointing
-----------------

Saves the best model during training based on validation performance.

.. code-block:: python

    checkpoint = keras.callbacks.ModelCheckpoint(
        'trajectory_model_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

Hyperparameter Tuning
===================

.. py:function:: tune_learning_rate(X_seq_train, X_lidar_train, y_train, X_seq_val, X_lidar_val, y_val, model, min_lr=1e-5, max_lr=1e-1, steps=20)
   
   Perform learning rate range test to find optimal learning rate.
   
   :param numpy.ndarray X_seq_train: Training sequence data
   :param numpy.ndarray X_lidar_train: Training LiDAR data
   :param numpy.ndarray y_train: Training target data
   :param numpy.ndarray X_seq_val: Validation sequence data
   :param numpy.ndarray X_lidar_val: Validation LiDAR data
   :param numpy.ndarray y_val: Validation target data
   :param tensorflow.keras.Model model: Model to tune
   :param float min_lr: Minimum learning rate to test
   :param float max_lr: Maximum learning rate to test
   :param int steps: Number of learning rates to test
   :return: Optimal learning rate
   :rtype: float

Data Augmentation
===============

.. py:function:: augment_sequence_data(X_seq, y_seq, jitter_factor=0.05, dropout_prob=0.1)
   
   Augment sequence data with noise and feature dropout for improved robustness.
   
   :param numpy.ndarray X_seq: Input sequence data
   :param numpy.ndarray y_seq: Target sequence data
   :return: Augmented input and target data
   :rtype: tuple
   
   Augmentation techniques:
   
   * Random jittering: Adding noise to continuous features
   * Feature dropout: Randomly setting features to zero
   * Sequence shifting: Shifting sequence by small offsets
   * Target jittering: Adding controlled noise to target values

.. py:function:: augment_lidar_data(X_lidar, rotation_range=15, translation_range=0.2)
   
   Augment LiDAR point cloud data with geometric transformations.
   
   :param numpy.ndarray X_lidar: Input LiDAR data
   :return: Augmented LiDAR data
   :rtype: numpy.ndarray
   
   Augmentation techniques:
   
   * Random rotation: Rotating point cloud around z-axis
   * Random translation: Shifting point cloud by small offsets
   * Point dropout: Randomly removing points
   * Intensity jittering: Adding noise to intensity values

Cross-Validation
==============

.. py:function:: perform_cross_validation(X_seq, X_lidar, y_seq, seg_info, model_fn, n_folds=5)
   
   Perform k-fold cross-validation for robust model evaluation.
   
   :param numpy.ndarray X_seq: Sequence data
   :param numpy.ndarray X_lidar: LiDAR data
   :param numpy.ndarray y_seq: Target data
   :param list seg_info: Segment information
   :param callable model_fn: Function to create model
   :param int n_folds: Number of folds
   :return: Cross-validation results
   :rtype: dict

   This function ensures that data from the same driving segment is not split between training and validation sets.

Training Visualization
===================

.. py:function:: plot_training_history(history)
   
   Plot training and validation metrics from model training history.
   
   :param tensorflow.keras.callbacks.History history: Training history
   
   Plots include:
   
   * Training vs. validation loss
   * Learning rate schedule
   * Metrics breakdown by epoch

.. code-block:: python

    # Example usage
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()