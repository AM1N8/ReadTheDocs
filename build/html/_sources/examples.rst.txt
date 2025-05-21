Examples
========

Basic Usage
-----------

.. code-block:: python

    # Load and preprocess data
    ego_df = pd.read_csv('carla_data/ego_data.csv')
    ego_df = calculate_vehicle_dynamics(ego_df)
    
    # Train model
    model = build_trajectory_prediction_model()
    model.fit(training_data, epochs=10)
    
    # Evaluate
    results = evaluate_model_performance(model, test_data)

Advanced Control
---------------

.. code-block:: python

    # Implement control system
    control_df = implement_trajectory_control(
        ego_df,
        trained_model,
        input_scaler,
        target_scaler,
        lidar_mapping
    )
    
    # Visualize results
    create_interactive_visualization(ego_df, control_df)

.. raw:: html
   :file: _static/interactive_visualization.html

Visualization
------------

The interactive visualization above demonstrates the trajectory control system in action. It shows:

- The ego vehicle's predicted path
- Actual trajectory followed
- Control inputs over time
- Environmental obstacles and boundaries

You can interact with the visualization by:

- Zooming in/out to examine specific areas
- Clicking on data points to see detailed information
- Using the timeline slider to view the simulation progression
- Toggling different data layers using the control panel