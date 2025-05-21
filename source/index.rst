.. CARLA Trajectory Prediction documentation master file

#############################################
Vehicle Trajectory Forecasting
#############################################

.. image:: https://img.shields.io/badge/CARLA-Trajectory%20Prediction-blue.svg
   :alt: CARLA Trajectory Prediction
   
*This project implements an advanced trajectory prediction system for autonomous vehicles by combining vehicle telemetry data with LiDAR point cloud information. The system predicts future vehicle paths while accounting for both the vehicle's dynamics and its surrounding environment.*

.. raw:: html

   <div class="container">
      <div class="row">
         <div class="col-lg-6">
            <div class="card">
               <div class="card-body">
                  <h5 class="card-title">Get Started</h5>
                  <p class="card-text">Follow our comprehensive installation guide to set up the framework on your system.</p>
                  <a href="installation.html" class="btn btn-primary">Installation Guide</a>
               </div>
            </div>
         </div>
         <div class="col-lg-6">
            <div class="card">
               <div class="card-body">
                  <h5 class="card-title">Try Examples</h5>
                  <p class="card-text">Explore our ready-to-use examples to understand how to implement trajectory prediction.</p>
                  <a href="examples.html" class="btn btn-primary">View Examples</a>
               </div>
            </div>
         </div>
      </div>
   </div>

About
=====

CARLA Trajectory Prediction is a framework designed for predicting the future trajectories 
of vehicles in urban environments using the CARLA simulator. This library provides 
tools for data collection, model training, evaluation, and integration with autonomous 
driving systems.

Key Features
===========

- Data collection and preprocessing utilities
- State-of-the-art trajectory prediction models
- Comprehensive evaluation metrics
- Seamless integration with CARLA simulator
- Visualization tools for trajectory analysis

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   overview
   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/data
   api/model
   api/training
   api/evaluation

.. toctree::
   :maxdepth: 1
   :caption: Resources
   :hidden:

   examples

Indices and Navigation
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <div class="footer-note">
      <p><em>CARLA Trajectory Prediction is an open-source project developed for research in autonomous driving systems.</em></p>
   </div>