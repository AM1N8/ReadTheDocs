.. CARLA Trajectory Prediction installation

#############################################
Installation
#############################################

Prerequisites
============

Before installing CARLA Trajectory Prediction, ensure you have the following prerequisites:

* Python 3.7+ (3.8 recommended)
* CARLA Simulator 0.9.10+ (for data collection)
* CUDA-compatible GPU (recommended for training)
* 8GB+ RAM

Required Dependencies
===================

* TensorFlow 2.4+
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Plotly (for interactive visualizations)
* PlyFile (for LiDAR data processing)
* tqdm

Installation Options
==================

Installing from PyPI
------------------

The easiest way to install CARLA Trajectory Prediction is through PyPI:

.. code-block:: bash

    pip install carla-trajectory-prediction

Installing from Source
--------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/username/carla-trajectory-prediction.git
    cd carla-trajectory-prediction
    pip install -e .

This will install the package in development mode, allowing you to modify the code and immediately see the effects.


CARLA Simulator Setup
===================

To collect data for training and evaluation, you need to set up the CARLA simulator:

1. Download CARLA from the `official website <https://carla.org/>`_
2. Extract the archive to a suitable location
3. Set the environment variable:

.. code-block:: bash

    export CARLA_ROOT=/path/to/carla

Verifying Installation
====================

To verify your installation is working correctly:

.. code-block:: python

    import carla_trajectory
    carla_trajectory.check_installation()

You should see a success message indicating that all components are properly installed.

Troubleshooting
=============

Common Issues
-----------

1. **TensorFlow GPU not detected**

   Ensure you have the correct CUDA and cuDNN versions installed for your TensorFlow version.

2. **LiDAR data loading errors**

   Check that the PlyFile library is properly installed: ``pip install plyfile``

3. **Import errors**

   Make sure all dependencies are installed: ``pip install -r requirements.txt``

System-Specific Notes
-------------------

**Linux**:
  
  For optimal performance, consider installing TensorFlow with specific optimizations:
  
  .. code-block:: bash
  
      pip install tensorflow==2.8.0

**Windows**:
  
  Make sure Visual C++ Redistributable is installed for PlyFile support.

**macOS**:
  
  TensorFlow performance may be limited. Consider using CPU-only mode.

Getting Help
==========

If you encounter any installation problems:

1. Check the `FAQ section <https://github.com/username/carla-trajectory-prediction/wiki/FAQ>`_
2. Open an issue on our `GitHub repository <https://github.com/username/carla-trajectory-prediction/issues>`_
3. Contact support at support@example.com