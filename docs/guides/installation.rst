Installation
============

The most recent stable version of ``OneCovariance`` should be installed directly from GitHub.

.. code-block::bash

    git clone https://github.com/KiDS-WL/covariance
    conda env create -f conda_env.yaml
    pip install .

If you do not want to use the conda environment, make sure that you have ``gfortran`` and ``gsl`` installed.
You can install both via ``conda``:

.. code-block:: bash

    conda install -c conda-forge gfortran
    conda install -c conda-forge gsl
