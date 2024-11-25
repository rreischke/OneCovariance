Installation
============

The most recent stable version of ``OneCovariance`` should be installed directly from GitHub.

.. code-block:: bash

    git clone git@github.com:rreischke/OneCovariance.git
    cd OneCovariance
    conda env create -f conda_env.yaml
    conda activate cov20_env
    pip install .

On some Linux servers you will have to install ``gxx_linux-64`` by hand and the installation will not work. This usually shows the following error message in the terminal:

.. code-block:: bash

    gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory

If this is the case just install it by typing

.. code-block:: bash
    
    conda install -c conda-forge gxx_linux-64

and redo ``pip install .``  .

If you do not want to use the conda environment, make sure that you have ``gfortran`` and ``gsl`` installed. **Note that there is an issue with pybind11 when using python >= 3.13, so make sure to use a version below that for the moment.**
You can install both via ``conda``:

.. code-block:: bash

    conda install -c conda-forge gfortran
    conda install -c conda-forge gsl
    conda install -c conda-forge gxx_linux-64
    git clone git@github.com:rreischke/OneCovariance.git
    cd OneCovariance    
    pip install .

Once you have carried out these steps, you are ready to run the code. 
Lastly, since the code uses simpson instead of trapezoidal rules for some integrations, ``scipy`` should be at least version 1.11.


Known issues
------------

There have been some issues reported when using MacOS Sonoma or higher with openmp resulting into a crash when running healpy or the implemented C++ integrators, causing the code to crash:

.. code-block:: bash

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

The code automatically checks for the OS version now. If you still encounter this problem a quick fix is to remove the if statement in ``line 11`` of the ``covariance.py`` and thus use

.. code-block:: python
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

by default. I have run many tests and never found incorrect results with this option set to ``True``. For any linux-based system this command is not required and is not set to True by default.