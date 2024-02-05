Some examples
=============

KiDS-1000 covariance
--------------------
The standard ``config.ini`` (after you pulled the directory) will run a simplified KiDS-1000-like cosmic shear setup. Not all parameters specified in the ``config.ini`` are used and it is merely used as an explanatory file to explain all the parameters which can be set.
Let us take a closer look at the output: Since a plot for the correlation coefficient was requested, we have:

.. image:: correlation_xpm.png
   :width: 790

With the corresponding covariance saved in ``covariance_matrix.mat``. A complete list of all the entries can be found in the ``covariance_list.dat`` file as shown below.

.. image:: covariance_list.png
   :width: 790

The first column specifies which combination of observables is considered, in this case :math:`\xi_{+}\xi_{+}`. The second and third column label the combination of the independent spatial variable of the corresponding summary statistic, here this are the two :math:`\theta` bins.
For bandpowers this would be the multipole bands and for COSEBIS the order. ``s1`` and ``s2`` label the sample bins in mass used (for the evaluation of the halo model integrals). ``tomoi``, ..., `tomol` are the tomographic bin combinations, which start counting at 1.
The total covariance is safed in the column ``cov``. If in the ``config.ini`` the variable ``split_gauss`` is set to true the Gaussian component of the covariance is split into a sample-variance, shot/shape noise and mix term labeled ``covg_sva``, ``covg_sn`` and ``covg_mix`` respectively.
Finally the last two columns show the non-Gaussian and the super-sample covariance term respecitvely, since they have been switched off in the ini-file they are set to zero.

We can calculate the covariance also for bandpowers and COSEBIs by setting:

``est_shear = bandpowers``

``est_shear = cosebi``

in the ini-file. Similarly the non-Gaussian and the super-sample covariance term can be requested by setting

``nongauss = True``

``ssc = True``

Using Input :math:`C_\ell`
--------------------------
In the directory ``input/Cell`` files for precomputed angular power spectra, :math:`C_\ell`, are provided. They should explain the required structure and can be passed to the code by setting

``Cell_directory = ./input``

``Cgg_file = Cell_gg.ascii``

``Cgm_file = Cell_gkappa.ascii``

``Cmm_file = Cell_kappakappa.ascii``

in the ini-file. In this way one can use the code to produce the covariance of the implemented summary statistic for any tracer for which a harmonic covariance has been calculated. 


3x2pt for :math:`C_\ell`:
We will calculate the full 3x2pt covariance matrix in harmonic space by running the covariance code with the in ``config_3x2pt.ini`` in the ``/config_files`` directory.
To obtain a good understanding, we will go through the ``.ini`` file section by section:

``[observables]
cosmic_shear = True
est_shear = C_ell
ggl = True
est_ggl = C_ell
clustering = True
est_clust = C_ell
cstellar_mf = False
cross_terms = True
unbiased_clustering = False``





