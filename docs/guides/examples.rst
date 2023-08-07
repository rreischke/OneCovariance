Crash course
============
Running the code with the unaltered ``config.ini`` will calculate the Gaussian real-space covariance matrix for a KiDS-1000 setting for a 3x2pt analysis
including photometric galaxy clustering, :math:`w(\theta)`, galaxy-galaxy-lensing, :math:`\gamma_\mathrm{t}(\theta)` and cosmic shear, 
:math:`\xi_{\pm}(\theta)`. Running the code will produce several files in the ``output`` directory:

- A plot ``correlation_coefficient.pdf`` of the correlation coeffcient of the covariance matrix.
- A text file in matrix format ``covariance.mat`` including the desired covariance matrix which can be used directly. The priority is always the following (from the slowest to the fastest index):
  Of course if a certain probe does not exist it is ignored. Furthermore, galaxy clustering and cosmic shear have :math:`n(n+1)/2` unique observables if :math:`n` is the number of tomographic lens and 
  source bins respectively. Galaxy-galaxy-lensing on the other hand has :math:`n_\mathrm{lens}n_\mathrm{source}` unique observables.
- A text file in a list format ``covariance.dat`` including all entries of the covariance matrix labeled by probe, tomographic bin combination and projected quantity.