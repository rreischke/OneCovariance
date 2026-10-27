Estimators from the real-space covariance
=========================================

By default, the covariance of COSEBIs, bandpowers and arbitrary summary statistics is
obtained by projecting the covariance of the angular power spectra with the Fourier-space
weights of the estimator. Alternatively, it can be obtained from the covariance of the
real-space correlation functions in fine angular bins:

.. code-block:: ini

    [observables]
    theta_to_obs = True

The estimators are then exactly the measured ones, i.e. linear combinations of the
correlation functions in the fine bins,

.. math::

    \mathcal{O}_m = \int \mathrm{d}\theta\,\theta\, R_m(\theta)\, w(\theta)\;, \qquad
    \mathcal{O}^{E/B}_m = \int \mathrm{d}\theta\,\theta\,
    \left[R_{+m}(\theta)\,\xi_+(\theta) \pm R_{-m}(\theta)\,\xi_-(\theta)\right]

(:math:`\gamma_t` as :math:`w`), and their covariance is

.. math::

    \mathrm{Cov}(\mathcal{O}_m, \mathcal{O}_n) = \sum_{ij} A_{mi}\,
    \mathrm{Cov}(\xi_i, \xi_j)\, A_{nj}\;, \qquad
    A_{mi} = \int_{\theta_i} \mathrm{d}\theta\,\theta\, R_m(\theta)\;,

with the integral over the fine bin :math:`i`. Every term of the real-space covariance
enters this way, independently of how it was obtained: e.g. the shot noise from the
measured pair counts (``npair_*_file``) or the mixed term from the triplet counts of a
catalogue (``mix_term_*``). All other settings of ``[covTHETAspace settings]`` are used as
given there.

Filters
-------
The filters :math:`R_m(\theta)` are

* ``cosebi``: COSEBIs for cosmic shear and Psi-statistics for clustering and galaxy-galaxy
  lensing, with the settings of ``[covCOSEBI settings]``,
* ``bandpowers``: the bandpowers of ``[covbandpowers settings]``,
* ``xi_pm``, ``gamma_t``, ``w``: the correlation functions in the bins of
  ``[covTHETAspace settings]`` (a rebinning of the fine bins, which is exact),
* the real-space filters of the arbitrary summary statistics (``arb_real_filter_*_file``)
  if ``do_arbitrary_obs = True``; the Fourier-space filters are not needed then.
  Filters which are top-hats of real-space bins are taken as averages over the bin; for
  cosmic shear the ``mm_p`` file is then :math:`\xi_+` and the ``mm_m`` file :math:`\xi_-`.

The output has the same format as without ``theta_to_obs``.

Fine bins
---------
The fine bins have all edges of the filters (support, apodisation, top-hats) as bin edges.
Their number over the full angular range and their spacing are set in
``[covTHETAspace settings]``:

.. code-block:: ini

    [covTHETAspace settings]
    theta_to_obs_bins = 500       ; or theta_to_obs_bins_clustering / _lensing
    theta_to_obs_type = log       ; log, lin or loglin

The estimators from the fine bins differ from those with the continuous filters by an
error that decreases with the square of the bin width. It is estimated from the estimators
measured from half as many bins and printed; a warning is given if it exceeds
:math:`10^{-3}`. The defaults are chosen for an error of about :math:`10^{-3}` of the
Gaussian covariance:

=================================  ===========  =========
Filters                            Fine bins    Spacing
=================================  ===========  =========
COSEBIs                            500          log
Psi-statistics                     400          lin
bandpowers, arbitrary filters      1000         loglin
w, gamma_t, xi_pm                  coarse bins  log
=================================  ===========  =========

``loglin`` is equally spaced in
:math:`\ln\theta/\ln(\theta_\mathrm{max}/\theta_\mathrm{min}) + \theta/(\theta_\mathrm{max}-\theta_\mathrm{min})`,
i.e. logarithmic at small and linear at large angles, as needed for filters which oscillate
with :math:`\ell\theta`.

Replacing terms of the real-space covariance
--------------------------------------------
``CovTHETASpace`` passes every block of the real-space covariance (``'ww'``, ``'wgt'``,
``'wxip'``, ``'wxim'``, ``'gtgt'``, ``'xipgt'``, ``'ximgt'``, ``'xipxip'``, ``'xipxim'``,
``'ximxim'``) through the method ``_theta_block(name, block)`` once it is complete. To replace
or modify a term, override this method in a subclass of ``CovTHETAtoOBS``, change the block
(given in the fine bins, with shape (theta bins, theta bins, sample bins, sample bins, tomo,
tomo, tomo, tomo)) and pass it on to ``CovTHETAtoOBS._theta_block``, which maps it to the
estimators.
