import time
import numpy as np
from scipy.special import jv
from scipy.signal import argrelextrema
import multiprocessing as mp

import levin

try:
    from onecov.cov_ell_space import CovELLSpace
except:
    from cov_ell_space import CovELLSpace


class CovARBsummary(CovELLSpace):
    """
    This class calculates the covariance of the same tracer using arbitrary
    summary statistics whose Fourier and Realspace filters are passed through file inputs.
    It also calculates their respective cross-covariance.
    The structure is very similar to the COSEBIs class, but requires a more careful treatment
    of the noise term. It should be noted that covCOSEBI, CovBandPowers and covTHETAspace are
    optimized for the respective estimators. This class uses a more brute force approach for
    the integration and will thus most likelily be slower or less accurate. It should only be
    used if cross-variances between different 2pt statistics is required or a new statistic
    needs to be calculated which has not been implemented in the code.

    Parameters
    ----------
    cov_dict : dictionary
        Specifies which terms of the covariance (Gaussian, non-Gaussian,
        super-sample covariance) should be calculated. To be passed from
        the read_input method of the Input class.
    obs_dict : dictionary
        with the following keys (To be passed from the read_input method
        of the Input class.)
        'observables' : dictionary
            Specifies which observables (cosmic shear, galaxy-galaxy
            lensing and/or clustering) should be calculated. Also,
            indicates whether cross-terms are evaluated.
        'ELLspace' : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        'THETAspace' : dictionary
            Specifies the exact details of the projection to real space,
            e.g., theta_min/max and the number of theta bins to be
            calculated.
        'bandpowers' : dictionary
            Specifies the exact details for the bandpower covariance.
    cosmo_dict : dictionary
        Specifies all cosmological parameters. To be passed from the
        read_input method of the Input class.
    bias_dict : dictionary
        Specifies all the information about the bias model. To be passed
        from the read_input method of the Input class.
    iA_dict: dictionary
        Specifies all the information about the intrinsic alignment model.
        To be passed from the read_input method in the Input class.
    hod_dict : dictionary
        Specifies all the information about the halo occupation
        distribution used. This defines the shot noise level of the
        covariance and includes the mass bin definition of the different
        galaxy populations. To be passed from the read_input method of
        the Input class.
    survey_params_dict : dictionary
        Specifies all the information unique to a specific survey.
        Relevant values are the effective number density of galaxies for
        all tomographic bins as well as the ellipticity dispersion for
        galaxy shapes. To be passed from the read_input method of the
        Input class.
    prec : dictionary
        with the following keys (To be passed from the read_input method
        of the Input class.)
        'hm' : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        'powspec' : dictionary
            Contains precision information about the power spectra, this
            includes k-range and spacing.
        'trispec' : dictionary
            Contains precision information about the trispectra, this
            includes k-range and spacing and the desired precision
            limits.
    read_in_tables : dictionary
        with the following keys (To be passed from the read_input method
        of the FileInput class.)
        'zclust' : dictionary
            Look-up table for the clustering redshifts and their number
            of tomographic bins. Relevant for clustering and galaxy-
            galaxy lensing estimators.
        'zlens' : dictionary
            Look-up table for the lensing redshifts and their number of
            tomographic bins. Relevant for cosmic shear and galaxy-
            galaxy lensing estimators.
        'survey_area_clust' : dictionary
            Contains information about the survey footprint that is read
            from the fits file of the survey (processed by healpy).
            Possible keys: 'area', 'ell', 'a_lm'
        'Pxy' : dictionary
            default : None
            Look-up table for the power spectra (matter-matter, tracer-
            tracer, matter-tracer, optional).
        'Cxy' : dictionary
            default : None
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer, optional).
        'effbias' : dictionary
            default : None
            Look-up table for the effective bias as a function of
            redshift(optional).
        'mor' : dictionary
            default : None
            Look-up table for the mass-observable relation (optional).
        'occprob' : dictionary
            default : None
            Look-up table for the occupation probability as a function
            of halo mass per galaxy sample (optional).
        'occnum' : dictionary
            default : None
            Look-up table for the occupation number as a function of
            halo mass per galaxy sample (optional).
        'tri': dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
    """

    def __init__(self,
                 cov_dict,
                 obs_dict,
                 output_dict,
                 cosmo_dict,
                 bias_dict,
                 iA_dict,
                 hod_dict,
                 survey_params_dict,
                 prec,
                 read_in_tables):
        CovELLSpace.__init__(self,
                             cov_dict,
                             obs_dict,
                             output_dict,
                             cosmo_dict,
                             bias_dict,
                             iA_dict,
                             hod_dict,
                             survey_params_dict,
                             prec,
                             read_in_tables)
        

    def __get_fourier_weights(self,
                              fourier_tabs):
        """
        This function reads in the Fourier weights from the input
        tables and sets them to private variables for later intergration.
        """
        self.fourier_ell = []
        self.fourier_weights = []
        if self.gg:
            for gg_summary in fourier_tabs['gg']:
                self.fourier_ell.append(gg_summary['ell'])
                self.fourier_weights.append(gg_summary['W_ell'])
        if self.gm:
            for gm_summary in fourier_tabs['gm']:
                self.fourier_ell.append(gm_summary['ell'])
                self.fourier_weights.append(gm_summary['W_ell'])
        if self.mm:
            for mm_summary in fourier_tabs['mm']:
                self.fourier_ell.append(mm_summary['ell'])
                self.fourier_weights.append(mm_summary['W_ell'])
        return True
    
    def __get_real_weights(self,
                           real_tabs):
        """
        This function reads in the realspace weights from the input
        tables and sets them to private variables for later intergration.
        """
        self.real_theta = []
        self.real_weights = []
        if self.gg:
            for gg_summary in fourier_tabs['gg']:
                self.real_theta.append(gg_summary['ell'])
                self.real_weights.append(gg_summary['W_ell'])
        if self.gm:
            for gm_summary in fourier_tabs['gm']:
                self.real_theta.append(gm_summary['ell'])
                self.real_weights.append(gm_summary['W_ell'])
        if self.mm:
            for mm_summary in fourier_tabs['mm']:
                self.real_theta.append(mm_summary['ell'])
                self.real_weights.append(mm_summary['W_ell'])
        return True