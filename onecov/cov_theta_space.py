import time
import numpy as np
from scipy.interpolate import UnivariateSpline
import levin
from scipy.special import jv
from scipy.signal import argrelextrema

try:
    from onecov.cov_ell_space import CovELLSpace
    from onecov.cov_discrete import *
    from onecov.cov_discrete_utils import *
except:
    from cov_ell_space import CovELLSpace
    from cov_discrete import *
    from cov_discrete_utils import *


class CovTHETASpace(CovELLSpace):
    """
    This class calculates the real space covariance for shear-shear
    correlation functions xi_+/-, for position-shear correlations
    gamma_t (galaxy-galaxy lensing), and position-position correlations
    (galaxy clustering). Inherits the functionality of the CovELLSpace
    class.

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
        'COSEBIs' : dictionary
            Specifies the exact details of the projection to COSEBIs,
            e.g. the number of modes to be calculated.
        'bandpowers' : dictionary
            Specifies the exact details of the projection to bandpowers,
            e.g. the ell modes and their spacing.
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

    Atrributes
    ----------
    see CovELLSpace class

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_theta_space import CovTHETASpace
    inp = Input()
    covterms, observables, output, cosmo, bias, iA hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    covtheta = CovTHETASpace(covterms, observables, output, cosmo, bias,
        iA, hod, survey_params, prec, read_in_tables)

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
        self.xi_pp = obs_dict['THETAspace']['xi_pp']
        self.xi_pm = obs_dict['THETAspace']['xi_pm']
        self.xi_mm = obs_dict['THETAspace']['xi_mm']
        self.accuracy = obs_dict['THETAspace']['theta_acc']
        self.theta_space_dict = obs_dict['THETAspace']
        self.thetabins, self.theta_ul_bins = \
            self.__set_theta_bins(obs_dict['THETAspace'])
        if ((obs_dict['observables']['est_shear'] == 'xi_pm' and obs_dict['observables']['cosmic_shear']) or (obs_dict['observables']['est_ggl'] == 'gamma_t' and obs_dict['observables']['ggl']) or obs_dict['observables']['est_clust'] == 'w' and obs_dict['observables']['clustering']):
            if self.gg or self.gm:
                save_n_eff_clust = survey_params_dict['n_eff_clust']
            if self.mm or self.gm:
                save_n_eff_lens = survey_params_dict['n_eff_lens']
            if self.sample_dim > 1:
                if self.gg or self.gm:
                    survey_params_dict['n_eff_clust'] = self.Ngal.T/self.arcmin2torad2
                if self.mm or self.gm:
                    survey_params_dict['n_eff_lens'] = survey_params_dict['n_eff_lens'][:, None]
            else:
                if self.gg or self.gm:
                    survey_params_dict['n_eff_clust'] = survey_params_dict['n_eff_clust'][:, None]
                if self.mm or self.gm:
                    survey_params_dict['n_eff_lens'] = survey_params_dict['n_eff_lens'][:, None]  
            if self.gg or self.gm:
                self.npair_gg, self.npair_gm, _ = \
                    self.get_npair([self.gg, self.gm, False],
                                self.theta_ul_bins_clustering,
                                self.theta_bins_clustering,
                                survey_params_dict,
                                read_in_tables['npair'])
            if self.mm:
                _, _, self.npair_mm = \
                    self.get_npair([False, False, self.mm],
                                self.theta_ul_bins_lensing,
                                self.theta_bins_lensing,
                                survey_params_dict,
                                read_in_tables['npair'])
            if self.gg or self.gm:    
                survey_params_dict['n_eff_clust'] = save_n_eff_clust
            if self.mm or self.gm:
                survey_params_dict['n_eff_lens'] = save_n_eff_lens
        if ((obs_dict['observables']['est_shear'] == 'xi_pm' and obs_dict['observables']['cosmic_shear']) or (obs_dict['observables']['est_ggl'] == 'gamma_t' and obs_dict['observables']['ggl']) or obs_dict['observables']['est_clust'] == 'w' and obs_dict['observables']['clustering']):
            self.__get_weights()
            self.ell_limits = []
            for mode in range(len(self.WXY_stack[:,0])):
                limits_at_mode = np.array(self.ell_fourier_integral[argrelextrema(self.WXY_stack[mode,:], np.less)[0][:]])[::self.integration_intervals]
                limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]) + 2)
                limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]
                limits_at_mode_append[0] = self.ell_fourier_integral[0]
                limits_at_mode_append[-1] = self.ell_fourier_integral[-1]
                self.ell_limits.append(limits_at_mode_append)
            self.levin_int_fourier = levin.Levin(0, 16, 32, obs_dict['THETAspace']['theta_acc']/np.sqrt(len(max(self.ell_limits, key=len))), self.integration_intervals, self.num_cores)
            self.levin_int_fourier.init_w_ell(self.ell_fourier_integral, self.WXY_stack.T)
            self.__get_signal(obs_dict)
        
    def __set_theta_bins(self,
                         covTHETAspacesettings):
        """
        Calculates the theta values at which the covariance is calculated

        Parameters
        ----------
        covTHETAspacesettings : dictionary
            Specifies the angular bins at which the covariance is
            evaluated.

        Returns
        -------
            thetabins : array
                with shape (observables['THETAspace']['theta_bins'])
            theta_ul_bins : array
                bin boundaries with shape (observables['THETAspace']['theta_bins'] + 1)
        """
        self.theta_bins_clustering = None
        self.theta_bins_lensing = None
        self.theta_ul_bins_clustering = None
        self.theta_ul_bins_lensing = None
        if self.gg or self.gm:
            if covTHETAspacesettings['theta_type_clustering'] == 'lin':
                self.theta_ul_bins_clustering = np.linspace(
                    covTHETAspacesettings['theta_min_clustering'],
                    covTHETAspacesettings['theta_max_clustering'],
                    covTHETAspacesettings['theta_bins_clustering'] + 1)
                self.theta_bins_clustering = .5 * (self.theta_ul_bins_clustering[1:] + self.theta_ul_bins_clustering[:-1])
            if covTHETAspacesettings['theta_type_clustering'] == 'log':
                self.theta_ul_bins_clustering = np.geomspace(
                    covTHETAspacesettings['theta_min_clustering'],
                    covTHETAspacesettings['theta_max_clustering'],
                    covTHETAspacesettings['theta_bins_clustering'] + 1)
                self.theta_bins_clustering = np.exp(.5 * (np.log(self.theta_ul_bins_clustering[1:])
                                        + np.log(self.theta_ul_bins_clustering[:-1])))
        if self.mm:
            if covTHETAspacesettings['theta_type_lensing'] == 'lin':
                self.theta_ul_bins_lensing = np.linspace(
                    covTHETAspacesettings['theta_min_lensing'],
                    covTHETAspacesettings['theta_max_lensing'],
                    covTHETAspacesettings['theta_bins_lensing'] + 1)
                self.theta_bins_lensing = .5 * (self.theta_ul_bins_lensing[1:] + self.theta_ul_bins_lensing[:-1])
            if covTHETAspacesettings['theta_type_lensing'] == 'log':
                self.theta_ul_bins_lensing = np.geomspace(
                    covTHETAspacesettings['theta_min_lensing'],
                    covTHETAspacesettings['theta_max_lensing'],
                    covTHETAspacesettings['theta_bins_lensing'] + 1)
                self.theta_bins_lensing = np.exp(.5 * (np.log(self.theta_ul_bins_lensing[1:])
                                        + np.log(self.theta_ul_bins_lensing[:-1])))


        if covTHETAspacesettings['theta_type'] == 'lin':
            theta_ul_bins = np.linspace(
                covTHETAspacesettings['theta_min'],
                covTHETAspacesettings['theta_max'],
                covTHETAspacesettings['theta_bins'] + 1)
            theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])
        if covTHETAspacesettings['theta_type'] == 'log':
            theta_ul_bins = np.geomspace(
                covTHETAspacesettings['theta_min'],
                covTHETAspacesettings['theta_max'],
                covTHETAspacesettings['theta_bins'] + 1)
            theta_bins = np.exp(.5 * (np.log(theta_ul_bins[1:])
                                    + np.log(theta_ul_bins[:-1])))
        if covTHETAspacesettings['theta_type'] == 'list':
            theta_ul_bins = covTHETAspacesettings['theta_list']
            theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])
        return theta_bins, theta_ul_bins

    def __get_signal(self, obs_dict):
        """
        Calculates the clustering signal which might be used in the
        clustering-z covariance

        Parameters
        ----------
        
        Returns
        -------
            Sets private variable self.w_gg with shape 
            (observables['THETAspace']['theta_bins'], sample_dim, n_tomo_clust, n_tomo_clust)
        """
        self.data_vector_length = 0
        if self.gg:
            self.data_vector_length += len(self.theta_bins_clustering)*self.n_tomo_clust*(self.n_tomo_clust+1)/2
            w_signal_shape = (len(self.theta_bins_clustering),
                              self.sample_dim,
                              self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust)
            original_shape = self.Cell_gg[0, :, :, :, :].shape
            w_signal = np.zeros(w_signal_shape)
            flat_length = self.sample_dim**2*self.n_tomo_clust**2
            Cell_gg_flat = np.reshape(
                self.Cell_gg, (len(self.ellrange), flat_length))
            w_signal_at_thetai_flat = np.zeros(flat_length)
            for i_theta in range(len(self.theta_bins_clustering)):
                integrand = Cell_gg_flat*self.ellrange[:, None]
                self.levin_int_fourier.init_integral(
                    self.ellrange, integrand, True, True)
                w_signal_at_thetai_flat = self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[i_theta],i_theta)
                w_signal[i_theta, :, :, :, :] = np.reshape(
                    w_signal_at_thetai_flat, original_shape)/2.0/np.pi
            self.w_gg = w_signal
        if self.gm:
            self.data_vector_length += len(self.theta_bins_clustering)*self.n_tomo_clust*self.n_tomo_lens
            gt_signal_shape = (len(self.theta_bins_clustering),
                              self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_lens)
            original_shape = self.Cell_gm[0, :, :, :].shape
            gt_signal = np.zeros(gt_signal_shape)
            flat_length = self.sample_dim*self.n_tomo_clust*self.n_tomo_lens
            Cell_gm_flat = np.reshape(
                self.Cell_gm, (len(self.ellrange), flat_length))
            gt_signal_at_thetai_flat = np.zeros(flat_length)
            for i_theta in range(len(self.theta_bins_clustering)):
                integrand = Cell_gm_flat*self.ellrange[:, None]
                self.levin_int_fourier.init_integral(
                    self.ellrange, integrand, True, True)
                gt_signal_at_thetai_flat = self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[i_theta + self.gg_summaries],i_theta + self.gg_summaries)
                gt_signal[i_theta, :, :, :] = np.reshape(
                    gt_signal_at_thetai_flat, original_shape)/2.0/np.pi
            self.gt = gt_signal
          ## define spline on finer theta range, theta_min = theta_min/5, theta_max = theta_ax*2
        if obs_dict['THETAspace']['mix_term_do_mix_for'] is not None:
            if 'xipxip' in obs_dict['THETAspace']['mix_term_do_mix_for'][:] or 'ximxim' in obs_dict['THETAspace']['mix_term_do_mix_for'][:]:
                if self.mm:
                    self.data_vector_length += len(self.theta_bins_lensing)*self.n_tomo_lens*(self.n_tomo_lens+1)
                    theta_ul_bins = np.geomspace(
                        self.theta_ul_bins[0]/5,
                        self.theta_ul_bins[-1]*40,
                        100)
                    theta_bins = np.exp(.5 * (np.log(theta_ul_bins[1:])
                                            + np.log(theta_ul_bins[:-1])))
                    xip_signal_shape = (len(theta_bins),
                                        self.sample_dim,
                                        self.n_tomo_lens, self.n_tomo_lens)
                    xip_signal = np.zeros(xip_signal_shape)
                    xim_signal = np.zeros(xip_signal_shape)
                    original_shape = self.Cell_mm[0, :, :, :].shape
                    flat_length = self.sample_dim*self.n_tomo_lens**2
                    Cell_mm_flat = np.reshape(
                        self.Cell_mm, (len(self.ellrange), flat_length))
                    xip_signal_at_thetai_flat = np.zeros(flat_length)
                    xim_signal_at_thetai_flat = np.zeros(flat_length)
                    self.xi_spline = {}
                    self.xi_spline["xip"] = [None]*int(self.n_tomo_lens*(self.n_tomo_lens + 1)/2)
                    self.xi_spline["xim"] = [None]*int(self.n_tomo_lens*(self.n_tomo_lens + 1)/2)
                    K_mmE = []
                    K_mmB = []
                    for i_theta in range(len(theta_bins)):
                        theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
                        theta_l = theta_ul_bins[i_theta]/60/180*np.pi
                        xu = self.ell_fourier_integral*theta_u
                        xl = self.ell_fourier_integral*theta_l
                        K_mmE.append(2/(xu**2 - xl**2)*(xu*jv(1,xu) - xl*jv(1,xl)))
                        K_mmB.append(2/(xu**2 - xl**2)*((xu - 8/xu)*jv(1,xu) - 8*jv(2,xu) - (xl-8/xl)*jv(1,xl) + 8*jv(2,xl)))
                    WXY_stack = []
                    K_mmE = np.array(K_mmE)
                    K_mmB = np.array(K_mmB)
                    for i_theta in range(len(theta_bins)):
                        WXY_stack.append(K_mmE[i_theta,:])
                    for i_theta in range(len(theta_bins)):
                        WXY_stack.append(K_mmB[i_theta,:])
                    WXY_stack = np.array(WXY_stack)
                    ell_limits = []
                    for mode in range(len(WXY_stack[:,0])):
                        limits_at_mode = np.array(self.ell_fourier_integral[argrelextrema(WXY_stack[mode,:], np.less)[0][:]])[::self.integration_intervals]
                        limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ellrange[-2])]) + 2)
                        limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ellrange[-2])]
                        limits_at_mode_append[0] = self.ell_fourier_integral[0]
                        limits_at_mode_append[-1] = self.ell_fourier_integral[-1]
                        ell_limits.append(limits_at_mode_append)
                    for i_theta in range(len(theta_bins)):
                        levin_int_fourier = levin.Levin(0, 16, 32, obs_dict['THETAspace']['theta_acc']/np.sqrt(len(max(self.ell_limits, key=len))), self.integration_intervals, self.num_cores)
                        aux_WXY_stack = []
                        aux_WXY_stack.append(WXY_stack[i_theta, :])
                        aux_WXY_stack.append(WXY_stack[i_theta + len(theta_bins),:])
                        aux_WXY_stack = np.array(aux_WXY_stack)
                        levin_int_fourier.init_w_ell(self.ell_fourier_integral, aux_WXY_stack.T)
                        integrand = Cell_mm_flat*self.ellrange[:, None]
                        levin_int_fourier.init_integral(
                            self.ellrange, integrand, True, True)
                        xip_signal_at_thetai_flat = levin_int_fourier.cquad_integrate_single_well(ell_limits[i_theta],0)
                        xip_signal[i_theta, :, :, :] = np.reshape(
                            xip_signal_at_thetai_flat, original_shape)/2.0/np.pi
                        xim_signal_at_thetai_flat = levin_int_fourier.cquad_integrate_single_well(ell_limits[i_theta + len(theta_bins)],1)
                        xim_signal[i_theta, :, :, :] = np.reshape(
                            xim_signal_at_thetai_flat, original_shape)/2.0/np.pi    
                    self.xip = xip_signal
                    self.xim = xim_signal
                    flat_idx = 0
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            self.xi_spline["xip"][flat_idx] = UnivariateSpline((theta_bins),(self.xip[:,0,i_tomo, j_tomo]), s=0, k= 1)
                            self.xi_spline["xim"][flat_idx] = UnivariateSpline((theta_bins),(self.xim[:,0,i_tomo, j_tomo]), s=0, k= 1)
                            flat_idx += 1
                    
                    
    def __get_weights(self):
        N_fourier = int(1e5)
        self.K_gg = []
        self.K_gm = []
        self.K_mmE = []
        self.K_mmB = []
        self.gg_summaries = 0
        self.gm_summaries = 0
        self.mmE_summaries = 0
        self.ell_fourier_integral = np.geomspace(1,1e5,N_fourier)
        if self.gg or self.gm:
            for i_theta in range(len(self.theta_bins_clustering)):
                theta_u = self.theta_ul_bins_clustering[i_theta+1]/60/180*np.pi
                theta_l = self.theta_ul_bins_clustering[i_theta]/60/180*np.pi
                xu = self.ell_fourier_integral*theta_u
                xl = self.ell_fourier_integral*theta_l
                self.K_gg.append(2/(xu**2 - xl**2)*(xu*jv(1,xu) - xl*jv(1,xl)))
                self.K_gm.append(2/(xu**2 - xl**2)*(-xu*jv(1,xu) + xl*jv(1,xl) -2*jv(0,xu) + 2*jv(0,xl)))
        if self.mm:
            for i_theta in range(len(self.theta_bins_lensing)):
                theta_u = self.theta_ul_bins_lensing[i_theta+1]/60/180*np.pi
                theta_l = self.theta_ul_bins_lensing[i_theta]/60/180*np.pi
                xu = self.ell_fourier_integral*theta_u
                xl = self.ell_fourier_integral*theta_l
                self.K_mmE.append(2/(xu**2 - xl**2)*(xu*jv(1,xu) - xl*jv(1,xl)))
                self.K_mmB.append(2/(xu**2 - xl**2)*((xu - 8/xu)*jv(1,xu) - 8*jv(2,xu) - (xl-8/xl)*jv(1,xl) + 8*jv(2,xl)))
        self.K_gg = np.array(self.K_gg)
        self.K_gm = np.array(self.K_gm)
        self.K_mmE = np.array(self.K_mmE)
        self.K_mmB = np.array(self.K_mmB)
        self.WXY_stack = []
        if self.gg:
            for i_theta in range(len(self.theta_bins_clustering)):
                self.WXY_stack.append(self.K_gg[i_theta,:])
            self.gg_summaries = len(self.theta_bins_clustering)
        if self.gm:
            for i_theta in range(len(self.theta_bins_clustering)):
                self.WXY_stack.append(self.K_gm[i_theta,:])
            self.gm_summaries = len(self.theta_bins_clustering)
        if self.mm:
            self.mmE_summaries = len(self.theta_bins_lensing)
            for i_theta in range(len(self.theta_bins_lensing)):
                self.WXY_stack.append(self.K_mmE[i_theta,:])
            for i_theta in range(len(self.theta_bins_lensing)):
                self.WXY_stack.append(self.K_mmB[i_theta,:])
        self.WXY_stack = np.array(self.WXY_stack)

    def __get_triplet_mix_term(self,
                               CovTHETASpace_settings,
                               survey_params_dict,
                               gauss_xipxip_mix,
                               gauss_xipxim_mix,
                               gauss_ximxim_mix):
        """
        Calculates the mixed term directly from a catalogue and therefore
        accounts for a more accurate prediction, especially at the survey
        edges
        """
        if 'xipxip' in CovTHETASpace_settings['mix_term_do_mix_for'][:] or 'ximxim' in CovTHETASpace_settings['mix_term_do_mix_for'][:] or 'xipxim' in CovTHETASpace_settings['mix_term_do_mix_for'][:]:
            print("")
            print("\rCalculating the mixed term from triplet counts")
            print("\rAllocating DiscreteDataClass")
            thisdata = DiscreteData(path_to_data=CovTHETASpace_settings['mix_term_file_path_catalog'], 
                    colname_weight=CovTHETASpace_settings['mix_term_col_name_weight'], 
                    colname_pos1=CovTHETASpace_settings['mix_term_col_name_pos1'], 
                    colname_pos2=CovTHETASpace_settings['mix_term_col_name_pos2'], 
                    colname_zbin=CovTHETASpace_settings['mix_term_col_name_zbin'], 
                    isspherical=CovTHETASpace_settings['mix_term_isspherical'], 
                    sigma2_eps= 4*survey_params_dict['ellipticity_dispersion']**2, 
                    target_patchsize=CovTHETASpace_settings['mix_term_target_patchsize'], 
                    do_overlap=CovTHETASpace_settings['mix_term_do_overlap'])
            if not thisdata.mixed_fail:
                print("\rBuilding patches")
                thisdata.gen_patches(func=cygnus_patches, 
                        func_args={"ra":thisdata.pos1, "dec":thisdata.pos2, 
                                    "g1":np.ones(len(thisdata.pos1)), "g2":np.ones(len(thisdata.pos1)), 
                                    "e1":np.ones(len(thisdata.pos1)), "e2":np.ones(len(thisdata.pos1)),
                                    "zbin":thisdata.zbin, "weight":thisdata.weight,
                                    "overlap_arcmin":CovTHETASpace_settings['mix_term_do_overlap']*self.theta_ul_bins[-1]})
                print("\rAllocating DiscreteCovTHETASpace")
                disccov = DiscreteCovTHETASpace(discrete=thisdata,
                                    xi_spl=self.xi_spline,
                                    bin_edges=self.theta_ul_bins,
                                    nmax=CovTHETASpace_settings['mix_term_nmax'],
                                    nbinsphi=CovTHETASpace_settings['mix_term_nbins_phi'],
                                    nsubbins=CovTHETASpace_settings['mix_term_nsubr'],
                                    do_ec=CovTHETASpace_settings['mix_term_do_ec'],
                                    nthreads=self.num_cores,
                                    savepath_triplets=CovTHETASpace_settings['mix_term_file_path_save_triplets'],
                                    loadpath_triplets=CovTHETASpace_settings['mix_term_file_path_load_triplets'],
                                    terms=CovTHETASpace_settings['mix_term_do_mix_for'])

                print("\rComputing triplets",end="")
                disccov.compute_triplets(fthin=CovTHETASpace_settings['mix_term_subsample'])
                print("\rComputing Mixed covariance",end="")
                gauxx_xipxip_mixed_reconsidered, gauxx_xipxim_mixed_reconsidered, gauxx_ximxim_mixed_reconsidered = \
                disccov.mixed_covariance()
                return gauxx_xipxip_mixed_reconsidered, gauxx_xipxim_mixed_reconsidered, gauxx_ximxim_mixed_reconsidered
            else:
                return gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix
        else:
            return None
        
    def __plot_signal(self,
                      output_dict):
        """
        Creates a corner plot of the signal
        
        Parameters
        ----------
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        """
        if not output_dict['make_plot']:
            return True
        else:
            import matplotlib.pyplot as plt
            ratio = self.data_vector_length / 140
            if output_dict['use_tex']:
                plt.rc('text', usetex=True)
            else:
                plt.rc('text', usetex=False)
            fig, ax = plt.subplots(1, 1, figsize=(12,12))            
            labels_position = []
            labels_position_y = []
            labels_text = []
            position = 0
            old_position = 0
            
            plt.yticks(labels_position_y, labels_text)
            plt.xticks(labels_position, labels_text)
            print("Plotting Signal")
        
    def calc_covTHETA(self,
                      obs_dict,
                      output_dict,
                      bias_dict,
                      hod_dict,
                      survey_params_dict,
                      prec,
                      read_in_tables):
        """
        Calculates the full covariance between all observables in real
        space as specified in the config file.

        Parameters
        ----------
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
            'COSEBIs' : dictionary
                Specifies the exact details of the projection to COSEBIs,
                e.g. the number of modes to be calculated.
            'bandpowers' : dictionary
                Specifies the exact details of the projection to bandpowers,
                e.g. the ell modes and their spacing.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
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

        Returns
        -------
        gauss, nongauss, ssc : list of arrays
            each with 10 entries for the observables
                ['ww', 'wgt', 'wxi+', 'wxi-', 'gtgt', 'xi+gt',
                 'xigt-', 'xi+xi+', 'xi+xi-', 'xi-xi-']
            each entry with shape (theta bins, theta bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        print("Calculating real space covariance from angular power " +
              "spectra (C_ell's).")
        
        
        if not self.cov_dict['split_gauss']:
            gauss_ww,     gauss_wgt,    gauss_wxip,  gauss_wxim, \
                gauss_gtgt,   gauss_xipgt,  gauss_ximgt, \
                gauss_xipxip, gauss_xipxim, \
                gauss_ximxim, \
                gauss_ww_sn,     gauss_gtgt_sn, \
                gauss_xipxip_sn, gauss_ximxim_sn = \
                self.covTHETA_gaussian(obs_dict['ELLspace'],
                                       survey_params_dict,
                                       read_in_tables['npair'])
            gauss = [gauss_ww + gauss_ww_sn, gauss_wgt, gauss_wxip, gauss_wxim,
                     gauss_gtgt + gauss_gtgt_sn, gauss_xipgt,  gauss_ximgt,
                     gauss_xipxip + gauss_xipxip_sn, gauss_xipxim,
                     gauss_ximxim + gauss_ximxim_sn]
        else:
            gauss = self.covTHETA_gaussian(obs_dict['ELLspace'],
                                           survey_params_dict,
                                           read_in_tables['npair'])
        #self.cov_dict['nongauss'] = False
        nongauss = self.covTHETA_non_gaussian(obs_dict['ELLspace'],
                                              survey_params_dict,
                                              output_dict,
                                              bias_dict,
                                              hod_dict,
                                              prec,
                                              read_in_tables['tri'], True)
 
        #self.cov_dict['ssc'] = False
        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            ssc = []
            for i_list in range(len(nongauss)):
                if nongauss[i_list] is not None:
                    ssc.append(nongauss[i_list]*0)
                else:
                    ssc.append(None)
        else:     
            ssc = self.covTHETA_ssc(obs_dict['ELLspace'],
                                    survey_params_dict,
                                    output_dict,
                                    bias_dict,
                                    hod_dict,
                                    prec,
                                    read_in_tables['tri'], True)
        return list(gauss), list(nongauss), list(ssc)

    def covTHETA_gaussian(self,
                          covELLspacesettings,
                          survey_params_dict,
                          calc_prefac=True):
        """
        Calculates the Gaussian (disconnected) covariance in real space
        for the all specified observables.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : boolean
            Specifies whether the ell-independent prefactor should be multiplied
            or not. Is set to True by default.

        Returns
        -------
        gauss_ww,     gauss_wgt,   gauss_wxip,  gauss_wxim, \
        gauss_gtgt,   gauss_xipgt, gauss_ximgt, \
        gauss_xipxip, gauss_xipxim, \
        gauss_ximxim, \
        gauss_ww_sn, gauss_gtgt_sn, gauss_xipm_sn : list of arrays
            with shape (theta bins, theta bins,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the diagonal terms of the covariance
        matrix, one needs to add gauss_xy + gauss_xy_sn. They
        are kept separate for later numerical reasons.

        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        print("Calculating gaussian real-space covariance from angular " +
              "power spectra (C_ell's).")

        if self.tomos_6x2pt_clust is not None:
            raise Exception("6x2pt not yet implement beyond C(l) covariance")

        gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
            gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
            gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
            gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
            gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
            gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
            gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
            gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipxip_sn, \
            gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
            gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn = \
            self.__covTHETA_split_gaussian(covELLspacesettings,
                                           survey_params_dict,
                                           calc_prefac)
        if self.theta_space_dict['mix_term_do_mix_for'] is not None:
            print("")
            print('\rDoing mixed term', end="")
            gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix = \
            self.__get_triplet_mix_term(self.theta_space_dict, survey_params_dict,
                                        gauss_xipxip_mix,gauss_xipxim_mix,gauss_ximxim_mix)
        if not self.cov_dict['split_gauss']:
            gauss_ww = gauss_ww_sva + gauss_ww_mix
            gauss_wgt = gauss_wgt_sva + gauss_wgt_mix
            gauss_wxip = gauss_wxip_sva + gauss_wxip_mix
            gauss_wxim = gauss_wxim_sva + gauss_wxim_mix
            gauss_gtgt = gauss_gtgt_sva + gauss_gtgt_mix
            gauss_xipgt = gauss_xipgt_sva + gauss_xipgt_mix
            gauss_ximgt = gauss_ximgt_sva + gauss_ximgt_mix
            gauss_xipxip = gauss_xipxip_sva + gauss_xipxip_mix
            gauss_xipxim = gauss_xipxim_sva + gauss_xipxim_mix
            gauss_ximxim = gauss_ximxim_sva + gauss_ximxim_mix
            return gauss_ww,     gauss_wgt,    gauss_wxip,  gauss_wxim, \
                gauss_gtgt,   gauss_xipgt,  gauss_ximgt, \
                gauss_xipxip, gauss_xipxim, \
                gauss_ximxim, \
                gauss_ww_sn, gauss_gtgt_sn, gauss_xipxip_sn, gauss_ximxim_sn
        else:
            return gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
                gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
                gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
                gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
                gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
                gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
                gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
                gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipxip_sn, \
                gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
                gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn

    def __covTHETA_split_gaussian(self,
                                  covELLspacesettings,
                                  survey_params_dict,
                                  calc_prefac):
        """
        Calculates the Gaussian (disconnected) covariance in real space
        for the specified observables and splits it into sample-variance
        (SVA), shot noise (SN) and SNxSVA(mix) terms.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : boolean
            Specifies whether the ell-independent prefactor should be multiplied
            or not. Is set to True by default.
        
        
        
        Returns:
        --------
            gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
            gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
            gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
            gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
            gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
            gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
            gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
            gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipxip_sn, \
            gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
            gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn : list of
                                                                 arrays
                with shape (theta bins, theta bins,
                            sample bins, sample bins,
                            n_tomo_clust/lens, n_tomo_clust/lens,
                            n_tomo_clust/lens, n_tomo_clust/lens)
                with gauss_xipxip_sn = gauss_ximxim_sn
        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        save_entry = self.cov_dict['split_gauss']
        self.cov_dict['split_gauss'] = True
        gaussELLgggg_sva, gaussELLgggg_mix, _, gaussELLgggm_sva, gaussELLgggm_mix, _, \
            gaussELLggmm_sva, _, _, gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
            gaussELLmmgm_sva, gaussELLmmgm_mix, _, gaussELLmmmm_sva, gaussELLmmmm_mix, _ = \
            self.covELL_gaussian(covELLspacesettings,
                                 survey_params_dict,
                                 calc_prefac=False)
        self.cov_dict['split_gauss'] = save_entry
        
        
        if self.gg or self.gm:
            kron_delta_tomo_clust = np.diag(np.ones(self.n_tomo_clust))
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))
        if self.mm or self.gm:
            kron_delta_tomo_lens = np.diag(survey_params_dict['ellipticity_dispersion']**2)
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))
        

        if self.gg:
            print("")
            original_shape = gaussELLgggg_sva[0, 0, :, :, :, :, :, :].shape
            covww_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                               self.sample_dim, self.sample_dim,
                               self.n_tomo_clust, self.n_tomo_clust,
                               self.n_tomo_clust, self.n_tomo_clust)
            gauss_ww_sva = np.zeros(covww_shape_sva)
            gauss_ww_mix = np.zeros(covww_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            gaussELLgggg_sva_flat = np.reshape(gaussELLgggg_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLgggg_mix_flat = np.reshape(gaussELLgggg_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)) **2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggg_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ww_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggg_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ww_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggg_sva_flat + gaussELLgggg_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ww_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                         
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance ww at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_ww_sn = \
                (kron_delta_tomo_clust[None, None, None, :, None, :, None]
                 * kron_delta_tomo_clust[None, None, None, None, :, None, :]
                 + kron_delta_tomo_clust[None, None, None, :, None, None, :]
                 * kron_delta_tomo_clust[None, None, None, None, :, :, None]) \
                 * kron_delta_mass_bins[None, :, :, None, None, None, None] \
                / self.npair_gg[:, :, None, :, :, None, None]
            gauss_ww_sn = \
                gauss_ww_sn[:, None, :, :, :, :, :, :] \
                * np.eye(len(self.theta_bins_clustering))[:, :, None, None, None, None, None, None]
        else:
            gauss_ww_sva, gauss_ww_mix, gauss_ww_sn = 0, 0 , 0

        if self.gg and self.gm and self.cross_terms:
            print("")
            original_shape = gaussELLgggm_sva[0, 0, :, :, :, :, :, :].shape
            covwgt_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                self.sample_dim, self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_clust,
                                self.n_tomo_clust, self.n_tomo_lens)
            gauss_wgt_sva = np.zeros(covwgt_shape_sva)
            gauss_wgt_mix = np.zeros(covwgt_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens*self.n_tomo_clust**3
            gaussELLgggm_sva_flat = np.reshape(gaussELLgggm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLgggm_mix_flat = np.reshape(gaussELLgggm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)) ** 2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggm_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_wgt_sva[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, self.gg_summaries)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggm_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_wgt_mix[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, self.gg_summaries)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgggm_sva_flat + gaussELLgggm_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_wgt_sva[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, self.gg_summaries)),original_shape)
                        
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance wgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
        else:
            gauss_wgt_sva, gauss_wgt_mix = 0, 0

        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = gaussELLggmm_sva[0, 0, :, :, :, :, :, :].shape
            covwxipm_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_lensing),
                                  self.sample_dim, 1,
                                  self.n_tomo_clust, self.n_tomo_clust,
                                  self.n_tomo_lens, self.n_tomo_lens)
            gauss_wxip_sva = np.zeros(covwxipm_shape_sva) if self.xi_pp else 0
            gauss_wxim_sva = np.zeros(covwxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            gaussELLggmm_sva_flat = np.reshape(gaussELLggmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)*len(self.theta_bins_lensing))
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLggmm_sva_flat)*self.ellrange,0,-1), True, True)
                    if self.xi_pp:
                        gauss_wxip_sva[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'], survey_params_dict['survey_area_lens'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    if self.xi_mm:
                        if len(local_ell_limit) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                        gauss_wxim_sva[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'], survey_params_dict['survey_area_lens'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance wxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
        else:
            gauss_wxip_sva, gauss_wxim_sva = 0, 0

        if self.gm:
            print("")
            original_shape = gaussELLgmgm_sva[0, 0, :, :, :, :, :, :].shape
            covgtgt_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                 self.sample_dim, self.sample_dim,
                                 self.n_tomo_clust, self.n_tomo_lens,
                                 self.n_tomo_clust, self.n_tomo_lens)
            gauss_gtgt_sva = np.zeros(covgtgt_shape_sva)
            gauss_gtgt_mix = np.zeros(covgtgt_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            gaussELLgmgm_sva_flat = np.reshape(gaussELLgmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLgmgm_mix_flat = np.reshape(gaussELLgmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)) **2
            for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgmgm_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_gtgt_sva[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgmgm_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_gtgt_mix[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLgmgm_sva_flat + gaussELLgmgm_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_gtgt_sva[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance gtgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_gtgt_sn = \
                kron_delta_tomo_clust[None, None, None, :, None, :, None] \
                * kron_delta_tomo_lens[None, None, None, None, :, None, :] \
                * kron_delta_mass_bins[None, :, :, None, None, None, None] \
                / self.npair_gm[:, :, None, :, :, None, None] 
            gauss_gtgt_sn = \
                gauss_gtgt_sn[:, None, :, :, :, :, :, :] \
                * np.eye(len(self.theta_bins_clustering))[:, :, None, None, None, None, None, None]
        

        else:
            gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn = 0, 0, 0

        if self.mm and self.gm and self.cross_terms:
            print("")
            original_shape = gaussELLmmgm_sva[0, 0, :, :, :, :, :, :].shape
            covxipmgt_shape_sva = (len(self.theta_bins_lensing), len(self.theta_bins_clustering),
                                   1, self.sample_dim,
                                   self.n_tomo_lens, self.n_tomo_lens,
                                   self.n_tomo_clust, self.n_tomo_lens)
            gauss_xipgt_sva = \
                np.zeros(covxipmgt_shape_sva) if self.xi_pp else 0
            gauss_ximgt_sva = \
                np.zeros(covxipmgt_shape_sva) if self.xi_mm else 0
            gauss_xipgt_mix = \
                np.zeros(covxipmgt_shape_sva) if self.xi_pp else 0
            gauss_ximgt_mix = \
                np.zeros(covxipmgt_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.n_tomo_clust*self.n_tomo_lens**3
            gaussELLmmgm_sva_flat = np.reshape(gaussELLmmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLmmgm_mix_flat = np.reshape(gaussELLmmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)*len(self.theta_bins_lensing))
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmgm_sva_flat)*self.ellrange,0,-1), True, True)
                        if self.xi_pp:
                            gauss_xipgt_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if self.xi_mm:
                            if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            gauss_ximgt_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmgm_mix_flat)*self.ellrange,0,-1), True, True)
                        local_ell_limit = self.ell_limits[m_mode][:]
                        if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                            local_ell_limit = self.ell_limits[n_mode][:]
                        if self.xi_pp:
                            gauss_xipgt_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if self.xi_mm:
                            if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            gauss_ximgt_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmgm_sva_flat + gaussELLmmgm_mix_flat)*self.ellrange,0,-1), True, True)
                        if self.xi_pp:
                            gauss_xipgt_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if self.xi_mm:
                            if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            gauss_ximgt_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance xipmgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")   
        else:
            gauss_xipgt_sva, gauss_ximgt_sva, gauss_xipgt_mix, gauss_ximgt_mix = 0, 0, 0 ,0

        if self.mm:
            print("")
            original_shape = gaussELLmmmm_sva[0, 0, :, :, :, :, :, :].shape
            covxipm_shape_sva = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                 1, 1,
                                 self.n_tomo_lens, self.n_tomo_lens,
                                 self.n_tomo_lens, self.n_tomo_lens)
            gauss_xipxip_sva = np.zeros(covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_sva = np.zeros(covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_sva = np.zeros(covxipm_shape_sva) if self.xi_mm else 0
            gauss_xipxip_mix = np.zeros(covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_mix = np.zeros(covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_mix = np.zeros(covxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.n_tomo_lens**4
            gaussELLmmmm_sva_flat = np.reshape(gaussELLmmmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLmmmm_mix_flat = np.reshape(gaussELLmmmm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_lensing)) **2
            

            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmmm_sva_flat)*self.ellrange,0,-1), True, True)
                        if self.xi_pp:
                            gauss_xipxip_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if self.xi_mm:
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            if len(self.ell_limits[m_mode + self.mmE_summaries][:]) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                            gauss_ximxim_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                            if self.cross_terms:
                                gauss_xipxim_sva[m_mode  - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmmm_mix_flat)*self.ellrange,0,-1), True, True)
                        if self.xi_pp:
                            local_ell_limit = self.ell_limits[m_mode][:]
                            if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                                local_ell_limit = self.ell_limits[n_mode][:]         
                            gauss_xipxip_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if self.xi_mm:
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[n_mode][:]        
                            gauss_ximxim_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                            if self.cross_terms:
                                gauss_xipxim_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELLmmmm_sva_flat + gaussELLmmmm_mix_flat)*self.ellrange,0,-1), True, True)
                        t01 = time.time()
                        if self.xi_pp:
                            gauss_xipxip_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        t01 = time.time()
                        if self.xi_mm:
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                            if len(self.ell_limits[m_mode + self.mmE_summaries][:]) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                                local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]        
                            gauss_ximxim_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                            if self.cross_terms:
                                gauss_xipxim_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape) 
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance xipmxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_xipm_sn = \
                (kron_delta_tomo_lens[None, None, None, :, None, :, None]
                 * kron_delta_tomo_lens[None, None, None, None, :, None, :]
                 + kron_delta_tomo_lens[None, None, None, :, None, None, :]
                 * kron_delta_tomo_lens[None, None, None, None, :, :, None]) \
                / self.npair_mm[:, :, None, :, :, None, None] / 0.5
            gauss_xipm_sn = \
                gauss_xipm_sn[:, None, :, :, :, :, :, :] \
                * np.eye(len(self.theta_bins_lensing))[:, :, None, None, None, None, None, None]
        else:
            gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva, gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix, gauss_xipm_sn = 0, 0, 0, 0, 0, 0, 0


        
        print("\nWrapping up all Gaussian real-space covariance contributions.")

        # all other shotnoise terms
        gauss_wgt_sn, gauss_wxip_sn, gauss_wxim_sn = 0, 0, 0
        gauss_xipgt_sn, gauss_ximgt_sn = 0, 0
        gauss_xipxim_sn = 0

        gauss_wxip_mix = 0
        gauss_wxim_mix = 0


        return gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
            gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
            gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
            gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
            gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
            gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
            gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
            gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipm_sn, \
            gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
            gauss_ximxim_sva, gauss_ximxim_mix, gauss_xipm_sn


    def covTHETA_non_gaussian(self,
                              covELLspacesettings,
                              survey_params_dict,
                              output_dict,
                              bias_dict,
                              hod_dict,
                              hm_prec,
                              tri_tab,
                              calc_prefac):
        """
        Calculates the non-Gaussian covariance between all observables in real
        space as specified in the config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
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
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        calc_prefac : boolean
            Specifies whether the ell-independent prefactor should be multiplied
            or not. Is set to True by default.

        Returns
        -------
            nongauss_ww, nongauss_wgt, nongauss_wxip, nongauss_wxim, nongauss_gtgt, \ 
            nongauss_xipgt, nongauss_ximgt, nongauss_xipxip, nongauss_xipxim, nongauss_ximxim
        
            each entry with shape (theta bins, theta bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covTHETA_4pt_projection(covELLspacesettings,
                                                  survey_params_dict,
                                                  output_dict,
                                                  bias_dict,
                                                  hod_dict,
                                                  hm_prec,
                                                  tri_tab,
                                                  calc_prefac,
                                                  True)

    def covTHETA_ssc(self,
                     covELLspacesettings,
                     survey_params_dict,
                     output_dict,
                     bias_dict,
                     hod_dict,
                     hm_prec,
                     tri_tab,
                     calc_prefac):
        """
        Calculates the super sample covariance between all observables in real
        space as specified in the config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
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
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        calc_prefac : boolean
            Specifies whether the ell-independent prefactor should be multiplied
            or not. Is set to True by default.

        Returns
        -------
            ssc_ww, ssc_wgt, ssc_wxip, ssc_wxim, ssc_gtgt, ssc_xipgt, ssc_ximgt, \
            ssc_xipxip, ssc_xipxim, ssc_ximxim
        
            each entry with shape (theta bins, theta bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covTHETA_4pt_projection(covELLspacesettings,
                                                  survey_params_dict,
                                                  output_dict,
                                                  bias_dict,
                                                  hod_dict,
                                                  hm_prec,
                                                  tri_tab,
                                                  calc_prefac,
                                                  False)

    def __covTHETA_4pt_projection(self,
                                  covELLspacesettings,
                                  survey_params_dict,
                                  output_dict,
                                  bias_dict,
                                  hod_dict,
                                  hm_prec,
                                  tri_tab,
                                  calc_prefac,
                                  connected):
        """
        Auxillary function to integrate four point functions in real space between all
        observables specified in the input file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
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
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        calc_prefac : boolean
            Specifies whether the ell-independent prefactor should be multiplied
            or not. Is set to True by default.
        connected : boolean
            If True the trispectrum is considered
            If False the SSC is considered

        Returns
        -------
            tri_ww, tri_wgt, tri_wxip, tri_wxim, tri_gtgt, \ 
            tri_xipgt, tri_ximgt, tri_xipxip, tri_xipxim, tri_ximxim
        
            each entry with shape (theta bins, theta bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        nongauss_ww = None
        nongauss_wgt = None
        nongauss_wxip = None
        nongauss_wxim = None
        nongauss_gtgt = None
        nongauss_xipgt = None
        nongauss_ximgt = None
        nongauss_xipxip = None
        nongauss_xipxim = None
        nongauss_ximxim = None
        self.levin_int_fourier.update_Levin(0, 16, 32,1e-3,1e-4)
        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, hm_prec, tri_tab)
            nongaussELLgggg1, nongaussELLgggm1, nongaussELLggmm1, nongaussELLgmgm1, nongaussELLmmgm1, nongaussELLmmmm1 = self.covELL_ssc(
                    bias_dict, hod_dict, hm_prec, survey_params_dict, covELLspacesettings)
            if self.gg:
                nongaussELLgggg = nongaussELLgggg/(survey_params_dict['survey_area_clust'] / self.deg2torad2) + nongaussELLgggg1
            if self.gg and self.gm and self.cross_terms:
                nongaussELLgggm = nongaussELLgggm/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) + nongaussELLgggm1
            if self.gg and self.mm and self.cross_terms:
                nongaussELLggmm = nongaussELLggmm/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2) + nongaussELLggmm1
            if self.gm:
                nongaussELLgmgm = nongaussELLggmm/(survey_params_dict['survey_area_ggl'] / self.deg2torad2) + nongaussELLgmgm1
            if self.mm and self.gm and self.cross_terms:
                nongaussELLmmgm = nongaussELLmmgm/(max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) + nongaussELLmmgm1
            if self.mm:
                nongaussELLmmmm = nongaussELLmmmm/(survey_params_dict['survey_area_lens'] / self.deg2torad2) + nongaussELLmmmm1
            connected = False
        else:
            if (connected):
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, hm_prec, tri_tab)
            else:
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_ssc(
                    bias_dict, hod_dict, hm_prec, survey_params_dict, covELLspacesettings)
        if self.gg:
            print("")
            original_shape = nongaussELLgggg[0, 0, :, :, :, :, :, :].shape
            covww_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                    self.sample_dim, self.sample_dim,
                                    self.n_tomo_clust, self.n_tomo_clust,
                                    self.n_tomo_clust, self.n_tomo_clust)
            nongauss_ww = np.zeros(covww_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            nongaussELLgggg_flat = np.reshape(nongaussELLgggg, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)) **2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLgggg_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))

                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_ww[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_ww[m_mode, n_mode, :, :, :, :, :, :] /= (survey_params_dict['survey_area_clust'] / self.deg2torad2)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)                    
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance ww at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")

        if self.gg and self.gm and self.cross_terms:
            print("")
            # cov_NG(w^ij(theta1) gt^kl(theta2))
            original_shape = nongaussELLgggm[0, 0, :, :, :, :, :, :].shape
            
            covwgt_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                     self.sample_dim, self.sample_dim,
                                     self.n_tomo_clust, self.n_tomo_clust,
                                     self.n_tomo_clust, self.n_tomo_lens)
            nongauss_wgt = np.zeros(covwgt_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust**3*self.n_tomo_lens
            nongaussELLgggm_flat = np.reshape(nongaussELLgggm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering))**2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLgggm_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_wgt[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_wgt[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance wgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = nongaussELLggmm[0, 0, :, :, :, :, :, :].shape
            covwxip_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_lensing),
                                      self.sample_dim, 1,
                                      self.n_tomo_clust, self.n_tomo_clust,
                                      self.n_tomo_lens, self.n_tomo_lens)
            nongauss_wxip = np.zeros(covwxip_shape_nongauss)
            covwxim_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_lensing),
                                      self.sample_dim, 1,
                                      self.n_tomo_clust, self.n_tomo_clust,
                                      self.n_tomo_lens, self.n_tomo_lens)
            nongauss_wxim = np.zeros(covwxim_shape_nongauss)
            flat_length = self.sample_dim * \
                self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELLggmm_flat = np.reshape(nongaussELLggmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)*len(self.theta_bins_lensing))
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    inner_integralE = np.zeros((len(self.ellrange), flat_length))
                    if self.xi_mm:
                        inner_integralB = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLggmm_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integralE[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                        if self.xi_mm:
                            inner_integralB[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode + self.mmE_summaries][:], n_mode + self.mmE_summaries))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralE*self.ellrange[:, None], True, True)
                    nongauss_wxip[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if self.xi_mm:
                        self.levin_int_fourier.init_integral(self.ellrange, inner_integralB*self.ellrange[:, None], True, True)
                        nongauss_wxim[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_wxip[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                        if self.xi_mm:
                            nongauss_wxim[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2)    
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                        'real-space covariance wxipm at ' +
                        str(round(theta/theta_comb*100, 1)) + '% in ' +
                        str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                        str(round(eta, 1)) + 'min', end="")
        if self.gm:
            print("")
            original_shape = nongaussELLgmgm[0, 0, :, :, :, :, :, :].shape
            covgtgt_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                      self.sample_dim, self.sample_dim,
                                      self.n_tomo_clust, self.n_tomo_lens,
                                      self.n_tomo_clust, self.n_tomo_lens)
            nongauss_gtgt = np.zeros(covgtgt_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELLgmgm_flat = np.reshape(nongaussELLgmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)) ** 2
            for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLgmgm_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_gtgt[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_gtgt[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_ggl']) / self.deg2torad2
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance gtgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
        if self.mm and self.gm and self.cross_terms:
            print("")
            #
            original_shape = nongaussELLmmgm[0, 0, :, :, :, :, :, :].shape
            covxipgt_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_clustering),
                                       1, self.sample_dim,
                                       self.n_tomo_lens, self.n_tomo_lens,
                                       self.n_tomo_clust, self.n_tomo_lens)
            nongauss_xipgt = np.zeros(covxipgt_shape_nongauss)
            covximgt_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_clustering),
                                       1, self.sample_dim,
                                       self.n_tomo_lens, self.n_tomo_lens,
                                       self.n_tomo_clust, self.n_tomo_lens)
            nongauss_ximgt = np.zeros(covximgt_shape_nongauss)
            flat_length = self.sample_dim * \
                self.n_tomo_clust*self.n_tomo_lens**3
            nongaussELLmmgm_flat = np.reshape(nongaussELLmmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_clustering)*len(self.theta_bins_lensing))
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLmmgm_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_xipgt[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if self.xi_mm:
                        nongauss_ximgt[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode + self.mmE_summaries][:], m_mode + self.mmE_summaries)),original_shape)
                    if connected:
                        nongauss_xipgt[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                        if self.xi_mm:
                            nongauss_ximgt[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                        'real-space covariance xipmgt at ' +
                        str(round(theta/theta_comb*100, 1)) + '% in ' +
                        str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                        str(round(eta, 1)) + 'min', end="")
        if self.mm:
            print("")
            original_shape = nongaussELLmmmm[0, 0, :, :, :, :, :, :].shape
            covxipxip_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                        1, 1,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_xipxip = np.zeros(covxipxip_shape_nongauss)
            covxipxim_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                        1, 1,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_xipxim = np.zeros(covxipxim_shape_nongauss)
            covximxim_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                        1, 1,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_ximxim = np.zeros(covximxim_shape_nongauss)
            flat_length = self.n_tomo_lens**4
            nongaussELLmmmm_flat = np.reshape(nongaussELLmmmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, theta = time.time(), 0
            theta_comb = (len(self.theta_bins_lensing))**2
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    inner_integralE = np.zeros((len(self.ellrange), flat_length))
                    if self.xi_mm:
                        inner_integralB = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELLmmmm_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integralE[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                        if self.xi_mm:
                            inner_integralB[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode + self.mmE_summaries][:], n_mode + self.mmE_summaries))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralE*self.ellrange[:, None], True, True)    
                    nongauss_xipxip[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if self.xi_mm:
                        self.levin_int_fourier.init_integral(self.ellrange, inner_integralB*self.ellrange[:, None], True, True)
                        nongauss_xipxim[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode ][:], m_mode)),original_shape)
                        nongauss_ximxim[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode + self.mmE_summaries][:], m_mode + self.mmE_summaries)),original_shape)
                    if connected:
                        nongauss_xipxip[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                        if self.xi_mm:
                            nongauss_xipxim[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                            nongauss_ximxim[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for connected term for the '
                          'real-space covariance xipmxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
        print("")
        return nongauss_ww, nongauss_wgt, nongauss_wxip, nongauss_wxim, nongauss_gtgt, nongauss_xipgt, nongauss_ximgt, nongauss_xipxip, nongauss_xipxim, nongauss_ximxim
