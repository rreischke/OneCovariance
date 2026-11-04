import time
from concurrent.futures import ThreadPoolExecutor
from math import factorial

import levin
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import jv

try:
    from onecov.cov_ell_space import CovELLSpace, LazyWeightsLevin
    from onecov.cov_orpheus import OrpheusCovTHETASpace
except:
    from cov_ell_space import CovELLSpace, LazyWeightsLevin
    from cov_orpheus import OrpheusCovTHETASpace


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
        self.bin_averaging = obs_dict['THETAspace']['bin_averaging']
        self.accuracy = obs_dict['THETAspace']['theta_acc']
        self.theta_space_dict = obs_dict['THETAspace']
        self.theta_space_discrete_dict = \
            obs_dict.get('THETAspace_discrete', {'do_discrete': False})
        # tabulated pair counts take precedence for the shot noise of the
        # discrete covariance
        self.npair_mm_tabulated = read_in_tables['npair']['npair_mm'] is not None
        self.thetabins, self.theta_ul_bins = \
            self.__set_theta_bins(obs_dict['THETAspace'])
        if ((obs_dict['observables']['est_shear'] == 'xi_pm' and obs_dict['observables']['cosmic_shear']) or (obs_dict['observables']['est_ggl'] == 'gamma_t' and obs_dict['observables']['ggl']) or obs_dict['observables']['est_clust'] == 'w' and obs_dict['observables']['clustering']):
            if self.gg or self.gm:
                save_n_eff_clust = survey_params_dict['n_eff_clust']
            if self.mm or self.gm:
                save_n_eff_lens = survey_params_dict['n_eff_lens']
            if self.sample_dim > 1:
                if self.gg or self.gm:
                    survey_params_dict['n_eff_clust'] = self._n_eff_clust_per_sample(survey_params_dict)
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
            if not self.cov_dict['sn_only']:
                self.__get_weights()
                self.ell_limits = []
                for mode in range(len(self.WXY_stack[:,0])):
                    limits_at_mode = np.array(self.ell_fourier_integral[argrelextrema(self.WXY_stack[mode,:], np.less)[0][:]])[::self.integration_intervals]
                    limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]) + 2)
                    limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]
                    # start integrating at the smallest multipole of the ell-space covariance
                    limits_at_mode_append[0] = self.ellrange[0]
                    limits_at_mode_append[-1] = self.ell_fourier_integral[-1]
                    self.ell_limits.append(limits_at_mode_append)
                # the weights are only passed to levin if a cquad integral needs them
                self.levin_int_fourier = LazyWeightsLevin(levin.Levin(0, 16, 32, obs_dict['THETAspace']['theta_acc']/np.sqrt(len(max(self.ell_limits, key=len))), self.integration_intervals, self.num_cores))
                self.levin_int_fourier.init_w_ell(self.ell_fourier_integral, self.WXY_stack.T)
                self._register_projection_weights(self.levin_int_fourier, self.ell_fourier_integral, self.WXY_stack,
                                                  obs_dict['THETAspace']['theta_acc'])
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
            if covTHETAspacesettings['theta_type_clustering'] == 'list':
                # bin edges given explicitly (e.g. the fine bins of CovTHETAtoOBS)
                self.theta_ul_bins_clustering = np.array(covTHETAspacesettings['theta_list_clustering'], dtype=float)
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
            if covTHETAspacesettings['theta_type_lensing'] == 'list':
                self.theta_ul_bins_lensing = np.array(covTHETAspacesettings['theta_list_lensing'], dtype=float)
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
            integrand = Cell_gg_flat*self.ellrange[:, None]*(self.pixel_weight**2)[:, None]
            projected = self._project_single(self.levin_int_fourier, self.ell_limits,
                                             list(range(len(self.theta_bins_clustering))), integrand)
            for i_theta in range(len(self.theta_bins_clustering)):
                w_signal[i_theta, :, :, :, :] = np.reshape(
                    projected[i_theta], original_shape)/2.0/np.pi
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
            integrand = Cell_gm_flat*self.ellrange[:, None]*(self.pixel_weight**2)[:, None]
            projected = self._project_single(self.levin_int_fourier, self.ell_limits,
                                             [i_theta + self.gg_summaries for i_theta in range(len(self.theta_bins_clustering))],
                                             integrand)
            for i_theta in range(len(self.theta_bins_clustering)):
                gt_signal[i_theta, :, :, :] = np.reshape(
                    projected[i_theta], original_shape)/2.0/np.pi
            self.gt = gt_signal
        if self.mm:
            self.data_vector_length += len(self.theta_bins_lensing)*self.n_tomo_lens*(self.n_tomo_lens +1)/2
            mm_signal_shape = (len(self.theta_bins_lensing),
                              1,
                              self.n_tomo_lens, self.n_tomo_lens)
            original_shape = self.Cell_mm[0, :, :, :].shape
            xip_signal = np.zeros(mm_signal_shape)
            xim_signal = np.zeros(mm_signal_shape)
            # cosmic shear has a single sample axis
            flat_length = self.n_tomo_lens*self.n_tomo_lens
            Cell_mm_flat = np.reshape(
                self.Cell_mm, (len(self.ellrange), flat_length))
            integrand = Cell_mm_flat*self.ellrange[:, None]*(self.pixel_weight**2)[:, None]
            n_theta = len(self.theta_bins_lensing)
            first = self.gg_summaries + self.gm_summaries
            projected = self._project_single(self.levin_int_fourier, self.ell_limits,
                                             [first + i for i in range(n_theta)]
                                             + [first + self.mmE_summaries + i for i in range(n_theta)],
                                             integrand)
            for i_theta in range(n_theta):
                xip_signal[i_theta, :, : ,:] = np.reshape(projected[i_theta], original_shape)/2.0/np.pi
                xim_signal[i_theta, :, : ,:] = np.reshape(projected[n_theta + i_theta], original_shape)/2.0/np.pi
            if self.Cell_mmB is not None:
                # intrinsic B-modes: xi_+ = int J_0 (C_EE + C_BB),
                # xi_- = int J_4 (C_EE - C_BB)
                n_theta = len(self.theta_bins_lensing)
                first = self.gg_summaries + self.gm_summaries
                xip_signal += self._Cell_mmB_projected(
                    self.levin_int_fourier, self.ell_limits,
                    [first + i_theta for i_theta in range(n_theta)],
                    [1/2.0/np.pi]*n_theta, self.pixel_weight)
                xim_signal += self._Cell_mmB_projected(
                    self.levin_int_fourier, self.ell_limits,
                    [first + self.mmE_summaries + i_theta for i_theta in range(n_theta)],
                    [-1/2.0/np.pi]*n_theta, self.pixel_weight)
            self.xi_plus = xip_signal
            self.xi_minus = xim_signal
            
    @staticmethod
    def __int_x_jn(x, n, j1=None):
        """
        Returns int_0^x t J_n(t) dt for n = 2 or 4, the antiderivative needed
        for the bin-averaged J_2 and J_4 weights. The closed form suffers from
        catastrophic cancellation for small x (the result goes like x^(n+2)),
        so the power series is used for x < 2. j1 = J_1(x) can be passed if
        it is known already.
        """
        if j1 is None:
            j1 = jv(1, x)
        if n == 2:
            result = -x*j1 - 2*jv(0,x) + 2
        elif n == 4:
            result = (x - 8/x)*j1 - 8*jv(2,x) + 4
        else:
            raise ValueError("only implemented for n = 2, 4")
        small = x < 2
        if np.any(small):
            x_series = x[small]
            series = np.zeros_like(x_series)
            for k in range(15):
                series += (-1)**k * x_series**(2*k + n + 2) \
                    / float(2**(2*k + n) * factorial(k) * factorial(k + n) * (2*k + n + 2))
            result[small] = series
        return result

    def __get_weights(self):
        """
        Sets the (bin-averaged) Bessel kernels of all real-space bins,
        self.WXY_stack with shape (gg, gm, mm_+ and mm_- bins,
        len(self.ell_fourier_integral)). The kernels are written directly
        into the stack, which is the only copy (it is large for many bins).
        With bin averaging, the antiderivatives int_0^x t J_n(t) dt are
        evaluated once per bin edge (neighbouring bins share them).
        """
        N_fourier = int(1e5)
        self.gg_summaries = 0
        self.gm_summaries = 0
        self.mmE_summaries = 0
        self.ell_fourier_integral = np.geomspace(1,1e5,N_fourier)
        if self.gg:
            self.gg_summaries = len(self.theta_bins_clustering)
        if self.gm:
            self.gm_summaries = len(self.theta_bins_clustering)
        if self.mm:
            self.mmE_summaries = len(self.theta_bins_lensing)
        self.WXY_stack = np.zeros((self.gg_summaries + self.gm_summaries + 2*self.mmE_summaries,
                                   N_fourier))
        # (Bessel order, first row) of the kernels of the clustering and the lensing bins
        groups = []
        if self.gg or self.gm:
            blocks = []
            if self.gg:
                blocks.append((0, 0))
            if self.gm:
                blocks.append((2, self.gg_summaries))
            groups.append((self.theta_ul_bins_clustering, self.theta_bins_clustering, blocks))
        if self.mm:
            first = self.gg_summaries + self.gm_summaries
            groups.append((self.theta_ul_bins_lensing, self.theta_bins_lensing,
                           [(0, first), (4, first + self.mmE_summaries)]))
        for theta_ul_bins, theta_bins, blocks in groups:
            if not self.bin_averaging:
                # at the bin centres (theta in arcmin)
                for i_theta in range(len(theta_bins)):
                    x = self.ell_fourier_integral*theta_bins[i_theta]/60/180*np.pi
                    for n, row in blocks:
                        self.WXY_stack[row + i_theta] = jv(n, x)
                continue
            def at_edge(edge, blocks=blocks):
                x = self.ell_fourier_integral*edge/60/180*np.pi
                j1 = jv(1, x)
                return x, {n: x*j1 if n == 0 else self.__int_x_jn(x, n, j1) for n, _ in blocks}

            # the Bessel functions of the edges in parallel (numpy/scipy ufuncs
            # release the GIL, no OpenMP involved), in chunks to bound the memory
            n_workers = max(1, int(getattr(self, 'num_cores', 1)))
            previous = None
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for start in range(0, len(theta_ul_bins), 4*n_workers):
                    chunk = theta_ul_bins[start:start + 4*n_workers]
                    for i_edge, (x, antiderivative) in enumerate(pool.map(at_edge, chunk), start=start):
                        if previous is not None:
                            x_lower, antiderivative_lower = previous
                            norm = 2/(x**2 - x_lower**2)
                            for n, row in blocks:
                                self.WXY_stack[row + i_edge - 1] = norm*(antiderivative[n] - antiderivative_lower[n])
                        previous = (x, antiderivative)

    def __project_gauss(self,
                        sva_flat,
                        mix_flat,
                        sva_jobs,
                        mix_jobs,
                        prefac,
                        original_shape,
                        name,
                        added=False):
        """
        Projects the Gaussian ell-space covariance to real space with
        CovELLSpace._project_gauss_jobs (with added=True the results are
        added to the target arrays, CovELLSpace._project_gauss_jobs_added).

        Parameters
        ----------
        sva_flat, mix_flat : arrays
            with shape (len(ellrange), len(ellrange), number of integrals),
            the sample variance and mixed term of the ell-space covariance.
            mix_flat can be None if there is no mixed term.
        sva_jobs, mix_jobs : list
            of jobs (m_mode, n_mode, target_array, target_index[,
            mirror_index]), the result of the integral over the weights
            m_mode and n_mode is stored in target_array[target_index] =
            prefac * result. For auto-correlation blocks the integral is
            symmetric in m_mode and n_mode, so it is also stored in
            target_array[mirror_index] if given. If the Gaussian term is not
            split, the sum of sva and mix is projected with the sva_jobs.
        """
        def with_limits(jobs):
            return [(self._ell_limits_double(self.ell_limits, job[0], job[1]), job[0], job[1], job[2], job[3], prefac) + tuple(job[4:])
                    for job in jobs]
        project = self._project_gauss_jobs_added if added else self._project_gauss_jobs
        project(self.levin_int_fourier,
                self._gauss_todo(sva_flat, mix_flat, with_limits(sva_jobs), with_limits(mix_jobs)),
                original_shape, 'real-space covariance ' + name)

    def __project_4pt(self,
                      ell_cov,
                      jobs,
                      area,
                      original_shape,
                      name):
        """
        Projects a four-point (non-Gaussian or super-sample) ell-space
        covariance to real space with CovELLSpace._project_4pt_jobs.

        Parameters
        ----------
        ell_cov : array
            with shape (len(ellrange), len(ellrange), ...)
        jobs : list
            of jobs (outer_mode, inner_mode, target_array, target_index)
        area : float or None
            if not None, the results are divided by it (connected term)
        """
        prefac = 1.0/(4.0*np.pi**2)
        self._project_4pt_jobs(self.levin_int_fourier, self.ell_limits, ell_cov,
                               [job + (prefac, area) for job in jobs],
                               original_shape, 'real-space covariance ' + name)

    def __get_xi_tables(self,
                        theta,
                        chunk=128):
        """
        Tabulates xi_+ and xi_- of all lensing bin pairs at arbitrary
        separations, e.g. far beyond the theta bins as needed by the
        discrete covariance. The Bessel weights are set up for at most
        'chunk' separations at a time, which bounds the memory.

        Parameters
        ----------
        theta : array
            Separations in arcmin.
        chunk : int
            Number of separations integrated together.

        Returns
        -------
        xip, xim : arrays
            with shape (len(theta), n_tomo_lens, n_tomo_lens)
        """
        def bessel_weights(theta_chunk):
            return np.array([jv(n_bessel, self.ell_fourier_integral*theta_i/60/180*np.pi)
                             for n_bessel in (0, 4) for theta_i in theta_chunk])

        chunks = [theta[i:i + chunk] for i in range(0, len(theta), chunk)]
        # the integration limits of all separations come first, since the
        # longest list of them sets the accuracy of levin
        ell_limits = []
        for theta_chunk in chunks:
            WXY_stack = bessel_weights(theta_chunk)
            limits_chunk = []
            for mode in range(len(WXY_stack[:,0])):
                limits_at_mode = np.array(self.ell_fourier_integral[argrelextrema(WXY_stack[mode,:], np.less)[0][:]])[::self.integration_intervals]
                limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]) + 2)
                limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.ellrange[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]
                # start integrating at the smallest multipole of the ell-space covariance
                limits_at_mode_append[0] = self.ellrange[0]
                limits_at_mode_append[-1] = self.ell_fourier_integral[-1]
                limits_chunk.append(limits_at_mode_append)
            ell_limits.append(limits_chunk)
        accuracy = self.accuracy/np.sqrt(max(len(limits) for limits_chunk in ell_limits for limits in limits_chunk))
        flat_length = self.n_tomo_lens*self.n_tomo_lens
        Cell_mm_flat = np.reshape(self.Cell_mm, (len(self.ellrange), flat_length))
        integrand = Cell_mm_flat*self.ellrange[:, None]*(self.pixel_weight**2)[:, None]
        xip = np.zeros((len(theta), self.n_tomo_lens, self.n_tomo_lens))
        xim = np.zeros((len(theta), self.n_tomo_lens, self.n_tomo_lens))
        start = 0
        for theta_chunk, limits_chunk in zip(chunks, ell_limits):
            n_theta = len(theta_chunk)
            levin_int_fourier = levin.Levin(0, 16, 32, accuracy, self.integration_intervals, self.num_cores)
            levin_int_fourier.init_w_ell(self.ell_fourier_integral, bessel_weights(theta_chunk).T)
            levin_int_fourier.init_integral(self.ellrange, integrand, True, True)
            for i_theta in range(n_theta):
                xip[start + i_theta, :, :] = np.reshape(levin_int_fourier.cquad_integrate_single_well(limits_chunk[i_theta], i_theta),
                                                        (self.n_tomo_lens, self.n_tomo_lens))/2.0/np.pi
                xim[start + i_theta, :, :] = np.reshape(levin_int_fourier.cquad_integrate_single_well(limits_chunk[i_theta + n_theta], i_theta + n_theta),
                                                        (self.n_tomo_lens, self.n_tomo_lens))/2.0/np.pi
            if self.Cell_mmB is not None:
                # the correlation functions of the full shear field include the
                # intrinsic B-modes, xi_+ = int J_0 (C_EE + C_BB) and
                # xi_- = int J_4 (C_EE - C_BB)
                xip[start:start + n_theta] += np.reshape(self._Cell_mmB_projected(levin_int_fourier, limits_chunk, list(range(n_theta)),
                                                                                  [1/2.0/np.pi]*n_theta, self.pixel_weight),
                                                         (n_theta, self.n_tomo_lens, self.n_tomo_lens))
                xim[start:start + n_theta] += np.reshape(self._Cell_mmB_projected(levin_int_fourier, limits_chunk, [n_theta + i for i in range(n_theta)],
                                                                                  [-1/2.0/np.pi]*n_theta, self.pixel_weight),
                                                         (n_theta, self.n_tomo_lens, self.n_tomo_lens))
            start += n_theta
        return xip, xim

    def __get_discrete_gaussian(self,
                                survey_params_dict,
                                gauss_xipxip_sva,
                                gauss_xipxim_sva,
                                gauss_ximxim_sva,
                                gauss_xipxip_mix,
                                gauss_xipxim_mix,
                                gauss_ximxim_mix,
                                gauss_xipxip_sn,
                                gauss_ximxim_sn):
        """
        Replaces the Gaussian xi_pm terms listed in [covTHETAspace
        discrete]: 'replace_terms' by the discrete covariance computed
        with orpheus (see OrpheusCovTHETASpace). The uncertainty of the
        multiplicative shear bias is added to the new sample-variance
        term as to the one it replaces. The shot noise is kept if the
        pair counts are tabulated, since they take precedence.

        Returns
        -------
        gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva, \
        gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix, \
        gauss_xipxip_sn, gauss_ximxim_sn : arrays
            the replaced blocks as returned by _theta_block, here with
            shape (theta bins, theta bins, 1, 1,
                   n_tomo_lens, n_tomo_lens, n_tomo_lens, n_tomo_lens)
        """
        replace_terms = self.theta_space_discrete_dict['replace_terms']
        disccov = OrpheusCovTHETASpace(self.theta_space_discrete_dict,
                                       self.theta_ul_bins_lensing,
                                       survey_params_dict['n_eff_lens'],
                                       survey_params_dict['ellipticity_dispersion'],
                                       survey_params_dict['survey_area_lens'][0],
                                       self.num_cores, self.cache_dir)
        xi_theta = disccov.get_xi_theta()
        if self.cov_dict['sn_only']:
            xip = np.zeros((len(xi_theta), self.n_tomo_lens, self.n_tomo_lens))
            xim = np.zeros((len(xi_theta), self.n_tomo_lens, self.n_tomo_lens))
        else:
            print("Tabulating xi_pm at " + str(len(xi_theta)) + " separations between " +
                  str(np.round(xi_theta[0], 4)) + " and " + str(np.round(xi_theta[-1], 1)) +
                  " arcmin for the discrete covariance.")
            xip, xim = self.__get_xi_tables(xi_theta)
        sva, mix, sn = disccov.calc_covTHETA_discrete(xi_theta, xip, xim)

        if 'sva' in replace_terms:
            if not self.cov_dict['sn_only']:
                sva['xipxip'] = self._theta_add_product(sva['xipxip'], self.gaussELLmmmm_sva_mult_shear_bias, self.xi_plus, self.xi_plus)
                sva['xipxim'] = self._theta_add_product(sva['xipxim'], self.gaussELLmmmm_sva_mult_shear_bias, self.xi_plus, self.xi_minus)
                sva['ximxim'] = self._theta_add_product(sva['ximxim'], self.gaussELLmmmm_sva_mult_shear_bias, self.xi_minus, self.xi_minus)
            if isinstance(gauss_xipxip_sva, np.ndarray):
                gauss_xipxip_sva = self._theta_block('xipxip', sva['xipxip'])
            if isinstance(gauss_xipxim_sva, np.ndarray):
                gauss_xipxim_sva = self._theta_block('xipxim', sva['xipxim'])
            if isinstance(gauss_ximxim_sva, np.ndarray):
                gauss_ximxim_sva = self._theta_block('ximxim', sva['ximxim'])
        if 'mix' in replace_terms:
            if isinstance(gauss_xipxip_mix, np.ndarray):
                gauss_xipxip_mix = self._theta_block('xipxip', mix['xipxip'])
            if isinstance(gauss_xipxim_mix, np.ndarray):
                gauss_xipxim_mix = self._theta_block('xipxim', mix['xipxim'])
            if isinstance(gauss_ximxim_mix, np.ndarray):
                gauss_ximxim_mix = self._theta_block('ximxim', mix['ximxim'])
        if 'sn' in replace_terms:
            if self.npair_mm_tabulated:
                print("The shot noise of xi_pm is kept from the tabulated pair counts.")
            else:
                gauss_xipxip_sn = self._theta_block('xipxip', sn['xipxip'])
                gauss_ximxim_sn = self._theta_block('ximxim', sn['ximxim'])

        return gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva, \
            gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix, \
            gauss_xipxip_sn, gauss_ximxim_sn

    def __plot_signal(self,
                      output_dict):
        """
        Creates a corner plot of the signal
        
        Parameters
        ----------
        output_dict : dictionary
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
        
    def _theta_target(self,
                      name,
                      shape):
        """
        The array for a block of the real-space covariance ('ww', 'wgt',
        ..., see _theta_block) with shape (theta bins, theta bins, ...),
        which the projections fill entry by entry, block[(i, j)] = value.
        Here a new zero array; subclasses can return an object that maps the
        entries elsewhere as they are set (CovTHETAtoOBS).
        """
        return np.zeros(shape)

    def _theta_diagonal(self,
                        name,
                        diagonal):
        """
        A block of the real-space covariance which is diagonal in the
        angular bins (shot noise) from its diagonal with shape
        (theta bins, ...).
        """
        return diagonal[:, None] \
            * np.eye(len(diagonal))[(slice(None), slice(None)) + (None,)*(diagonal.ndim - 1)]

    def _theta_add_product(self,
                           block,
                           matrix,
                           left,
                           right):
        """
        block + matrix x left(theta_1) x right(theta_2), the contribution of
        the multiplicative shear bias, with matrix of shape (sample bins,
        sample bins, tomo, tomo, tomo, tomo) and the signals left and right
        with shape (theta bins, sample bins, tomo, tomo).
        """
        return block + matrix[None, None]*(left[:, None, :, None, :, :, None, None]*right[None, :, None, :, None, None, :, :])

    def _theta_block(self,
                     name,
                     block):
        """
        Called with every block of the real-space covariance as soon as it
        is complete (all terms that are added to it included), returns what
        is kept in its place. Here the block itself. Subclasses can replace
        or modify terms here, or map the block to other summary statistics
        (CovTHETAtoOBS) so that the blocks in many real-space bins are not
        kept.

        Parameters
        ----------
        name : string
            'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 'ximgt', 'xipxip',
            'xipxim' or 'ximxim'
        block : array or 0
            with shape (theta bins, theta bins, sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)
        """
        return block

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
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
            if self.csmf:
                gauss_ww,     gauss_wgt,    gauss_wxip,  gauss_wxim, \
                    gauss_gtgt,   gauss_xipgt,  gauss_ximgt, \
                    gauss_xipxip, gauss_xipxim, \
                    gauss_ximxim, \
                    gauss_ww_sn,     gauss_gtgt_sn, \
                    gauss_xipxip_sn, gauss_ximxim_sn, \
                    csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim = \
                    self.covTHETA_gaussian(obs_dict['ELLspace'],
                                        survey_params_dict,
                                        read_in_tables['npair'])
                gauss = [gauss_ww + gauss_ww_sn, gauss_wgt, gauss_wxip, gauss_wxim,
                        gauss_gtgt + gauss_gtgt_sn, gauss_xipgt,  gauss_ximgt,
                        gauss_xipxip + gauss_xipxip_sn, gauss_xipxim,
                        gauss_ximxim + gauss_ximxim_sn,
                        csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim]
            else:
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
        return self._split_csmf_terms(list(gauss), list(nongauss), list(ssc))

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

        if self.csmf:
            gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
                gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
                gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
                gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
                gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
                gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
                gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
                gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipxip_sn, \
                gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
                gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn, \
                csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim = \
                self.__covTHETA_split_gaussian(covELLspacesettings,
                                            survey_params_dict,
                                            calc_prefac)
        else:
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
        if self.theta_space_discrete_dict['do_discrete'] and self.mm:
            print("")
            print("Calculating the discrete Gaussian covariance of xi_pm with orpheus.")
            gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva, \
                gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix, \
                gauss_xipxip_sn, gauss_ximxim_sn = \
                self.__get_discrete_gaussian(survey_params_dict,
                                             gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva,
                                             gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix,
                                             gauss_xipxip_sn, gauss_ximxim_sn)

        if self.csmf:
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
                    gauss_ww_sn, gauss_gtgt_sn, gauss_xipxip_sn, gauss_ximxim_sn, \
                    csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim

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
                    gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn, \
                    csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim
        elif not self.cov_dict['split_gauss']:
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
        if self.csmf:
            gaussELLgggg_sva, gaussELLgggg_mix, _, \
            gaussELLgggm_sva, gaussELLgggm_mix, _, \
            gaussELLggmm_sva, _, _, \
            gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
            gaussELLmmgm_sva, gaussELLmmgm_mix, _, \
            gaussELLmmmm_sva, gaussELLmmmm_mix, _, \
            csmf_auto, csmf_gg, csmf_gm, csmf_mm = self.covELL_gaussian(
                covELLspacesettings, survey_params_dict, False)
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, _, \
                gaussELLgggm_sva, gaussELLgggm_mix, _, \
                gaussELLggmm_sva, _, _, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, _, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, _ = self.covELL_gaussian(
                    covELLspacesettings, survey_params_dict, False)
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
            gauss_ww_sva = self._theta_target('ww', covww_shape_sva)
            gauss_ww_mix = self._theta_target('ww', covww_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            gaussELLgggg_sva_flat = self._ell_cov_flat(gaussELLgggg_sva, flat_length)
            gaussELLgggg_mix_flat = self._ell_cov_flat(gaussELLgggg_mix, flat_length)
            if not self.cov_dict['sn_only']:
                sva_jobs, mix_jobs = [], []
                for m_mode in range(self.gg_summaries):
                    for n_mode in range(m_mode, self.gg_summaries):
                        idx, mirror = (m_mode, n_mode), (n_mode, m_mode)
                        sva_jobs.append((m_mode, n_mode, gauss_ww_sva, idx, mirror))
                        mix_jobs.append((m_mode, n_mode, gauss_ww_mix, idx, mirror))
                self.__project_gauss(gaussELLgggg_sva_flat, gaussELLgggg_mix_flat, sva_jobs, mix_jobs, 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2), original_shape, 'ww')
            gauss_ww_sn = \
                (kron_delta_tomo_clust[None, None, None, :, None, :, None]
                 * kron_delta_tomo_clust[None, None, None, None, :, None, :]
                 + kron_delta_tomo_clust[None, None, None, :, None, None, :]
                 * kron_delta_tomo_clust[None, None, None, None, :, :, None]) \
                 * self._sample_overlap_sn(1.0/self.npair_gg, True)[:, :, :, :, :, None, None]
            gauss_ww_sn = self._theta_diagonal('ww', gauss_ww_sn)
            gauss_ww_sva = self._theta_block('ww', gauss_ww_sva)
            gauss_ww_mix = self._theta_block('ww', gauss_ww_mix)
            gauss_ww_sn = self._theta_block('ww', gauss_ww_sn)
        else:
            gauss_ww_sva, gauss_ww_mix, gauss_ww_sn = 0, 0 , 0

        if self.gg and self.gm and self.cross_terms:
            print("")
            original_shape = gaussELLgggm_sva[0, 0, :, :, :, :, :, :].shape
            covwgt_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                self.sample_dim, self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_clust,
                                self.n_tomo_clust, self.n_tomo_lens)
            gauss_wgt_sva = self._theta_target('wgt', covwgt_shape_sva)
            gauss_wgt_mix = self._theta_target('wgt', covwgt_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens*self.n_tomo_clust**3
            gaussELLgggm_sva_flat = self._ell_cov_flat(gaussELLgggm_sva, flat_length)
            gaussELLgggm_mix_flat = self._ell_cov_flat(gaussELLgggm_mix, flat_length)
            
            if not self.cov_dict['sn_only']:
                sva_jobs, mix_jobs = [], []
                for m_mode in range(self.gg_summaries):
                    for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                        idx = (m_mode, n_mode - self.gg_summaries)
                        sva_jobs.append((m_mode, n_mode, gauss_wgt_sva, idx))
                        mix_jobs.append((m_mode, n_mode, gauss_wgt_mix, idx))
                self.__project_gauss(gaussELLgggm_sva_flat, gaussELLgggm_mix_flat, sva_jobs, mix_jobs, 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2), original_shape, 'wgt')
            gauss_wgt_sva = self._theta_block('wgt', gauss_wgt_sva)
            gauss_wgt_mix = self._theta_block('wgt', gauss_wgt_mix)
        else:
            gauss_wgt_sva, gauss_wgt_mix = 0, 0

        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = gaussELLggmm_sva[0, 0, :, :, :, :, :, :].shape
            covwxipm_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_lensing),
                                  self.sample_dim, 1,
                                  self.n_tomo_clust, self.n_tomo_clust,
                                  self.n_tomo_lens, self.n_tomo_lens)
            gauss_wxip_sva = self._theta_target('wxip', covwxipm_shape_sva) if self.xi_pp else 0
            gauss_wxim_sva = self._theta_target('wxim', covwxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            gaussELLggmm_sva_flat = self._ell_cov_flat(gaussELLggmm_sva, flat_length)
            if not self.cov_dict['sn_only']:
                sva_jobs = []
                for m_mode in range(self.gg_summaries):
                    for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                        idx = (m_mode, n_mode - self.gg_summaries - self.gm_summaries)
                        if self.xi_pp:
                            sva_jobs.append((m_mode, n_mode, gauss_wxip_sva, idx))
                        if self.xi_mm:
                            sva_jobs.append((m_mode, n_mode + self.mmE_summaries, gauss_wxim_sva, idx))
                self.__project_gauss(gaussELLggmm_sva_flat, None, sva_jobs, [], 1./(2.*np.pi*max(survey_params_dict['survey_area_clust'], survey_params_dict['survey_area_lens'])/self.deg2torad2), original_shape, 'wxipm')
            gauss_wxip_sva = self._theta_block('wxip', gauss_wxip_sva)
            gauss_wxim_sva = self._theta_block('wxim', gauss_wxim_sva)
        else:
            gauss_wxip_sva, gauss_wxim_sva = 0, 0

        if self.gm:
            print("")
            original_shape = gaussELLgmgm_sva[0, 0, :, :, :, :, :, :].shape
            covgtgt_shape_sva = (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                 self.sample_dim, self.sample_dim,
                                 self.n_tomo_clust, self.n_tomo_lens,
                                 self.n_tomo_clust, self.n_tomo_lens)
            gauss_gtgt_sva = self._theta_target('gtgt', covgtgt_shape_sva)
            gauss_gtgt_mix = self._theta_target('gtgt', covgtgt_shape_sva)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            gaussELLgmgm_sva_flat = self._ell_cov_flat(gaussELLgmgm_sva, flat_length)
            gaussELLgmgm_mix_flat = self._ell_cov_flat(gaussELLgmgm_mix, flat_length)
            if not self.cov_dict['sn_only']:
                sva_jobs, mix_jobs = [], []
                for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    for n_mode in range(m_mode, self.gm_summaries + self.gg_summaries):
                        idx = (m_mode - self.gg_summaries, n_mode - self.gg_summaries)
                        mirror = (idx[1], idx[0])
                        sva_jobs.append((m_mode, n_mode, gauss_gtgt_sva, idx, mirror))
                        mix_jobs.append((m_mode, n_mode, gauss_gtgt_mix, idx, mirror))
                self.__project_gauss(gaussELLgmgm_sva_flat, gaussELLgmgm_mix_flat, sva_jobs, mix_jobs, 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2), original_shape, 'gtgt')
                gauss_gtgt_sva = self._theta_add_product(gauss_gtgt_sva, self.gaussELLgmgm_sva_mult_shear_bias, self.gt, self.gt)
            gauss_gtgt_sn = \
                kron_delta_tomo_clust[None, None, None, :, None, :, None] \
                * kron_delta_tomo_lens[None, None, None, None, :, None, :] \
                * self._sample_overlap_sn(1.0/self.npair_gm, False)[:, :, :, :, :, None, None]
            gauss_gtgt_sn = self._theta_diagonal('gtgt', gauss_gtgt_sn)
            gauss_gtgt_sva = self._theta_block('gtgt', gauss_gtgt_sva)
            gauss_gtgt_mix = self._theta_block('gtgt', gauss_gtgt_mix)
            gauss_gtgt_sn = self._theta_block('gtgt', gauss_gtgt_sn)
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
                self._theta_target('xipgt', covxipmgt_shape_sva) if self.xi_pp else 0
            gauss_ximgt_sva = \
                self._theta_target('ximgt', covxipmgt_shape_sva) if self.xi_mm else 0
            gauss_xipgt_mix = \
                self._theta_target('xipgt', covxipmgt_shape_sva) if self.xi_pp else 0
            gauss_ximgt_mix = \
                self._theta_target('ximgt', covxipmgt_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.n_tomo_clust*self.n_tomo_lens**3
            gaussELLmmgm_sva_flat = self._ell_cov_flat(gaussELLmmgm_sva, flat_length)
            gaussELLmmgm_mix_flat = self._ell_cov_flat(gaussELLmmgm_mix, flat_length)
            
            if not self.cov_dict['sn_only']:
                sva_jobs, mix_jobs = [], []
                for m_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                        idx = (m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries)
                        if self.xi_pp:
                            sva_jobs.append((m_mode, n_mode, gauss_xipgt_sva, idx))
                            mix_jobs.append((m_mode, n_mode, gauss_xipgt_mix, idx))
                        if self.xi_mm:
                            sva_jobs.append((m_mode + self.mmE_summaries, n_mode, gauss_ximgt_sva, idx))
                            mix_jobs.append((m_mode + self.mmE_summaries, n_mode, gauss_ximgt_mix, idx))
                self.__project_gauss(gaussELLmmgm_sva_flat, gaussELLmmgm_mix_flat, sva_jobs, mix_jobs, 1./(2.*np.pi*max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2), original_shape, 'xipmgt')
                if self.xi_mm:
                    gauss_ximgt_sva = self._theta_add_product(gauss_ximgt_sva, self.gaussELLmmgm_sva_mult_shear_bias, self.xi_minus, self.gt)
                if self.xi_pp:
                    gauss_xipgt_sva = self._theta_add_product(gauss_xipgt_sva, self.gaussELLmmgm_sva_mult_shear_bias, self.xi_plus, self.gt)
            gauss_xipgt_sva = self._theta_block('xipgt', gauss_xipgt_sva)
            gauss_xipgt_mix = self._theta_block('xipgt', gauss_xipgt_mix)
            gauss_ximgt_sva = self._theta_block('ximgt', gauss_ximgt_sva)
            gauss_ximgt_mix = self._theta_block('ximgt', gauss_ximgt_mix)
        else:
            gauss_xipgt_sva, gauss_ximgt_sva, gauss_xipgt_mix, gauss_ximgt_mix = 0, 0, 0 ,0

        if self.mm:
            print("")
            original_shape = gaussELLmmmm_sva[0, 0, :, :, :, :, :, :].shape
            covxipm_shape_sva = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                 1, 1,
                                 self.n_tomo_lens, self.n_tomo_lens,
                                 self.n_tomo_lens, self.n_tomo_lens)
            gauss_xipxip_sva = self._theta_target('xipxip', covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_sva = self._theta_target('xipxim', covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_sva = self._theta_target('ximxim', covxipm_shape_sva) if self.xi_mm else 0
            gauss_xipxip_mix = self._theta_target('xipxip', covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_mix = self._theta_target('xipxim', covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_mix = self._theta_target('ximxim', covxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.n_tomo_lens**4
            gaussELLmmmm_sva_flat = self._ell_cov_flat(gaussELLmmmm_sva, flat_length)
            gaussELLmmmm_mix_flat = self._ell_cov_flat(gaussELLmmmm_mix, flat_length)
            if not self.cov_dict['sn_only']:
                sva_jobs, mix_jobs = [], []
                for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                        idx = (m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries)
                        mirror = (idx[1], idx[0])
                        # xi+xi+ and xi-xi- are symmetric in the two theta bins, xi+xi- is not
                        if self.xi_pp and n_mode >= m_mode:
                            sva_jobs.append((m_mode, n_mode, gauss_xipxip_sva, idx, mirror))
                            mix_jobs.append((m_mode, n_mode, gauss_xipxip_mix, idx, mirror))
                        if self.xi_mm:
                            if n_mode >= m_mode:
                                sva_jobs.append((m_mode + self.mmE_summaries, n_mode + self.mmE_summaries, gauss_ximxim_sva, idx, mirror))
                                mix_jobs.append((m_mode + self.mmE_summaries, n_mode + self.mmE_summaries, gauss_ximxim_mix, idx, mirror))
                            if self.cross_terms and self.xi_pm:
                                sva_jobs.append((m_mode, n_mode + self.mmE_summaries, gauss_xipxim_sva, idx))
                                mix_jobs.append((m_mode, n_mode + self.mmE_summaries, gauss_xipxim_mix, idx))
                self.__project_gauss(gaussELLmmmm_sva_flat, gaussELLmmmm_mix_flat, sva_jobs, mix_jobs, 1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2), original_shape, 'xipmxipm')
                if self.gaussELLmmmm_BB is not None:
                    # intrinsic B-modes, xi_+ = int J_0 (C_EE + C_BB) and
                    # xi_- = int J_4 (C_EE - C_BB): the B-mode covariance
                    # enters with the same weights, with a minus sign for
                    # xi_+ xi_-
                    BB_sva_flat = self._ell_cov_flat(self.gaussELLmmmm_BB[0], flat_length)
                    BB_mix_flat = self._ell_cov_flat(self.gaussELLmmmm_BB[1], flat_length)
                    negative = (id(gauss_xipxim_sva), id(gauss_xipxim_mix))
                    for sign in [1, -1]:
                        self.__project_gauss(BB_sva_flat, BB_mix_flat,
                                             [job for job in sva_jobs if (id(job[2]) in negative) == (sign < 0)],
                                             [job for job in mix_jobs if (id(job[2]) in negative) == (sign < 0)],
                                             sign/(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2),
                                             original_shape, 'xipmxipm (intrinsic B-modes)', added=True)
                if self.xi_mm:
                    gauss_ximxim_sva = self._theta_add_product(gauss_ximxim_sva, self.gaussELLmmmm_sva_mult_shear_bias, self.xi_minus, self.xi_minus)
                if self.xi_pm:
                    gauss_xipxim_sva = self._theta_add_product(gauss_xipxim_sva, self.gaussELLmmmm_sva_mult_shear_bias, self.xi_plus, self.xi_minus)
                if self.xi_pp:
                    gauss_xipxip_sva = self._theta_add_product(gauss_xipxip_sva, self.gaussELLmmmm_sva_mult_shear_bias, self.xi_plus, self.xi_plus)
            gauss_xipm_sn = \
                (kron_delta_tomo_lens[None, None, None, :, None, :, None]
                 * kron_delta_tomo_lens[None, None, None, None, :, None, :]
                 + kron_delta_tomo_lens[None, None, None, :, None, None, :]
                 * kron_delta_tomo_lens[None, None, None, None, :, :, None]) \
                / self.npair_mm[:, :, None, :, :, None, None] / 0.5
            # the shot noise of xi_+ xi_+ and xi_- xi_- is the same, as blocks they are separate
            gauss_xipxip_sn = self._theta_block('xipxip', self._theta_diagonal('xipxip', gauss_xipm_sn))
            gauss_ximxim_sn = self._theta_block('ximxim', self._theta_diagonal('ximxim', gauss_xipm_sn))
            del gauss_xipm_sn
            gauss_xipxip_sva = self._theta_block('xipxip', gauss_xipxip_sva)
            gauss_xipxip_mix = self._theta_block('xipxip', gauss_xipxip_mix)
            gauss_xipxim_sva = self._theta_block('xipxim', gauss_xipxim_sva)
            gauss_xipxim_mix = self._theta_block('xipxim', gauss_xipxim_mix)
            gauss_ximxim_sva = self._theta_block('ximxim', gauss_ximxim_sva)
            gauss_ximxim_mix = self._theta_block('ximxim', gauss_ximxim_mix)
        else:
            gauss_xipxip_sva, gauss_xipxim_sva, gauss_ximxim_sva, gauss_xipxip_mix, gauss_xipxim_mix, gauss_ximxim_mix = 0, 0, 0, 0, 0, 0
            gauss_xipxip_sn, gauss_ximxim_sn = 0, 0

        if self.csmf:
            # the projections of the cross-covariance with the SMF and, for
            # the split output, of its super-sample part (see _split_csmf_terms)
            csmf_cross_parts = [(csmf_gg, csmf_gm, csmf_mm)]
            if self.cov_dict['ssc'] and getattr(self, 'csmf_parts', None) is not None:
                csmf_cross_parts.append(tuple(self.csmf_parts['cross_ssc']))
            csmf_projected = []
            for csmf_gg, csmf_gm, csmf_mm in csmf_cross_parts:
                if self.gg:
                    csmf_w = np.zeros((len(self.theta_bins_clustering), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
                    if not self.cov_dict['sn_only'] and not self.csmf_auto_only:
                        original_shape = csmf_gg[0, :, :, :, :, :].shape
                        flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust**2*self.n_tomo_csmf
                        csmf_AS_flat = np.reshape(csmf_gg, (len(self.ellrange), flat_length))
                        modes = list(range(self.gg_summaries))
                        projected = self._project_single(self.levin_int_fourier, self.ell_limits, modes,
                                                         csmf_AS_flat*self.ellrange[:, None])
                        for i, m_mode in enumerate(modes):
                            csmf_w[m_mode, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(projected[i], original_shape)
                else:
                    csmf_w = 0
                if self.gm:
                    csmf_gt = np.zeros((len(self.theta_bins_clustering), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
                    if not self.cov_dict['sn_only'] and not self.csmf_auto_only:
                        original_shape = csmf_gm[0, :, :, :, :, :].shape
                        flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust*self.n_tomo_lens*self.n_tomo_csmf
                        csmf_AS_flat = np.reshape(csmf_gm, (len(self.ellrange), flat_length))
                        modes = list(range(self.gg_summaries, self.gm_summaries + self.gg_summaries))
                        projected = self._project_single(self.levin_int_fourier, self.ell_limits, modes,
                                                         csmf_AS_flat*self.ellrange[:, None])
                        for i, m_mode in enumerate(modes):
                            csmf_gt[m_mode - self.gg_summaries, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(projected[i], original_shape)
                else:
                    csmf_gt = 0
            
                if self.mm:
                    csmf_xip = np.zeros((len(self.theta_bins_lensing), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                    csmf_xim = np.zeros((len(self.theta_bins_lensing), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                    if not self.cov_dict['sn_only'] and not self.csmf_auto_only:
                        original_shape = csmf_mm[0, :, :, :, :, :].shape
                        flat_length = len(self.log10csmf_mass_bins)*self.n_tomo_lens**2*self.n_tomo_csmf
                        csmf_AS_flat = np.reshape(csmf_mm, (len(self.ellrange), flat_length))
                        first = self.gg_summaries + self.gm_summaries
                        modes = list(range(first, first + self.mmE_summaries))
                        projected = self._project_single(self.levin_int_fourier, self.ell_limits,
                                                         modes + [m_mode + self.mmE_summaries for m_mode in modes],
                                                         csmf_AS_flat*self.ellrange[:, None])
                        for i, m_mode in enumerate(modes):
                            csmf_xip[m_mode - first, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(projected[i], original_shape)
                            csmf_xim[m_mode - first, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(projected[len(modes) + i], original_shape)

                else:
                    csmf_xip, csmf_xim = 0, 0
                csmf_projected.append((csmf_w, csmf_gt, csmf_xip, csmf_xim))
            csmf_w, csmf_gt, csmf_xip, csmf_xim = csmf_projected[0]
            self.csmf_projected_ssc = list(csmf_projected[1]) if len(csmf_projected) > 1 else None
        
        print("\nWrapping up all Gaussian real-space covariance contributions.")

        # all other shotnoise terms
        gauss_wgt_sn, gauss_wxip_sn, gauss_wxim_sn = 0, 0, 0
        gauss_xipgt_sn, gauss_ximgt_sn = 0, 0
        gauss_xipxim_sn = 0

        gauss_wxip_mix = 0
        gauss_wxim_mix = 0

        if self.csmf:
            return gauss_ww_sva, gauss_ww_mix, gauss_ww_sn, \
                gauss_wgt_sva, gauss_wgt_mix, gauss_wgt_sn, \
                gauss_wxip_sva, gauss_wxip_mix, gauss_wxip_sn, \
                gauss_wxim_sva, gauss_wxim_mix, gauss_wxim_sn, \
                gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn, \
                gauss_xipgt_sva, gauss_xipgt_mix, gauss_xipgt_sn, \
                gauss_ximgt_sva, gauss_ximgt_mix, gauss_ximgt_sn, \
                gauss_xipxip_sva, gauss_xipxip_mix, gauss_xipxip_sn, \
                gauss_xipxim_sva, gauss_xipxim_mix, gauss_xipxim_sn, \
                gauss_ximxim_sva, gauss_ximxim_mix, gauss_ximxim_sn, \
                csmf_auto, csmf_w, csmf_gt, csmf_xip, csmf_xim
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
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
        self.levin_int_fourier.update_Levin(0, 16, 32,1e-3,1e-3)
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
                nongaussELLgmgm = nongaussELLgmgm/(survey_params_dict['survey_area_ggl'] / self.deg2torad2) + nongaussELLgmgm1
            if self.mm and self.gm and self.cross_terms:
                nongaussELLmmgm = nongaussELLmmgm/(max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) + nongaussELLmmgm1
            if self.mm:
                nongaussELLmmmm = nongaussELLmmmm/(survey_params_dict['survey_area_lens'] / self.deg2torad2) + nongaussELLmmmm1
            connected = False
        elif (connected):
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                covELLspacesettings, output_dict, bias_dict, hod_dict, hm_prec, tri_tab)
        else:
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_ssc(
                bias_dict, hod_dict, hm_prec, survey_params_dict, covELLspacesettings)
            
        # jobs: (outer_mode, inner_mode, target_array, target_index)
        if self.gg:
            print("")
            original_shape = self._tomo_shape(nongaussELLgggg)
            nongauss_ww = self._theta_target('ww', (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                    self.sample_dim, self.sample_dim,
                                    self.n_tomo_clust, self.n_tomo_clust,
                                    self.n_tomo_clust, self.n_tomo_clust))
            jobs = [(m_mode, n_mode, nongauss_ww, (m_mode, n_mode))
                    for m_mode in range(self.gg_summaries) for n_mode in range(self.gg_summaries)]
            self.__project_4pt(nongaussELLgggg, jobs,
                               (survey_params_dict['survey_area_clust'] / self.deg2torad2) if connected else None,
                               original_shape, 'ww')
            nongauss_ww = self._theta_block('ww', nongauss_ww)

        if self.gg and self.gm and self.cross_terms:
            print("")
            # cov_NG(w^ij(theta1) gt^kl(theta2))
            original_shape = self._tomo_shape(nongaussELLgggm)
            nongauss_wgt = self._theta_target('wgt', (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                     self.sample_dim, self.sample_dim,
                                     self.n_tomo_clust, self.n_tomo_clust,
                                     self.n_tomo_clust, self.n_tomo_lens))
            jobs = [(m_mode, n_mode, nongauss_wgt, (m_mode, n_mode - self.gg_summaries))
                    for m_mode in range(self.gg_summaries) for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries)]
            self.__project_4pt(nongaussELLgggm, jobs,
                               (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) if connected else None,
                               original_shape, 'wgt')
            nongauss_wgt = self._theta_block('wgt', nongauss_wgt)

        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = self._tomo_shape(nongaussELLggmm)
            covwxipm_shape_nongauss = (len(self.theta_bins_clustering), len(self.theta_bins_lensing),
                                       self.sample_dim, 1,
                                       self.n_tomo_clust, self.n_tomo_clust,
                                       self.n_tomo_lens, self.n_tomo_lens)
            nongauss_wxip = self._theta_target('wxip', covwxipm_shape_nongauss)
            nongauss_wxim = self._theta_target('wxim', covwxipm_shape_nongauss)
            jobs = []
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    idx = (m_mode, n_mode - self.gg_summaries - self.gm_summaries)
                    jobs.append((m_mode, n_mode, nongauss_wxip, idx))
                    if self.xi_mm:
                        jobs.append((m_mode, n_mode + self.mmE_summaries, nongauss_wxim, idx))
            self.__project_4pt(nongaussELLggmm, jobs,
                               (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2) if connected else None,
                               original_shape, 'wxipm')
            nongauss_wxip = self._theta_block('wxip', nongauss_wxip)
            nongauss_wxim = self._theta_block('wxim', nongauss_wxim)

        if self.gm:
            print("")
            original_shape = self._tomo_shape(nongaussELLgmgm)
            nongauss_gtgt = self._theta_target('gtgt', (len(self.theta_bins_clustering), len(self.theta_bins_clustering),
                                      self.sample_dim, self.sample_dim,
                                      self.n_tomo_clust, self.n_tomo_lens,
                                      self.n_tomo_clust, self.n_tomo_lens))
            jobs = [(m_mode, n_mode, nongauss_gtgt, (m_mode - self.gg_summaries, n_mode - self.gg_summaries))
                    for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries) for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries)]
            self.__project_4pt(nongaussELLgmgm, jobs,
                               (survey_params_dict['survey_area_ggl']) / self.deg2torad2 if connected else None,
                               original_shape, 'gtgt')
            nongauss_gtgt = self._theta_block('gtgt', nongauss_gtgt)

        if self.mm and self.gm and self.cross_terms:
            print("")
            original_shape = self._tomo_shape(nongaussELLmmgm)
            covxipmgt_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_clustering),
                                        1, self.sample_dim,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_clust, self.n_tomo_lens)
            nongauss_xipgt = self._theta_target('xipgt', covxipmgt_shape_nongauss)
            nongauss_ximgt = self._theta_target('ximgt', covxipmgt_shape_nongauss)
            jobs = []
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                    idx = (m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries)
                    jobs.append((m_mode, n_mode, nongauss_xipgt, idx))
                    if self.xi_mm:
                        jobs.append((m_mode + self.mmE_summaries, n_mode, nongauss_ximgt, idx))
            self.__project_4pt(nongaussELLmmgm, jobs,
                               (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2) if connected else None,
                               original_shape, 'xipmgt')
            nongauss_xipgt = self._theta_block('xipgt', nongauss_xipgt)
            nongauss_ximgt = self._theta_block('ximgt', nongauss_ximgt)

        if self.mm:
            print("")
            original_shape = self._tomo_shape(nongaussELLmmmm)
            covxipm_shape_nongauss = (len(self.theta_bins_lensing), len(self.theta_bins_lensing),
                                      1, 1,
                                      self.n_tomo_lens, self.n_tomo_lens,
                                      self.n_tomo_lens, self.n_tomo_lens)
            nongauss_xipxip = self._theta_target('xipxip', covxipm_shape_nongauss)
            nongauss_xipxim = self._theta_target('xipxim', covxipm_shape_nongauss)
            nongauss_ximxim = self._theta_target('ximxim', covxipm_shape_nongauss)
            jobs = []
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    idx = (m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries)
                    jobs.append((m_mode, n_mode, nongauss_xipxip, idx))
                    if self.xi_mm:
                        jobs.append((m_mode, n_mode + self.mmE_summaries, nongauss_xipxim, idx))
                        jobs.append((m_mode + self.mmE_summaries, n_mode + self.mmE_summaries, nongauss_ximxim, idx))
            self.__project_4pt(nongaussELLmmmm, jobs,
                               (survey_params_dict['survey_area_lens'] / self.deg2torad2) if connected else None,
                               original_shape, 'xipmxipm')
            nongauss_xipxip = self._theta_block('xipxip', nongauss_xipxip)
            nongauss_xipxim = self._theta_block('xipxim', nongauss_xipxim)
            nongauss_ximxim = self._theta_block('ximxim', nongauss_ximxim)
        print("")
        return nongauss_ww, nongauss_wgt, nongauss_wxip, nongauss_wxim, nongauss_gtgt, nongauss_xipgt, nongauss_ximgt, nongauss_xipxip, nongauss_xipxim, nongauss_ximxim
