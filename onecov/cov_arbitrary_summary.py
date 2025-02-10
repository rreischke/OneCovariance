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
        'arbitrary_summary' dictionary
                Specifies a few details for the arbitrary summary covariance
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
        'arb_summary': dictionary
            Look-up table for the arbitrary summary statistics filters
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
        self.__get_fourier_weights(read_in_tables['arb_summary'])
        self.__get_real_weights(read_in_tables['arb_summary'])
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
        self.dnpair_gg, self.dnpair_gm, self.dnpair_mm, self.theta_gg, self.theta_gm, self.theta_mm  = self.get_dnpair([self.gg, self.gm, self.mm],
                                                                                                                        self.theta_real_integral,
                                                                                                                        survey_params_dict,
                                                                                                                        read_in_tables['npair'])
        
        if self.gg or self.gm:    
            survey_params_dict['n_eff_clust'] = save_n_eff_clust
        if self.mm or self.gm:
            survey_params_dict['n_eff_lens'] = save_n_eff_lens
        
        self.ell_limits = []
        for mode in range(len(self.WXY_stack[:,0])):
            limits_at_mode = np.array(self.ell_fourier_integral[argrelextrema(self.WXY_stack[mode,:], np.less)[0][:]])[::obs_dict['arbitrary_summary']['oscillations_straddle']]
            limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.ell_fourier_integral[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]) + 2)
            limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.ell_fourier_integral[1]) & (limits_at_mode < self.ell_fourier_integral[-2])]
            limits_at_mode_append[0] = self.ell_fourier_integral[0]
            limits_at_mode_append[-1] = self.ell_fourier_integral[-1]
            self.ell_limits.append(limits_at_mode_append)
        self.levin_int_fourier = levin.Levin(0, 16, 32, obs_dict['arbitrary_summary']['arbitrary_accuracy']/np.sqrt(len(max(self.ell_limits, key=len))), 50, self.num_cores)
        self.levin_int_fourier.init_w_ell(self.ell_fourier_integral, self.WXY_stack.T)
    
        self.theta_limits = []
        for mode in range(len(self.RXY_stack[:,0])):
            limits_at_mode = np.array(self.theta_real_integral[argrelextrema(self.RXY_stack[mode,:], np.less)[0][:]])[::obs_dict['arbitrary_summary']['oscillations_straddle']]
            limits_at_mode_append = np.zeros(len(limits_at_mode) + 2)
            limits_at_mode_append[1:-1] = limits_at_mode
            if len(np.where(self.RXY_stack[mode,:]==0)[0]) > 1:
                if np.where(self.RXY_stack[mode,:]==0)[0][0] == 0 and  np.where(self.RXY_stack[mode,:]==0)[0][1] == 1:
                    limits_at_mode_append[0] = self.theta_real_integral[np.where(self.RXY_stack[mode,:]!=0)[0][0] -1]
                else:
                    limits_at_mode_append[0] = self.theta_real_integral[0]
                if np.where(self.RXY_stack[mode,:]==0)[0][-2] == len(self.RXY_stack[mode,:])-2 and np.where(self.RXY_stack[mode,:]==0)[0][-1] == len(self.RXY_stack[mode,:])-1:
                    limits_at_mode_append[-1] = self.theta_real_integral[np.where(self.RXY_stack[mode,:]!=0)[0][-1]]
                else:
                    limits_at_mode_append[-1] = self.theta_real_integral[-1]
            else:
                limits_at_mode_append[-1] = self.theta_real_integral[-1]
                limits_at_mode_append[0] = self.theta_real_integral[0]
            self.theta_limits.append(limits_at_mode_append/60/180*np.pi)
        self.levin_int_real = levin.Levin(0, 16, 32, obs_dict['arbitrary_summary']['arbitrary_accuracy']/np.sqrt(len(max(self.theta_limits, key=len))), 50, self.num_cores)
        self.levin_int_real.init_w_ell(self.theta_real_integral/60/180*np.pi, self.RXY_stack.T)
        self.__get_shotnoise_integrals()
        self.__detect_rscf(survey_params_dict, read_in_tables)
        self.__get_arbitrary_signal()

    def __get_arbitrary_signal(self):
        """
        Calculates the signal of the arbitrary summary in all tomographic bin
        combination and all tracers specified.

        """

        self.arbE_mm = np.zeros((self.mmE_summaries, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens))
        self.arbB_mm = np.zeros((self.mmB_summaries, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens))
        self.arbE_gm = np.zeros((self.gm_summaries, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_lens))
        self.arbE_gg = np.zeros((self.gg_summaries, self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust))
        
        if self.mm:
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries
            original_shape = self.Cell_mm[0, :, :, :].shape
            flat_length = self.n_tomo_lens**2*self.sample_dim
            Cell_mm_flat = np.reshape(self.Cell_mm, (len(
                self.ellrange), flat_length))
            for mode in range(self.mmE_summaries):
                self.levin_int_fourier.init_integral(self.ellrange, Cell_mm_flat*self.ellrange[:,None], True, True)
                self.arbE_mm[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[mode + self.gg_summaries + self.gm_summaries][:], mode + self.gg_summaries + self.gm_summaries)),original_shape)
                self.arbB_mm[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[mode + self.gg_summaries + self.gm_summaries + self.mmE_summaries][:], mode + self.gg_summaries + self.gm_summaries + self.mmE_summaries)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rArbitrary E/B-mode calculation for lensing at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
            print(" ")

        if self.gm:
            t0, tcomb = time.time(), 1
            tcombs = self.gm_summaries
            original_shape = self.Cell_gm[0, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens*self.n_tomo_clust
            Cell_gm_flat = np.reshape(self.Cell_gm, (len(
                self.ellrange), flat_length))
            for mode in range(self.gm_summaries):
                self.levin_int_fourier.init_integral(self.ellrange, Cell_gm_flat*self.ellrange[:,None], True, True)
                self.arbE_gm[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[mode + self.gg_summaries][:], mode + self.gg_summaries)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rArbitrary E-mode calculation for GGL at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
                
            print(" ")

        if self.gg:
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries
            original_shape = self.Cell_gg[0, :, :, :, :].shape
            flat_length = self.sample_dim**2*self.n_tomo_clust**2
            Cell_gg_flat = np.reshape(self.Cell_gg, (len(
                self.ellrange), flat_length))
            for mode in range(self.gg_summaries):
                self.levin_int_fourier.init_integral(self.ellrange, Cell_gg_flat*self.ellrange[:,None], True, True)
                self.arbE_gg[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[mode ][:], mode)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rArbitrary E-mode calculation for clustering at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
            print(" ")

    def __detect_rscf(self,
                      survey_params_dict,
                       read_in_tables):
        '''
        Function checking which of the arbitrary summary statistics are real space correlation
        functions to correct for discreteness effects.
        '''
        self.index_realspace_clustering = []
        self.index_realspace_ggl = []
        self.index_realspace_lensing = []
        self.theta_ul_realspace_clustering = []
        self.theta_ul_realspace_ggl = []
        self.theta_ul_realspace_lensing = []
        realspace_counts = 0
        for i in range(len(self.RXY_stack[:,0])):
            index = np.where(self.RXY_stack[i,:] != 0)[0]
            test_array = self.RXY_stack[i,index]*self.theta_real_integral[index]
            if np.all(np.round(test_array/test_array[0],4) == 1.):
                if i < self.gg_summaries_real + self.gm_summaries_real + self.mmE_summaries_real:
                    if i >= self.gg_summaries_real + self.gm_summaries_real and i < self.gg_summaries_real + self.gm_summaries_real + self.mmE_summaries_real:
                        self.index_realspace_lensing.append(i)
                        if realspace_counts == 0:
                            self.theta_ul_realspace_lensing.append(self.theta_real_integral[index[0]])
                            self.theta_ul_realspace_lensing.append(self.theta_real_integral[index[-1]])
                        else:
                            self.theta_ul_realspace_lensing.append(self.theta_real_integral[index[-1]])
                    if i >= self.gg_summaries_real and i < self.gg_summaries_real + self.gm_summaries_real:
                        self.index_realspace_ggl.append(i)
                        if realspace_counts == 0:
                            self.theta_ul_realspace_ggl.append(self.theta_real_integral[index[0]])
                            self.theta_ul_realspace_ggl.append(self.theta_real_integral[index[-1]])
                        else:
                            self.theta_ul_realspace_ggl.append(self.theta_real_integral[index[-1]])
                    if  i < self.gg_summaries_real:
                        self.index_realspace_clustering.append(i)
                        if realspace_counts == 0:
                            self.theta_ul_realspace_clustering.append(self.theta_real_integral[index[0]])
                            self.theta_ul_realspace_clustering.append(self.theta_real_integral[index[-1]])
                        else:
                            self.theta_ul_realspace_clustering.append(self.theta_real_integral[index[-1]])
                    realspace_counts += 1
        if realspace_counts == 0:
            self.index_realspace_lensing = None
            self.index_realspace_ggl = None
            self.index_realspace_clustering = None
            self.theta_ul_realspace_lensing = None
            self.theta_realspace_lensing = None
            self.theta_ul_realspace_ggl = None
            self.theta_realspace_ggl = None
            self.theta_ul_realspace_clustering = None
            self.theta_realspace_clustering = None
        else:
            if len(self.theta_ul_realspace_lensing) != 0:
                self.index_realspace_lensing = np.array(self.index_realspace_lensing)
                self.theta_ul_realspace_lensing = np.array(self.theta_ul_realspace_lensing)
                dtheta = self.theta_ul_realspace_lensing[1:] - self.theta_ul_realspace_lensing[:-1]
                if np.all(np.round(dtheta/dtheta[0],4) == 1):
                    self.theta_realspace_lensing = .5 * (self.theta_ul_realspace_lensing[1:] + self.theta_ul_realspace_lensing[:-1])
                else:
                    self.theta_realspace_lensing = np.exp(.5 * (np.log(self.theta_ul_realspace_lensing[1:])
                                        + np.log(self.theta_ul_realspace_lensing[:-1])))
            else:
                self.theta_ul_realspace_lensing = None
                self.theta_realspace_lensing = None
                self.index_realspace_lensing = None
            if len(self.theta_ul_realspace_ggl) != 0:
                self.index_realspace_ggl = np.array(self.index_realspace_ggl)
                self.theta_ul_realspace_ggl = np.array(self.theta_ul_realspace_ggl)
                dtheta = self.theta_ul_realspace_ggl[1:] - self.theta_ul_realspace_ggl[:-1]
                if np.all(np.round(dtheta/dtheta[0],4) == 1):
                    self.theta_realspace_ggl = .5 * (self.theta_ul_realspace_ggl[1:] + self.theta_ul_realspace_ggl[:-1])
                else:
                    self.theta_realspace_ggl = np.exp(.5 * (np.log(self.theta_ul_realspace_ggl[1:])
                                        + np.log(self.theta_ul_realspace_ggl[:-1])))
            else:
                self.theta_ul_realspace_ggl = None
                self.theta_realspace_ggl = None
                self.index_realspace_ggl = None
            if len(self.theta_ul_realspace_clustering) != 0:
                self.index_realspace_clustering = np.array(self.index_realspace_clustering)
                self.theta_ul_realspace_clustering = np.array(self.theta_ul_realspace_clustering)
                dtheta = self.theta_ul_realspace_clustering[1:] - self.theta_ul_realspace_clustering[:-1]
                if np.all(np.round(dtheta/dtheta[0],4) == 1):
                    self.theta_realspace_clustering = .5 * (self.theta_ul_realspace_clustering[1:] + self.theta_ul_realspace_clustering[:-1])
                else:
                    self.theta_realspace_clustering = np.exp(.5 * (np.log(self.theta_ul_realspace_clustering[1:])
                                        + np.log(self.theta_ul_realspace_clustering[:-1])))
            else:
                self.theta_ul_realspace_clustering = None
                self.theta_realspace_clustering = None
                self.index_realspace_clustering = None
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
        if self.gg and self.theta_ul_realspace_clustering is not None:
            self.npair_gg, _, _ = \
                self.get_npair([self.gg, False, False],
                            self.theta_ul_realspace_clustering,
                            self.theta_realspace_clustering,
                            survey_params_dict,
                            read_in_tables['npair'])
        if self.mm and self.theta_ul_realspace_lensing is not None:
            _, _, self.npair_mm = \
                self.get_npair([False, False, self.mm],
                            self.theta_ul_realspace_lensing,
                            self.theta_realspace_lensing,
                            survey_params_dict,
                            read_in_tables['npair'])
        if self.gm and self.theta_ul_realspace_ggl is not None:
            _, self.npair_gm, _ = \
                self.get_npair([False, self.gm, False],
                            self.theta_ul_realspace_ggl,
                            self.theta_realspace_ggl,
                            survey_params_dict,
                            read_in_tables['npair'])
        if self.gg or self.gm:    
            survey_params_dict['n_eff_clust'] = save_n_eff_clust
        if self.mm or self.gm:
            survey_params_dict['n_eff_lens'] = save_n_eff_lens
        if self.mm and self.index_realspace_lensing is not None:
            correction = []
            for i in range(self.index_realspace_lensing[0]):
                correction.append(np.ones_like(self.SN_integral_mmmm[0,0,:, :, :]))
            for i, iv in enumerate(self.index_realspace_lensing):
                correction.append(np.sqrt(1/(self.SN_integral_mmmm[iv,iv, : ,: ,:]*self.npair_mm[i, :, :, :])))
            correction = np.array(correction)
            for i in range(len(self.SN_integral_mmmm[:,0,0,0,0])):
                for j in range(len(self.SN_integral_mmmm[0,:,0,0,0])):
                    local_correction = 1
                    if i in self.index_realspace_lensing:
                        local_correction *= correction[i, :, :, :]
                    if j in self.index_realspace_lensing:
                        local_correction *= correction[j, :, :, :]
                    self.SN_integral_mmmm[i,j, : ,: ,:] *= local_correction
        if self.gm and self.index_realspace_ggl is not None:
            correction = []
            for i in range(self.index_realspace_ggl[0]):
                correction.append(np.ones_like(self.SN_integral_gmgm[0,0,:, :, :]))
            for i, iv in enumerate(self.index_realspace_ggl):
                correction.append(np.sqrt(1/(self.SN_integral_gmgm[iv,iv, : ,: ,:]*self.npair_gm[i, :, :, :])))
            correction = np.array(correction)
            for i in range(len(self.SN_integral_gmgm[:,0,0,0,0])):
                for j in range(len(self.SN_integral_gmgm[0,:,0,0,0])):
                    local_correction = 1
                    if i in self.index_realspace_ggl:
                        local_correction *= correction[i, :, :, :]
                    if j in self.index_realspace_ggl:
                        local_correction *= correction[j, :, :, :]
                    self.SN_integral_gmgm[i,j, : ,: ,:] *= local_correction
        if self.gg and self.index_realspace_clustering is not None:
            correction = []
            for i in range(self.index_realspace_clustering[0]):
                correction.append(np.ones_like(self.SN_integral_gggg[0,0,:, :, :]))
            for i, iv in enumerate(self.index_realspace_clustering):
                correction.append(np.sqrt(1/(self.SN_integral_gggg[iv,iv, : ,: ,:]*self.npair_gg[i, :, :, :])))
            correction = np.array(correction)
            for i in range(len(self.SN_integral_gggg[:,0,0,0,0])):
                for j in range(len(self.SN_integral_gggg[0,:,0,0,0])):
                    local_correction = 1
                    if i in self.index_realspace_clustering:
                        local_correction *= correction[i, :, :, :]
                    if j in self.index_realspace_clustering:
                        local_correction *= correction[j, :, :, :]
                    self.SN_integral_gggg[i,j, : ,: ,:] *= local_correction




    def __get_fourier_weights(self,
                              arb_summary_tables):
        """
        This function reads in the Fourier filters from the input
        tables and sets them to private variables for later intergration.

        Parameters:
        -----------
        arb_summary_tables: dictionary
            Look-up table for the arbitrary summary statistics filters
        """
        self.gg_summaries = 0    
        self.gm_summaries = 0
        self.mmE_summaries = 0
        self.mmB_summaries = 0
        if self.gg:
            self.fourier_ell_gg = []
            self.fourier_weights_gg = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['ell_gg']:
                self.fourier_ell_gg.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.gg_summaries += 1
            for W_summary in arb_summary_tables['WL_gg']:
                self.fourier_weights_gg.append(W_summary)
            aux_fourier_ell = np.zeros((self.gg_summaries, len(longest_ell)))
            aux_fourier_weights = np.zeros_like(aux_fourier_ell)
            for i_summaries in range(self.gg_summaries):
                aux_fourier_weights[i_summaries,:] = np.interp(longest_ell, self.fourier_ell_gg[i_summaries], self.fourier_weights_gg[i_summaries], left=0,right=0)
            self.fourier_ell_gg = longest_ell
            self.fourier_weights_gg = aux_fourier_weights
        if self.gm:
            self.fourier_ell_gm = []
            self.fourier_weights_gm = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['ell_gm']:
                self.fourier_ell_gm.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.gm_summaries += 1
            for W_summary in arb_summary_tables['WL_gm']:
                self.fourier_weights_gm.append(W_summary)
            aux_fourier_ell = np.zeros((self.gm_summaries, len(longest_ell)))
            aux_fourier_weights = np.zeros_like(aux_fourier_ell)
            for i_summaries in range(self.gm_summaries):
                aux_fourier_weights[i_summaries,:] = np.interp(longest_ell, self.fourier_ell_gm[i_summaries], self.fourier_weights_gm[i_summaries], left=0,right=0)
            self.fourier_ell_gm = longest_ell
            self.fourier_weights_gm = aux_fourier_weights
        if self.mm:
            self.fourier_ell_mmE = []
            self.fourier_weights_mmE = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['ell_mmE']:
                self.fourier_ell_mmE.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.mmE_summaries += 1
            for W_summary in arb_summary_tables['WL_mmE']:
                self.fourier_weights_mmE.append(W_summary)
            aux_fourier_ell = np.zeros((self.mmE_summaries, len(longest_ell)))
            aux_fourier_weights = np.zeros_like(aux_fourier_ell)
            for i_summaries in range(self.mmE_summaries):
                aux_fourier_weights[i_summaries,:] = np.interp(longest_ell, self.fourier_ell_mmE[i_summaries], self.fourier_weights_mmE[i_summaries], left=0,right=0)
            self.fourier_ell_mmE = longest_ell
            self.fourier_weights_mmE = aux_fourier_weights

            self.fourier_ell_mmB = []
            self.fourier_weights_mmB = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['ell_mmB']:
                self.fourier_ell_mmB.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.mmB_summaries += 1
            for W_summary in arb_summary_tables['WL_mmB']:
                self.fourier_weights_mmB.append(W_summary)
            aux_fourier_ell = np.zeros((self.mmB_summaries, len(longest_ell)))
            aux_fourier_weights = np.zeros_like(aux_fourier_ell)
            for i_summaries in range(self.mmB_summaries):
                aux_fourier_weights[i_summaries,:] = np.interp(longest_ell, self.fourier_ell_mmB[i_summaries], self.fourier_weights_mmB[i_summaries], left=0,right=0)
            self.fourier_ell_mmB = longest_ell
            self.fourier_weights_mmB = aux_fourier_weights


        self.ell_fourier_integral = np.zeros(2)
        self.WXY_stack = []
        if self.gg:
            if len(self.ell_fourier_integral) < len(self.fourier_ell_gg):
                self.ell_fourier_integral = self.fourier_ell_gg
        if self.gm:
            if len(self.ell_fourier_integral) < len(self.fourier_ell_gm):
                self.ell_fourier_integral = self.fourier_ell_gm
        if self.mm:
            if len(self.ell_fourier_integral) < len(self.fourier_ell_mmE):
                self.ell_fourier_integral = self.fourier_ell_mmE
            if len(self.ell_fourier_integral) < len(self.fourier_ell_mmB):
                self.ell_fourier_integral = self.fourier_ell_mmB
        if self.gg:
            for i_summaries in range(self.gg_summaries):
                self.WXY_stack.append(np.interp(self.ell_fourier_integral,self.fourier_ell_gg, self.fourier_weights_gg[i_summaries], left=0,right=0))
        if self.gm:
            for i_summaries in range(self.gm_summaries):
                self.WXY_stack.append(np.interp(self.ell_fourier_integral,self.fourier_ell_gm,self.fourier_weights_gm[i_summaries], left=0,right=0))
        if self.mm:
            for i_summaries in range(self.mmE_summaries):
                self.WXY_stack.append(np.interp(self.ell_fourier_integral,self.fourier_ell_mmE,self.fourier_weights_mmE[i_summaries], left=0,right=0))
            for i_summaries in range(self.mmB_summaries):
                self.WXY_stack.append(np.interp(self.ell_fourier_integral,self.fourier_ell_mmB,self.fourier_weights_mmB[i_summaries], left=0,right=0))
        self.WXY_stack = np.array(self.WXY_stack)
        return True
    
    def __get_real_weights(self,
                           arb_summary_tables):
        """
        This function reads in the real filters from the input
        tables and sets them to private variables for later intergration.

        Parameters:
        -----------
        arb_summary_tables: dictionary
            Look-up table for the arbitrary summary statistics filters
        """
        self.gg_summaries_real = 0
        self.gm_summaries_real = 0
        self.mmE_summaries_real = 0
        self.mmB_summaries_real = 0   
        if self.gg:
            self.real_theta_gg = []
            self.real_weights_gg = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['theta_gg']:
                self.real_theta_gg.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.gg_summaries_real += 1
            for W_summary in arb_summary_tables['RL_gg']:
                self.real_weights_gg.append(W_summary)
            aux_real_ell = np.zeros((self.gg_summaries_real, len(longest_ell)))
            aux_real_weights = np.zeros_like(aux_real_ell)
            for i_summaries_real in range(self.gg_summaries_real):
                aux_real_weights[i_summaries_real,:] = np.interp(longest_ell, self.real_theta_gg[i_summaries_real], self.real_weights_gg[i_summaries_real], left=0,right=0)
            self.real_theta_gg = longest_ell
            self.real_weights_gg = aux_real_weights
            if self.gg_summaries_real != self.gg_summaries:
                raise Exception("InputError: The number of Fourier and Real space filters for the arbitrary " +
                                "summary statistics does not match for gg, please check the files passed via arb_fourier_filter_gg_file "+
                                "and arb_real_filter_gg_file in the config file")
        if self.gm:
            self.real_theta_gm = []
            self.real_weights_gm = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['theta_gm']:
                self.real_theta_gm.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.gm_summaries_real += 1
            for W_summary in arb_summary_tables['RL_gm']:
                self.real_weights_gm.append(W_summary)
            aux_real_ell = np.zeros((self.gm_summaries_real, len(longest_ell)))
            aux_real_weights = np.zeros_like(aux_real_ell)
            for i_summaries_real in range(self.gm_summaries_real):
                aux_real_weights[i_summaries_real,:] = np.interp(longest_ell, self.real_theta_gm[i_summaries_real], self.real_weights_gm[i_summaries_real], left=0,right=0)
            self.real_theta_gm = longest_ell
            self.real_weights_gm = aux_real_weights
            if self.gm_summaries_real != self.gm_summaries:
                raise Exception("InputError: The number of Fourier and Real space filters for the arbitrary " +
                                "summary statistics does not match for gm, please check the files passed via arb_fourier_filter_gm_file "+
                                "and arb_real_filter_gm_file in the config file")
        if self.mm:
            self.real_theta_mmE = []
            self.real_weights_mmE = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['theta_mm_p']:
                self.real_theta_mmE.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.mmE_summaries_real += 1
            for W_summary in arb_summary_tables['RL_mm_p']:
                self.real_weights_mmE.append(W_summary)
            aux_real_ell = np.zeros((self.mmE_summaries_real, len(longest_ell)))
            aux_real_weights = np.zeros_like(aux_real_ell)
            for i_summaries_real in range(self.mmE_summaries_real):
                aux_real_weights[i_summaries_real,:] = np.interp(longest_ell, self.real_theta_mmE[i_summaries_real], self.real_weights_mmE[i_summaries_real], left=0,right=0)
            self.real_theta_mmE = longest_ell
            self.real_weights_mmE = aux_real_weights
            if self.mmE_summaries_real != self.mmE_summaries:
                raise Exception("InputError: The number of Fourier and Real space filters for the arbitrary " +
                                "summary statistics does not match for mmE, please check the files passed via arb_fourier_filter_mmE_file "+
                                "and arb_real_filter_mm_p_file in the config file")

            self.real_theta_mmB = []
            self.real_weights_mmB = []
            longest_ell = np.zeros(2)
            for ell_summary in arb_summary_tables['theta_mm_m']:
                self.real_theta_mmB.append(ell_summary)
                if len(longest_ell) < len(ell_summary):
                    longest_ell = ell_summary
                self.mmB_summaries_real += 1
            for W_summary in arb_summary_tables['RL_mm_m']:
                self.real_weights_mmB.append(W_summary)
            aux_real_ell = np.zeros((self.mmB_summaries_real, len(longest_ell)))
            aux_real_weights = np.zeros_like(aux_real_ell)
            for i_summaries_real in range(self.mmB_summaries_real):
                aux_real_weights[i_summaries_real,:] = np.interp(longest_ell, self.real_theta_mmB[i_summaries_real], self.real_weights_mmB[i_summaries_real], left=0,right=0)
            self.real_theta_mmB = longest_ell
            self.real_weights_mmB = aux_real_weights
            if self.mmB_summaries_real != self.mmB_summaries:
                raise Exception("InputError: The number of Fourier and Real space filters for the arbitrary " +
                                "summary statistics does not match for mmB, please check the files passed via arb_fourier_filter_mmB_file "+
                                "and arb_real_filter_mm_m_file in the config file")



        self.theta_real_integral = np.zeros(2)
        self.RXY_stack = []
        if self.gg:
            if len(self.theta_real_integral) < len(self.real_theta_gg):
                self.theta_real_integral = self.real_theta_gg
        if self.gm:
            if len(self.theta_real_integral) < len(self.real_theta_gm):
                self.theta_real_integral = self.real_theta_gm
        if self.mm:
            if len(self.theta_real_integral) < len(self.real_theta_mmE):
                self.theta_real_integral = self.real_theta_mmE
            if len(self.theta_real_integral) < len(self.real_theta_mmB):
                self.theta_real_integral = self.real_theta_mmB
        if self.gg:
            for i_summaries_real in range(self.gg_summaries_real):
                self.RXY_stack.append(np.interp(self.theta_real_integral,self.real_theta_gg, self.real_weights_gg[i_summaries_real], left=0,right=0))
        if self.gm:
            for i_summaries_real in range(self.gm_summaries_real):
                self.RXY_stack.append(np.interp(self.theta_real_integral,self.real_theta_gm,self.real_weights_gm[i_summaries_real], left=0,right=0))
        if self.mm:
            for i_summaries_real in range(self.mmE_summaries_real):
                self.RXY_stack.append(np.interp(self.theta_real_integral,self.real_theta_mmE,self.real_weights_mmE[i_summaries_real], left=0,right=0))
            for i_summaries_real in range(self.mmB_summaries_real):
                self.RXY_stack.append(np.interp(self.theta_real_integral,self.real_theta_mmB,self.real_weights_mmB[i_summaries_real], left=0,right=0))
        self.RXY_stack = np.array(self.RXY_stack)

    def __get_shotnoise_integrals(self):
        """
        Function precomputing the integrals for the shot/shape noise over theta
        """
        
        if self.gg:
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries**2
            self.SN_integral_gggg = np.zeros((self.gg_summaries, self.gg_summaries, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust))
            original_shape = (self.sample_dim,self.n_tomo_clust, self.n_tomo_clust)
            dnpair_gg_flat = np.reshape(self.dnpair_gg, (len(self.theta_real_integral), self.n_tomo_clust**2*self.sample_dim))
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries):
                    local_theta_limit = self.theta_limits[m_mode][:]
                    if len(self.theta_limits[m_mode][:]) < len(self.theta_limits[n_mode][:]):
                        local_theta_limit = self.theta_limits[n_mode][:]
                    if self.theta_limits[n_mode][0] == self.theta_limits[m_mode][-1] or self.theta_limits[m_mode][0] == self.theta_limits[n_mode][-1]:
                        tcomb += 1
                        continue
                    integrand = 1/(dnpair_gg_flat* 60*180/np.pi)*(self.theta_real_integral[:, None]/60/180*np.pi)**2
                    self.levin_int_real.init_integral(self.theta_real_integral/60/180*np.pi, integrand, True, True)
                    self.SN_integral_gggg[m_mode, n_mode, :, :, :] = np.reshape(np.array(self.levin_int_real.cquad_integrate_double_well(local_theta_limit, m_mode, n_mode)), original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCalculating Shot Noise integrals for gggg arbitrary summary covariance '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        if self.gm:
            t0, tcomb = time.time(), 1
            tcombs = self.gm_summaries**2
            self.SN_integral_gmgm = np.zeros((self.gm_summaries, self.gm_summaries, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens))
            original_shape = (self.sample_dim,self.n_tomo_clust, self.n_tomo_lens)
            dnpair_gm_flat = np.reshape(self.dnpair_gm, (len(self.theta_real_integral), self.n_tomo_clust*self.n_tomo_lens*self.sample_dim))
            for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_theta_limit = self.theta_limits[m_mode][:]
                    if len(self.theta_limits[m_mode][:]) < len(self.theta_limits[n_mode][:]):
                        local_theta_limit = self.theta_limits[n_mode][:]
                    if self.theta_limits[n_mode][0] == self.theta_limits[m_mode][-1] or self.theta_limits[m_mode][0] == self.theta_limits[n_mode][-1]:
                        tcomb += 1
                        continue
                    integrand = 1/(dnpair_gm_flat* 60*180/np.pi)*(self.theta_real_integral[:, None]/60/180*np.pi)**2
                    self.levin_int_real.init_integral(self.theta_real_integral/60/180*np.pi, integrand, True, True)
                    self.SN_integral_gmgm[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :] = np.reshape(np.array(self.levin_int_real.cquad_integrate_double_well(local_theta_limit, m_mode, n_mode)), original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCalculating Shot Noise integrals for gmgm arbitrary summary covariance '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        if self.mm:
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries**2
            self.SN_integral_mmmm = np.zeros((self.mmE_summaries, self.mmE_summaries,  1, self.n_tomo_lens, self.n_tomo_lens))
            original_shape = (1,self.n_tomo_lens, self.n_tomo_lens)
            dnpair_mm_flat = np.reshape(self.dnpair_mm, (len(self.theta_real_integral), self.n_tomo_lens*self.n_tomo_lens))
            
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.gm_summaries + self.gg_summaries + self.mmE_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gm_summaries + self.gg_summaries + self.mmE_summaries):
                    local_theta_limit = self.theta_limits[m_mode][:]
                    if len(self.theta_limits[m_mode][:]) < len(self.theta_limits[n_mode][:]):
                        local_theta_limit = self.theta_limits[n_mode][:]
                    if self.theta_limits[n_mode][0] == self.theta_limits[m_mode][-1] or self.theta_limits[m_mode][0] == self.theta_limits[n_mode][-1]:
                        tcomb += 1
                        continue
                    integrand = 1/(dnpair_mm_flat* 60*180/np.pi)*(self.theta_real_integral[:, None]/60/180*np.pi)**2
                    self.levin_int_real.init_integral(self.theta_real_integral/60/180*np.pi, integrand, True, True)
                    self.SN_integral_mmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :] = np.reshape(np.array(self.levin_int_real.cquad_integrate_double_well(local_theta_limit, m_mode, n_mode)), original_shape)
                    self.SN_integral_mmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :] += np.reshape(np.array(self.levin_int_real.cquad_integrate_double_well(local_theta_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)), original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCalculating Shot Noise integrals for mmmm arbitrary summary covariance '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
    
    def calc_covarbsummary(self,
                           obs_dict,
                           output_dict,
                           bias_dict,
                           hod_dict,
                           survey_params_dict,
                           prec,
                           read_in_tables):
        """
        Calculates the full covariance between all observables for bandpowers
        as specified in the config file.

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
                ['CEgggg', 'CEgggm', 'CEggmm', 'CBggmm', 'CEgmgm', 'CEmmgm',
                 'CBmmgm', 'CEEmmmm', 'CEBmmmm', 'CBBmmmm']
            each entry with shape (theta bins, theta bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['split_gauss']:
            if self.csmf:
                gauss_CEgggg, gauss_CEgggm, gauss_CEggmm, gauss_CBggmm, \
                    gauss_CEgmgm, gauss_CEmmgm, gauss_CBmmgm, \
                    gauss_CEEmmmm, gauss_CEBmmmm, \
                    gauss_CBBmmmm, \
                    gauss_CEgggg_sn, gauss_CEgmgm_sn, gauss_CEEmmmm_sn, gauss_CBBmmmm_sn, \
                    csmf_AS_auto, csmf_AS_gg, csmf_AS_gm, csmf_AS_mmE, csmf_AS_mmB = \
                    self.covarbsummary_gaussian(obs_dict,
                                        survey_params_dict)
                gauss = [gauss_CEgggg + gauss_CEgggg_sn, gauss_CEgggm, gauss_CEggmm, gauss_CBggmm,
                        gauss_CEgmgm + gauss_CEgmgm_sn, gauss_CEmmgm, gauss_CBmmgm,
                        gauss_CEEmmmm + gauss_CEEmmmm_sn, gauss_CEBmmmm,
                        gauss_CBBmmmm + gauss_CBBmmmm_sn,
                        csmf_AS_auto, csmf_AS_gg, csmf_AS_gm, csmf_AS_mmE, csmf_AS_mmB]
            else:
                gauss_CEgggg, gauss_CEgggm, gauss_CEggmm, gauss_CBggmm, \
                    gauss_CEgmgm, gauss_CEmmgm, gauss_CBmmgm, \
                    gauss_CEEmmmm, gauss_CEBmmmm, \
                    gauss_CBBmmmm, \
                    gauss_CEgggg_sn, gauss_CEgmgm_sn, gauss_CEEmmmm_sn, gauss_CBBmmmm_sn = \
                    self.covarbsummary_gaussian(obs_dict,
                                        survey_params_dict)
                gauss = [gauss_CEgggg + gauss_CEgggg_sn, gauss_CEgggm, gauss_CEggmm, gauss_CBggmm,
                        gauss_CEgmgm + gauss_CEgmgm_sn, gauss_CEmmgm, gauss_CBmmgm,
                        gauss_CEEmmmm + gauss_CEEmmmm_sn, gauss_CEBmmmm,
                        gauss_CBBmmmm + gauss_CBBmmmm_sn]
        else:
            gauss = self.covarbsummary_gaussian(obs_dict,
                                       survey_params_dict)
        
        nongauss = self.covarbsummary_non_gaussian(obs_dict['ELLspace'],
                                                    survey_params_dict,
                                                    output_dict,
                                                    bias_dict,
                                                    hod_dict,
                                                    prec,
                                                    read_in_tables['tri'])

        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            ssc = []
            for i_list in range(len(nongauss)):
                if nongauss[i_list] is not None:
                    ssc.append(nongauss[i_list]*0)
                else:
                    ssc.append(None)
        else:
            ssc = self.covarbsummary_ssc(obs_dict['ELLspace'],
                                    survey_params_dict,
                                    output_dict,
                                    bias_dict,
                                    hod_dict,
                                    prec,
                                    read_in_tables['tri'])

        return list(gauss), list(nongauss), list(ssc)

    def covarbsummary_gaussian(self,
                               obs_dict,
                               survey_params_dict):
        """
        Calculates the Gaussian (disconnected) covariance for arbitrary summary
        for the all specified observables.

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
            'bandpowers' : dictionary
                Specifies the exact details for the bandpower covariance.
            'arbitrary_summary' dictionary
                Specifies a few details for the arbitrary summary covariance
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        
        Returns
        -------
        Depending whether split_gaussian is true or not.
        split_gauss == True:
            gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
            gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
            gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
            gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
            gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
            gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
            gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
            gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
            gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
            gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn : list of arrays
            
            each with shape (ell bins, ell bins,
                             sample bins, sample bins,
                             n_tomo_clust/lens, n_tomo_clust/lens,
                             n_tomo_clust/lens, n_tomo_clust/lens)
        split_gauss == False
            gauss_ASgggg, gauss_ASgggm, gauss_ASEggmm, gauss_ASBggmm, \
            gauss_ASgmgm, gauss_ASEmmgm, gauss_ASBmmgm, \
            gauss_ASEEmmmm, gauss_ASEBmmmm, \
            gauss_ASBBmmmm, \
            gauss_ASgggg_sn, gauss_ASgmgm_sn, gauss_ASEEmmmm_sn, gauss_ASBBmmmm_sn

            each with shape (ell bins, ell bins,
                             sample bins, sample bins,
                             n_tomo_clust/lens, n_tomo_clust/lens,
                             n_tomo_clust/lens, n_tomo_clust/lens)
        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the diagonal terms of the covariance
        matrix, one needs to add gauss_xy + gauss_xy_sn. They
        are kept separated.

        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        print("Calculating gaussian arbitrary summary covariance from angular " +
              "correlations.")

        if self.csmf:
            gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn, \
                csmf_AS_auto, csmf_AS_gg, csmf_AS_gm, csmf_AS_mmE, csmf_AS_mmB = \
                self.__covarbsummary_split_gaussian(obs_dict,
                                            survey_params_dict)
        else:
            gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn = \
                self.__covarbsummary_split_gaussian(obs_dict,
                                            survey_params_dict)

        if self.csmf:
            if not self.cov_dict['split_gauss']:
                gauss_ASgggg = gauss_ASgggg_sva + gauss_ASgggg_mix
                gauss_ASgggm = gauss_ASgggm_sva + gauss_ASgggm_mix
                gauss_ASEggmm = gauss_ASEggmm_sva + gauss_ASEggmm_mix
                gauss_ASBggmm = gauss_ASBggmm_sva + gauss_ASBggmm_mix
                gauss_ASgmgm = gauss_ASgmgm_sva + gauss_ASgmgm_mix
                gauss_ASEmmgm = gauss_ASEmmgm_sva + gauss_ASEmmgm_mix
                gauss_ASBmmgm = gauss_ASBmmgm_sva + gauss_ASBmmgm_mix
                gauss_ASEEmmmm = gauss_ASEEmmmm_sva + gauss_ASEEmmmm_mix
                gauss_ASEBmmmm = gauss_ASEBmmmm_sva + gauss_ASEBmmmm_mix
                gauss_ASBBmmmm = gauss_ASBBmmmm_sva + gauss_ASBBmmmm_mix
                return gauss_ASgggg, gauss_ASgggm, gauss_ASEggmm, gauss_ASBggmm, \
                    gauss_ASgmgm, gauss_ASEmmgm, gauss_ASBmmgm, \
                    gauss_ASEEmmmm, gauss_ASEBmmmm, \
                    gauss_ASBBmmmm, \
                    gauss_ASgggg_sn, gauss_ASgmgm_sn, gauss_ASEEmmmm_sn, gauss_ASBBmmmm_sn, \
                    csmf_AS_auto, csmf_AS_gg, csmf_AS_gm, csmf_AS_mmE, csmf_AS_mmB
            else:
                return gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                        gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                        gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                        gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                        gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                        gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                        gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                        gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                        gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                        gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn, \
                        csmf_AS_auto, csmf_AS_gg, csmf_AS_gm, csmf_AS_mmE, csmf_AS_mmB
        else:
            if not self.cov_dict['split_gauss']:
                gauss_ASgggg = gauss_ASgggg_sva + gauss_ASgggg_mix
                gauss_ASgggm = gauss_ASgggm_sva + gauss_ASgggm_mix
                gauss_ASEggmm = gauss_ASEggmm_sva + gauss_ASEggmm_mix
                gauss_ASBggmm = gauss_ASBggmm_sva + gauss_ASBggmm_mix
                gauss_ASgmgm = gauss_ASgmgm_sva + gauss_ASgmgm_mix
                gauss_ASEmmgm = gauss_ASEmmgm_sva + gauss_ASEmmgm_mix
                gauss_ASBmmgm = gauss_ASBmmgm_sva + gauss_ASBmmgm_mix
                gauss_ASEEmmmm = gauss_ASEEmmmm_sva + gauss_ASEEmmmm_mix
                gauss_ASEBmmmm = gauss_ASEBmmmm_sva + gauss_ASEBmmmm_mix
                gauss_ASBBmmmm = gauss_ASBBmmmm_sva + gauss_ASBBmmmm_mix
                return gauss_ASgggg, gauss_ASgggm, gauss_ASEggmm, gauss_ASBggmm, \
                    gauss_ASgmgm, gauss_ASEmmgm, gauss_ASBmmgm, \
                    gauss_ASEEmmmm, gauss_ASEBmmmm, \
                    gauss_ASBBmmmm, \
                    gauss_ASgggg_sn, gauss_ASgmgm_sn, gauss_ASEEmmmm_sn, gauss_ASBBmmmm_sn
            else:
                return gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                        gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                        gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                        gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                        gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                        gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                        gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                        gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                        gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                        gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn


    def __covarbsummary_split_gaussian(self,
                                       obs_dict,
                                       survey_params_dict):
        """
        Calculates the Gaussian (disconnected) covariance for arbitrary
        summary statistics as specified by input tables
        for the specified observables and splits it into sample-variance
        (SVA), shot noise (SN) and SNxSVA(mix) terms.

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
            'bandpowers' : dictionary
                Specifies the exact details for the bandpower covariance.
            'arbitrary_summary' dictionary
                Specifies a few details for the arbitrary summary covariance
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
            gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
            gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
            gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
            gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
            gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
            gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
            gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
            gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
            gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
            gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn : list of arrays
            
            each with shape (spatial_scales, spatial_scales, 
                            sample bins, sample bins,
                            n_tomo_clust/lens, n_tomo_clust/lens,
                            n_tomo_clust/lens, n_tomo_clust/lens)
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
                obs_dict['ELLspace'], survey_params_dict, False)
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, _, \
                gaussELLgggm_sva, gaussELLgggm_mix, _, \
                gaussELLggmm_sva, _, _, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, _, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, _ = self.covELL_gaussian(
                    obs_dict['ELLspace'], survey_params_dict, False)
        self.cov_dict['split_gauss'] = save_entry
        if self.gg or self.gm:
            kron_delta_tomo_clust = np.diag(np.ones(self.n_tomo_clust))
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))
        if self.mm or self.gm:
            kron_delta_tomo_lens = np.diag(survey_params_dict['ellipticity_dispersion']**2)
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))

        if self.gg:
            gauss_ASgggg_sva = np.zeros(
                (self.gg_summaries, self.gg_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            gauss_ASgggg_mix = np.zeros_like(gauss_ASgggg_sva)
            gauss_ASgggg_sn = np.zeros_like(gauss_ASgggg_sva)
            original_shape = gauss_ASgggg_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            gaussELL_sva_flat = np.reshape(gaussELLgggg_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgggg_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries**2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggg_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggg_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:    
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggg_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./(2.0*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                            
                    gauss_ASgggg_sn[m_mode, n_mode, :, :, :, :, :, :] =  (kron_delta_tomo_clust[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_clust[None, None, None, :, None, :]
                                                                            + kron_delta_tomo_clust[None, None, :, None, None, :]
                                                                            * kron_delta_tomo_clust[None, None, None, :, :, None]) \
                                                                            * kron_delta_mass_bins[:, :, None, None, None, None]*self.SN_integral_gggg[m_mode, n_mode, None, :, :, : ,None, None] 
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'gggg term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn = 0, 0, 0

        if self.gg and self.gm and self.cross_terms:
            gauss_ASgggm_sva = np.zeros(
                (self.gg_summaries, self.gm_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            gauss_ASgggm_mix = np.zeros_like(gauss_ASgggm_sva)
            original_shape = gauss_ASgggm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**3*self.n_tomo_lens
            gaussELL_sva_flat = np.reshape(gaussELLgggm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgggm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries*self.gm_summaries
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggm_sva[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggm_mix[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgggm_sva[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'gggm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
                    gauss_ASgggm_sn = 0
            print("")
        else:
            gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn = 0, 0, 0

        if self.gg and self.mm and self.cross_terms:
            gauss_ASEggmm_sva = np.zeros(
                (self.gg_summaries, self.mmE_summaries, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            gauss_ASBggmm_sva = np.zeros(
                (self.gg_summaries, self.mmB_summaries, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))  
            original_shape = gauss_ASEggmm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            gaussELL_sva_flat = np.reshape(gaussELLggmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries*self.mmE_summaries
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                    gauss_ASEggmm_sva[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    if len(local_ell_limit) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                        local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                    gauss_ASBggmm_sva[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'ggmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
                    gauss_ASEggmm_mix = 0
                    gauss_ASBggmm_mix = 0
                    gauss_ASEggmm_sn = 0
                    gauss_ASBggmm_sn = 0
            print("")
        else:
            gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn = 0, 0, 0
            gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn = 0, 0, 0

        if self.gm:
            gauss_ASgmgm_sva = np.zeros(
                (self.gm_summaries, self.gm_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            gauss_ASgmgm_mix = np.zeros_like(gauss_ASgmgm_sva)
            gauss_ASgmgm_sn = np.zeros_like(gauss_ASgmgm_sva)
            original_shape = gauss_ASgmgm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            gaussELL_sva_flat = np.reshape(gaussELLgmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gm_summaries**2
            for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgmgm_sva[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgmgm_mix[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASgmgm_sva[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        
                    gauss_ASgmgm_sn[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = (kron_delta_tomo_clust[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_lens[None, None, None, :, None, :]) \
                                                                            * kron_delta_mass_bins[:,:, None, None, None, None] \
                                                                            * self.SN_integral_gmgm[m_mode - self.gg_summaries, n_mode - self.gg_summaries, None, :, :, : ,None, None] 
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'gmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            adding = self.gaussELLgmgm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbE_gm[:, None, :, None, :, :, None, None]*self.arbE_gm[None, :, None, :, None, None, :, :])
            gauss_ASgmgm_sva[:, :, :, :, :, :, :, :] = gauss_ASgmgm_sva[:, :, :, :, :, :, :, :] + adding[:, :, :, :, :, :, :, :]
            print("")
        else:
            gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn = 0, 0, 0


        if self.mm and self.gm and self.cross_terms:
            gauss_ASEmmgm_sva = np.zeros(
                (self.mmE_summaries, self.gm_summaries, 1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            gauss_ASEmmgm_mix = np.zeros_like(gauss_ASEmmgm_sva)
            gauss_ASBmmgm_sva = np.zeros_like(gauss_ASEmmgm_sva)
            gauss_ASBmmgm_mix = np.zeros_like(gauss_ASEmmgm_sva)
            
            original_shape = gauss_ASEmmgm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_clust*self.n_tomo_lens**3
            gaussELL_sva_flat = np.reshape(gaussELLmmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLmmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries*self.gm_summaries
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEmmgm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                        gauss_ASBmmgm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                        local_ell_limit = self.ell_limits[m_mode][:]
                        if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                            local_ell_limit = self.ell_limits[n_mode][:]
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEmmgm_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                        gauss_ASBmmgm_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEmmgm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                        gauss_ASBmmgm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1./(2.*np.pi*survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode)),original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'mmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
                    gauss_ASEmmgm_sn = 0
                    gauss_ASBmmgm_sn = 0
            adding = self.gaussELLmmgm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbE_mm[:, None, :, None, :, :, None, None]*self.arbE_gm[None, :, None, :, None, None, :, :])
            gauss_ASEmmgm_sva[:, :, 0, :, :, :, :, :] = gauss_ASEmmgm_sva[:, :, 0, :, :, :, :, :] + adding[:, :, 0, :, :, :, :, :]
            adding = self.gaussELLmmgm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbB_mm[:, None, :, None, :, :, None, None]*self.arbE_gm[None, :, None, :, None, None, :, :])
            gauss_ASBmmgm_sva[:, :, 0, :, :, :, :, :] = gauss_ASBmmgm_sva[:, :, 0, :, :, :, :, :] + adding[:, :, 0, :, :, :, :, :]
            print("")
        else:
            gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn = 0, 0, 0
            gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn = 0, 0, 0

        if self.mm:
            gauss_ASEEmmmm_sva = np.zeros(
                (self.mmE_summaries, self.mmE_summaries, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            gauss_ASEEmmmm_mix = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASEEmmmm_sn = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASBBmmmm_sva = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASBBmmmm_sn = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASBBmmmm_mix = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASEBmmmm_sva = np.zeros_like(gauss_ASEEmmmm_sva)
            gauss_ASEBmmmm_mix = np.zeros_like(gauss_ASEEmmmm_sva)
            
            original_shape = gauss_ASEEmmmm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.n_tomo_lens**4
            gaussELL_sva_flat = np.reshape(gaussELLmmmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLmmmm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries**2
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEEmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                        gauss_ASEBmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                        gauss_ASBBmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                        local_ell_limit = self.ell_limits[m_mode][:]
                        if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                            local_ell_limit = self.ell_limits[n_mode][:]
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEEmmmm_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                        gauss_ASEBmmmm_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[m_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[m_mode + self.mmE_summaries][:]
                        gauss_ASBBmmmm_mix[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                    else:
                        self.levin_int_fourier.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gauss_ASEEmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        if len(local_ell_limit) < len(self.ell_limits[n_mode + self.mmE_summaries][:]):
                            local_ell_limit = self.ell_limits[n_mode + self.mmE_summaries][:]
                        gauss_ASEBmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.mmE_summaries)),original_shape)
                        if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                            local_ell_limit = self.ell_limits[n_mode][:]
                        gauss_ASBBmmmm_sva[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] =  1./(2.*np.pi*survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_double_well(local_ell_limit, m_mode + self.mmE_summaries, n_mode + self.mmE_summaries)),original_shape)
                    
                    gauss_ASEEmmmm_sn[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = (kron_delta_tomo_lens[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_lens[None, None, None, :, None, :]
                                                                            + kron_delta_tomo_lens[None, None, :, None, None, :]
                                                                            * kron_delta_tomo_lens[None, None, None, :, :, None]) \
                                                                            * self.SN_integral_mmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, None, :, :, : ,None, None]/0.5
                    gauss_ASBBmmmm_sn[n_mode - self.gg_summaries - self.gm_summaries, m_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = gauss_ASEEmmmm_sn[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :]
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the Gaussian '
                            'mmmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            gauss_ASEBmmmm_sn = 0
            adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbB_mm[:, None, :, None, :, :, None, None]*self.arbB_mm[None, :, None, :, None, None, :, :])
            gauss_ASBBmmmm_sva[:, :, 0, 0, :, :, :, :] = gauss_ASBBmmmm_sva[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
            adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbE_mm[:, None, :, None, :, :, None, None]*self.arbB_mm[None, :, None, :, None, None, :, :])
            gauss_ASEBmmmm_sva[:, :, 0, 0, :, :, :, :] = gauss_ASEBmmmm_sva[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
            adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.arbE_mm[:, None, :, None, :, :, None, None]*self.arbE_mm[None, :, None, :, None, None, :, :])
            gauss_ASEEmmmm_sva[:, :, 0, 0, :, :, :, :] = gauss_ASEEmmmm_sva[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
            print("")        
        else:
            gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn = 0, 0, 0
            gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn = 0, 0, 0
            gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn = 0, 0, 0

        if self.csmf:
            if self.gg:
                csmf_ASgg = np.zeros((self.gg_summaries, len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
                original_shape = csmf_gg[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust**2*self.n_tomo_csmf
                csmf_AS_flat = np.reshape(csmf_gg, (len(self.ellrange), flat_length))
                for m_mode in range(self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    self.levin_int_fourier.init_integral(self.ellrange, csmf_AS_flat, True, True)
                    csmf_ASgg[m_mode, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(local_ell_limit, m_mode)),original_shape)            
            else:
                csmf_ASgg = 0
            if self.gm:
                csmf_ASgm = np.zeros((self.gm_summaries, len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
                original_shape = csmf_gm[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust*self.n_tomo_lens*self.n_tomo_csmf
                csmf_AS_flat = np.reshape(csmf_gm, (len(self.ellrange), flat_length))
                for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    self.levin_int_fourier.init_integral(self.ellrange, csmf_AS_flat, True, True)
                    csmf_ASgm[m_mode - self.gg_summaries, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(local_ell_limit, m_mode)),original_shape)            
            else:
                csmf_ASgm = 0
            
            if self.mm:
                csmf_ASmmE = np.zeros((self.mmE_summaries, len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                csmf_ASmmB = np.zeros((self.mmB_summaries, len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                original_shape = csmf_mm[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins)*self.n_tomo_lens**2*self.n_tomo_csmf
                csmf_AS_flat = np.reshape(csmf_mm, (len(self.ellrange), flat_length))
                for m_mode in range(self.gg_summaries + self.gm_summaries, self.gm_summaries + self.gg_summaries + self.mmE_summaries):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    self.levin_int_fourier.init_integral(self.ellrange, csmf_AS_flat, True, True)
                    csmf_ASmmE[m_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(local_ell_limit, m_mode)),original_shape)            
                    csmf_ASmmB[m_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(local_ell_limit, m_mode + self.mmE_summaries)),original_shape)            

            else:
                csmf_ASmmE, csmf_ASmmB = 0, 0

        print("\nWrapping up all Gaussian arbitrary summary covariance contributions.")

        if self.csmf:
            return gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn, \
                csmf_auto, csmf_ASgg, csmf_ASgm, csmf_ASmmE, csmf_ASmmB
        else:
            return gauss_ASgggg_sva, gauss_ASgggg_mix, gauss_ASgggg_sn, \
                gauss_ASgggm_sva, gauss_ASgggm_mix, gauss_ASgggm_sn, \
                gauss_ASEggmm_sva, gauss_ASEggmm_mix, gauss_ASEggmm_sn, \
                gauss_ASBggmm_sva, gauss_ASBggmm_mix, gauss_ASBggmm_sn, \
                gauss_ASgmgm_sva, gauss_ASgmgm_mix, gauss_ASgmgm_sn, \
                gauss_ASEmmgm_sva, gauss_ASEmmgm_mix, gauss_ASEmmgm_sn, \
                gauss_ASBmmgm_sva, gauss_ASBmmgm_mix, gauss_ASBmmgm_sn, \
                gauss_ASEEmmmm_sva, gauss_ASEEmmmm_mix, gauss_ASEEmmmm_sn, \
                gauss_ASEBmmmm_sva, gauss_ASEBmmmm_mix, gauss_ASEBmmmm_sn, \
                gauss_ASBBmmmm_sva, gauss_ASBBmmmm_mix, gauss_ASBBmmmm_sn
        

    def covarbsummary_non_gaussian(self,
                                   covELLspacesettings,
                                   survey_params_dict,
                                   output_dict,
                                   bias_dict,
                                   hod_dict,
                                   prec,
                                   tri_tab):
        """
        Calculates the non-Gaussian covariance between all observables for the
        arbitrary summary as specified in the config file.

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
        
        Returns
        -------
            nongauss_ASgggg, nongauss_ASgggm, nongauss_ASEggmm, nongauss_ASBggmm, \
            nongauss_ASgmgm, nongauss_ASEmmgm, nongauss_ASBmmgm, nongauss_ASEEmmmm, \
            nongauss_ASEBmmmm, nongauss_ASBBmmmm : list of arrays

            each entry with shape (spatial_scales, spatial_scales, 
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        
        """

        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covarbsummary_4pt_projection(covELLspacesettings,
                                                       survey_params_dict,
                                                       output_dict,
                                                       bias_dict,
                                                       hod_dict,
                                                       prec,
                                                       tri_tab,
                                                       True)
    def covarbsummary_ssc(self,
                          covELLspacesettings,
                          survey_params_dict,
                          output_dict,
                          bias_dict,
                          hod_dict,
                          prec,
                          tri_tab):
        """
        Calculates the super sample covariance between all observables for
        arbitrary summary as specified in the config file.

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
        
        Returns
        -------
            nongauss_ASgggg, nongauss_ASgggm, nongauss_ASEggmm, nongauss_ASBggmm, \
            nongauss_ASgmgm, nongauss_ASEmmgm, nongauss_ASBmmgm, nongauss_ASEEmmmm, \
            nongauss_ASEBmmmm, nongauss_ASBBmmmm : list of arrays

            each entry with shape (spatial_scales, spatial_scales, 
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covarbsummary_4pt_projection(covELLspacesettings,
                                                       survey_params_dict,
                                                       output_dict,
                                                       bias_dict,
                                                       hod_dict,
                                                       prec,
                                                       tri_tab,
                                                       False)

    def __covarbsummary_4pt_projection(self,
                                       covELLspacesettings,
                                       survey_params_dict,
                                       output_dict,
                                       bias_dict,
                                       hod_dict,
                                       prec,
                                       tri_tab,
                                       connected):
        """
        Auxillary function to integrate four point functions from ell space to
        arbitrary summary space for all observables specified in the input file.

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
        connected : boolean
            If True the trispectrum is considered
            If False the SSC is considered

        Returns
        -------
            nongauss_CEgggg, nongauss_CEgggm, nongauss_CEggmm, nongauss_CBggmm, \
            nongauss_CEgmgm, nongauss_CEmmgm, nongauss_CBmmgm, nongauss_CEEmmmm, \
            nongauss_CEBmmmm, nongauss_CBBmmmm
        
            each entry with shape (spatial_scales, spatial_scales, 
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        nongauss_ASgggg = None
        nongauss_ASgggm = None
        nongauss_ASEggmm = None
        nongauss_ASBggmm = None
        nongauss_ASgmgm = None
        nongauss_ASEmmgm = None
        nongauss_ASBmmgm = None
        nongauss_ASEEmmmm = None
        nongauss_ASEBmmmm = None
        nongauss_ASBBmmmm = None
        self.levin_int_fourier.update_Levin(0, 16, 32,1e-3,1e-4)
        
        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab)
            nongaussELLgggg1, nongaussELLgggm1, nongaussELLggmm1, nongaussELLgmgm1, nongaussELLmmgm1, nongaussELLmmmm1 = self.covELL_ssc(
                    bias_dict, hod_dict, prec, survey_params_dict, covELLspacesettings)
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
        else:
            if (connected):
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab)
            else:
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_ssc(
                    bias_dict, hod_dict, prec, survey_params_dict, covELLspacesettings)
        if self.gg:
            nongauss_ASgggg = np.zeros(
                (self.gg_summaries, self.gg_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            original_shape = nongauss_ASgggg[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            nongaussELL_flat = np.reshape(nongaussELLgggg, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries**2
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_ASgggg[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_ASgggg[m_mode, n_mode, :, :, :, :, :, :] /= (survey_params_dict['survey_area_clust'] / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\Arbitrary summary covariance calculation for the '
                            'nonGaussian gggg term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            nongauss_ASgggg = 0

        if self.gg and self.gm and self.cross_terms:
            nongauss_ASgggm = np.zeros(
                (self.gg_summaries, self.gm_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            original_shape = nongauss_ASgggm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**3*self.n_tomo_lens
            nongaussELL_flat = np.reshape(nongaussELLgggm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gm_summaries* self.gg_summaries
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_ASgggm[m_mode, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_ASgggm[m_mode, n_mode, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the '
                            'nonGaussian gggm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            nongauss_ASgggm = 0

        if self.gg and self.mm and self.cross_terms:
            nongauss_ASEggmm = np.zeros(
                (self.gg_summaries, self.mmE_summaries, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            nongauss_ASBggmm = np.zeros_like(nongauss_ASEggmm)
            original_shape = nongauss_ASEggmm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELL_flat = np.reshape(nongaussELLggmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gg_summaries * self.mmE_summaries
            for m_mode in range(self.gg_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.gg_summaries + self.gm_summaries + self.mmE_summaries):
                    inner_integralE = np.zeros((len(self.ellrange), flat_length))
                    inner_integralB = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integralE[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                        inner_integralB[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode + self.mmE_summaries][:], n_mode + self.mmE_summaries))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralE*self.ellrange[:, None], True, True)
                    nongauss_ASEggmm[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralB*self.ellrange[:, None], True, True)
                    nongauss_ASBggmm[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_ASEggmm[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                        nongauss_ASBggmm[m_mode, n_mode - self.gg_summaries - self.gm_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the '
                            'nonGaussian ggmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            nongauss_ASEggmm = 0
            nongauss_ASBggmm = 0

        if self.gm:
            nongauss_ASgmgm = np.zeros(
                (self.gm_summaries, self.gm_summaries, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            original_shape = nongauss_ASgmgm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELL_flat = np.reshape(nongaussELLgmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.gm_summaries**2
            for m_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                for n_mode in range(self.gg_summaries, self.gm_summaries + self.gg_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_ASgmgm[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongauss_ASgmgm[m_mode - self.gg_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_ggl']) / self.deg2torad2
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the '
                            'nonGaussian gmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            nongauss_ASgmgm = 0


        if self.mm and self.gm and self.cross_terms:
            nongauss_ASEmmgm = np.zeros(
                (self.mmE_summaries, self.gm_summaries, 1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            nongauss_ASBmmgm = np.zeros_like(nongauss_ASEmmgm)
            original_shape = nongauss_ASEmmgm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_clust*self.n_tomo_lens**3
            nongaussELL_flat = np.reshape(nongaussELLmmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries*self.gm_summaries
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries, self.gg_summaries + self.gm_summaries):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongauss_ASEmmgm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    nongauss_ASBmmgm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode + self.mmE_summaries][:], m_mode + self.mmE_summaries)),original_shape)
                    if connected:
                        nongauss_ASEmmgm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                        nongauss_ASBmmgm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the '
                            'nonGaussian mmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
        else:
            nongauss_ASEmmgm, nongauss_ASBmmgm = 0, 0

        if self.mm:
            nongauss_ASEEmmmm = np.zeros(
                (self.mmE_summaries, self.mmE_summaries, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            nongauss_ASEBmmmm = np.zeros_like(nongauss_ASEEmmmm)
            nongauss_ASBBmmmm = np.zeros_like(nongauss_ASEEmmmm)
            original_shape = nongauss_ASEEmmmm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.n_tomo_lens**4
            nongaussELL_flat = np.reshape(nongaussELLmmmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.mmE_summaries**2
            for m_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                for n_mode in range(self.gg_summaries + self.gm_summaries, self.mmE_summaries + self.gg_summaries + self.gm_summaries):
                    inner_integralE = np.zeros((len(self.ellrange), flat_length))
                    inner_integralB = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int_fourier.init_integral(self.ellrange, nongaussELL_flat[:, i_ell, :]*self.ellrange[:, None], True, True)
                        inner_integralE[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                        inner_integralB[i_ell, :] = np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[n_mode + self.mmE_summaries][:], n_mode + self.mmE_summaries))
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralE*self.ellrange[:, None], True, True)
                    nongauss_ASEEmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    self.levin_int_fourier.init_integral(self.ellrange, inner_integralB*self.ellrange[:, None], True, True)
                    nongauss_ASEBmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    nongauss_ASBBmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int_fourier.cquad_integrate_single_well(self.ell_limits[m_mode + self.mmE_summaries][:], m_mode + self.mmE_summaries)),original_shape)
                    if connected:
                        nongauss_ASEEmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                        nongauss_ASEBmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                        nongauss_ASBBmmmm[m_mode - self.gg_summaries - self.gm_summaries, n_mode - self.gg_summaries + self.gm_summaries, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rArbitrary summary covariance calculation for the '
                            'nonGaussian mmmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print("")
            
        else:
            nongauss_ASEEmmmm, nongauss_ASBBmmmm, nongauss_ASEBmmmm = 0, 0, 0

        return nongauss_ASgggg, nongauss_ASgggm, nongauss_ASEggmm, nongauss_ASBggmm, \
            nongauss_ASgmgm, nongauss_ASEmmgm, nongauss_ASBmmgm, nongauss_ASEEmmmm, \
            nongauss_ASEBmmmm, nongauss_ASBBmmmm
