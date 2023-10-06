import time
import numpy as np
from scipy.interpolate import UnivariateSpline
import levin

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
        self.thetabins, self.theta_ul_bins = \
            self.__set_theta_bins(obs_dict['THETAspace'])
        if ((obs_dict['observables']['est_shear'] == 'xi_pm' and obs_dict['observables']['cosmic_shear']) or (obs_dict['observables']['est_ggl'] == 'gamma_t' and obs_dict['observables']['ggl']) or obs_dict['observables']['est_clust'] == 'w' and obs_dict['observables']['clustering']):
            self.npair_gg, self.npair_gm, self.npair_mm = \
                self.get_npair([self.gg, self.gm, self.mm],
                            self.theta_ul_bins,
                            self.thetabins,
                            survey_params_dict,
                            read_in_tables['npair'])
        self.__get_signal_ww()

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

    def __get_signal_ww(self):
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
        if self.gg:
            w_signal_shape = (len(self.thetabins),
                              self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust)
            original_shape = self.Cell_gg[0, :, :, :].shape
            w_signal = np.zeros(w_signal_shape)
            flat_length = self.sample_dim*self.n_tomo_clust**2
            Cell_gg_flat = np.reshape(
                self.Cell_gg, (len(self.ellrange), flat_length))
            w_signal_at_thetai_flat = np.zeros(flat_length)
            lev = levin.Levin(0, 16, 32, self.accuracy, self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                theta_i = self.thetabins[i_theta] / 60 * np.pi / 180
                integrand = Cell_gg_flat*self.ellrange[:, None]
                lev.init_integral(
                    self.ellrange, integrand, True, True)
                w_signal_at_thetai_flat = lev.single_bessel(
                    theta_i, 0, self.ellrange[0], self.ellrange[-1])
                w_signal[i_theta, :, :, :] = np.reshape(
                    w_signal_at_thetai_flat, original_shape)/2.0/np.pi
            self.w_gg = w_signal
        
        ## define spline on finer theta range, theta_min = theta_min/5, theta_max = theta_ax*2
        if self.mm:
            theta_ul_bins = np.geomspace(
                self.theta_ul_bins[0]/5,
                self.theta_ul_bins[-1]*4,
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
            lev = levin.Levin(0, 16, 32, self.accuracy, self.integration_intervals)
            self.xi_spline = {}
            self.xi_spline["xip"] = [None]*int(self.n_tomo_lens*(self.n_tomo_lens + 1)/2)
            self.xi_spline["xim"] = [None]*int(self.n_tomo_lens*(self.n_tomo_lens + 1)/2)
            for i_theta in range(len(theta_bins)):
                theta_i = theta_bins[i_theta] / 60 * np.pi / 180
                integrand = Cell_mm_flat*self.ellrange[:, None]
                lev.init_integral(
                    self.ellrange, integrand, True, True)
                xip_signal_at_thetai_flat = lev.single_bessel(
                    theta_i, 0, self.ellrange[0], self.ellrange[-1])
                xip_signal[i_theta, :, :, :] = np.reshape(
                    xip_signal_at_thetai_flat, original_shape)/2.0/np.pi
                xim_signal_at_thetai_flat = lev.single_bessel(
                    theta_i, 4, self.ellrange[0], self.ellrange[-1])
                xim_signal[i_theta, :, :, :] = np.reshape(
                    xim_signal_at_thetai_flat, original_shape)/2.0/np.pi        
            self.xip = xip_signal
            self.xim = xim_signal
            flat_idx = 0
            for i_tomo in range(self.n_tomo_lens):
                for j_tomo in range(i_tomo, self.n_tomo_lens):
                    self.xi_spline["xip"][flat_idx] = UnivariateSpline((theta_bins),(self.xip[:,0,i_tomo, j_tomo]), s=0)
                    self.xi_spline["xim"][flat_idx] = UnivariateSpline((theta_bins),(self.xim[:,0,i_tomo, j_tomo]), s=0)
                    flat_idx += 1
            
    def __get_triplet_mix_term(self,
                               CovTHETASpace_settings,
                               survey_params_dict):
        """
        Calculates the mixed term directly from a catalogue and therefore
        accounts for a more accurate prediction, especially at the survey
        edges
        """
        if CovTHETASpace_settings['mix_term_do_mix_for'] == 'xipxip' or CovTHETASpace_settings['mix_term_do_mix_for'] == 'ximxim':
            
            thisdata = DiscreteData(path_to_data=CovTHETASpace_settings['mix_term_file_path_catalog'], 
                    colname_weight=CovTHETASpace_settings['mix_term_col_name_weight'], 
                    colname_pos1=CovTHETASpace_settings['mix_term_col_name_pos1'], 
                    colname_pos2=CovTHETASpace_settings['mix_term_col_name_pos2'], 
                    colname_zbin=CovTHETASpace_settings['mix_term_col_name_zbin'], 
                    isspherical=CovTHETASpace_settings['mix_term_isspherical'], 
                    sigma2_eps= 4*survey_params_dict['ellipticity_dispersion']**2, 
                    target_patchsize=CovTHETASpace_settings['mix_term_target_patchsize'], 
                    do_overlap=CovTHETASpace_settings['mix_term_do_overlap'])
            thisdata.gen_patches(func=cygnus_patches, 
                     func_args={"ra":thisdata.pos1, "dec":thisdata.pos2, 
                                "g1":np.ones(len(thisdata.pos1)), "g2":np.ones(len(thisdata.pos1)), 
                                "e1":np.ones(len(thisdata.pos1)), "e2":np.ones(len(thisdata.pos1)),
                                "zbin":thisdata.zbin, "weight":thisdata.weight,
                                "overlap_arcmin":CovTHETASpace_settings['mix_term_do_overlap']*self.theta_ul_bins[-1]})
            disccov = DiscreteCovTHETASpace(discrete=thisdata,
                                xi_spl=self.xi_spline,
                                bin_edges=self.theta_ul_bins,
                                nmax=CovTHETASpace_settings['mix_term_nmax'],
                                nbinsphi=CovTHETASpace_settings['mix_term_nbins_phi'],
                                do_ec=CovTHETASpace_settings['mix_term_do_ec'],
                                savepath_triplets=CovTHETASpace_settings['mix_term_file_path_save_triplets'],
                                loadpath_triplets=CovTHETASpace_settings['mix_term_file_path_load_triplets'],
                                dpix_min_force=CovTHETASpace_settings['mix_term_dpix_min'],
                                terms=CovTHETASpace_settings['mix_term_do_mix_for'])
            disccov.compute_triplets()
            allmixedmeas, allshape = disccov.mixed_covariance()

        else:
            return None
        
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
        if self.mm or self.gm:
            kron_delta_tomo_lens = np.diag(np.ones(self.n_tomo_lens))

        if self.gg:
            print("")
            original_shape = gaussELLgggg_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_ww = self.__calc_prefac_covreal(
                    survey_params_dict['survey_area_clust'])
            else:
                prefac_ww = np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_ww = prefac_ww[:, :, None, None, None, None, None, None]

            covww_shape_sva = (len(self.thetabins), len(self.thetabins),
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
            
            lev = levin.Levin(2, 16, 32, self.accuracy/2.0, self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) **2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    integrand = np.moveaxis(np.diagonal(gaussELLgggg_sva_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    gauss_ww_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    integrand = np.moveaxis(np.diagonal(gaussELLgggg_mix_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    gauss_ww_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance ww at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_ww_sva *= prefac_ww
            gauss_ww_mix *= prefac_ww
            gauss_ww_sn = \
                (kron_delta_tomo_clust[None, :, None, :, None]
                 * kron_delta_tomo_clust[None, None, :, None, :]
                 + kron_delta_tomo_clust[None, :, None, None, :]
                 * kron_delta_tomo_clust[None, None, :, :, None]) \
                / self.npair_gg[:, :, :, None, None]
            gauss_ww_sn = \
                gauss_ww_sn[:, None, None, :, :, :, :] \
                * np.eye(len(self.thetabins))[:, :, None, None, None, None, None, None]
        else:
            gauss_ww_sva, gauss_ww_mix, gauss_ww_sn = 0, 0 , 0

        if self.gg and self.gm and self.cross_terms:
            print("")
            original_shape = gaussELLgggm_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_wgt = self.__calc_prefac_covreal(
                    max(survey_params_dict['survey_area_clust'],
                        survey_params_dict['survey_area_ggl']))
            else:
                prefac_wgt = np.ones(
                    (len(self.thetabins), len(self.thetabins)))
            prefac_wgt = prefac_wgt[:, :, None, None, None, None, None, None]

            covwgt_shape_sva = (len(self.thetabins), len(self.thetabins),
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
            
            lev = levin.Levin(2, 16, 32, self.accuracy/np.sqrt(8.), self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) ** 2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    integrand = np.moveaxis(np.diagonal(gaussELLgggm_sva_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand /= self.ellrange[:,None] 
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat -= 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2.*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2.*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    
                    gauss_wgt_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    integrand = np.moveaxis(np.diagonal(gaussELLgggm_mix_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand /= self.ellrange[:,None] 
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat -= 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2.*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2.*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    
                    gauss_wgt_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance wgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_wgt_sva *= prefac_wgt
            gauss_wgt_mix *= prefac_wgt
        else:
            gauss_wgt_sva, gauss_wgt_mix = 0, 0

        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = gaussELLggmm_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_wxipm = self.__calc_prefac_covreal(
                    max(survey_params_dict['survey_area_clust'],
                        survey_params_dict['survey_area_lens']))
            else:
                prefac_wxipm = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_wxipm = prefac_wxipm[:, :,
                                        None, None, None, None, None, None]

            covwxipm_shape_sva = (len(self.thetabins), len(self.thetabins),
                                  self.sample_dim, self.sample_dim,
                                  self.n_tomo_clust, self.n_tomo_clust,
                                  self.n_tomo_lens, self.n_tomo_lens)
            gauss_wxip_sva = np.zeros(covwxipm_shape_sva) if self.xi_pp else 0
            gauss_wxim_sva = np.zeros(covwxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            gaussELLggmm_sva_flat = np.reshape(gaussELLggmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            lev = levin.Levin(2, 16, 32, self.accuracy/np.sqrt(12.), self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins))**2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    if self.xi_pp:
                        integrand = np.moveaxis(np.diagonal(gaussELLggmm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_wxip_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    
                    if self.xi_mm:
                        integrand = np.moveaxis(np.diagonal(gaussELLggmm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                                self.ellrange, integrand, True, True)    
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_wxim_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)

                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance wxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_wxip_sva *= prefac_wxipm
            gauss_wxim_sva *= prefac_wxipm
        else:
            gauss_wxip_sva, gauss_wxim_sva = 0, 0

        if self.gm:
            print("")
            original_shape = gaussELLgmgm_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_gtgt = self.__calc_prefac_covreal(
                    survey_params_dict['survey_area_ggl'])
            else:
                prefac_gtgt = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_gtgt = prefac_gtgt[:, :, None, None, None, None, None, None]


            covgtgt_shape_sva = (len(self.thetabins), len(self.thetabins),
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
            lev = levin.Levin(2, 16, 32, self.accuracy/4.0, self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) **2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    integrand = np.moveaxis(np.diagonal(gaussELLgmgm_sva_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand /= self.ellrange[:,None] 
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat += 2*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand /= self.ellrange[:,None]
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat += 4.0*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 4.0*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 4.0*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 4.0*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    
                    gauss_gtgt_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    integrand = np.moveaxis(np.diagonal(gaussELLgmgm_mix_flat)/self.ellrange,0,-1)
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    
                    integrand /= self.ellrange[:,None] 
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat += 2*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_ui*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_li*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 2*theta_uj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2*theta_lj*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 0, 1, self.ellrange[0], self.ellrange[-1]))
                    

                    
                    
                    integrand /= self.ellrange[:,None]
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat += 4.0*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_uj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 4.0*np.nan_to_num(lev.double_bessel(
                        theta_ui, theta_lj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= 4.0*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_uj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 4.0*np.nan_to_num(lev.double_bessel(
                        theta_li, theta_lj, 0, 0, self.ellrange[0], self.ellrange[-1]))
                    
                    gauss_gtgt_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance gtgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")

            gauss_gtgt_sva *= prefac_gtgt
            gauss_gtgt_mix *= prefac_gtgt
    
            gauss_gtgt_sn = \
                kron_delta_tomo_clust[None, :, None, :, None] \
                * kron_delta_tomo_lens[None, None, :, None, :] \
                / self.npair_gm[:, :, :, None, None]  \
                * survey_params_dict['ellipticity_dispersion'][None, None, :, None, None]**2
            gauss_gtgt_sn = \
                gauss_gtgt_sn[:, None, None, :, :, :, :] \
                * np.eye(len(self.thetabins))[:, :, None, None, None, None, None, None]
        else:
            gauss_gtgt_sva, gauss_gtgt_mix, gauss_gtgt_sn = 0, 0, 0

        if self.mm and self.gm and self.cross_terms:
            print("")
            original_shape = gaussELLmmgm_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_xipmgt = self.__calc_prefac_covreal(
                    max(survey_params_dict['survey_area_ggl'],
                        survey_params_dict['survey_area_lens']))
            else:
                prefac_xipmgt = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_xipmgt = prefac_xipmgt[:, :,
                                          None, None, None, None, None, None]

            covxipmgt_shape_sva = (len(self.thetabins), len(self.thetabins),
                                   self.sample_dim, self.sample_dim,
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
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust*self.n_tomo_lens**3
            gaussELLmmgm_sva_flat = np.reshape(gaussELLmmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLmmgm_mix_flat = np.reshape(gaussELLmmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            
            lev = levin.Levin(2, 16, 32, self.accuracy/np.sqrt(24.), self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins))**2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    
                    if self.xi_pp:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmgm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 2.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        gauss_xipgt_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    
                    if self.xi_mm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmgm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 2*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 2*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 8*theta_uj/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_lj/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_uj/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_lj/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 16/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        gauss_ximgt_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)

                    if self.xi_pp:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmgm_mix_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 2.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        gauss_xipgt_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)

                    if self.xi_mm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmgm_mix_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = -theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 2*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 2*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 2*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 8*theta_uj/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_lj/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8*theta_uj/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8*theta_lj/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 0, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None] 
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 16/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16/theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 16/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 16/theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 0, self.ellrange[0], self.ellrange[-1]))
                        gauss_ximgt_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)

                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance xipmgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_xipgt_sva *= prefac_xipmgt
            gauss_ximgt_sva *= prefac_xipmgt
            gauss_xipgt_mix *= prefac_xipmgt
            gauss_ximgt_mix *= prefac_xipmgt        
        else:
            gauss_xipgt_sva, gauss_ximgt_sva, gauss_xipgt_mix, gauss_ximgt_mix = 0, 0, 0 ,0

        if self.mm:
            print("")
            original_shape = gaussELLmmmm_sva[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                prefac_xipm = self.__calc_prefac_covreal(
                    survey_params_dict['survey_area_lens'])
            else:
                prefac_xipm = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_xipm = prefac_xipm[:, :, None, None, None, None, None, None]

            covxipm_shape_sva = (len(self.thetabins), len(self.thetabins),
                                 self.sample_dim, self.sample_dim,
                                 self.n_tomo_lens, self.n_tomo_lens,
                                 self.n_tomo_lens, self.n_tomo_lens)
            gauss_xipxip_sva = np.zeros(covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_sva = np.zeros(covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_sva = np.zeros(covxipm_shape_sva) if self.xi_mm else 0
            gauss_xipxip_mix = np.zeros(covxipm_shape_sva) if self.xi_pp else 0
            gauss_xipxim_mix = np.zeros(covxipm_shape_sva) if self.xi_pm else 0
            gauss_ximxim_mix = np.zeros(covxipm_shape_sva) if self.xi_mm else 0
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens**4
            gaussELLmmmm_sva_flat = np.reshape(gaussELLmmmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELLmmmm_mix_flat = np.reshape(gaussELLmmmm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            lev = levin.Levin(2, 16, 32, self.accuracy/6.0, self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) **2
            

            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    
                    
                    if self.xi_pp:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))

                        
                        gauss_xipxip_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_pm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))        
                        cov_at_thetaij_flat += 8.*theta_li/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_xipxim_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_mm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_sva_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        
                        cov_at_thetaij_flat -= 8.*(theta_ui/theta_uj + theta_uj/theta_ui)*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*(theta_ui/theta_lj + theta_lj/theta_ui)*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*(theta_li/theta_uj + theta_uj/theta_li)*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*(theta_li/theta_lj + theta_lj/theta_li)*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64.*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64.*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64.*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64.*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        
                        cov_at_thetaij_flat += 64./theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 64./theta_ui/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_ui/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_li/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_li/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_ximxim_sva[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)

                    if self.xi_pp:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_mix_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_xipxip_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    
                    if self.xi_pm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_mix_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                    
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_xipxim_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                        
                    if self.xi_mm:
                        integrand = np.moveaxis(np.diagonal(gaussELLmmmm_mix_flat)/self.ellrange,0,-1)
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_ui*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += theta_li*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.*(theta_ui/theta_uj + theta_uj/theta_ui)*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*(theta_ui/theta_lj + theta_lj/theta_ui)*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.*(theta_li/theta_uj + theta_uj/theta_li)*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 8.*(theta_li/theta_lj + theta_lj/theta_li)*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64.*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64.*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64.*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64.*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 2, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 64./theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_ui*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_li*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 2, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand /= self.ellrange[:,None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat += 64./theta_ui/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_ui/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_ui, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= 64./theta_li/theta_uj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_uj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 64./theta_li/theta_lj*np.nan_to_num(lev.double_bessel(
                            theta_li, theta_lj, 1, 1, self.ellrange[0], self.ellrange[-1]))
                        gauss_ximxim_mix[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                        
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for Gaussian term for the '
                          'real-space covariance xipmxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            gauss_xipxip_sva *= prefac_xipm                
            gauss_xipxip_mix *= prefac_xipm                
            gauss_xipxim_sva *= prefac_xipm                
            gauss_xipxim_mix *= prefac_xipm                
            gauss_ximxim_sva *= prefac_xipm                
            gauss_ximxim_mix *= prefac_xipm                
            gauss_xipm_sn = \
                (kron_delta_tomo_lens[None, :, None, :, None]
                 * kron_delta_tomo_lens[None, None, :, None, :]
                 + kron_delta_tomo_lens[None, :, None, None, :]
                 * kron_delta_tomo_lens[None, None, :, :, None]) \
                / self.npair_mm[:, :, :, None, None] / 0.5 \
                * survey_params_dict['ellipticity_dispersion'][None, None, :, None, None]**4 
            gauss_xipm_sn = \
                gauss_xipm_sn[:, None, None, :, :, :, :] \
                * np.eye(len(self.thetabins))[:, :, None, None, None, None, None, None]

           
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

    def __calc_prefac_covreal(self,
                              Amax):
        """
        Calculates the prefactor for the real-sapce covariance matrix.
        In particular for the SVA and mix terms.

        Parameters
        ----------
        Amax : float
            Maximum of the survey area between the two probes under
            consideration.

        Returns:
        --------
        prefac : array
            Prefactor of shape (theta_bins, theta bins)
        """

        prefac = 1 / 2 / np.pi / Amax * self.deg2torad2 \
            * (2 / (self.theta_ul_bins[1:]**2
                    - self.theta_ul_bins[:-1]**2)[:, None]) \
            * (2 / (self.theta_ul_bins[1:]**2
                    - self.theta_ul_bins[:-1]**2)[None, :]) \
            * self.arcmin2torad2**2

        return prefac


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
        if (connected):
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                covELLspacesettings, output_dict, bias_dict, hod_dict, hm_prec, tri_tab)
        else:
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_ssc(
                bias_dict, hod_dict, hm_prec, survey_params_dict, covELLspacesettings)
        if self.gg:
            print("")
            original_shape = nongaussELLgggg[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_ww = self.__calc_prefac_covreal(
                        survey_params_dict['survey_area_clust'])/2/np.pi
                else:
                    prefac_ww = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_ww = np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_ww = prefac_ww[:, :, None, None, None, None, None, None]

            covww_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                    self.sample_dim, self.sample_dim,
                                    self.n_tomo_clust, self.n_tomo_clust,
                                    self.n_tomo_clust, self.n_tomo_clust)
            nongauss_ww = np.zeros(covww_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            nongaussELLgggg_flat = np.reshape(nongaussELLgggg, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) **2
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    # cov_NG(w^ij(theta1) w^kl(theta2))
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral = np.zeros(
                        (len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        integrand = nongaussELLgggg_flat[i_ell, :, :]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] = theta_uj*np.nan_to_num(lev.single_bessel(
                            theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] -= theta_lj*np.nan_to_num(lev.single_bessel(
                            theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand = inner_integral
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                        theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                        theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                    nongauss_ww[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    nongauss_ww[j_theta, i_theta, :, :, :, :, :,
                                :] = nongauss_ww[i_theta, j_theta, :, :, :, :, :, :]

                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance ww at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            nongauss_ww *= prefac_ww

        if self.gg and self.gm and self.cross_terms:
            # cov_NG(w^ij(theta1) gt^kl(theta2))
            original_shape = nongaussELLgggm[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_wgt = self.__calc_prefac_covreal(
                        max(survey_params_dict['survey_area_clust'],
                            survey_params_dict['survey_area_ggl']))/2/np.pi
                else:
                    prefac_wgt = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_wgt = np.ones(
                    (len(self.thetabins), len(self.thetabins)))
            prefac_wgt = prefac_wgt[:, :, None, None, None, None, None, None]

            covwgt_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                     self.sample_dim, self.sample_dim,
                                     self.n_tomo_clust, self.n_tomo_clust,
                                     self.n_tomo_clust, self.n_tomo_lens)
            nongauss_wgt = np.zeros(covwgt_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust**3*self.n_tomo_lens
            nongaussELLgggm_flat = np.reshape(nongaussELLgggm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins))**2
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral = np.zeros(
                        (len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        integrand = nongaussELLgggm_flat[i_ell, :, :]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] = -theta_uj*np.nan_to_num(lev.single_bessel(
                            theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += theta_lj*np.nan_to_num(lev.single_bessel(
                            theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = nongaussELLgggm_flat[i_ell,
                                                         :, :]/self.ellrange[:, None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] -= 2.*np.nan_to_num(lev.single_bessel(
                            theta_uj, 0, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += 2.*np.nan_to_num(lev.single_bessel(
                            theta_lj, 0, self.ellrange[0], self.ellrange[-1]))
                    integrand = inner_integral
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                        theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                        theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                    nongauss_wgt[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance wgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            nongauss_wgt *= prefac_wgt

        if self.gg and self.mm and self.cross_terms:
            print("")
            original_shape = nongaussELLggmm[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_wxipm = self.__calc_prefac_covreal(
                        max(survey_params_dict['survey_area_clust'],
                            survey_params_dict['survey_area_lens']))/2/np.pi
                else:
                    prefac_wxipm = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_wxipm = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_wxipm = prefac_wxipm[:, :,
                                        None, None, None, None, None, None]

            covwxip_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                      self.sample_dim, self.sample_dim,
                                      self.n_tomo_clust, self.n_tomo_clust,
                                      self.n_tomo_lens, self.n_tomo_lens)
            nongauss_wxip = np.zeros(covwxip_shape_nongauss)
            covwxim_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                      self.sample_dim, self.sample_dim,
                                      self.n_tomo_clust, self.n_tomo_clust,
                                      self.n_tomo_clust, self.n_tomo_clust)
            nongauss_wxim = np.zeros(covwxim_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELLggmm_flat = np.reshape(nongaussELLggmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) ** 2
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral_xip = np.zeros(
                        (len(self.ellrange), flat_length))
                    inner_integral_xim = np.zeros(
                        (len(self.ellrange), flat_length))
                    if self.xi_pp:
                        for i_ell in range(len(self.ellrange)):
                            integrand = nongaussELLggmm_flat[i_ell, :, :]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xip[i_ell, :] = theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xip[i_ell, :] -= theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral_xip
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_wxip[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_mm:
                        for i_ell in range(len(self.ellrange)):
                            integrand = nongaussELLggmm_flat[i_ell, :, :]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] = theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] -= theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                            integrand = nongaussELLggmm_flat[i_ell,
                                                             :, :]/self.ellrange[:, None]**2
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] -= 8.0/theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] += 8.0/theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                            integrand = nongaussELLggmm_flat[i_ell,
                                                             :, :]/self.ellrange[:, None]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] -= 8.0*np.nan_to_num(lev.single_bessel(
                                theta_uj, 2, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] += 8.0*np.nan_to_num(lev.single_bessel(
                                theta_lj, 2, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral_xim
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_wxim[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                        theta += 1
                        eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                        print('\rProjection for non-Gaussian term for the '
                          'real-space covariance wxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
                nongauss_wxim *= prefac_wxipm
                nongauss_wxip *= prefac_wxipm

        if self.gm:
            print("")
            original_shape = nongaussELLgmgm[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_gtgt = self.__calc_prefac_covreal(
                        survey_params_dict['survey_area_ggl'])/2/np.pi
                else:
                    prefac_gtgt = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_gtgt = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_gtgt = prefac_gtgt[:, :, None, None, None, None, None, None]
            covgtgt_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                      self.sample_dim, self.sample_dim,
                                      self.n_tomo_clust, self.n_tomo_lens,
                                      self.n_tomo_clust, self.n_tomo_lens)
            nongauss_gtgt = np.zeros(covgtgt_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust**2*self.n_tomo_lens**2
            nongaussELLgmgm_flat = np.reshape(nongaussELLgmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins)) ** 2
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral = np.zeros(
                        (len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        integrand = nongaussELLgmgm_flat[i_ell, :, :]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] = -theta_uj*np.nan_to_num(lev.single_bessel(
                            theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += theta_lj*np.nan_to_num(lev.single_bessel(
                            theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = nongaussELLgmgm_flat[i_ell,
                                                         :, :]/self.ellrange[:, None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] -= 2.*np.nan_to_num(lev.single_bessel(
                            theta_uj, 0, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += 2.*np.nan_to_num(lev.single_bessel(
                            theta_lj, 0, self.ellrange[0], self.ellrange[-1]))
                    integrand = inner_integral
                    lev.init_integral(
                        self.ellrange, integrand, True, True)
                    cov_at_thetaij_flat = -theta_ui*np.nan_to_num(lev.single_bessel(
                        theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += theta_li*np.nan_to_num(lev.single_bessel(
                        theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                    integrand = inner_integral/self.ellrange[:, None]
                    cov_at_thetaij_flat -= 2.*np.nan_to_num(lev.single_bessel(
                        theta_ui, 0, self.ellrange[0], self.ellrange[-1]))
                    cov_at_thetaij_flat += 2.*np.nan_to_num(lev.single_bessel(
                        theta_li, 0, self.ellrange[0], self.ellrange[-1]))
                    nongauss_gtgt[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                        cov_at_thetaij_flat, original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for non-Gaussian term for the '
                          'real-space covariance gtgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            nongauss_wgt *= prefac_wgt

        if self.mm and self.gm and self.cross_terms:
            print("")
            #
            original_shape = nongaussELLmmgm[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_xipmgt = self.__calc_prefac_covreal(
                        max(survey_params_dict['survey_area_ggl'],
                            survey_params_dict['survey_area_lens']))/2/np.pi
                else:
                    prefac_xipmgt = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_xipmgt = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_xipmgt = prefac_xipmgt[:, :,
                                          None, None, None, None, None, None]
            covxipgt_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                       self.sample_dim, self.sample_dim,
                                       self.n_tomo_lens, self.n_tomo_lens,
                                       self.n_tomo_clust, self.n_tomo_lens)
            nongauss_xipgt = np.zeros(covxipgt_shape_nongauss)
            covximgt_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                       self.sample_dim, self.sample_dim,
                                       self.n_tomo_lens, self.n_tomo_lens,
                                       self.n_tomo_clust, self.n_tomo_lens)
            nongauss_ximgt = np.zeros(covximgt_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * \
                self.n_tomo_clust*self.n_tomo_lens**3
            nongaussELLmmgm_flat = np.reshape(nongaussELLmmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins))**2
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral = np.zeros(
                        (len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        integrand = nongaussELLmmgm_flat[i_ell, :, :]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] = -theta_uj*np.nan_to_num(lev.single_bessel(
                            theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += theta_lj*np.nan_to_num(lev.single_bessel(
                            theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = nongaussELLmmgm_flat[i_ell,
                                                         :, :]/self.ellrange[:, None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        inner_integral[i_ell, :] -= 2.*np.nan_to_num(lev.single_bessel(
                            theta_uj, 0, self.ellrange[0], self.ellrange[-1]))
                        inner_integral[i_ell, :] += 2.*np.nan_to_num(lev.single_bessel(
                            theta_lj, 0, self.ellrange[0], self.ellrange[-1]))
                    if self.xi_pp:
                        integrand = inner_integral
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_xipgt[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_mm:
                        integrand = inner_integral
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral/self.ellrange[:, None]**2
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.0/theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.0/theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral/self.ellrange[:, None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.0*np.nan_to_num(lev.single_bessel(
                            theta_ui, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.0*np.nan_to_num(lev.single_bessel(
                            theta_li, 2, self.ellrange[0], self.ellrange[-1]))
                        nongauss_ximgt[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                        theta += 1
                        eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                        print('\rProjection for non-Gaussian term for the '
                          'real-space covariance xipmgt at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            nongauss_xipgt *= prefac_xipmgt
            nongauss_ximgt *= prefac_xipmgt

        if self.mm:
            print("")
            original_shape = nongaussELLmmmm[0, 0, :, :, :, :, :, :].shape
            if calc_prefac:
                if connected:
                    prefac_xipm = self.__calc_prefac_covreal(
                        survey_params_dict['survey_area_lens'])/2/np.pi
                else:
                    prefac_xipm = self.__calc_prefac_covreal(
                        self.deg2torad2)/2/np.pi
            else:
                prefac_xipm = \
                    np.ones((len(self.thetabins), len(self.thetabins)))
            prefac_xipm = prefac_xipm[:, :, None, None, None, None, None, None]
            covxipxip_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                        self.sample_dim, self.sample_dim,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_xipxip = np.zeros(covxipxip_shape_nongauss)
            covxipxim_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                        self.sample_dim, self.sample_dim,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_xipxim = np.zeros(covxipxim_shape_nongauss)
            covximxim_shape_nongauss = (len(self.thetabins), len(self.thetabins),
                                        self.sample_dim, self.sample_dim,
                                        self.n_tomo_lens, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens)
            nongauss_ximxim = np.zeros(covximxim_shape_nongauss)
            flat_length = self.sample_dim*self.sample_dim * self.n_tomo_lens**4
            nongaussELLmmmm_flat = np.reshape(nongaussELLmmmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            cov_at_thetaij_flat = np.zeros(flat_length)
            t0, theta = time.time(), 0
            theta_comb = (len(self.thetabins))**2
            lev = levin.Levin(0, 16, 32, self.accuracy/6., self.integration_intervals)
            for i_theta in range(len(self.thetabins)):
                for j_theta in range(len(self.thetabins)):
                    theta_li = self.theta_ul_bins[i_theta] / 60 * np.pi / 180
                    theta_ui = self.theta_ul_bins[i_theta+1] / 60 * np.pi / 180
                    theta_lj = self.theta_ul_bins[j_theta] / 60 * np.pi / 180
                    theta_uj = self.theta_ul_bins[j_theta+1] / 60 * np.pi / 180
                    inner_integral = np.zeros(
                        (len(self.ellrange), flat_length))
                    inner_integral_xip = np.zeros(
                        (len(self.ellrange), flat_length))
                    inner_integral_xim = np.zeros(
                        (len(self.ellrange), flat_length))
                    if self.xi_pp:
                        for i_ell in range(len(self.ellrange)):
                            integrand = nongaussELLmmmm_flat[i_ell, :, :]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xip[i_ell, :] = theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xip[i_ell, :] -= theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral_xip
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_xipxip[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_mm:
                        for i_ell in range(len(self.ellrange)):
                            integrand = nongaussELLmmmm_flat[i_ell, :, :]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] = theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] -= theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                            integrand = nongaussELLmmmm_flat[i_ell,
                                                             :, :]/self.ellrange[:, None]**2
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] -= 8.0/theta_uj*np.nan_to_num(lev.single_bessel(
                                theta_uj, 1, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] += 8.0/theta_lj*np.nan_to_num(lev.single_bessel(
                                theta_lj, 1, self.ellrange[0], self.ellrange[-1]))
                            integrand = nongaussELLmmmm_flat[i_ell,
                                                             :, :]/self.ellrange[:, None]
                            lev.init_integral(
                                self.ellrange, integrand, True, True)
                            inner_integral_xim[i_ell, :] -= 8.0*np.nan_to_num(lev.single_bessel(
                                theta_uj, 2, self.ellrange[0], self.ellrange[-1]))
                            inner_integral_xim[i_ell, :] += 8.0*np.nan_to_num(lev.single_bessel(
                                theta_lj, 2, self.ellrange[0], self.ellrange[-1]))
                    if self.xi_pp:
                        integrand = inner_integral_xip
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_xipxip[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_mm:
                        integrand = inner_integral_xim
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral_xim / \
                            self.ellrange[:, None]**2
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.0/theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.0/theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        integrand = inner_integral_xim/self.ellrange[:, None]
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat -= 8.0*np.nan_to_num(lev.single_bessel(
                            theta_ui, 2, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat += 8.0*np.nan_to_num(lev.single_bessel(
                            theta_li, 2, self.ellrange[0], self.ellrange[-1]))
                        nongauss_ximxim[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    if self.xi_pm:
                        integrand = inner_integral_xim
                        lev.init_integral(
                            self.ellrange, integrand, True, True)
                        cov_at_thetaij_flat = theta_ui*np.nan_to_num(lev.single_bessel(
                            theta_ui, 1, self.ellrange[0], self.ellrange[-1]))
                        cov_at_thetaij_flat -= theta_li*np.nan_to_num(lev.single_bessel(
                            theta_li, 1, self.ellrange[0], self.ellrange[-1]))
                        nongauss_xipxim[i_theta, j_theta, :, :, :, :, :, :] = np.reshape(
                            cov_at_thetaij_flat, original_shape)
                    theta += 1
                    eta = (time.time()-t0)/60 * (theta_comb/theta-1)
                    print('\rProjection for connected term for the '
                          'real-space covariance xipmxipm at ' +
                          str(round(theta/theta_comb*100, 1)) + '% in ' +
                          str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                          str(round(eta, 1)) + 'min', end="")
            nongauss_xipxip *= prefac_xipm
            nongauss_ximxim *= prefac_xipm
            nongauss_xipxim *= prefac_xipm
        print("")
        return nongauss_ww, nongauss_wgt, nongauss_wxip, nongauss_wxim, nongauss_gtgt, nongauss_xipgt, nongauss_ximgt, nongauss_xipxip, nongauss_xipxim, nongauss_ximxim
