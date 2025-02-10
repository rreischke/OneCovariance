import numpy as np
import configparser
import glob
from os import walk, path
from astropy.io import ascii, fits
from scipy.interpolate import interp1d



class Input:
    """
    Provides methods to read-in all parameters from a configuration file 
    that are needed for the covariance code. Some parameters are 
    compulsory, others are optional. Whenever an optional parameter is 
    not set explicitly and a fall-back value is inserted, a message will 
    be displayed to inform the user what the value has been set to. 
    Lastly, a new configuration file can be generated that lists all 
    values used. A full list of all input values is given in the config.ini .

    Attributes
    ----------
    Too many to list, but all relevant ones are put into
    dictionaries and explained in the method 'read_input'

    Example :
    ---------
    from cov_input import Input
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()

    """

    def __init__(self):
        self.config_name = None

        # terms that enter the covariance
        self.covterms = dict()
        self.gauss = None
        self.split_gauss = None
        self.nongauss = None
        self.ssc = None

        # observables and their estimator
        self.observables = dict()
        self.observables_abr = dict()
        self.cosmicshear = None
        self.est_shear = None
        self.ggl = None
        self.est_ggl = None
        self.clustering = None
        self.est_clust = None
        self.cross_terms = None
        self.clustering_z = None
        self.unbiased_clustering = None

        # Conditional stellar mass function
        self.cstellar_mf = None
        self.csmf_log10Mmin = None
        self.csmf_log10Mmax = None
        self.csmf_N_log10M_bin = None
        self.csmf_log10M_bins = None
        self.csmf_diagonal = None

        

        # output settings
        self.output = dict()
        self.output_abr = dict()
        self.output_dir = None
        self.output_file = None
        self.output_style = None
        self.make_plot = None
        self.save_configs = None
        self.save_Cells = None
        self.save_trispecs = None
        self.save_alms = None
        self.use_tex = None
        self.list_style_spatial_first = None
        self.save_as_binary = None
        # for lensing in projected Fourier space
        self.covELLspace_settings = dict()
        self.covELLspace_settings_abr = dict()
        self.ell_min = None
        self.ell_max = None
        self.ell_bins = None
        self.ell_type = None
        self.delta_z = None
        self.integration_steps = None
        self.tri_delta_z = None
        self.nz_polyorder = None
        self.limber = None
        self.nglimber = None
        self.multiplicative_shear_bias_uncertainty = None
        self.pixelised_cell = None
        self.pixel_Nside = None
        self.n_spec = None
        self.ell_spec_min = None
        self.ell_spec_max = None
        self.ell_spec_bins = None
        self.ell_spec_type = None
        self.ell_photo_min = None
        self.ell_photo_max = None
        self.ell_photo_bins = None
        self.ell_photo_type = None
        self.ell_min_lensing = None
        self.ell_max_lensing = None
        self.ell_bins_lensing = None
        self.ell_type_lensing = None
        self.ell_min_clustering = None
        self.ell_max_clustering = None
        self.ell_bins_clustering = None
        self.ell_type_clustering = None
        

        # for cosmic shear in projected real space
        self.covTHETAspace_settings = dict()
        self.covTHETAspace_settings_abr = dict()
        self.theta_min = None
        self.theta_max = None
        self.theta_bins = None
        self.theta_type = None
        self.theta_min_clustering = None
        self.theta_max_clustering = None
        self.theta_bins_clustering = None
        self.theta_type_clustering = None
        self.theta_min_lensing = None
        self.theta_max_lensing = None
        self.theta_bins_lensing = None
        self.theta_type_lensing = None
        self.xi_pp = None
        self.xi_pm = None
        self.xi_mm = None
        self.theta_list = None
        self.theta_acc = None
        self.integration_intervals = None
        self.mix_term_file_path_catalog = None
        self.mix_term_col_name_weight = None
        self.mix_term_col_name_pos1 = None
        self.mix_term_col_name_pos2 = None
        self.mix_term_col_name_zbin = None
        self.mix_term_isspherical = None
        self.mix_term_target_patchsize = None
        self.mix_term_do_overlap = None
        self.mix_term_do_mix_for = None
        self.mix_term_nbins_phi = None
        self.mix_term_nmax = None
        self.mix_term_do_ec = None
        self.mix_term_subsample = None
        self.mix_term_nsubr = None
        self.mix_term_file_path_save_triplets = None
        self.mix_term_file_path_load_triplets = None


        # for COSEBIs
        self.covCOSEBI_settings = dict()
        self.covCOESBI_settings_abr = dict()
        self.En_modes = None
        self.theta_min_cosebi = None
        self.theta_max_cosebi = None
        self.En_modes_clustering = None
        self.theta_min_cosebi_clustering = None
        self.theta_max_cosebi_clustering = None
        self.En_modes_lensing = None
        self.theta_min_cosebi_lensing = None
        self.theta_max_cosebi_lensing = None
        self.dimensionless_cosebi = None
        
        self.En_acc = None
        self.Wn_style = None
        self.Wn_acc = None

        # for bandpowers
        self.covbandpowers_settings = dict()
        self.covbandpowers_settings_abr = dict()
        self.apodisation_log_width_clustering = None
        self.theta_lo_clustering = None
        self.theta_up_clustering = None
        self.bp_ell_min_clustering = None
        self.bp_ell_max_clustering = None
        self.bp_ell_bins_clustering = None
        self.bp_ell_type_clustering = None
        self.apodisation_log_width_lensing = None
        self.theta_lo_lensing = None
        self.theta_up_lensing = None
        self.bp_ell_min_lensing = None
        self.bp_ell_max_lensing = None
        self.bp_ell_bins_lensing = None
        self.bp_ell_type_lensing = None
        
        self.theta_binning = None
        self.bandpower_accuracy = None
        
        # for arbitrary summary statistics
        self.arbitrary_summary_settings = dict()
        self.arbitrary_summary_settings_abr = dict()
        self.do_arbitrary_obs = None
        self.oscillations_straddle = None
        self.arbitrary_accuracy = None

        # for GGL in real space
        self.covRspace_settings = dict()
        self.covRspace_settings_abr = dict()
        self.projected_radius_min = None
        self.projected_radius_max = None
        self.projected_radius_bins = None
        self.projected_radius_type = None
        self.mean_redshift = None
        self.projection_length_clustering = None

        # cosmological parameters
        self.cosmo = dict()
        self.cosmo_abr = dict()
        self.sigma8 = None
        self.As = None
        self.h = None
        self.omegam = None
        self.omegab = None
        self.omegade = None
        self.w0 = None
        self.wa = None
        self.ns = None
        self.neff = None
        self.mnu = None
        self.Tcmb0 = None

        # galaxy bias parameters
        self.bias = dict()
        self.bias_abr = dict()
        self.bias_model = None
        self.bias_2h = None
        self.Mc_relation_cen = None
        self.Mc_relation_sat = None
        self.norm_Mc_relation_cen = None
        self.norm_Mc_relation_sat = None
        self.logmass_bins = None
        self.sampledim = None

        # intrinsic alignment parameters
        self.intrinsic_alignments = dict()
        self.intrinsic_alignments_abr = dict()
        self.A_IA = None
        self.eta_IA = None
        self.z_pivot_IA = None

        # halo occupation distribution parameters
        self.hod = dict()
        self.hod_abr = dict()
        # hod: mass-observable relation
        self.hod_model_mor_cen = None
        self.hod_model_mor_sat = None
        # hod: if the mass-observable relation is 'double_powerlaw'
        self.dpow_logM0_cen = None
        self.dpow_logM1_cen = None
        self.dpow_a_cen = None
        self.dpow_b_cen = None
        self.dpow_norm_cen = None
        self.dpow_logM0_sat = None
        self.dpow_logM1_sat = None
        self.dpow_a_sat = None
        self.dpow_b_sat = None
        self.dpow_norm_sat = None
        # hod: scattering relation
        self.hod_model_scatter_cen = None
        self.hod_model_scatter_sat = None
        # hod: if scattering relation is 'lognormal'
        self.logn_sigma_c_cen = None
        self.logn_sigma_c_sat = None
        # hod: if the scattering relation is 'modschechter'
        # (modified schechter)
        self.modsch_logMref_cen = None
        self.modsch_alpha_s_cen = None
        self.modsch_b_cen = None
        self.modsch_logMref_sat = None
        self.modsch_alpha_s_sat = None
        self.modsch_b_sat = None

        # parameters unique to a specific survey
        self.survey_params = dict()
        self.survey_params_abr = dict()
        self.mask_dir = None
        self.mask_file_clust = None
        self.read_mask_clust = None
        self.alm_file_clust = None
        self.read_alm_clust = None
        self.survey_area_clust = None
        self.n_eff_clust = None
        self.shot_noise_clust = None
        self.tomos_6x2pt_clust = None
        self.mask_file_ggl = None
        self.read_mask_ggl = None
        self.alm_file_ggl = None
        self.read_alm_ggl = None
        self.survey_area_ggl = None
        self.n_eff_ggl = None
        self.mask_file_lens = None
        self.read_mask_lens = None
        self.alm_file_lens = None
        self.read_alm_lens = None
        self.survey_area_lens = None
        self.n_eff_lens = None
        self.sigma_eps = None
        self.shot_noise_gamma = None
        self.alm_file_clust_lens = None
        self.read_mask_clust_lens = None
        self.read_alm_clust_lens = None
        self.alm_file_clust_ggl = None
        self.read_mask_clust_ggl = None
        self.read_alm_clust_ggl = None
        self.alm_file_lens_ggl = None
        self.read_mask_lens_ggl = None
        self.read_alm_lens_ggl = None

        # mass range and other quantities for the halo model as defined
        # by the hmf class by Steven Murray
        self.hm_prec = dict()
        self.hm_prec_abr = dict()
        self.M_bins = None
        self.log10M_min = None
        self.log10M_max = None
        self.hmf_model = None
        self.mdef_model = None
        self.mdef_params = None
        self.disable_mass_conversion = None
        self.delta_c = None
        self.transfer_model = None

        # k-range for the power spectra
        self.powspec_prec = dict()
        self.powspec_prec_abr = dict()
        self.nl_model = None
        self.log10k_bins = None
        self.log10k_min = None
        self.log10k_max = None
        self.small_k_damping = None
        self.HMCode_logT_AGN = None
        self.HMCode_A_baryon = None
        self.HMCode_eta_baryon = None

        # k-range and precision values for the trispectra
        self.trispec_prec = dict()
        self.trispec_prec_abr = dict()
        self.matter_klim = None
        self.matter_mulim = None
        self.tri_logk_bins = None
        self.tri_logk_min = None
        self.tri_logk_max = None
        self.tri_small_k_damping = None
        self.lower_calc_limit = None

        # misc
        self.misc = dict()
        self.num_cores = None

    def __read_in_cov_dict(self,
                           config,
                           config_name):
        """
        Reads in which parts of the covariance should be calculated.
        Every value that is not specified either raises an exception or
        gets a fall-back value which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covariance terms' in config:
            if 'gauss' in config['covariance terms']:
                self.gauss = config['covariance terms'].getboolean('gauss')
            else:
                self.gauss = True
                print("The Gaussian covariance will be calculated.")

            if 'split_gauss' in config['covariance terms']:
                self.split_gauss = \
                    config['covariance terms'].getboolean('split_gauss')
            else:
                self.split_gauss = False

            if 'nongauss' in config['covariance terms']:
                self.nongauss = \
                    config['covariance terms'].getboolean('nongauss')
            else:
                self.nongauss = True
                print("The non-Gaussian covariance will be calculated.")

            if 'ssc' in config['covariance terms']:
                self.ssc = config['covariance terms'].getboolean('ssc')
            else:
                self.ssc = True
                print("The super-sample covariance will be calculated.")
        else:
            self.gauss = True
            print("The Gaussian covariance will be calculated.")
            self.split_gauss = False
            self.nongauss = True
            print("The non-Gaussian covariance will be calculated.")
            self.ssc = True
            print("The super-sample covariance will be calculated.")

        if not self.gauss and not self.nongauss and not self.ssc:
            raise Exception("ConfigError: All covariance terms are ignored. " +
                            "Must be adjusted in file " + config_name + ", [covariance " +
                            "terms]: 'gauss' /'nongauss' /'ssc'.")

        if not self.gauss and self.split_gauss:
            self.gauss = True

        return True

    def __read_in_obs_dict(self,
                           config,
                           config_name):
        """
        Reads in for which observables the covariance should be 
        calculated. Every value that is not specified either raises an 
        exception or gets a fall-back value which is reported to the 
        user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'observables' in config:
            if 'cosmic_shear' in config['observables']:
                self.cosmicshear = \
                    config['observables'].getboolean('cosmic_shear')
            else:
                self.cosmicshear = False
            if 'clustering_z' in config['observables']:
                self.clustering_z = \
                    config['observables'].getboolean('clustering_z')
            else:
                self.clustering_z = False
            if 'unbiased_clustering' in config['observables']:
                self.unbiased_clustering = config['observables'].getboolean('unbiased_clustering')
            else:
                self.unbiased_clustering = False
            if 'est_shear' in config['observables']:
                self.est_shear = config['observables']['est_shear']
            if 'ggl' in config['observables']:
                self.ggl = config['observables'].getboolean('ggl')
            else:
                self.ggl = False
            if 'est_ggl' in config['observables']:
                self.est_ggl = config['observables']['est_ggl']
            if 'clustering' in config['observables']:
                self.clustering = \
                    config['observables'].getboolean('clustering')
            else:
                self.clustering = False
            if 'est_clust' in config['observables']:
                self.est_clust = config['observables']['est_clust']
            if 'cross_terms' in config['observables']:
                self.cross_terms = \
                    config['observables'].getboolean('cross_terms')
            else:
                self.cross_terms = True
            if 'cstellar_mf' in config['observables']:
                self.cstellar_mf = config['observables'].getboolean('cstellar_mf')
            else:
                self.cstellar_mf = False
        else:
            raise Exception("ConfigError: The section [observables] is " +
                            "missing in config file " + config_name + ". Compulsory " +
                            "inputs are either 'cosmic_shear', 'ggl', or 'clustering' " +
                            "and at least one specified estimator.")
        if 'csmf settings' in config and self.cstellar_mf:
            if 'csmf_log10M_bins' in config['csmf settings']:
                self.csmf_log10M_bins = np.array(config['csmf settings']['csmf_log10M_bins'].split(',')).astype(float)
            if 'csmf_diagonal' in config['csmf settings']:
                self.csmf_diagonal = config['csmf settings'].getboolean('csmf_diagonal')
            else:
                self.csmf_diagonal = False
            if not isinstance(self.csmf_log10M_bins, np.ndarray):
                if 'csmf_log10Mmin' in config['csmf settings']:
                    self.csmf_log10Mmin = float(config['csmf settings']['csmf_log10Mmin'])
                if 'csmf_log10Mmax' in config['csmf settings']:
                    self.csmf_log10Mmax = float(config['csmf settings']['csmf_log10Mmax'])
                if 'csmf_N_log10M_bin' in config['csmf settings']:
                    self.csmf_N_log10M_bin = int(config['csmf settings']['csmf_N_log10M_bin'])
                if self.csmf_N_log10M_bin is not None and self.csmf_log10Mmax is not None and self.csmf_log10Mmin is not None:
                    self.csmf_log10M_bins = np.linspace(self.csmf_log10Mmin, self.csmf_log10Mmax, self.csmf_N_log10M_bin + 1)
            if self.csmf_log10M_bins is None:
                raise Exception("ConfigError: You requested the stellar mass function as an observable, "+
                                "but did not specify the mass bins. Please adjust "
                                + config_name + " in the [csmf settings] section ")
        else:
            self.cstellar_mf = False
        if not self.cosmicshear and \
           not self.ggl and \
           not self.clustering:
            raise Exception("ConfigError: No observables are specified. " +
                            "Must be adjusted in file " + config_name + ", " +
                            "[observables]: 'cosmic_shear' / 'ggl', / 'clustering'.")

        if self.cosmicshear and self.est_shear is None:
            self.est_shear = 'k_space'
            print("The estimator for cosmic shear is set to k_space.")
        if self.ggl and self.est_ggl is None:
            self.est_ggl = 'k_space'
            print("The estimator for galaxy-galaxy lensing is set to k_space.")
        if self.clustering and self.est_clust is None:
            self.est_clust = 'k_space'
            print("The estimator for clustering is set to k_space.")

        obs_str = ['C_ell', 'xi_pm', 'cosebi', 'k_space', 'bandpowers']
        if self.cosmicshear:
            if self.est_shear not in obs_str:
                raise Exception("ConfigError: The observable 'cosmic_shear' " +
                                "is supposed to be calculated but no valid estimator is " +
                                "specified. Must be adjusted in file " + config_name +
                                ", [observables]: 'est_shear = " +
                                ' / '.join(map(str, obs_str)) + ".")

        obs_str = ['C_ell', 'gamma_t', 'k_space', 'projected_real','bandpowers','cosebi']
        if self.ggl:
            if self.est_ggl not in obs_str:
                raise Exception("ConfigError: The observable 'galaxy-galaxy " +
                                "lensing' is supposed to be calculated but no valid " +
                                "estimator is specified. Must be adjusted in file " +
                                config_name + ", [observables]: 'est_ggl = " +
                                ' / '.join(map(str, obs_str)) + ".")

        obs_str = ['C_ell', 'w', 'k_space', 'projected_real','bandpowers','cosebi']
        if self.clustering:
            if self.est_clust not in obs_str:
                raise Exception("ConfigError: The observable 'clustering' " +
                                "is supposed to be calculated but no valid estimator is " +
                                "specified. Must be adjusted in file " + config_name +
                                ", [observables]: 'est_clust = " +
                                ' / '.join(map(str, obs_str)) + ".")
        if self.clustering and self.ggl:
            if self.est_clust != self.est_ggl:
                if not (self.est_clust == "w" and self.est_ggl == "gamma_t"):
                    raise Exception("ConfigError: The observable 'clustering' and 'ggl " +
                                "are supposed to be calculated but no valid estimator combination is " +
                                "specified. Must be adjusted in file " + config_name)
        if self.clustering and self.cosmicshear:
            if self.est_clust != self.est_shear:
                if not (self.est_clust == "w" and self.est_shear == "xi_pm"):
                    raise Exception("ConfigError: The observable 'clustering' and 'lensing " +
                                "are supposed to be calculated but no valid estimator combination is " +
                                "specified. Must be adjusted in file " + config_name)
        if self.ggl and self.cosmicshear:
            if self.est_ggl != self.est_shear:
                if not (self.est_ggl == "gamma_t" and self.est_shear == "xi_pm"):
                    raise Exception("ConfigError: The observable 'ggl' and 'lensing " +
                                "are supposed to be calculated but no valid estimator combination is " +
                                "specified. Must be adjusted in file " + config_name)
        
        
                

        return True

    def __read_in_covELLspace_settings(self,
                                       config,
                                       config_name):
        """
        Reads in further information needed to calculate the covariance 
        for the estimator 'C_ell'. Every value that is not specified 
        either raises an exception or gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covELLspace settings' in config:
            if self.cosmicshear or self.ggl:
                if 'mult_shear_bias' in config['covELLspace settings']:
                    self.multiplicative_shear_bias_uncertainty = np.array(config['covELLspace settings']['mult_shear_bias'].split(',')).astype(float)
                else:
                    self.multiplicative_shear_bias_uncertainty = np.array([0.0])
                    print("Multiplicative shear bias uncertainty is set to 0. Adjust in " + config_name + " with one sigma value per source bin ")
            if 'limber' in config['covELLspace settings']:
                self.limber = config['covELLspace settings'].getboolean('limber')
                if not self.limber:
                    print("Note that the full non-Limber calculation slows down the code significantly.")
            if 'pixelised_cell' in config['covELLspace settings']:
                self.pixelised_cell = config['covELLspace settings'].getboolean('pixelised_cell')
                if self.pixelised_cell:
                    self.pixel_Nside = int(config['covELLspace settings']['pixel_Nside'])
            if 'nglimber' in config['covELLspace settings']:
                self.nglimber = config['covELLspace settings'].getboolean('nglimber')
                if not self.nglimber:
                    print("Note that the full non-Limber calculation slows down the code significantly.")
            else:
                self.nglimber = True
            if 'ell_min' in config['covELLspace settings']:
                self.ell_min = float(
                    config['covELLspace settings']['ell_min'])
            if 'ell_max' in config['covELLspace settings']:
                self.ell_max = float(
                    config['covELLspace settings']['ell_max'])
                if self.pixelised_cell:
                    self.ell_max = 3*self.pixel_Nside - 1
            if 'ell_bins' in config['covELLspace settings']:
                self.ell_bins = int(
                    config['covELLspace settings']['ell_bins'])
            if 'ell_type' in config['covELLspace settings']:
                self.ell_type = config['covELLspace settings']['ell_type']
            if 'delta_z' in config['covELLspace settings']:
                self.delta_z = float(
                    config['covELLspace settings']['delta_z'])
            if 'integration_steps' in config['covELLspace settings']:
                self.integration_steps = int(
                    config['covELLspace settings']['integration_steps'])
            if 'tri_delta_z' in config['covELLspace settings']:
                self.tri_delta_z = float(
                    config['covELLspace settings']['tri_delta_z'])
            if 'nz_interpolation_polynom_order' in \
                    config['covELLspace settings']:
                self.nz_polyorder = int(config['covELLspace settings']
                                        ['nz_interpolation_polynom_order'])
                if self.nz_polyorder != 1:
                    print("InputWarning: Choosing a value for '[covELLspace " +
                          "settings]: nz_interpolation_polynom_order' which " +
                          "is not 1, is not recommended and can lead to " +
                          "non-positive values in the splined object.")
            if 'n_spec' in config['covELLspace settings']:
                self.n_spec = int(config['covELLspace settings']['n_spec'])
                if 'ell_spec_min' in config['covELLspace settings']:
                    self.ell_spec_min = float(
                        config['covELLspace settings']['ell_spec_min'])
                if 'ell_spec_max' in config['covELLspace settings']:
                    self.ell_spec_max = float(
                        config['covELLspace settings']['ell_spec_max'])
                if 'ell_spec_bins' in config['covELLspace settings']:
                    self.ell_spec_bins = int(
                        config['covELLspace settings']['ell_spec_bins'])
                if 'ell_spec_type' in config['covELLspace settings']:
                    self.ell_spec_type = config['covELLspace settings']['ell_spec_type']
                if 'ell_photo_min' in config['covELLspace settings']:
                    self.ell_photo_min = float(
                        config['covELLspace settings']['ell_photo_min'])
                if 'ell_photo_max' in config['covELLspace settings']:
                    self.ell_photo_max = float(
                        config['covELLspace settings']['ell_photo_max'])
                if 'ell_photo_bins' in config['covELLspace settings']:
                    self.ell_photo_bins = int(
                        config['covELLspace settings']['ell_photo_bins'])
                if 'ell_photo_type' in config['covELLspace settings']:
                    self.ell_photo_type = config['covELLspace settings']['ell_photo_type']
            if 'ell_min_clustering' in config['covELLspace settings']:
                self.ell_min_clustering = float(
                    config['covELLspace settings']['ell_min_clustering'])
            if 'ell_max_clustering' in config['covELLspace settings']:
                self.ell_max_clustering = float(
                    config['covELLspace settings']['ell_max_clustering'])
                if self.pixelised_cell:
                    self.ell_max_clustering = 3*self.pixel_Nside - 1
            if 'ell_bins_clustering' in config['covELLspace settings']:
                self.ell_bins_clustering = int(
                    config['covELLspace settings']['ell_bins_clustering'])
            if 'ell_type_clustering' in config['covELLspace settings']:
                self.ell_type_clustering = config['covELLspace settings']['ell_type_clustering']
            if 'ell_min_lensing' in config['covELLspace settings']:
                self.ell_min_lensing = float(
                    config['covELLspace settings']['ell_min_lensing'])
            if 'ell_max_lensing' in config['covELLspace settings']:
                self.ell_max_lensing = float(
                    config['covELLspace settings']['ell_max_lensing'])
                if self.pixelised_cell:
                    self.ell_max_lensing = 3*self.pixel_Nside - 1
            if 'ell_bins_lensing' in config['covELLspace settings']:
                self.ell_bins_lensing = int(
                    config['covELLspace settings']['ell_bins_lensing'])
            if 'ell_type_lensing' in config['covELLspace settings']:
                self.ell_type_lensing = config['covELLspace settings']['ell_type_lensing']
        else:
            ...
        #if self.pixelised_cell is None:
            #self.pixelised_cell = False
            
        if (self.cosmicshear and self.est_shear == 'C_ell') or \
           (self.ggl and self.est_ggl == 'C_ell') or \
           (self.clustering and self.est_clust == 'C_ell'):
            if self.limber is None:
                self.limber = True
            if self.nglimber is None:
                self.nglimber = True
            
            else:
                if self.pixel_Nside is None and self.pixelised_cell:
                    raise Exception("ConfigError: C_ells are required to be pixelised " +
                                "but 'pixelised_cell = True', however Nside is not set in " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", specifiy e.g. [covELLspace settings]: 'pixel_Nside = " +
                                "2048")

            if self.cosmicshear:
                if self.ell_min_lensing is not None:
                    self.ell_min = self.ell_min_lensing
                if self.ell_max_lensing is not None:
                    self.ell_max = self.ell_max_lensing
            if self.clustering or self.ggl:
                if self.ell_min_clustering is not None:
                    if self.ell_min is not None:
                        self.ell_min = min(self.ell_min_clustering, self.ell_min)
                    else:
                        self.ell_min = self.ell_min_clustering
                if self.ell_max_clustering is not None:
                    if self.ell_min is not None:
                        self.ell_max = max(self.ell_max_clustering, self.ell_max)
                    else:
                        self.ell_max = self.ell_max_clustering

            if self.ell_min is None:
                self.ell_min = 2
            if self.ell_max is None:
                self.ell_max = int(1e4)
            if self.ell_bins is None:
                self.ell_bins = 100
            if self.ell_type is None:
                self.ell_type = 'log'

                    

            elif self.ell_type != 'lin' and self.ell_type != 'log':
                raise Exception("ConfigError: The binning type for ell bins " +
                                "[covELLspace settings]: 'ell_type = " +
                                config['covELLspace settings']['ell_type'] + "' is not " +
                                "recognised. Must be either 'lin' or 'log'.")
            if self.n_spec is not None and self.n_spec != 0:
                if self.ell_spec_min is None:
                    raise Exception("ConfigError: An estimator is " +
                                "'C_ell' but no minimum ell for the spectroscopic projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covELLspace settings]: 'ell_spec_min = 100'.")
                if self.ell_spec_max is None:
                    raise Exception("ConfigError: An estimator is " +
                                "'C_ell' but no maximum ell for the spectroscopic projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covELLspace settings]: 'ell_spec_max = 2000'.")
                if self.ell_spec_bins is None:
                    raise Exception("ConfigError: An estimator is " +
                                    "'C_ell' but no number of spectroscopic ell bins is specified. Must " +
                                    "be adjusted in config file " + config_name + ", " +
                                    "[covELLspace settings]: 'ell_spec_bins = 10'.")
                if self.ell_spec_type is None:
                    self.ell_spec_type = 'log'
                    print("The binning type for spectroscopic ell bins " +
                        "[covELLspace settings]: 'ell_spec_type' is set to 'log'.")
                if self.ell_photo_min is None:
                    raise Exception("ConfigError: An estimator is " +
                                "'C_ell' but no minimum ell for the photometric projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covELLspace settings]: 'ell_photo_min = 100'.")
                if self.ell_photo_max is None:
                    raise Exception("ConfigError: An estimator is " +
                                "'C_ell' but no maximum ell for the photometric projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covELLspace settings]: 'ell_photo_max = 2000'.")
                if self.ell_photo_bins is None:
                    raise Exception("ConfigError: An estimator is " +
                                    "'C_ell' but no number of photometric ell bins is specified. Must " +
                                    "be adjusted in config file " + config_name + ", " +
                                    "[covELLspace settings]: 'ell_photo_bins = 10'.")
                if self.ell_photo_type is None:
                    self.ell_photo_type = 'log'
                    print("The binning type for photometric ell bins " +
                        "[covELLspace settings]: 'ell_photo_type' is set to 'log'.")
                if self.ell_spec_min < self.ell_min:
                    self.ell_min = self.ell_spec_min
                if self.ell_photo_min < self.ell_min:
                    self.ell_min = self.ell_photo_min
                if self.ell_photo_max > self.ell_max:
                    self.ell_max = self.ell_spec_max
                if self.ell_photo_max > self.ell_max:
                    self.ell_max = self.ell_photo_max

            if self.delta_z is None:
                self.delta_z = 0.02
                print("The redshift spacing for the C_ell covariance l.o.s. " +
                      "integration [covELLspace settings]: 'delta_z' is set " +
                      "to '0.02'.")
            if self.integration_steps is None:
                self.integration_steps = 1000
                print("The integration steps for the C_ell covariance " +
                      "l.o.s. integration [covELLspace settings]: " +
                      "'integration_steps' is set to '1000'.")
            if self.tri_delta_z is None and self.nongauss:
                self.tri_delta_z = 0.5
                try:
                    config['tabulated inputs files']['trispec_file']
                except KeyError:
                    print("The redshift spacing for calculating the " +
                          "trispectra is set to [covELLspace settings]: " +
                          "'tri_delta_z = 0.5'.")
            if self.nz_polyorder is None:
                self.nz_polyorder = 1
                print("The redshift distribution is linearly interpolated " +
                      "between bins [covELLspace settings]: " +
                      "'nz_interpolation_polynom_order = 1' (put 0 for a " +
                      "histogram interpretation)")
        if self.ell_min is None:
            self.ell_min = 2
        if self.ell_max is None:
            self.ell_max = int(1e4)
        if self.ell_bins is None:
            self.ell_bins = 100
        if self.ell_type is None:
            self.ell_type = 'log'
        if self.cosmicshear and self.est_shear == 'C_ell' and (self.ell_min_lensing is not None and self.ell_bins_lensing is not None and self.ell_max_lensing is not None and self.ell_type_lensing is not None):
            self.ell_min = self.ell_min_lensing
            self.ell_max = self.ell_max_lensing
            self.ell_bins = 100
            self.ell_type = 'log'
        if ((self.ggl and self.est_ggl == 'C_ell') or (self.clustering and self.est_clust == 'C_ell')) and (self.ell_min_lensing is not None and self.ell_bins_lensing is not None and self.ell_max_lensing is not None and self.ell_type_lensing is not None):
            if self.ell_min > self.ell_min_clustering:
                self.ell_min = self.ell_min_clustering
            if self.ell_max < self.ell_max_clustering:
                self.ell_max = self.ell_max_clustering
            self.ell_bins = 100
            self.ell_type = 'log'
        
        return True

    def __read_in_covTHETAspace_settings(self,
                                         config,
                                         config_name):
        """
        Reads in further information needed to calculate the covariance 
        for the estimator 'xi_pm'. Every value that is not specified 
        either raises an exception or gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covTHETAspace settings' in config:
            if 'theta_min' in config['covTHETAspace settings']:
                self.theta_min = float(
                    config['covTHETAspace settings']['theta_min'])
            if 'theta_max' in config['covTHETAspace settings']:
                self.theta_max = float(
                    config['covTHETAspace settings']['theta_max'])
            if 'theta_bins' in config['covTHETAspace settings']:
                self.theta_bins = int(
                    config['covTHETAspace settings']['theta_bins'])
            if 'theta_type' in config['covTHETAspace settings']:
                self.theta_type = \
                    config['covTHETAspace settings']['theta_type']
            
            if 'theta_min_clustering' in config['covTHETAspace settings']:
                self.theta_min_clustering = float(
                    config['covTHETAspace settings']['theta_min_clustering'])
            if 'theta_max_clustering' in config['covTHETAspace settings']:
                self.theta_max_clustering = float(
                    config['covTHETAspace settings']['theta_max_clustering'])
            if 'theta_bins_clustering' in config['covTHETAspace settings']:
                self.theta_bins_clustering = int(
                    config['covTHETAspace settings']['theta_bins_clustering'])
            if 'theta_type_clustering' in config['covTHETAspace settings']:
                self.theta_type_clustering = \
                    config['covTHETAspace settings']['theta_type_clustering']
            
            if 'theta_min_lensing' in config['covTHETAspace settings']:
                self.theta_min_lensing = float(
                    config['covTHETAspace settings']['theta_min_lensing'])
            if 'theta_max_lensing' in config['covTHETAspace settings']:
                self.theta_max_lensing = float(
                    config['covTHETAspace settings']['theta_max_lensing'])
            if 'theta_bins_lensing' in config['covTHETAspace settings']:
                self.theta_bins_lensing = int(
                    config['covTHETAspace settings']['theta_bins_lensing'])
            if 'theta_type_lensing' in config['covTHETAspace settings']:
                self.theta_type_lensing = \
                    config['covTHETAspace settings']['theta_type_lensing']

            if 'theta_list' in config['covTHETAspace settings'] and self.theta_type == 'list':
                self.theta_list = np.array(
                    config['covTHETAspace settings']['theta_list'].split(','))
            if 'xi_pp' in config['covTHETAspace settings']:
                self.xi_pp = \
                    config['covTHETAspace settings'].getboolean('xi_pp')
            else:
                self.xi_pp = True
            if 'xi_mm' in config['covTHETAspace settings']:
                self.xi_mm = \
                    config['covTHETAspace settings'].getboolean('xi_mm')
            else:
                self.xi_mm = True
            if 'theta_accuracy' in config['covTHETAspace settings']:
                self.theta_acc = float(config['covTHETAspace settings']['theta_accuracy'])
            if 'integration_intervals' in config['covTHETAspace settings']:
                self.integration_intervals = int(config['covTHETAspace settings']['integration_intervals'])
                
            if 'mix_term_do_mix_for' in config['covTHETAspace settings']:
                self.mix_term_do_mix_for = config['covTHETAspace settings']['mix_term_do_mix_for'].split(',')
                if self.mix_term_do_mix_for[0] == '':
                    self.mix_term_do_mix_for = None
            if self.mix_term_do_mix_for:
                if 'mix_term_file_path_load_triplets' in config['covTHETAspace settings']:
                    self.mix_term_file_path_load_triplets = config['covTHETAspace settings']['mix_term_file_path_load_triplets']
                    if self.mix_term_file_path_load_triplets == '':
                        self.mix_term_file_path_load_triplets = None
                else:
                    self.mix_term_file_path_load_triplets = None
            if self.mix_term_do_mix_for:
                if 'mix_term_file_path_catalog' in config['covTHETAspace settings']:
                    self.mix_term_file_path_catalog = config['covTHETAspace settings']['mix_term_file_path_catalog']
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the path to the catalogue. Please update [mix_term_file_path_catalog] in your config")
                if 'mix_term_col_name_weight' in config['covTHETAspace settings']:
                    self.mix_term_col_name_weight = config['covTHETAspace settings']['mix_term_col_name_weight']
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not the column name of the weight. Please update [mix_term_col_name_weight] in your config")
                if 'mix_term_col_name_pos1' in config['covTHETAspace settings']:
                    self.mix_term_col_name_pos1 = config['covTHETAspace settings']['mix_term_col_name_pos1']
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the column name of the first position. Please update [mix_term_col_name_pos1] in your config")
                if 'mix_term_col_name_pos2' in config['covTHETAspace settings']:
                    self.mix_term_col_name_pos2 = config['covTHETAspace settings']['mix_term_col_name_pos2']
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the column name of the second position. Please update [mix_term_col_name_pos2] in your config")
                if 'mix_term_col_name_zbin' in config['covTHETAspace settings']:
                    self.mix_term_col_name_zbin = config['covTHETAspace settings']['mix_term_col_name_zbin']
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the column name of the zbin. Please update [mix_term_col_name_zbin] in your config")
                if 'mix_term_isspherical' in config['covTHETAspace settings']:
                    self.mix_term_isspherical = config['covTHETAspace settings'].getboolean('mix_term_isspherical')
                else:
                    print("ConfigWarning: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify mix_term_isspherical. mix_term_isspherical is set to false")
                    self.mix_term_isspherical = False
                if 'mix_term_target_patchsize' in config['covTHETAspace settings']:
                    self.mix_term_target_patchsize = float(config['covTHETAspace settings']['mix_term_target_patchsize'])
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify patchsize. Please update [mix_term_target_patchsize] in your config")
                if 'mix_term_do_overlap' in config['covTHETAspace settings']:
                    self.mix_term_do_overlap = config['covTHETAspace settings'].getboolean('mix_term_do_overlap')
                else:
                    print("ConfigWarning: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specif whether the overlap should be considered. mix_term_do_overlap is set to false")
                    self.mix_term_do_overlap = False
                if 'mix_term_nbins_phi' in config['covTHETAspace settings']:
                    self.mix_term_nbins_phi = int(config['covTHETAspace settings']['mix_term_nbins_phi'])
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the number of phi bins. Please update [mix_term_nbins_phi] in your config")
                if 'mix_term_nmax' in config['covTHETAspace settings']:
                    self.mix_term_nmax = int(config['covTHETAspace settings']['mix_term_nmax'])
                else:
                    raise Exception("ConfigError: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify the N_max for the mixterm. Please update [mix_term_nmax] in your config")
                if 'mix_term_do_ec' in config['covTHETAspace settings']:
                    self.mix_term_do_ec = config['covTHETAspace settings'].getboolean('mix_term_do_ec')
                else:
                    print("ConfigWarning: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify whether the edge correction should be done. mix_term_do_ec is set to false")
                    self.mix_term_do_ec = False
                    
                if 'mix_term_subsample' in config['covTHETAspace settings']:
                    self.mix_term_subsample = int(config['covTHETAspace settings']['mix_term_subsample'])
                else:
                    print("ConfigWarning: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify how much the original catalog should be downsampled. mix_term_subsample is set to 10")
                    self.mix_term_subsample = 10
                if 'mix_term_nsubr' in config['covTHETAspace settings']:
                    self.mix_term_nsubr = int(config['covTHETAspace settings']['mix_term_nsubr'])
                else:
                    print("ConfigWarning: You want to calculate the mixterm for", self.mix_term_do_mix_for, "but did not specify how many subintervals are required for the triplet calculation. mix_term_nsubr is set to 5")
                    self.mix_term_nsubr = 5
                if 'mix_term_file_path_save_triplets' in config['covTHETAspace settings']:
                    self.mix_term_file_path_save_triplets = config['covTHETAspace settings']['mix_term_file_path_save_triplets']
                else:
                    print("ConfigWarning: You want to save the triplets for the mixterm for", self.mix_term_do_mix_for, "but did not specify a path to save. Will not be saved")
                    self.mix_term_file_path_save_triplets = None
        else:
            if self.cosmicshear and self.est_shear == 'xi_pm':
                self.xi_pp = True
                self.xi_mm = True
        
        if self.limber is None:
            self.limber = True
        if self.nglimber is None:
            self.nglimber = True
            
        
        if (self.clustering and self.est_clust == 'w') or (self.ggl and self.est_ggl == 'gamma_t'):
            if self.theta_min is None:
                if self.theta_min_clustering is None:
                    raise Exception("ConfigError: The clustering is " +
                                    "'w' or 'gamma_t' but no minimum theta for the projection is " +
                                    "specified. Must be adjusted in config file " +
                                    config_name + ", [covTHETAspace settings]: 'theta_min = " +
                                    "10' [arcmin].")
                else:
                    self.theta_min = self.theta_min_clustering
            if self.theta_min_clustering is None:
                self.theta_min_clustering = self.theta_min
            if self.theta_max is None:
                if self.theta_max_clustering is None:
                    raise Exception("ConfigError: The clustering is " +
                                    "'w' or 'gamma_t' but no maximum theta for the projection is " +
                                    "specified. Must be adjusted in config file " +
                                    config_name + ", [covTHETAspace settings]: 'theta_max = " +
                                    "100'. [arcmax")
                else:
                    self.theta_max = self.theta_max_clustering
            if self.theta_max_clustering is None:
                self.theta_max_clustering = self.theta_max      
            if self.theta_bins is None:
                if self.theta_bins_clustering is None:
                    raise Exception("ConfigError: The clustering is " +
                                    "'w' or 'gamma_t' but no number of theta bins is specified. Must " +
                                    "be adjusted in config file " + config_name + ", " +
                                    "[covTHETAspace settings]: 'theta_bins = 10'.")
                else:
                    self.theta_bins = self.theta_bins_clustering
            if self.theta_bins_clustering is None:
                self.theta_bins_clustering = self.theta_bins
            if self.theta_type is None:
                self.theta_type = 'log'
                print("The binning type for theta bins " +
                      "[covTHETAspace settings]: 'theta_type' is set to " +
                      "'log'.")
            if self.theta_type_clustering is None:
                self.theta_type_clustering = self.theta_type
                
            if self.theta_min is None:
                if self.cosmicshear:
                    self.theta_min = self.theta_min_lensing   
                if self.clustering or self.ggl:
                    self.theta_min = self.theta_min_clustering
                if self.cosmicshear and (self.clustering  or self.ggl):  
                    self.theta_min = min(self.theta_min_clustering, self.theta_min_lensing)
            
            if self.theta_max is None:
                if self.cosmicshear:
                    self.theta_max = self.theta_max_lensing   
                if self.clustering or self.ggl:
                    self.theta_max = self.theta_max_clustering
                if self.cosmicshear and (self.clustering  or self.ggl):  
                    self.theta_max = max(self.theta_max_clustering, self.theta_max_lensing)
            
            if self.theta_type is None:
                if self.cosmicshear:
                    self.theta_type = self.theta_type_lensing   
                if self.clustering or self.ggl:
                    self.theta_type = self.theta_type_clustering
            
            if self.theta_bins is None:
                if self.cosmicshear:
                    self.theta_bins = self.theta_bins_lensing   
                if self.clustering or self.ggl:
                    self.theta_bins = self.theta_bins_clustering
                if self.cosmicshear and (self.clustering  or self.ggl):  
                    self.theta_bins = max(self.theta_bins_clustering, self.theta_bins_lensing)
        if self.cosmicshear and self.est_shear == 'xi_pm':
            if self.theta_min is None:
                if self.theta_min_lensing is None:
                    raise Exception("ConfigError: The cosmic shear estimator is " +
                                    "'xi_pm' but no minimum theta for the projection is " +
                                    "specified. Must be adjusted in config file " +
                                    config_name + ", [covTHETAspace settings]: 'theta_min = " +
                                    "10' [arcmin].")
                else:
                    self.theta_min = self.theta_min_lensing
            if self.theta_min_lensing is None:
                self.theta_min_lensing = self.theta_min
            if self.theta_max is None:
                if self.theta_max_lensing is None:
                    raise Exception("ConfigError: The cosmic shear estimator is " +
                                    "'xi_pm' but no maximum theta for the projection is " +
                                    "specified. Must be adjusted in config file " +
                                    config_name + ", [covTHETAspace settings]: 'theta_max = " +
                                    "100'. [arcmax")
                else:
                    self.theta_max = self.theta_max_lensing
            if self.theta_max_lensing is None:
                self.theta_max_lensing = self.theta_max      
            if self.theta_bins is None:
                if self.theta_bins_lensing is None:
                    raise Exception("ConfigError: The cosmic shear estimator is " +
                                    "'xi_pm' but no number of theta bins is specified. Must " +
                                    "be adjusted in config file " + config_name + ", " +
                                    "[covTHETAspace settings]: 'theta_bins = 10'.")
                else:
                    self.theta_bins = self.theta_bins_lensing
            if self.theta_bins_lensing is None:
                self.theta_bins_lensing = self.theta_bins
            if self.theta_type is None:
                self.theta_type = 'log'
            if self.theta_type_lensing is None:
                self.theta_type_lensing = self.theta_type
            elif self.theta_type != 'lin' and self.theta_type != 'log':
                if self.theta_type_lensing_p != 'lin' and self.theta_type_lensing_p != 'log' and self.theta_type_lensing_m != 'lin' and self.theta_type_lensing_m != 'log':
                    raise Exception("ConfigError: The binning type for theta " +
                                    "bins [covTHETAspace settings]: 'theta_type = " +
                                    config['covTHETAspace settings']['theta_type'] + "' is " +
                                    "not recognised. Must be either 'lin' or 'log'.")
            if self.theta_acc is None:
                self.theta_acc = 1e-5
                print("The accuracy for the theta space covariance " +
                      "[covTHETAspace settings]: 'theta_accuracy' is set to  " +
                      str(self.theta_acc))
            if self.integration_intervals is None:
                self.integration_intervals = 40
                print("The number of integration intervals for the theta space covariance " +
                      "[covTHETAspace settings]: 'integration_intervals' is set to  " +
                      str(self.integration_intervals))
            
            if self.limber is None:
                self.limber = True
            if self.nglimber is None:
                self.nglimber = True
            if self.ell_min is None:
                self.ell_min = 2
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_min' is not specified. It is set to '2'.")
            if self.multiplicative_shear_bias_uncertainty is None:
                self.multiplicative_shear_bias_uncertainty = np.array([0.0])
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "mult_shear_bias' is not specified. It is set to '0'.")
            if self.ell_max is None:
                self.ell_max = 1e5
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '1e5'.")
            if self.ell_bins is None:
                self.ell_bins = int(np.log10(self.ell_max)*30)
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '" +
                      str(self.ell_bins) + "'.")
            if self.ell_type is None:
                self.ell_type = 'log'
            if self.delta_z is None:
                self.delta_z = 0.02
            if self.integration_steps is None:
                self.integration_steps = 1000
            if self.nz_polyorder is None:
                self.nz_polyorder = 1
            if self.tri_delta_z is None:
                self.tri_delta_z = 0.5

            if not self.xi_pp and not self.xi_mm:
                raise Exception("ConfigError: The shear estimator type is " +
                                "set to [observables]: 'est_shear = xi_pm', but none of " +
                                "the covariances for correlation functions ought to be " +
                                "calculated. Must be adjusted in [covTHETAspace " +
                                "settings]: 'xi_pp/mm = True' to go on.")

        if self.xi_pp and self.xi_mm and self.cross_terms:
            self.xi_pm = True
        else:
            self.xi_pm = False
        
        if self.mix_term_do_mix_for and not self.mix_term_file_path_load_triplets:
            if not self.mix_term_file_path_catalog:
                raise Exception("ConfigError: The mix term is supposed to be calculated " + 
                                "for " + self.mix_term_do_mix_for + ", however, the path to the " +
                                "catalog is not specified. Please specify " +
                                "'mix_term_file_path_catalog = path' in " + config_name)

        return True

    def __read_in_covCOSEBI_settings(self,
                                     config,
                                     config_name):
        """
        Reads in further information needed to calculate the covariance 
        for the estimator 'cosebi'. Every value that is not specified 
        either raises an exception or gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covCOSEBI settings' in config:
            if 'En_modes' in config['covCOSEBI settings']:
                self.En_modes = int(config['covCOSEBI settings']['En_modes'])
            if 'theta_min' in config['covCOSEBI settings']:
                self.theta_min_cosebi = \
                    float(config['covCOSEBI settings']['theta_min'])
            if 'theta_max' in config['covCOSEBI settings']:
                self.theta_max_cosebi = \
                    float(config['covCOSEBI settings']['theta_max'])
            
            if 'En_modes_clustering' in config['covCOSEBI settings']:
                self.En_modes_clustering = int(config['covCOSEBI settings']['En_modes_clustering'])
            if 'theta_min_clustering' in config['covCOSEBI settings']:
                self.theta_min_cosebi_clustering = \
                    float(config['covCOSEBI settings']['theta_min_clustering'])
            if 'theta_max_clustering' in config['covCOSEBI settings']:
                self.theta_max_cosebi_clustering = \
                    float(config['covCOSEBI settings']['theta_max_clustering'])
            
            if 'En_modes_lensing' in config['covCOSEBI settings']:
                self.En_modes_lensing = int(config['covCOSEBI settings']['En_modes_lensing'])
            if 'theta_min_lensing' in config['covCOSEBI settings']:
                self.theta_min_cosebi_lensing = \
                    float(config['covCOSEBI settings']['theta_min_lensing'])
            if 'theta_max_lensing' in config['covCOSEBI settings']:
                self.theta_max_cosebi_lensing = \
                    float(config['covCOSEBI settings']['theta_max_lensing'])
            
            if 'En_accuracy' in config['covCOSEBI settings']:
                self.En_acc = \
                    float(config['covCOSEBI settings']['En_accuracy'])
            else:
                self.En_acc = 1e-4
                print("The precision for the En calculation is not " +
                      "specified in '[covCOSEBI settings]: 'En_accuracy'. It " +
                      "is set to '1e-4'.")
            if 'Wn_style' in config['covCOSEBI settings']:
                self.Wn_style = \
                    config['covCOSEBI settings']['Wn_style']
            else:
                self.Wn_style = 'log'
            if 'Wn_accuracy' in config['covCOSEBI settings']:
                self.Wn_acc = \
                    float(config['covCOSEBI settings']['Wn_accuracy'])
            else:
                self.Wn_acc = 1e-6
            if 'dimensionless_cosebi' in config['covCOSEBI settings']:
                self.dimensionless_cosebi = \
                    config['covCOSEBI settings'].getboolean('dimensionless_cosebi')
        else:
            if self.cosmicshear and self.est_shear == 'cosebi':
                self.En_acc = 1e-4
                print("The precision for the En calculation is not " +
                      "specified in '[covCOSEBI settings]: 'En_accuracy'. It " +
                      "is set to '1e-4'.")
                self.Wn_style = 'log'
                print("The COSEBIs will be based on logarithmic kernel " +
                      "functions. Can be specified in " +
                      "'[covCOSEBI settings]: 'Wn_style'. It is set to 'log'.")
                self.Wn_acc = 1e-6

        if self.cosmicshear and self.est_shear == 'cosebi' and not self.do_arbitrary_obs:
            if self.En_modes is None:
                raise Exception("ConfigError: The cosmic shear estimator is " +
                                "'cosebi' but no number of E modes is specified. Must " +
                                "be adjusted in config file " + config_name + ", " +
                                "[covCOSEBI settings]: 'En_modes = 5'.")
            elif self.En_modes < 1:
                raise Exception("ConfigError: The cosmic shear estimator is " +
                                "'cosebi' but the number of E modes is less than 1. " +
                                "Must be adjusted in config file " + config_name + ", " +
                                "[covCOSEBI settings]: 'En_modes'.")

            if self.theta_min_cosebi is None:
                raise Exception("ConfigError: The cosmic shear estimator is " +
                                "'cosebi' but no minimum theta for the projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covCOSEBI settings]: 'theta_min = 1'. " +
                                "[arcmin")

            if self.theta_max_cosebi is None:
                raise Exception("ConfigError: The cosmic shear estimator is " +
                                "'cosebi' but no maximum theta for the projection is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covCOSEBI settings]: 'theta_max = " +
                                "100'. [arcmin")

            elif self.Wn_style != 'lin' and self.Wn_style != 'log':
                raise Exception("ConfigError: The kernel function type " +
                                "[covCOESBI settings]: 'Wn_style = " +
                                config['covCOSEBI settings']['Wn_style'] + "' is " +
                                "not recognised. Must be either 'lin' or 'log'.")
            
            if self.dimensionless_cosebi is None:
                self.dimensionless_cosebi = False

            if self.limber is None:
                self.limber = True

            if self.nglimber is None:
                self.nglimber = True

            if self.ell_min is None:
                self.ell_min = 2
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_min' is not specified. It is set to '2'.")
            if self.multiplicative_shear_bias_uncertainty is None:
                self.multiplicative_shear_bias_uncertainty = 0
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "mult_shear_bias' is not specified. It is set to '0'.")
            if self.ell_max is None:
                self.ell_max = 1e5
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '1e5'.")
            if self.ell_bins is None:
                self.ell_bins = int(np.log10(self.ell_max)*10)
                print("The shear-shear correlation functions are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '" +
                      str(self.ell_bins) + "'.")
            if self.ell_type is None:
                self.ell_type = 'log'
            if self.delta_z is None:
                self.delta_z = 0.02
            if self.integration_steps is None:
                self.integration_steps = 500
            if self.nz_polyorder is None:
                self.nz_polyorder = 1
            if self.tri_delta_z is None:
                self.tri_delta_z = 0.5
        if self.En_modes_clustering is None:
            self.En_modes_clustering = self.En_modes
        if self.theta_min_cosebi_clustering is None:
            self.theta_min_cosebi_clustering = self.theta_min_cosebi
        if self.theta_max_cosebi_clustering is None:
            self.theta_max_cosebi_clustering = self.theta_max_cosebi

        if self.En_modes_lensing is None:
            self.En_modes_lensing = self.En_modes
        if self.theta_min_cosebi_lensing is None:
            self.theta_min_cosebi_lensing = self.theta_min_cosebi
        if self.theta_max_cosebi_lensing is None:
            self.theta_max_cosebi_lensing = self.theta_max_cosebi
        return True
    
    def __read_in_covbandpowers_settings(self,
                                         config,
                                         config_name):
        """
        Reads in further information needed to calculate the covariance 
        for the estimator 'bandpowers'. Every value that is not specified 
        either raises an exception or gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covbandpowers settings' in config:
            if 'apodisation_log_width_clustering' in config['covbandpowers settings']:
                self.apodisation_log_width_clustering = float(config['covbandpowers settings']['apodisation_log_width_clustering'])
            if 'theta_lo_clustering' in config['covbandpowers settings']:
                self.theta_lo_clustering = float(config['covbandpowers settings']['theta_lo_clustering'])
            if 'theta_up_clustering' in config['covbandpowers settings']:
                self.theta_up_clustering = \
                    float(config['covbandpowers settings']['theta_up_clustering'])
            if 'ell_min_clustering' in config['covbandpowers settings']:
                self.bp_ell_min_clustering = \
                    float(config['covbandpowers settings']['ell_min_clustering'])
            if 'ell_max_clustering' in config['covbandpowers settings']:
                self.bp_ell_max_clustering = \
                    float(config['covbandpowers settings']['ell_max_clustering'])
            if 'ell_bins_clustering' in config['covbandpowers settings']:
                self.bp_ell_bins_clustering = \
                    int(config['covbandpowers settings']['ell_bins_clustering'])  
            if 'ell_type_clustering' in config['covbandpowers settings']:
                self.bp_ell_type_clustering = \
                    str(config['covbandpowers settings']['ell_type_clustering'])

            if 'apodisation_log_width_lensing' in config['covbandpowers settings']:
                self.apodisation_log_width_lensing = float(config['covbandpowers settings']['apodisation_log_width_lensing'])
            if 'theta_lo_lensing' in config['covbandpowers settings']:
                self.theta_lo_lensing = float(config['covbandpowers settings']['theta_lo_lensing'])
            if 'theta_up_lensing' in config['covbandpowers settings']:
                self.theta_up_lensing = \
                    float(config['covbandpowers settings']['theta_up_lensing'])
            if 'ell_min_lensing' in config['covbandpowers settings']:
                self.bp_ell_min_lensing = \
                    float(config['covbandpowers settings']['ell_min_lensing'])
            if 'ell_max_lensing' in config['covbandpowers settings']:
                self.bp_ell_max_lensing = \
                    float(config['covbandpowers settings']['ell_max_lensing'])
            if 'ell_bins_lensing' in config['covbandpowers settings']:
                self.bp_ell_bins_lensing = \
                    int(config['covbandpowers settings']['ell_bins_lensing'])  
            if 'ell_type_lensing' in config['covbandpowers settings']:
                self.bp_ell_type_lensing = \
                    str(config['covbandpowers settings']['ell_type_lensing'])    
            
            if 'theta_binning' in config['covbandpowers settings']:
                self.theta_binning = \
                    int(config['covbandpowers settings']['theta_binning'])
            if 'bandpower_accuracy' in config['covbandpowers settings']:
                self.bandpower_accuracy = float(config['covbandpowers settings']['bandpower_accuracy'])


        if self.est_shear == 'bandpowers' or self.est_ggl == 'bandpowers' or self.est_clust == self.est_shear == 'bandpowers' and not self.do_arbitrary_obs:
            if self.clustering or self.ggl:
                if self.apodisation_log_width_clustering is None:
                    if 'apodisation_log_width' in config['covbandpowers settings']:
                        self.apodisation_log_width_clustering = float(config['covbandpowers settings']['apodisation_log_width'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers' " +
                                        "but no apodisation_log_width_clustering is specified. Must " +
                                        "be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'apodisation_log_width_clustering = ...'.")
                if self.theta_lo_clustering is None:
                    if 'theta_lo' in config['covbandpowers settings']:
                        self.theta_lo_clustering = float(config['covbandpowers settings']['theta_lo'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers" +
                                        "but the lower limit of the apodisation for clustering is not set." +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'theta_lo_clustering = ...'.")

                if self.theta_up_clustering is None:
                    if 'theta_up' in config['covbandpowers settings']:
                        self.theta_up_clustering = float(config['covbandpowers settings']['theta_up'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers" +
                                        "but the upper limit of the apodisation for clustering is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'theta_up_clustering = ...'.")
                if self.bp_ell_min_clustering is None:
                    if 'ell_min' in config['covbandpowers settings']:
                        self.bp_ell_min_clustering = float(config['covbandpowers settings']['ell_min'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers" +
                                        "but the lower multipole limit of the bandpowers for clustering is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_min_clustering = ...'.")
                
                if self.bp_ell_max_clustering is None:
                    if 'ell_max' in config['covbandpowers settings']:
                        self.bp_ell_max_clustering = float(config['covbandpowers settings']['ell_max'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers" +
                                        "but the upper multipole limit of the bandpowers for clustering is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_max_clustering = ...'.")

                if self.bp_ell_bins_clustering is None:
                    if 'ell_bins' in config['covbandpowers settings']:
                        self.bp_ell_bins_clustering = int(config['covbandpowers settings']['ell_bins'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers" +
                                        "but the number of multipoles for the bandpowers for clustering is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_bins_clustering = ...'.")
                
                if self.bp_ell_type_clustering is None:
                    if 'ell_type' in config['covbandpowers settings']:
                        self.bp_ell_type_clustering = str(config['covbandpowers settings']['ell_type'])
                    else:
                        raise Exception("ConfigError: A clustering estimator is set to 'bandpowers' " +
                                        "but the tpye of multipole spacing for the bandpowers for clustering is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_type_clustering = ...'.")
            
            if self.cosmicshear:
                if self.apodisation_log_width_lensing is None:
                    if 'apodisation_log_width' in config['covbandpowers settings']:
                        self.apodisation_log_width_lensing = float(config['covbandpowers settings']['apodisation_log_width'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers' " +
                                        "but no apodisation_log_width_lensing is specified. Must " +
                                        "be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'apodisation_log_width_lensing = ...'.")
                if self.theta_lo_lensing is None:
                    if 'theta_lo' in config['covbandpowers settings']:
                        self.theta_lo_lensing = float(config['covbandpowers settings']['theta_lo'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers" +
                                        "but the lower limit of the apodisation for lensing is not set." +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'theta_lo_lensing = ...'.")

                if self.theta_up_lensing is None:
                    if 'theta_up' in config['covbandpowers settings']:
                        self.theta_up_lensing = float(config['covbandpowers settings']['theta_up'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers" +
                                        "but the upper limit of the apodisation for lensing is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'theta_up_lensing = ...'.")
                if self.bp_ell_min_lensing is None:
                    if 'ell_min' in config['covbandpowers settings']:
                        self.bp_ell_min_lensing = float(config['covbandpowers settings']['ell_min'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers" +
                                        "but the lower multipole limit of the bandpowers for lensing is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_min_lensing = ...'.")
                
                if self.bp_ell_max_lensing is None:
                    if 'ell_max' in config['covbandpowers settings']:
                        self.bp_ell_max_lensing = float(config['covbandpowers settings']['ell_max'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers" +
                                        "but the upper multipole limit of the bandpowers for lensing is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_max_lensing = ...'.")

                if self.bp_ell_bins_lensing is None:
                    if 'ell_bins' in config['covbandpowers settings']:
                        self.bp_ell_bins_lensing = int(config['covbandpowers settings']['ell_bins'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers" +
                                        "but the number of multipoles for the bandpowers for lensing is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_bins_lensing = ...'.")
                
                if self.bp_ell_type_lensing is None:
                    if 'ell_type' in config['covbandpowers settings']:
                        self.bp_ell_type_lensing = str(config['covbandpowers settings']['ell_type'])
                    else:
                        raise Exception("ConfigError: A lensing estimator is set to 'bandpowers' " +
                                        "but the tpye of multipole spacing for the bandpowers for lensing is not set. " +
                                        "Must be adjusted in config file " + config_name + ", " +
                                        "[covbandpowers settings]: 'ell_type_lensing = ...'.")
            
            if self.theta_binning is None:
                self.theta_binning = 300
                print("ConfigWarning: The bandpower theta_binning has not been specified. Using fallback value of", self.theta_binning)
            if self.bandpower_accuracy is None:
                self.bandpower_accuracy = 1e-7
                print("ConfigWarning: The bandpower accuracy has not been specified. Using fallback value of", self.bandpower_accuracy)

            if self.limber is None:
                self.limber = True

            if self.nglimber is None:
                self.nglimber = True

            if self.ell_min is None:
                self.ell_min = 2
                print("Band powers are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_min' is not specified. It is set to '2'.")
            if self.multiplicative_shear_bias_uncertainty is None:
                self.multiplicative_shear_bias_uncertainty = 0
                print("Band powers are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "mult_shear_bias' is not specified. It is set to '0'.")
            if self.ell_max is None:
                self.ell_max = 1e5
                print("Band powerss are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '1e5'.")
            if self.ell_bins is None:
                self.ell_bins = int(np.log10(self.ell_max)*10)
                print("Band powers are calculated " +
                      "from the C_ells but '[covELLspace settings]: " +
                      "ell_max' is not specified. It is set to '" +
                      str(self.ell_bins) + "'.")
            if self.ell_type is None:
                self.ell_type = 'log'
            if self.delta_z is None:
                self.delta_z = 0.02
            if self.integration_steps is None:
                self.integration_steps = 500
            if self.nz_polyorder is None:
                self.nz_polyorder = 1
            if self.tri_delta_z is None:
                self.tri_delta_z = 0.5

        return True

    def __read_in_arbitrary_summary_settings(self,
                                             config,
                                             config_name):
        """
        Reads in further information needed to calculate the covariance 
        for arbitrary summary statistics. Every value that is not specified 
        either raises an exception or gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'arbitrary_summary' in config:
            if 'do_arbitrary_obs' in config['arbitrary_summary']:
                self.do_arbitrary_obs = config['arbitrary_summary'].getboolean('do_arbitrary_obs')
            else:
                self.do_arbitrary_obs = False
            if 'oscillations_straddle' in config['arbitrary_summary']:
                self.oscillations_straddle = int(config['arbitrary_summary']['oscillations_straddle'])
            else:
                self.oscillations_straddle = 20
            if 'arbitrary_accuracy' in config['arbitrary_summary']:
                self.arbitrary_accuracy = float(config['arbitrary_summary']['arbitrary_accuracy'])
            else:
                self.arbitrary_accuracy = 1e-5
            if self.limber is None:
                self.limber = True

            if self.nglimber is None:
                self.nglimber = True
            if self.do_arbitrary_obs:
                if self.ell_min is None:
                    self.ell_min = 2
                    print("Arbitrary summary statistics are calculated " +
                        "from the C_ells but '[covELLspace settings]: " +
                        "ell_min' is not specified. It is set to '2'.")
                if self.multiplicative_shear_bias_uncertainty is None:
                    self.multiplicative_shear_bias_uncertainty = 0
                    print("Arbitrary summary statistics are calculated " +
                        "from the C_ells but '[covELLspace settings]: " +
                        "mult_shear_bias' is not specified. It is set to '0'.")
                if self.ell_max is None:
                    self.ell_max = 1e5
                    print("Arbitrary summary statisticss are calculated " +
                        "from the C_ells but '[covELLspace settings]: " +
                        "ell_max' is not specified. It is set to '1e5'.")
                if self.ell_bins is None:
                    self.ell_bins = int(np.log10(self.ell_max)*10)
                    print("Arbitrary summary statistics are calculated " +
                        "from the C_ells but '[covELLspace settings]: " +
                        "ell_max' is not specified. It is set to '" +
                        str(self.ell_bins) + "'.")
                if self.ell_type is None:
                    self.ell_type = 'log'
                if self.delta_z is None:
                    self.delta_z = 0.08
                if self.integration_steps is None:
                    self.integration_steps = 500
                if self.nz_polyorder is None:
                    self.nz_polyorder = 1
                if self.tri_delta_z is None:
                    self.tri_delta_z = 0.5

        return True


    def __read_in_covRspace_settings(self,
                                     config,
                                     config_name):
        """
        Reads in further information needed to calculate the covariance 
        for the estimator 'projected_real'. Every value that is not 
        specified either raises an exception or gets a fall-back value 
        which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'covRspace settings' in config:
            if 'projected_radius_min' in config['covRspace settings']:
                self.projected_radius_min = float(
                    config['covRspace settings']['projected_radius_min'])
            if 'projected_radius_max' in config['covRspace settings']:
                self.projected_radius_max = float(
                    config['covRspace settings']['projected_radius_max'])
            if 'projected_radius_bins' in config['covRspace settings']:
                self.projected_radius_bins = int(
                    config['covRspace settings']['projected_radius_bins'])
            if 'projected_radius_type' in config['covRspace settings']:
                self.projected_radius_type = config['covRspace settings'][
                    'projected_radius_type']
            if 'mean_redshift' in config['covRspace settings']:
                self.mean_redshift = np.array(
                    config['covRspace settings']['mean_redshift'].split(','))
                try:
                    self.mean_redshift = (self.mean_redshift).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[covRspace settings]: 'mean_redshift = " +
                                    config['covRspace settings']['mean_redshift'] + "' " +
                                    "to numpy array. Must be adjusted in config file " +
                                    config_name + ".")
            if 'projection_length_clustering' in config['covRspace settings']:
                self.projection_length_clustering = float(
                    config['covRspace settings']['projection_length_clustering'])
        else:
            ...

        if (self.ggl and self.est_ggl == 'projected_real') or \
           (self.clustering and self.est_clust == 'projected_real'):
            if self.projected_radius_min is None:
                raise Exception("ConfigError: At least one of the chosen " +
                                "estimators is 'projected_real' but no minimum radius " +
                                "for the projection is specified. Must be adjusted in " +
                                "config file " + config_name + ", [covRspace settings]: " +
                                "'projected_radius_min = 0.01' Mpc.")

            if self.projected_radius_max is None:
                raise Exception("ConfigError: At least one of the chosen " +
                                "estimators is 'projected_real' but no maximum radius " +
                                "for the projection is specified. Must be adjusted in " +
                                "config file " + config_name + ", [covRspace settings]: " +
                                "'projected_radius_max = 100' Mpc.")

            if self.projected_radius_bins is None:
                raise Exception("ConfigError: At least one of the chosen " +
                                "estimators is 'projected_real' but no number of radial " +
                                "bins is specified. Must be adjusted in config file " +
                                config_name + ", [covRspace settings]: " +
                                "'projected_radius_bins = 10'.")

            if self.projected_radius_type is None:
                self.projected_radius_type = 'log'
                print("The binning type for radial bins [covRspace " +
                      "settings]: 'projected_radius_type' is set to 'log'.")
            elif self.projected_radius_type != 'lin' and \
                    self.projected_radius_type != 'log':
                self.projected_radius_type = 'log'
                raise Exception("ConfigError: The binning type for radial " +
                                "bins [covRspace settings]: 'projected_radius_type = " +
                                config['covRspace settings']['projected_radius_type'] +
                                "' is not recognised. Must be either 'lin' or 'log'.")

            if self.mean_redshift is None:  # array for tomographic bins
                raise Exception("ConfigError: At least one of the chosen " +
                                "estimators is 'projected_real' but no mean redshift is " +
                                "specified. Must be adjusted in config file " +
                                config_name + ", [covRspace settings]: 'mean_redshift = " +
                                "0.4, 0.5' (as many entries as there are tomographic " +
                                "foreground bins.")

            if self.projection_length_clustering is None:
                self.projection_length_clustering = 100
                print("The projection length of clustering " +
                      "[covRspace settings]: 'projection_length_clustering' " +
                      "is set to 100 Mpc/h.")

        return True

    def __read_in_output_dict(self,
                              config):
        """
        Reads in the path, name and style for the output files. This 
        includes the output file for the covariance as well as the 
        configuration file. If no output file should be printed the 
        result will be displayed in the terminal, the return of the 
        read_input method ('self.output_dir+self.output_file') will be 
        0. Every value that is not specified gets a fall-back value.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """

        if 'output settings' in config:
            if 'directory' in config['output settings']:
                self.output_dir = config['output settings']['directory']

            if 'style' in config['output settings']:
                self.output_style = \
                    config['output settings']['style'].replace(
                        " ", "").split(',')
            else:
                self.output_style = ['list']
                print("The style of the output file [output settings]: " +
                      "'style' will be 'list'.")
            if 'save_as_binary' in config['output settings']:
                self.save_as_binary = config['output settings'].getboolean('save_as_binary')
            else:
                self.save_as_binary = False

            if 'list_style_spatial_first' in config['output settings']:
                self.list_style_spatial_first = config['output settings'].getboolean('list_style_spatial_first')
            else:
                self.list_style_spatial_first = False

            self.output_file = False
            if 'list' in self.output_style or 'matrix' in self.output_style:
                self.output_file = True
                if 'file' in config['output settings']:
                    if (config['output settings']['file'].casefold() == 'true' or
                            config['output settings']['file'].casefold() == 'false'):
                        self.output_file = \
                            config['output settings'].getboolean('file')
                    else:
                        self.output_file = \
                            config['output settings']['file'].replace(
                                " ", "").split(',')
                if type(self.output_file) is bool:
                    if self.output_file:
                        self.output_file = ['covariance_' + style + '.dat'
                                            for style in self.output_style]
                        if 'covariance_terminal.dat' in self.output_file:
                            self.output_file.replace(
                                'covariance_terminal.dat', '')
                        print("The name of the output file is set to " +
                              ', '.join(self.output_file) + ".")
                    else:
                        ...

            self.make_plot = True
            if 'corrmatrix_plot' in config['output settings']:
                if (config['output settings']
                        ['corrmatrix_plot'].casefold() == 'true' or
                    config['output settings']
                        ['corrmatrix_plot'].casefold() == 'false'):
                    self.make_plot = \
                        config['output settings'].getboolean('corrmatrix_plot')
                else:
                    self.make_plot = \
                        config['output settings']['corrmatrix_plot']
            if type(self.make_plot) is bool and self.make_plot:
                self.make_plot = 'corrcoeff.pdf'
                print("The name of the plotting file is set to " +
                      "'corrcoeff.pdf'.")

            self.save_configs = True
            if 'save_configs' in config['output settings']:
                if (config['output settings']
                        ['save_configs'].casefold() == 'true' or
                    config['output settings']
                        ['save_configs'].casefold() == 'false'):
                    self.save_configs = \
                        config['output settings'].getboolean('save_configs')
                else:
                    self.save_configs = \
                        config['output settings']['save_configs']
            if type(self.save_configs) is bool and self.save_configs:
                self.save_configs = 'save_configs.ini'

            self.save_Cells = False
            if 'save_Cells' in config['output settings']:
                if (config['output settings']
                        ['save_Cells'].casefold() == 'true' or
                    config['output settings']
                        ['save_Cells'].casefold() == 'false'):
                    self.save_Cells = \
                        config['output settings'].getboolean('save_Cells')
                else:
                    self.save_Cells = \
                        config['output settings']['save_Cells']
            if type(self.save_Cells) is bool and self.save_Cells:
                self.save_Cells = 'Cell.ascii'

            self.save_trispecs = False
            if 'save_trispectra' in config['output settings']:
                if (config['output settings']
                        ['save_trispectra'].casefold() == 'true' or
                    config['output settings']
                        ['save_trispectra'].casefold() == 'false'):
                    self.save_trispecs = \
                        config['output settings'].getboolean('save_trispectra')
                else:
                    self.save_trispecs = \
                        config['output settings']['save_trispectra']
            if type(self.save_trispecs) is bool and self.save_trispecs:
                self.save_trispecs = 'trispectra.ascii'

            self.save_alms = False
            if 'save_alms' in config['output settings']:
                if (config['output settings']
                        ['save_alms'].casefold() == 'true' or
                    config['output settings']
                        ['save_alms'].casefold() == 'false'):
                    self.save_alms = \
                        config['output settings'].getboolean('save_alms')
                else:
                    self.save_alms = \
                        config['output settings']['save_alms']
            if type(self.save_alms) is bool and self.save_alms:
                self.save_alms = 'alms'
            if 'use tex' in config['output settings']:
                self.use_tex = config['output settings'].getboolean('use_tex')
            else:
                self.use_tex = False
        else:
            self.output_style = ['list', 'matrix']
            print("The style of the output file [output settings]: 'style' " +
                  "will be 'list, matrix'.")
            self.output_file = ['covariance_list.dat', 'covariance_matrix.dat']
            print("The names of the output files [output settings]: 'file' " +
                  "will be 'covariance_list.dat, covariance_matrix.dat'.")
            self.make_plot = 'corrcoeff.pdf'
            print("The name of the plotting file [output settings]: " +
                  "'corrmatrix_plot' will be 'corrcoeff.pdf'.")
            self.save_configs = 'save_configs.ini'
            print("The current configuration will be saved in the of file " +
                  "'save_configs.ini'.")
            self.save_Cells = False
            self.save_trispecs = False
            self.save_alms = False
            self.use_tex = False
            self.list_style_spatial_first = False

        if self.output_style and \
           len(self.output_style) != len(self.output_file):
            self.output_file = ['covariance_' + style + '.dat'
                                for style in self.output_style]
            if 'covariance_terminal.dat' in self.output_file:
                self.output_file.replace(
                    'covariance_terminal.dat', '')
            print("ConfigWarning: The number of entries for [output " +
                  "settings]: 'style' does not match the number for 'file.' " +
                  "The output files will be renamed to " +
                  ', '.join(self.output_file) + ".")

        allowed_entries = ['list', 'matrix', 'terminal']
        check = [itm in allowed_entries for itm in self.output_style]
        if not all(check):
            fallback = \
                [itm for itm, yes in zip(self.output_style, check) if yes]
            if fallback == []:
                fallback = ['list', 'matrix']
            print("ConfigWarning: At least one style of the output [output " +
                  "settings]: 'style = " + ', '.join(self.output_style) +
                  "' is not recognized. Available options are 'list', " +
                  "'matrix', and 'terminal'. Fallback to " +
                  ', '.join(fallback) + ".")
            self.output_style = fallback
            if not self.output_file:
                self.output_file = ['covariance_list.dat',
                                    'covariance_matrix.dat']
                print("The names of the output files are set to " +
                      "'covariance_list.dat, covariance_matrix.dat'.")

        return True

    def __read_in_cosmo_dict(self,
                             config):
        """
        Reads in the parameters for the cosmological model for which the 
        covariance should be calculated. Every value that is not 
        specified gets a fall-back value which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """

        if 'cosmo' in config:
            if 'sigma8' in config['cosmo']:
                self.sigma8 = float(config['cosmo']['sigma8'])
            else:
                self.As = 0.811
                print("The clustering amplitude sigma8 [cosmo]: 'sigma8' " +
                      "is set to 0.811 (Planck18).")

            if 'A_s' in config['cosmo']:
                self.As = float(config['cosmo']['A_s'])

            if 'h' in config['cosmo']:
                self.h = float(config['cosmo']['h'])
            else:
                self.h = 0.674
                print("The Hubble parameter h [cosmo]: 'h' is set to 0.674 " +
                      "(Planck18).")

            if 'omega_m' in config['cosmo']:
                self.omegam = float(config['cosmo']['omega_m'])
            else:
                self.omegam = 0.315
                print("The dimensionless matter density parameter Omega_m " +
                      "[cosmo]: 'omega_m' is set to 0.315 (Planck18).")

            if 'omega_b' in config['cosmo']:
                self.omegab = float(config['cosmo']['omega_b'])
            else:
                self.omegab = 0.049
                print("The dimensionless baryon density parameter Omega_b " +
                      "[cosmo]: 'omega_b' is set to 0.049 (Planck18).")

            if 'omega_de' in config['cosmo']:
                self.omegade = float(config['cosmo']['omega_de'])
                if np.abs(self.omegade + self.omegam - 1) < 1e-5:
                    self.omegade = 1.0 - self.omegam
            else:
                self.omegade = 1.0 - self.omegam
                print("The dimensionless dark energy density parameter " +
                      "Omega_Lambda [cosmo]: 'omega_de' is set to " +
                      "1-Omega_m (vanilla LCDM value).")

            if 'w0' in config['cosmo']:
                self.w0 = float(config['cosmo']['w0'])
            else:
                self.w0 = -1
                print("The w0 parameter for the parametrised dark energy " +
                      "equation of state [cosmo]: 'w0' is set to -1 " +
                      "(vanilla LCDM value).")

            if 'wa' in config['cosmo']:
                self.wa = float(config['cosmo']['wa'])
            else:
                self.wa = 0
                print("The wa parameter for the parametrised dark energy " +
                      "equation of state [cosmo]: 'wa' is set to 0 " +
                      "(vanilla LCDM value).")

            if 'ns' in config['cosmo']:
                self.ns = float(config['cosmo']['ns'])
            else:
                self.ns = 0.965
                print("The slope for the initial power spectrum n_s " +
                      "[cosmo]: 'ns' is set to 0.965 (Planck18).")

            if 'neff' in config['cosmo']:
                self.neff = float(config['cosmo']['neff'])
            else:
                self.neff = 3.046
                print("The effective number relativistic species [cosmo]: " +
                      "'neff' is set to 3.046 (vanilla LCDM value).")

            if 'm_nu' in config['cosmo']:
                self.mnu = float(config['cosmo']['m_nu'])
            else:
                self.mnu = 0
                print("The summed mass of the neutrinos [cosmo]: 'm_nu' is " +
                      "set to 0 eV (Planck18).")

            if 'Tcmb0' in config['cosmo']:
                self.Tcmb0 = float(config['cosmo']['Tcmb0'])
            else:
                self.Tcmb0 = 2.725

        else:
            self.sigma8 = 0.811
            print("The clustering amplitude sigma8 [cosmo]: 'sigma8' is " +
                  "set to 0.811 (Planck18).")
            self.h = 0.674
            print("The Hubble parameter h [cosmo]: 'h' is set to 0.674 " +
                  "(Planck18).")
            self.omegam = 0.315
            print("The dimensionless matter density parameter Omega_m " +
                  "[cosmo]: 'Omega_m' is set to 0.315 (Planck18).")
            self.omegab = 0.049
            print("The dimensionless baryon density parameter Omega_b " +
                  "[cosmo]: 'Omega_b' is set to 0.049 (Planck18).")
            self.omegade = 1.0 - self.omegam
            print("The dimensionless dark energy density parameter " +
                  "Omega_Lambda [cosmo]: 'Omega_de' is set to 1-Omega_m " +
                  "(vanilla LCDM value).")
            self.w0 = -1
            print("The w0 parameter for the parametrised dark energy " +
                  "equation of state [cosmo]: 'w0' is set to -1 (vanilla " +
                  "LCDM value).")
            self.wa = 0
            print("The wa parameter for the parametrised dark energy " +
                  "equation of state [cosmo]: 'wa' is set to 0 (vanilla " +
                  "LCDM value).")
            self.ns = 0.965
            print("The slope for the initial power spectrum n_s [cosmo]: " +
                  "'ns' is set to 0.965 (Planck18).")
            self.neff = 3.046
            print("The effective number relativistic species [cosmo]: " +
                  "'neff' is set to 3.046 (vanilla LCDM value).")
            self.mnu = 0
            print("The summed mass of the neutrinos [cosmo]: 'm_nu' is set " +
                  "to 0 eV (Planck18).")
            self.Tcmb0 = 2.725

        return True

    def __read_in_bias_dict(self,
                            config,
                            config_name):
        """
        Reads in relevant information for the bias model for which the 
        covariance should be calculated. Every value that is not 
        specified either raises an exception or gets a fall-back value 
        which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'bias' in config:
            if 'model' in config['bias']:
                self.bias_model = config['bias']['model']
            else:
                self.bias_model = 'Tinker10'
             
            if 'bias_2h' in config['bias']:
                self.bias_2h = float(config['bias']['bias_2h'])
            else:
                self.bias_2h = 1
              
            if 'Mc_relation_cen' in config['bias']:
                self.Mc_relation_cen = config['bias']['Mc_relation_cen']
            else:
                self.Mc_relation_cen = 'duffy08'
               
            if 'Mc_relation_sat' in config['bias']:
                self.Mc_relation_sat = config['bias']['Mc_relation_sat']
            if self.Mc_relation_sat is None:
                self.Mc_relation_sat = 'duffy08'
            if 'norm_Mc_relation_cen' in config['bias']:
                self.norm_Mc_relation_cen = float(config['bias']['norm_Mc_relation_cen'])
            else:
                self.norm_Mc_relation_cen = 1.0
            if 'norm_Mc_relation_sat' in config['bias']:
                self.norm_Mc_relation_sat = float(config['bias']['norm_Mc_relation_sat'])
            else:
                self.norm_Mc_relation_sat = 1.0
            
            if 'log10mass_bins' in config['bias']:
                self.logmass_bins = \
                    np.array(config['bias']['log10mass_bins'].split(','))
                try:
                    self.logmass_bins = (self.logmass_bins).astype(float)
                    self.sampledim = len(self.logmass_bins) - 1
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[bias]: 'logmass_bins = " +
                                    config['bias']['log10mass_bins'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")
            else:
                self.sampledim = 1
            if self.logmass_bins is not None and len(self.logmass_bins) == 1:
                raise Exception("ConfigError: Only one value is given for " +
                                "[bias]: 'log10mass_bins' but at least two values must "
                                "be given (for lower and upper bound). Must be adjusted " +
                                "in config file " + config_name + ", [bias]: " +
                                "'log10mass_bins = 12,13,18' for two log10mass bins " +
                                "[12,13] and [13,18].")
        else:
            self.bias_model = 'Tinker10'
            print("The bias model [bias]: 'model' is set to Tinker10.")
            self.bias_2h = 1
            print("The bias for the 2-halo term [bias]: 'bias_2h' is set " +
                  "to 1.")
            self.Mc_relation_cen = 'duffy08'
            self.Mc_relation_sat = 'duffy08'
            self.norm_Mc_relation_cen = 1.0
            self.norm_Mc_relation_sat = 1.0
            print("The mass-concentration relation for the centrals [bias]: " +
                  "'Mc_relation_cen' is set to duffy08.")
            self.sampledim = 1

        return True
    
    def __read_in_IA_dict(self,
                          config,
                          config_name):
        """
        Reads in relevant information for the intrinsic alignment
        model for which the covariance should be calculated. Every 
        value that is not specified either raises an exception or gets
        a fall-back value which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """
        self.intrinsic_alignments = dict()
        self.intrinsic_alignments_abr = dict()

        if self.cosmicshear or self.ggl:    
            if 'IA' in config:
                if 'A_IA' in config['IA']:
                    self.A_IA = float(config['IA']['A_IA'])
                else:
                    self.A_IA = 0.4
                    print("The intrinsic alignment amplitude is set to 0.4")
                if 'eta_IA' in config['IA']:
                    self.eta_IA = float(config['IA']['eta_IA'])
                else:
                    self.eta_IA = 0.0
                    print("The redshift dependence of the alignment strength, " + 
                        "eta_IA, is set to 0")
                if 'z_pivot_IA' in config['IA']:
                    self.z_pivot_IA = float(config['IA']['z_pivot_IA'])
                else:
                    self.z_pivot_IA = 0.3
                    print("The pivot scale of the redshift dependence of IA, " +
                        "z_pivot, is set to 0.3")
            else:
                self.A_IA = 0.4
                print("The intrinsic alignment amplitude is set to 0.4")
                self.eta_IA = 0.0
                print("The redshift dependence of the alignment strength, " + 
                    "eta_IA, is set to 0")
                self.z_pivot_IA = 0.3
                print("The pivot scale of the redshift dependence of IA, " +
                        "z_pivot, is set to 0.3")
        
        return True
        
    def __read_in_hod_dict(self,
                           config,
                           config_name):
        """
        Reads in relevant information for the halo occupation 
        distribution model for which the covariance should be 
        calculated. Every value that is not specified either raises an 
        exception or gets a fall-back value which is reported to the 
        user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'hod' in config:
            if 'model_mor_cen' in config['hod']:
                self.hod_model_mor_cen = \
                    'self.' + config['hod']['model_mor_cen']
            if 'model_mor_sat' in config['hod']:
                self.hod_model_mor_sat = \
                    'self.' + config['hod']['model_mor_sat']

            if 'dpow_logM0_cen' in config['hod']:
                self.dpow_logM0_cen = float(config['hod']['dpow_logM0_cen'])
            if 'dpow_logM1_cen' in config['hod']:
                self.dpow_logM1_cen = float(config['hod']['dpow_logM1_cen'])
            if 'dpow_a_cen' in config['hod']:
                self.dpow_a_cen = float(config['hod']['dpow_a_cen'])
            if 'dpow_b_cen' in config['hod']:
                self.dpow_b_cen = float(config['hod']['dpow_b_cen'])
            if 'dpow_norm_cen' in config['hod']:
                self.dpow_norm_cen = float(config['hod']['dpow_norm_cen'])
            if 'dpow_logM0_sat' in config['hod']:
                self.dpow_logM0_sat = float(config['hod']['dpow_logM0_sat'])
            if 'dpow_logM1_sat' in config['hod']:
                self.dpow_logM1_sat = float(config['hod']['dpow_logM1_sat'])
            if 'dpow_a_sat' in config['hod']:
                self.dpow_a_sat = float(config['hod']['dpow_a_sat'])
            if 'dpow_b_sat' in config['hod']:
                self.dpow_b_sat = float(config['hod']['dpow_b_sat'])
            if 'dpow_norm_sat' in config['hod']:
                self.dpow_norm_sat = float(config['hod']['dpow_norm_sat'])

            if 'model_scatter_cen' in config['hod']:
                self.hod_model_scatter_cen = \
                    'self.' + config['hod']['model_scatter_cen']
            if 'model_scatter_sat' in config['hod']:
                self.hod_model_scatter_sat = \
                    'self.' + config['hod']['model_scatter_sat']

            if 'logn_sigma_c_cen' in config['hod']:
                self.logn_sigma_c_cen = \
                    float(config['hod']['logn_sigma_c_cen'])
            if 'logn_sigma_c_sat' in config['hod']:
                self.logn_sigma_c_sat = \
                    float(config['hod']['logn_sigma_c_sat'])

            if 'modsch_logMref_cen' in config['hod']:
                self.modsch_logMref_cen = \
                    float(config['hod']['modsch_logMref_cen'])
            if 'modsch_alpha_s_cen' in config['hod']:
                self.modsch_alpha_s_cen = \
                    float(config['hod']['modsch_alpha_s_cen'])
            if 'modsch_b_cen' in config['hod']:
                self.modsch_b_cen = \
                    np.array(config['hod']['modsch_b_cen'].split(','))
                try:
                    self.modsch_b_cen = (self.modsch_b_cen).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[hod]: 'modsch_b_cen = " +
                                    config['hod']['modsch_b_cen'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")
            if 'modsch_logMref_sat' in config['hod']:
                self.modsch_logMref_sat = \
                    float(config['hod']['modsch_logMref_sat'])
            if 'modsch_alpha_s_sat' in config['hod']:
                self.modsch_alpha_s_sat = \
                    float(config['hod']['modsch_alpha_s_sat'])
            if 'modsch_b_sat' in config['hod']:
                self.modsch_b_sat = \
                    np.array(config['hod']['modsch_b_sat'].split(','))
                try:
                    self.modsch_b_sat = (self.modsch_b_sat).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[hod]: 'modsch_b_sat = " +
                                    config['hod']['modsch_b_sat'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")
        else:
            self.hod_model_mor_cen = 'self.double_powerlaw'
            self.hod_model_mor_sat = 'self.double_powerlaw'
            self.hod_model_scatter_cen = 'self.lognormal'
            self.hod_model_scatter_sat = 'self.modschechter'
            self.dpow_logM0_cen = 10.6
            self.dpow_logM1_cen = 11.2
            
        if self.hod_model_mor_cen is not None and \
           self.hod_model_mor_cen != 'self.double_powerlaw':
            raise Exception("ConfigError: The chosen HOD model is " +
                            "[hod]: 'hod_model_mor_cen = " +
                            self.hod_model_mor_cen[5:] + "'. Which is not yet " +
                            "configured. Available options are: 'double_powerlaw'.")

        if self.hod_model_mor_cen == 'self.double_powerlaw':
            if self.dpow_logM0_cen is None:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_cen = double_powerlaw'. The " +
                                "value 'dpow_logM0_cen' must be provided for the " +
                                "mass-observable relation for centrals. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_logM0_cen = 10.6'.")
            if self.dpow_logM1_cen is None:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_cen = double_powerlaw'. The " +
                                "value 'dpow_logM1_cen' must be provided for the " +
                                "mass-observable relation for centrals. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_logM1_cen = 11.25'.")
            if self.dpow_a_cen is None:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_cen = double_powerlaw'. The " +
                                "value 'dpow_a_cen' must be provided for the " +
                                "mass-observable relation for centrals. ustt be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_a_cen = 3.41'.")
            if self.dpow_b_cen is None:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_cen = double_powerlaw'. The " +
                                "value 'dpow_b_cen' must be provided for the " +
                                "mass-observable relation for centrals. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_b_cen = 0.99'.")
            if self.dpow_norm_cen is None:
                self.dpow_norm_cen = 1
                print("The chosen HOD model is " +
                      "[hod]: 'hod_model_mor_cen = double_powerlaw'. The " +
                      "normalisation 'dpow_norm_cen' is set to 1.")

        if self.hod_model_mor_sat is not None and \
           self.hod_model_mor_sat != 'self.double_powerlaw':
            raise Exception("ConfigError: The chosen HOD model is " +
                            "[hod]: 'hod_model_mor_sat = " + self.hod_model_mor_sat[5:] +
                            "'. Which is not yet configured. Available options are: " +
                            "'double_powerlaw'.")

        if self.hod_model_mor_sat is None:
            self.hod_model_mor_sat = self.hod_model_mor_cen
        if self.hod_model_mor_sat == 'self.double_powerlaw':
            if self.dpow_logM0_sat is not None:
                ...
            elif self.dpow_logM0_sat is None and self.dpow_logM0_cen is not None:
                self.dpow_logM0_sat = self.dpow_logM0_cen
            else:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_sat = double_powerlaw'. The " +
                                "value 'dpow_logM0_sat' must be provided for the " +
                                "mass-observable relation for satellites. Can be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_logM0_sat = 10.6'.")
            if self.dpow_logM1_sat is not None:
                ...
            elif self.dpow_logM1_sat is None and self.dpow_logM1_cen is not None:
                self.dpow_logM1_sat = self.dpow_logM1_cen
            else:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_sat = double_powerlaw'. The " +
                                "value 'dpow_logM1_sat' must be provided for the " +
                                "mass-observable relation for satellites. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_logM1_sat = 11.25'.")
            if self.dpow_a_sat is not None:
                ...
            elif self.dpow_a_sat is None and self.dpow_a_cen is not None:
                self.dpow_a_sat = self.dpow_a_cen
            else:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_sat = double_powerlaw'. The " +
                                "value 'dpow_a_sat' must be provided for the " +
                                "mass-observable relation for satellites. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_a_sat = 3.41'.")
            if self.dpow_b_sat is not None:
                ...
            elif self.dpow_b_sat is None and self.dpow_b_cen is not None:
                self.dpow_b_sat = self.dpow_b_cen
            else:
                raise Exception("ConfigError: The chosen HOD model is " +
                                "[hod]: 'hod_model_mor_sat = double_powerlaw'. The " +
                                "value 'dpow_b_sat' must be provided for the " +
                                "mass-observable relation for satellites. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'dpow_b_sat = 0.99'.")
            if self.dpow_norm_sat is not None:
                ...
            elif self.dpow_norm_sat is None and self.dpow_norm_cen is not None:
                self.dpow_norm_sat = self.dpow_norm_cen
            else:
                self.dpow_norm_sat = 1
                print("The chosen HOD model is " +
                      "[hod]: 'hod_model_mor_sat = double_powerlaw'. The " +
                      "normalisation 'dpow_norm_sat' is set to 1.")

        if self.hod_model_scatter_cen is not None and \
           self.hod_model_scatter_cen != 'self.lognormal' and \
           self.hod_model_scatter_cen != 'self.modschechter':
            raise Exception("ConfigError: The chosen HOD model is " +
                            "[hod]: 'hod_model_scatter_cen = " +
                            self.hod_model_scatter_cen[5:] + "'. Which is not yet " +
                            "configured. Available options are: 'lognormal', " +
                            "'modschechter'.")

        if self.hod_model_scatter_cen == 'self.lognormal':
            if self.logn_sigma_c_cen is None:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for centrals is 'lognormal', " +
                                "thus 'logn_sigma_c_cen' must be provided. Must be " +
                                "adjusted in config file " + config_name + ", [hod]: " +
                                "'logn_sigma_c_cen = 0.35'.")

        if self.hod_model_scatter_sat is not None and \
           self.hod_model_scatter_sat != 'self.lognormal' and \
           self.hod_model_scatter_sat != 'self.modschechter':
            raise Exception("ConfigError: The chosen HOD model is " +
                            "[hod]: 'hod_model_scatter_sat = " +
                            self.hod_model_scatter_sat[5:] + "'. Which is not yet " +
                            "configured. Available options are: 'lognormal', " +
                            "'modschechter'.")

        if self.hod_model_scatter_sat is None:
            self.hod_model_scatter_sat = self.hod_model_scatter_cen
        if self.hod_model_scatter_sat == 'self.lognormal':
            if self.logn_sigma_c_sat is not None:
                ...
            elif self.logn_sigma_c_sat is None and self.logn_sigma_c_cen is not None:
                self.logn_sigma_c_sat = self.logn_sigma_c_cen
            else:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for satellites is " +
                                "'lognormal', thus 'logn_sigma_c_sat' must be provided. " +
                                "Must be adjusted in config file " + config_name + ", " +
                                "[hod]: 'logn_sigma_c_sat = 0.35'.")
        if self.hod_model_scatter_cen == 'self.modschechter':
            if self.modsch_logMref_cen is None:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for centrals is " +
                                "'modschechter', thus 'modsch_logMref_cen' must be " +
                                "provided. Must be adjusted in config file " +
                                config_name + ", [hod]: 'modsch_logMref_cen = 12'.")
            if self.modsch_alpha_s_cen is None:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for centrals is " +
                                "'modschechter', thus 'modsch_alpha_s_cen' must be " +
                                "provided. Must be adjusted in config file " +
                                config_name + ", [hod]: 'modsch_alpha_s_cen = -1.34'.")
            if self.modsch_b_cen is None:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for centrals is " +
                                "'modschechter', thus 'modsch_b_cen' must be provided. " +
                                "Must be adjusted in config file " + config_name + ", " +
                                "[hod]: 'modsch_b_cen = 0'.")

        if self.hod_model_scatter_sat is None:
            self.hod_model_scatter_sat = self.hod_model_scatter_cen
        if self.hod_model_scatter_sat == 'self.modschechter':
            if self.modsch_logMref_sat is not None:
                ...
            elif self.modsch_logMref_sat is None and \
                    self.modsch_logMref_cen is not None:
                self.modsch_logMref_sat = self.modsch_logMref_cen
            else:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for satellites is " +
                                "'modschechter', thus 'modsch_logMref_sat' must be " +
                                "provided. Must be adjusted in config file " +
                                config_name + ", [hod]: 'modsch_logMref_sat = 12'.")
            if self.modsch_alpha_s_sat is not None:
                ...
            elif self.modsch_alpha_s_sat is None and \
                    self.modsch_logMref_cen is not None:
                self.modsch_alpha_s_sat = self.modsch_alpha_s_cen
            else:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for satellites is " +
                                "'modschechter', thus 'modsch_alpha_s_sat' must be " +
                                "provided. Must be adjusted in config file " +
                                config_name + ", [hod]: 'modsch_alpha_s_sat = -1.34'.")
            if self.modsch_b_sat is not None:
                ...
            elif self.modsch_b_sat is None and self.modsch_b_cen is not None:
                self.modsch_b_sat = self.modsch_b_cen
            else:
                raise Exception("ConfigError: The scatter of the " +
                                "mass-observable relation for satellites is " +
                                "'modschechter', thus 'modsch_b_sat' must be provided. " +
                                "Must be adjusted in config file " + config_name + ", " +
                                "[hod]: 'modsch_b_sat = 0'.")

        if self.modsch_b_cen is not None and self.sampledim is not None:
            if len(self.modsch_b_cen) == 1:
                self.modsch_b_cen = \
                    np.ones(self.sampledim) * self.modsch_b_cen[0]
            if len(self.modsch_b_cen) != self.sampledim:
                raise Exception("ConfigError: The number of galaxy sample " +
                                "bins is " + str(self.sampledim) + " but 'modsch_b_cen' " +
                                "has " + str(len(self.modsch_b_cen)) + " entries. Must " +
                                "be adjusted to go on.")

        return True

    def __read_in_survey_params_dict(self,
                                     config,
                                     config_name):
        """
        Reads in relevant information for the specific survey for which 
        the covariance should be calculated. Every value that is not
        specified either raises an exception or gets a fall-back value 
        which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """

        if 'survey specs' in config:
            if 'mask_directory' in config['survey specs']:
                self.mask_dir = \
                    config['survey specs']['mask_directory']
            else:
                self.mask_dir = ''

            self.survey_area_clust = []
            if 'survey_area_clust_specz_in_deg2' in config['survey specs']:
                self.survey_area_clust.append(float(config['survey specs']
                                                    ['survey_area_clust_specz_in_deg2']))
            if len(self.survey_area_clust) == 1:
                if 'survey_area_clust_photz_in_deg2' in config['survey specs']:
                    self.survey_area_clust.append(float(config['survey specs']
                                                        ['survey_area_clust_photz_in_deg2']))
                else:
                    self.survey_area_clust.append(float(config['survey specs']
                                                        ['survey_area_clust_specz_in_deg2']))
                    print("The survey area for a photometric sample of " +
                          "galaxy clustering is implicitely set to [survey " +
                          "specs]: 'survey_area_clust_photz_in_deg2 = " +
                          config['survey specs']['survey_area_clust_specz_in_deg2'] +
                          " sqdeg. This parameter belongs to the extended " +
                          "6x2pt-analysis functionality.")
            elif 'survey_area_clust_photz_in_deg2' in config['survey specs']:
                raise Exception("ConfigError: The survey area for a " +
                                "photometric sample of galaxy clustering [survey " +
                                "specs]: 'survey_area_clust_photz_in_deg2' is given but " +
                                "no [survey specs]: 'survey_area_clust_specz_in_deg2'. " +
                                "Both parameters belong to the extended 6x2pt-analysis " +
                                "functionality and the 'specz' parameter must be passed.")
            if len(self.survey_area_clust) == 2:
                if 'survey_area_clust_cross_in_deg2' in config['survey specs']:
                    self.survey_area_clust.append(float(config['survey specs']
                                                        ['survey_area_clust_cross_in_deg2']))
                else:
                    if self.survey_area_clust[0] == self.survey_area_clust[1]:
                        self.survey_area_clust.append(float(config['survey specs']
                                                            ['survey_area_clust_specz_in_deg2']))
                    else:
                        print("The survey area overlap for a spectroscopic " +
                              "and a photometric sample of galaxy " +
                              "clustering will be set to the larger area of " +
                              "both samples [survey specs]: " +
                              "'survey_area_clust_cross_in_deg2 = " +
                              str(max(self.survey_area_clust)) + " sqdeg'. " +
                              "This parameter belongs to the extended " +
                              "6x2pt-analysis functionality.")
                self.survey_area_clust = np.array(self.survey_area_clust)
            elif 'survey_area_clust_cross_in_deg2' in config['survey specs']:
                raise Exception("ConfigError: The survey area overlap for a " +
                                "spectroscopic and a photometric sample of galaxy " +
                                "clustering [survey specs]: " +
                                "'survey_area_clust_cross_in_deg2' is given but no " +
                                "[survey specs]: 'survey_area_clust_specz_in_deg2' " +
                                "and 'survey_area_clust_photz_in_deg2'. All parameters " +
                                "belong to the extended 6x2pt-analysis functionality " +
                                "and the 'specz' and 'photz' parameter must be passed.")
            if len(self.survey_area_clust) == 0:
                if 'survey_area_clust_in_deg2' in config['survey specs']:
                    self.survey_area_clust = np.array(config['survey specs']
                                                      ['survey_area_clust_in_deg2'].split(','))
                    if self.n_spec is not None and len(self.survey_area_clust) < 2 and self.n_spec != 0 :
                        self.survey_area_clust = np.append(self.survey_area_clust,self.survey_area_clust[0])
                    try:
                        self.survey_area_clust = \
                            (self.survey_area_clust).astype(float)
                    except ValueError:
                        raise Exception("ConfigError: Cannot convert string " +
                                        "in [survey specs]: 'survey_area_clust_in_deg2' " +
                                        "= " +
                                        config['survey specs']['survey_area_clust_in_deg2'] +
                                        "' to numpy array. Must be adjusted in config " +
                                        "file " + config_name + ".")
                else:
                    self.survey_area_clust = None

            self.mask_file_clust = []
            if 'mask_file_clust_specz' in config['survey specs']:
                self.mask_file_clust.append(
                    config['survey specs']['mask_file_clust_specz'])
            if len(self.mask_file_clust) == 1:
                if 'mask_file_clust_photz' in config['survey specs']:
                    self.mask_file_clust.append(
                        config['survey specs']['mask_file_clust_photz'])
                else:
                    self.mask_file_clust.append(
                        config['survey specs']['mask_file_clust_specz'])
                    if self.survey_area_clust is None:
                        print("The name of the mask file for a photometric " +
                              "sample of galaxy clustering is implicitely set " +
                              "to [survey specs]: 'mask_file_clust_photz = " +
                              config['survey specs']['mask_file_clust_specz'] +
                              ". This parameter belongs to the extended " +
                              "6x2pt-analysis functionality.")
            elif 'mask_file_clust_photz' in config['survey specs']:
                raise Exception("ConfigError: The mask file for a " +
                                "photometric sample of galaxy clustering [survey " +
                                "specs]: 'mask_file_clust_photz' is given but no " +
                                "[survey specs]: 'mask_file_clust_specz'. Both " +
                                "parameters belong to the extended 6x2pt-analysis " +
                                "functionality and the 'specz' parameter must be passed.")
            if len(self.mask_file_clust) == 2:
                if 'mask_file_clust_cross' in config['survey specs']:
                    self.mask_file_clust.append(
                        config['survey specs']['mask_file_clust_cross'])
                else:
                    if self.mask_file_clust[0] == self.mask_file_clust[1]:
                        self.mask_file_clust = \
                            [config['survey specs']['mask_file_clust_specz']]
                    else:
                        if self.survey_area_clust is None:
                            raise Exception("The mask file for the survey " +
                                            "area overlap for a spectroscopic and a " +
                                            "photometric sample of galaxy clustering " +
                                            "is missing. Please add [survey specs]: " +
                                            "'mask_file_clust_cross' to continue. This " +
                                            "parameter belongs to the extended " +
                                            "6x2pt-analysis functionality.")
            elif 'mask_file_clust_cross' in config['survey specs']:
                raise Exception("ConfigError: The mask file for the survey " +
                                "area overlap for a spectroscopic and a photometric " +
                                "sample of galaxy clustering [survey specs]: " +
                                "'mask_file_clust_cross' is given but no [survey specs]: " +
                                "'mask_file_clust_specz' and 'mask_file_clust_photz'. " +
                                "All parameters belong to the extended 6x2pt-analysis " +
                                "functionality and the 'specz' and 'photz' parameter " +
                                "must be passed.")
            if len(self.mask_file_clust) == 0:
                if 'mask_file_clust' in config['survey specs']:
                    self.mask_file_clust = (config['survey specs']
                                            ['mask_file_clust'].replace(" ", "")).split(',')
                else:
                    self.mask_file_clust = None

            self.survey_area_ggl = []
            if 'survey_area_ggl_specz_in_deg2' in config['survey specs']:
                self.survey_area_ggl.append(float(config['survey specs']
                                                  ['survey_area_ggl_specz_in_deg2']))
            if len(self.survey_area_ggl) == 1:
                if 'survey_area_ggl_photz_in_deg2' in config['survey specs']:
                    self.survey_area_ggl.append(float(config['survey specs']
                                                      ['survey_area_ggl_photz_in_deg2']))
                else:
                    self.survey_area_ggl.append(float(config['survey specs']
                                                      ['survey_area_ggl_specz_in_deg2']))
                    print("The survey area for a photometric sample of " +
                          "galaxy-galaxy lensing is implicitely set to " +
                          "[survey specs]: 'survey_area_ggl_photz_in_deg2 = " +
                          config['survey specs']['survey_area_ggl_specz_in_deg2'] +
                          " sqdeg. This parameter belongs to the extended " +
                          "6x2pt-analysis functionality.")
                self.survey_area_ggl = np.array(self.survey_area_ggl)
            elif 'survey_area_ggl_photz_in_deg2' in config['survey specs']:
                raise Exception("ConfigError: The survey area for a " +
                                "photometric sample of galaxy-galaxy lensing [survey " +
                                "specs]: 'survey_area_ggl_photz_in_deg2' is given but " +
                                "no [survey specs]: 'survey_area_ggl_specz_in_deg2'. " +
                                "Both parameters belong to the extended 6x2pt-analysis " +
                                "functionality and the 'specz' parameter must be passed.")
            if len(self.survey_area_ggl) == 0:
                if 'survey_area_ggl_in_deg2' in config['survey specs']:
                    self.survey_area_ggl = np.array(config['survey specs']
                                                    ['survey_area_ggl_in_deg2'].split(','))
                    try:
                        self.survey_area_ggl = \
                            (self.survey_area_ggl).astype(float)
                    except ValueError:
                        raise Exception("ConfigError: Cannot convert string " +
                                        "in [survey specs]: 'survey_area_ggl_in_deg2' = " +
                                        config['survey specs']['survey_area_ggl_in_deg2'] +
                                        "' to numpy array. Must be adjusted in config " +
                                        "file " + config_name + ".")
                else:
                    self.survey_area_ggl = None
            if self.n_spec is not None and self.n_spec != 0 and len(self.survey_area_ggl) < 2 and len(self.survey_area_ggl) > 0:
                self.survey_area_ggl = np.append(self.survey_area_ggl,self.survey_area_ggl[0])
            self.mask_file_ggl = []
            if 'mask_file_ggl_specz' in config['survey specs']:
                self.mask_file_ggl.append(
                    config['survey specs']['mask_file_ggl_specz'])
            if len(self.mask_file_ggl) == 1:
                if 'mask_file_ggl_photz' in config['survey specs']:
                    self.mask_file_ggl.append(
                        config['survey specs']['mask_file_ggl_photz'])
                else:
                    if self.survey_area_ggl is None:
                        print("The name of the mask file for a photometric " +
                              "sample of galaxy-galaxy lensing is implicitely " +
                              "set to [survey specs]: 'mask_file_ggl_photz = " +
                              config['survey specs']['mask_file_ggl_specz'] +
                              ". This parameter belongs to the extended " +
                              "6x2pt-analysis functionality.")
            elif 'mask_file_ggl_photz' in config['survey specs']:
                raise Exception("ConfigError: The mask file for a " +
                                "photometric sample of galaxy-galaxy lensing [survey " +
                                "specs]: 'mask_file_ggl_photz' is given but no [survey " +
                                "specs]: 'mask_file_ggl_specz'. Both " +
                                "parameters belong to the extended 6x2pt-analysis " +
                                "functionality and the 'specz' parameter must be passed.")
            if len(self.mask_file_ggl) == 0:
                if 'mask_file_ggl' in config['survey specs']:
                    self.mask_file_ggl = (config['survey specs']
                                          ['mask_file_ggl'].replace(" ", "")).split(',')
                else:
                    self.mask_file_ggl = None

            if 'survey_area_lensing_in_deg2' in config['survey specs']:
                self.survey_area_lens = np.array(config['survey specs']
                                                 ['survey_area_lensing_in_deg2'].split(','))
                try:
                    self.survey_area_lens = \
                        (self.survey_area_lens).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'survey_area_lensing_in_deg2' = " +
                                    config['survey specs']['survey_area_lensing_in_deg2'] +
                                    "' to numpy array. Must be adjusted in config file " +
                                    config_name + ".")

            if 'mask_file_lensing' in config['survey specs']:
                self.mask_file_lens = (config['survey specs']
                                       ['mask_file_lensing'].replace(" ", "")).split(',')

            if 'alm_file_clust' in config['survey specs']:
                self.alm_file_clust = (config['survey specs']
                                       ['alm_file_clust'].replace(" ", "")).split(',')
            if 'alm_file_ggl' in config['survey specs']:
                self.alm_file_ggl = (config['survey specs']
                                     ['alm_file_ggl'].replace(" ", "")).split(',')
            if 'alm_file_lensing' in config['survey specs']:
                self.alm_file_lens = (config['survey specs']
                                      ['alm_file_lensing'].replace(" ", "")).split(',')
            if 'alm_file_clust_lens' in config['survey specs']:
                self.alm_file_clust_lens = (config['survey specs']
                                            ['alm_file_clust_lensing'].replace(" ", "")).split(',')
            if 'alm_file_clust_ggl' in config['survey specs']:
                self.alm_file_clust_ggl = (config['survey specs']
                                           ['alm_file_clust_ggl'].replace(" ", "")).split(',')
            if 'alm_file_lens_ggl' in config['survey specs']:
                self.alm_file_lens = (config['survey specs']
                                      ['alm_file_lensing_ggl'].replace(" ", "")).split(',')

            if 'ellipticity_dispersion' in config['survey specs']:
                self.sigma_eps = \
                    np.array(config['survey specs']
                             ['ellipticity_dispersion'].split(','))
                try:
                    self.sigma_eps = (self.sigma_eps).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'ellipticity_dispersion = " +
                                    config['survey specs']['ellipticity_dispersion'] +
                                    "' to numpy array. Must be adjusted in config file " +
                                    config_name + ".")
            else:
                self.sigma_eps = np.array([.3])
                if self.cosmicshear or self.ggl:
                    print("The ellipticity dispersion of observed galaxies " +
                        "[survey specs]: 'ellipticity_dispersion' is set to 0.3")

            if 'n_eff_clust_specz' in config['survey specs'] and \
               'n_eff_clust_photz' in config['survey specs']:
                self.n_eff_clust = np.array(
                    config['survey specs']['n_eff_clust_specz'].split(',') +
                    config['survey specs']['n_eff_clust_photz'].split(','))
                try:
                    self.n_eff_clust = (self.n_eff_clust).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'n_eff_clust = " +
                                    config['survey specs']['n_eff_clust'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")
            elif ('n_eff_clust_specz' in config['survey specs']) != \
                 ('n_eff_clust_photz' in config['survey specs']):
                raise Exception("ConfigError: The parameters for the " +
                                "extended 6x2pt analysis [survey specs]: " +
                                "'n_eff_clust_specz' and 'n_eff_clust_photz' must " +
                                "always be passed together.")

            if self.n_eff_clust is None and \
               'n_eff_clust' in config['survey specs']:
                self.n_eff_clust = \
                    np.array(config['survey specs']['n_eff_clust'].split(','))
                try:
                    self.n_eff_clust = (self.n_eff_clust).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'n_eff_clust = " +
                                    config['survey specs']['n_eff_clust'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")

            if 'n_eff_ggl_specz' in config['survey specs'] and \
               'n_eff_ggl_photz' in config['survey specs']:
                self.n_eff_ggl = np.array(
                    config['survey specs']['n_eff_ggl_specz'].split(',') +
                    config['survey specs']['n_eff_ggl_photz'].split(','))
                try:
                    self.n_eff_ggl = (self.n_eff_ggl).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'n_eff_ggl = " +
                                    config['survey specs']['n_eff_ggl'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")
            elif ('n_eff_ggl_specz' in config['survey specs']) != \
                 ('n_eff_ggl_photz' in config['survey specs']):
                raise Exception("ConfigError: The parameters for the " +
                                "extended 6x2pt analysis [survey specs]: " +
                                "'n_eff_ggl_specz' and 'n_eff_ggl_photz' must " +
                                "always be passed together.")
            elif 'n_eff_ggl' in config['survey specs']:
                self.n_eff_ggl = \
                    np.array(config['survey specs']['n_eff_ggl'].split(','))
                try:
                    self.n_eff_ggl = (self.n_eff_ggl).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'n_eff_ggl = " +
                                    config['survey specs']['n_eff_ggl'] + "' to numpy " +
                                    "array. Must be adjusted in config file " +
                                    config_name + ".")

            if 'n_eff_lensing' in config['survey specs']:
                self.n_eff_lens = \
                    np.array(config['survey specs']
                             ['n_eff_lensing'].split(','))
                if(len(self.n_eff_lens) != len(self.sigma_eps)):
                    raise Exception("ConfigWarning: the number of entries for the " +
                                    "ellipticity dispersion in " + config_name +
                                    " is not the same as the number of n_eff_lensing")
                try:
                    self.n_eff_lens = (self.n_eff_lens).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'n_eff_lensing = " +
                                    config['survey specs']['n_eff_lensing'] + "' to " +
                                    "numpy array. Must be adjusted in config file " +
                                    config_name + ".")

            if 'shot_noise_clust' in config['survey specs']:
                self.shot_noise_clust = \
                    np.array(config['survey specs']
                             ['shot_noise_clust'].split(','))
                try:
                    self.shot_noise_clust = \
                        (self.shot_noise_clust).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'shot_noise_clust = " +
                                    config['survey specs']['shot_noise_clust'] + "' to  " +
                                    "numpy array. Must be adjusted in config file " +
                                    config_name + ".")
            if 'shot_noise_gamma' in config['survey specs']:
                self.shot_noise_gamma = \
                    np.array(config['survey specs']
                             ['shot_noise_gamma'].split(','))
                try:
                    self.shot_noise_gamma = \
                        (self.shot_noise_gamma).astype(float)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'shot_noise_gamma = " +
                                    config['survey specs']['shot_noise_gamma'] + "' to  " +
                                    "numpy array. Must be adjusted in config file " +
                                    config_name + ".")

            if 'tomos_6x2pt_clust' in config['survey specs']:
                self.tomos_6x2pt_clust = \
                    np.array(config['survey specs']
                             ['tomos_6x2pt_clust'].split(','))
                try:
                    self.tomos_6x2pt_clust = \
                        (self.tomos_6x2pt_clust).astype(int)
                except ValueError:
                    raise Exception("ConfigError: Cannot convert string in " +
                                    "[survey specs]: 'tomos_6x2pt_clust = " +
                                    config['survey specs']['tomos_6x2pt_clust'] + "' to " +
                                    "numpy int array. Must be adjusted in config file " +
                                    config_name + ".")
        else:
            self.sigma_eps = np.array([.3])
            print("The ellipticity dispersion of observed galaxies [survey " +
                  "specs]: 'ellipticity_dispersion' is set to 0.3")

        if self.n_eff_clust is None:
            if self.clustering and self.est_clust != 'k_space':
                raise Exception("ConfigError: No effective number density " +
                                "of clustering galaxies is specified. Must be adjusted " +
                                "in config file " + config_name + ", [survey specs]: " +
                                "'n_eff_clust = 1.1, 2.2' (as many entries as there are " +
                                "tomographic clustering bins.")
        if self.n_eff_ggl is None:
            if self.ggl and self.est_ggl == 'k_space':
                raise Exception("ConfigError: No effective number density " +
                                "of galaxy-galaxy lensing is specified. Must be " +
                                "adjusted in config file " + config_name + ", [survey " +
                                "specs]: 'n_eff_clust = 1.1, 2.2' (as many entries as " +
                                "there are tomographic clustering bins.")
        if self.n_eff_lens is None:
            if self.cosmicshear and self.est_shear != 'k_space':
                raise Exception("ConfigError: No effective number density " +
                                "of lensing galaxies is specified. Must be adjusted in " +
                                "config file " + config_name + ", [survey specs]: " +
                                "'n_eff_lens = 1.1, 2.2' (as many entries as there are " +
                                "tomographic lensing bins.")

        self.read_mask_clust = False
        self.read_alm_clust = False
        if (self.clustering and
            self.est_clust != 'k_space' and
                self.est_clust != 'projected_real'):
            if self.ssc:
                if self.alm_file_clust is not None:
                    self.read_alm_clust = True
                    if self.mask_file_clust is not None and \
                       self.survey_area_clust is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_clust]
                        print("Omitting clustering mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_clust_in_deg2' and a " +
                              "seperate [survey specs]: 'alm_file_clust' is " +
                              "given.")
                elif self.mask_file_clust is not None:
                    self.read_mask_clust = True
                else:
                    print("ConfigWarning: No survey modes for the clustering area given."
                          + "Will proceed by assuming a spherical top-hat as a footprint and "
                          + "estimate the survey variance from the theory.")
                    # raise Exception("ConfigError: The survey modes for the " +
                    #                "clustering area are missing. Either provide a " +
                    #                "[survey specs]: 'mask_file_clust' or " +
                    #                "[survey specs]: 'alm_file_clust'.")
            else:
                if self.survey_area_clust is None:
                    if self.mask_file_clust is not None:
                        self.read_mask_clust = True
                    else:
                        raise Exception("ConfigError: The survey area for " +
                                        "the clustering measurements is missing. Either " +
                                        "provide a [survey specs]: 'mask_file_clust' or " +
                                        "[survey specs]: 'survey_area_clust_in_deg2'.")
                else:
                    if self.mask_file_clust is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_clust]
                        print("Omitting clustering mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_clust_in_deg2' is given.")

        self.read_mask_ggl = False
        self.read_alm_ggl = False
        if (self.ggl and
            self.est_ggl != 'k_space' and
                self.est_ggl != 'projected_real'):
            if self.ssc:
                if self.alm_file_ggl is not None:
                    self.read_alm_ggl = True
                    if self.mask_file_ggl is not None and \
                       self.survey_area_ggl is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_ggl]
                        print("Omitting galaxy-galaxy lensing mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_ggl_in_deg2' and a " +
                              "seperate [survey specs]: 'alm_file_ggl' is " +
                              "given.")
                elif self.mask_file_ggl is not None:
                    self.read_mask_ggl = True
                else:
                    print("ConfigWarning: No survey modes for the galaxy-galaxy lensing area given."
                          + "Will proceed by assuming a spherical top-hat as a footprint and "
                          + "estimate the survey variance from the theory.")
                    # raise Exception("ConfigError: The survey modes for the " +
                    #                "galaxy-galaxy lensing area are missing. Either " +
                    #                "provide a [survey specs]: 'mask_file_ggl' or " +
                    #                "[survey specs]: 'alm_file_ggl'.")
            else:
                if self.survey_area_ggl is None:
                    if self.mask_file_ggl is not None:
                        self.read_mask_ggl = True
                    else:
                        raise Exception("ConfigError: The survey area for " +
                                        "the galaxy-galaxy lensing measurements is " +
                                        "missing. Either provide a [survey specs]: " +
                                        "'mask_file_ggl' or [survey specs]: " +
                                        "'survey_area_ggl_in_deg2'.")
                else:
                    if self.mask_file_ggl is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_ggl]
                        print("Omitting galaxy-galaxy lensing mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_ggl_in_deg2' is given.")

        self.read_mask_lens = False
        self.read_alm_lens = False
        if self.cosmicshear and self.est_shear != 'k_space':
            if self.ssc:
                if self.alm_file_lens is not None:
                    self.read_alm_lens = True
                    if self.mask_file_lens is not None and \
                       self.survey_area_lens is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_lens]
                        print("Omitting lensing mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_lensing_in_deg2' and a " +
                              "seperate [survey specs]: 'alm_file_lensing' " +
                              "is given.")
                elif self.mask_file_lens is not None:
                    self.read_mask_lens = True
                else:
                    print("ConfigWarning: No survey modes for the lensing area given."
                          + "Will proceed by assuming a spherical top-hat as a footprint and "
                          + "estimate the survey variance from the theory.")
                    # raise Exception("ConfigError: The survey modes for " +
                    #                "the lensing area are missing. Either provide a " +
                    #                "[survey specs]: 'mask_file_lensing' or " +
                    #                "[survey specs]: 'alm_file_lensing'.")
            else:
                if self.survey_area_lens is None:
                    if self.mask_file_lens is not None:
                        self.read_mask_lens = True
                    else:
                        raise Exception("ConfigError: The survey area for " +
                                        "the lensing measurements is missing. Either " +
                                        "provide a [survey specs]: 'mask_file_lensing' " +
                                        "or [survey specs]: " +
                                        "'survey_area_lensing_in_deg2'.")
                else:
                    if self.mask_file_lens is not None:
                        fn = [path.join(self.mask_dir, mfile)
                              for mfile in self.mask_file_lens]
                        print("Omitting lensing mask file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'survey_area_lensing_in_deg2' is " +
                              "given.")

        # logical order of how it is called later on gg -> mm -> gm
        if self.read_mask_clust and self.read_mask_lens:
            if self.mask_file_clust == self.mask_file_lens:
                self.read_mask_lens = False
        if self.read_mask_clust and self.read_mask_ggl:
            if self.mask_file_clust == self.mask_file_ggl:
                self.read_mask_ggl = False
        if self.read_mask_lens and self.read_mask_ggl:
            if self.mask_file_lens == self.mask_file_ggl:
                self.read_mask_ggl = False
        if self.read_alm_clust and self.read_alm_lens:
            if self.alm_file_clust == self.alm_file_lens:
                self.read_alm_lens = False
        if self.read_alm_clust and self.read_alm_ggl:
            if self.alm_file_clust == self.alm_file_ggl:
                self.read_alm_ggl = False
        if self.read_alm_lens and self.read_alm_ggl:
            if self.alm_file_lens == self.alm_file_ggl:
                self.read_alm_ggl = False

        # onto the cross terms
        if self.ssc and self.cross_terms:
            self.read_mask_clust_lens = False
            self.read_alm_clust_lens = False
            self.read_mask_clust_ggl = False
            self.read_alm_clust_ggl = False
            self.read_mask_lens_ggl = False
            self.read_alm_lens_ggl = False
            if (self.clustering and
                self.est_clust != 'k_space' and
                    self.est_clust != 'projected_real'):
                if (self.cosmicshear and
                        self.est_shear != 'k_space'):
                    if self.alm_file_clust_lens is not None and \
                       self.alm_file_clust is not None and \
                       self.alm_file_clust != self.alm_file_lens:
                        self.read_alm_clust_lens = True
                    elif self.mask_file_clust is not None and \
                            self.mask_file_clust != self.mask_file_lens:
                        self.read_mask_clust_lens = True
                    if not self.read_alm_clust_lens and \
                            self.alm_file_clust_lens is not None:
                        fn = [path.join(self.mask_dir, afile)
                              for afile in self.alm_file_clust_lens]
                        print("Omitting clustering x lensing a_lm file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'alm_file_clust' and " +
                              "'alm_file_lensing' are the same.")
                if (self.ggl and
                    self.est_ggl != 'k_space' and
                        self.est_ggl != 'projected_real'):
                    if self.alm_file_clust_ggl is not None and \
                       self.alm_file_clust != self.alm_file_ggl:
                        self.read_alm_clust_ggl = True
                    elif self.mask_file_clust is not None and \
                            self.mask_file_clust != self.mask_file_ggl:
                        self.read_mask_clust_ggl = True
                    if not self.read_alm_clust_ggl and \
                            self.alm_file_clust_ggl is not None:
                        fn = [path.join(self.mask_dir, afile)
                              for afile in self.alm_file_clust_ggl]
                        print("Omitting clustering x galaxy-galaxy " +
                              "lensing a_lm file " +
                              ', '.join(map(str, fn)) + " since [survey " +
                              "specs]: 'alm_file_clust' and " +
                              "'alm_file_ggl' are the same.")
            if (self.cosmicshear and
                    self.est_shear != 'k_space'):
                if (self.ggl and
                    self.est_ggl != 'k_space' and
                        self.est_ggl != 'projected_real'):
                    if self.alm_file_lens is not None and \
                       self.alm_file_lens != self.alm_file_ggl:
                        self.read_alm_lens_ggl = True
                    elif self.mask_file_lens is not None and \
                            self.mask_file_lens != self.mask_file_ggl:
                        self.read_mask_lens_ggl = True
                    if not self.read_alm_lens_ggl and \
                            self.alm_file_lens_ggl is not None:
                        fn = [path.join(self.mask_dir, afile)
                              for afile in self.alm_file_lens_ggl]
                        print("Omitting lensing x galaxy-galaxy lensing" +
                              "a_lm file " + ', '.join(map(str, fn)) +
                              " since [survey specs]: " +
                              "'alm_file_lensing' and 'alm_file_ggl' " +
                              "are the same.")

        return True

    def __read_in_hm_prec(self,
                          config):
        """
        Reads in information needed to call the mass function from the
        hmf module (by Steven Murray). Every value that is not specified 
        gets a fall-back value which is reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """

        if 'halomodel evaluation' in config:
            if 'M_bins' in config['halomodel evaluation']:
                self.M_bins = int(config['halomodel evaluation']['M_bins'])
            else:
                self.M_bins = 500
                print("The number of mass bins [halomodel evaluation]: " +
                      "'M_bins' is set to 500.")

            if 'log10M_min' in config['halomodel evaluation']:
                self.log10M_min = float(
                    config['halomodel evaluation']['log10M_min'])
            else:
                self.log10M_min = 9
                print("The minimum logarithmic10 mass [halomodel " +
                      "evaluation]: 'log10M_min' is set to 9.")

            if 'log10M_max' in config['halomodel evaluation']:
                self.log10M_max = float(
                    config['halomodel evaluation']['log10M_max'])
            else:
                self.log10M_max = 18
                print("The maximum logarithmic10 mass [halomodel " +
                      "evaluation]: 'log10M_max' is set to 18.")

            if 'hmf_model' in config['halomodel evaluation']:
                self.hmf_model = config['halomodel evaluation']['hmf_model']
            else:
                self.hmf_model = 'Tinker10'
                print("The parameter 'hmf_model' for the hmf module " +
                      "[halomodel evaluation]: 'hmf_model' is set to " +
                      "Tinker10.")

            if 'mdef_model' in config['halomodel evaluation']:
                self.mdef_model = config['halomodel evaluation']['mdef_model']
            else:
                self.mdef_model = 'SOMean'
                print("The parameter 'mdef_model' for the hmf module " +
                      "[halomodel evaluation]: 'mdef_model' is set to SOMean.")

            if 'mdef_params' in config['halomodel evaluation']:
                self.mdef_params = \
                    config['halomodel evaluation']['mdef_params'].replace(
                        " ", "").split(',')
                self.mdef_params = \
                    dict(zip(self.mdef_params[::2], self.mdef_params[1::2]))
                for key in self.mdef_params.keys():
                    self.mdef_params[key] = float(self.mdef_params[key])

            else:
                self.mdef_params = {'overdensity': 200}
                print("The parameter 'mdef_params' for the hmf module " +
                      "[halomodel evaluation]: 'mdef_params' is set to " +
                      "overdensity: 200.")

            if 'disable_mass_conversion' in config['halomodel evaluation']:
                self.disable_mass_conversion = config['halomodel evaluation'].getboolean(
                    'disable_mass_conversion')
            else:
                self.disable_mass_conversion = True
                print("The parameter 'disable_mass_conversion' for the hmf " +
                      "module [halomodel evaluation]: " +
                      "'disable_mass_conversion' is set to True.")

            if 'delta_c' in config['halomodel evaluation']:
                self.delta_c = float(config['halomodel evaluation']['delta_c'])
            else:
                self.delta_c = 1.686
                print("The parameter 'delta_c' for the hmf module " +
                      "[halomodel evaluation]: 'delta_c' is set to 1.686.")

            if 'transfer_model' in config['halomodel evaluation']:
                self.transfer_model = \
                    config['halomodel evaluation']['transfer_model']
            else:
                self.transfer_model = 'CAMB'
                print("The parameter 'transfer_model' for the hmf module " +
                      "[halomodel evaluation]: 'transfer_model' is set to EH.")
            if 'small_k_damping_for1h' in config['halomodel evaluation']:
                self.small_k_damping = \
                    config['halomodel evaluation']['small_k_damping_for1h']
            else:
                if config['powspec evaluation']['non_linear_model'] != \
                        'mead2015':
                    self.small_k_damping = 'damped'
                    print("The suppression of power for the 1-halo term on " +
                          "large scales / small wavenumbers [halomodel " +
                          "evaluation]: 'small_k_damping_for1h' is set to " +
                          "'damped'.")
        else:
            self.M_bins = 500
            print("The number of mass bins [halomodel evaluation]: 'M_bins' " +
                  "is set to 500.")
            self.log10M_min = 9
            print("The minimum logarithmic10 mass [halomodel evaluation]: " +
                  "'log10M_min' is set to 9.")
            self.log10M_max = 18
            print("The maximum logarithmic10 mass [halomodel evaluation]: " +
                  "'log10M_max' is set to 18.")
            self.hmf_model = 'Tinker10'
            print("The parameter 'hmf_model' for the hmf module [halomodel " +
                  "evaluation]: 'hmf_model' is set to Tinker10.")
            self.mdef_model = 'SOMean'
            print("The parameter 'mdef_model' for the hmf module [halomodel " +
                  "evaluation]: 'mdef_model' is set to SOMean.")
            self.mdef_params = {'overdensity': 200}
            print("The parameter 'mdef_params' for the hmf module " +
                  "[halomodel evaluation]: 'mdef_params' is set to " +
                  "overdensity: 200.")
            self.disable_mass_conversion = True
            print("The parameter 'disable_mass_conversion' for the hmf " +
                  "module [halomodel evaluation]: 'disable_mass_conversion' " +
                  "is set to True.")
            self.delta_c = 1.686
            print("The parameter 'delta_c' for the hmf module [halomodel " +
                  "evaluation]: 'delta_c' is set to 1.686.")
            self.transfer_model = 'EH'
            print("The parameter 'transfer_model' for the hmf module " +
                  "[halomodel evaluation]: 'transfer_model' is set to EH.")
            self.small_k_damping = 'damped'
            print("The suppression of power for the 1-halo term on large " +
                  "scales / small wavenumbers [halomodel evaluation]: " +
                  "'small_k_damping_for1h' is set to 'damped'.")

        if self.bias_model != self.hmf_model:
            print("The bias model [bias]: 'model' is set to " +
                  self.bias_model + " while the mass function [halomodel " +
                  "evaluation]: 'hmf_model' is adjusted to " + self.hmf_model +
                  ". This might cause a bias later on.")

        return True

    def __read_in_powspec_prec(self,
                               config):
        """
        Reads in information needed to calculate power spectra. Every 
        value that is not specified gets a fall-back value which is 
        reported to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """
        estimators = [self.est_shear, self.est_ggl, self.est_clust]

        if 'powspec evaluation' in config:
            if 'non_linear_model' in config['powspec evaluation']:
                self.nl_model = \
                    config['powspec evaluation']['non_linear_model']
            else:
                self.nl_model = 'mead2015'
                print("The model for the nonlinear part of the " +
                      "matter-matter power spectrum [powspec evaluation]: " +
                      "'non_linear_model' is set to mead2015.")
            if 'log10k_bins' in config['powspec evaluation']:
                self.log10k_bins = \
                    int(config['powspec evaluation']['log10k_bins'])
            else:
                self.log10k_bins = 200
                if 'k_space' in estimators or \
                   'projected_real' in estimators:
                    print("The number of wavenumber bins [powspec " +
                          "evaluation]: 'log10k_bins' is set to 200.")
            if 'log10k_min' in config['powspec evaluation']:
                self.log10k_min = \
                    float(config['powspec evaluation']['log10k_min'])
            else:
                self.log10k_min = -5
                if 'k_space' in estimators or \
                   'projected_real' in estimators:
                    print("The minimum logarithmic wavenumber [powspec " +
                          "evaluation]: 'log10k_min' is set to -5.")
            if 'log10k_max' in config['powspec evaluation']:
                self.log10k_max = \
                    float(config['powspec evaluation']['log10k_max'])
            else:
                self.log10k_max = 4
                if 'k_space' in estimators or \
                   'projected_real' in estimators:
                    print("The maximum logarithmic wavenumber [powspec " +
                          "evaluation]: 'log10k_max' is set to 4.")
            if self.nl_model == 'mead2020_feedback':
                if 'HMCode_logT_AGN' in config['powspec evaluation']:
                    self.HMCode_logT_AGN = float(config['powspec evaluation']['HMCode_logT_AGN'])
                else:
                    self.HMCode_logT_AGN = 7.8
            if self.nl_model == 'mead2015' or self.nl_model == 'mead2016':
                if 'HMCode_A_baryon' in config['powspec evaluation']:
                    self.HMCode_A_baryon = float(config['powspec evaluation']['HMCode_A_baryon'])
                else:
                    self.HMCode_A_baryon = 3.13
                if 'HMCode_eta_baryon' in config['powspec evaluation']:
                    self.HMCode_eta_baryon = float(config['powspec evaluation']['HMCode_eta_baryon'])
                else:
                    self.HMCode_eta_baryon = 0.603
            


        else:
            self.nl_model = 'mead2015'
            self.HMCode_A_baryon = 3.13
            self.HMCode_eta_baryon = 0.603
            print("The model for the nonlinear part of the matter-matter " +
                  "power spectrum [powspec evaluation]: 'non_linear_model' " +
                  "is set to mead2015.")
            self.log10k_bins = 200
            if 'k_space' in estimators or \
               'projected_real' in estimators:
                print("The number of wavenumber bins [powspec evaluation]: " +
                      "'log10k_bins' is set to 200.")
            self.log10k_min = -5
            if 'k_space' in estimators or \
               'projected_real' in estimators:
                print("The minimum logarithmic wavenumber [powspec " +
                      "evaluation]: 'log10k_min' is set to -5.")
            self.log10k_max = 4
            if 'k_space' in estimators or \
               'projected_real' in estimators:
                print("The maximum logarithmic wavenumber [powspec " +
                      "evaluation]: 'log10k_max' is set to 4.")

        return True

    def __read_in_trispec_prec(self,
                               config):
        """
        Reads in information needed to calculate trispectra. Every value 
        that is not specified gets a fall-back value which is reported 
        to the user.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """
        if not self.nongauss:
            self.tri_logk_bins = 60
            self.tri_logk_min = -4
            self.tri_logk_max = 2
            return True

        if 'trispec evaluation' in config:
            if 'matter_klim' in config['trispec evaluation']:
                self.matter_klim = \
                    float(config['trispec evaluation']['matter_klim'])
            else:
                self.matter_klim = 1e-3
                print("The minimum wavenumber that is considered to " +
                      "calculate the trispectrum [trispec evaluation]: " +
                      "'matter_klim' is set to 1e-3.")

            if 'matter_mulim' in config['trispec evaluation']:
                self.matter_mulim = \
                    float(config['trispec evaluation']['matter_mulim'])
            else:
                self.matter_mulim = 1e-3
                print("The minimum angle between to k-vectors that is " +
                      "considered to calculate the trispectrum [trispec " +
                      "evaluation]: 'matter_mulim' is set to 1e-3.")

            if 'log10k_bins' in config['trispec evaluation']:
                self.tri_logk_bins = \
                    int(config['trispec evaluation']['log10k_bins'])
            else:
                self.tri_logk_bins = 60
                print("The number of wavenumber bins [trispec evaluation]: " +
                      "'log10k_bins' is set to 100.")

            if 'log10k_min' in config['trispec evaluation']:
                self.tri_logk_min = \
                    float(config['trispec evaluation']['log10k_min'])
                if self.tri_logk_min < self.log10k_min:
                    raise Exception("ConfigError: The trispectrum is " +
                                    "evaluated on smaller wavenumbers " +
                                    "(log10 k_tri_min = " + str(self.tri_logk_min) + ") " +
                                    "than the halo model integrals (log10 k_min = " +
                                    str(self.log10k_min) + ") this can lead to " +
                                    "unexpected behaviour.")
            else:
                self.tri_logk_min = -4
                if self.tri_logk_min < self.log10k_min:
                    self.tri_logk_min = self.log10k_min
                print("The minimum logarithmic10 wavenumber [trispec " +
                      "evaluation]: 'log10k_min' is set to " +
                      str(self.tri_logk_min) + ".")

            if 'log10k_max' in config['trispec evaluation']:
                self.tri_logk_max = \
                    float(config['trispec evaluation']['log10k_max'])
                if self.tri_logk_max > self.log10k_max:
                    raise Exception("ConfigError: The trispectrum is " +
                                    "evaluated on larger wavenumbers (log10 k_tri_max = " +
                                    str(self.tri_logk_max) + ") than the halo model " +
                                    "integrals (log10 k_max =" +
                                    str(self.log10k_max) + ") this can lead " +
                                    "to unexpected behaviour.")
            else:
                self.tri_logk_max = 2
                if self.tri_logk_max > self.log10k_max:
                    self.tri_logk_max = self.log10k_max
                print("The maximum logarithmic10 wavenumber [trispec " +
                      "evaluation]: 'log10k_max' is set to " +
                      str(self.tri_logk_max) + ".")
            if 'small_k_damping_for1h' in config['trispec evaluation']:
                self.tri_small_k_damping = \
                    config['trispec evaluation']['small_k_damping_for1h']
            else:
                self.tri_small_k_damping = 'damped'
                print("The suppression of power for the 1-halo term on " +
                      "large scales / small wavenumbers [trispectra " +
                      "evaluation]: 'small_k_damping_for1h' is set to " +
                      "'damped'.")
            if 'lower_calc_limit' in config['trispec evaluation']:
                self.lower_calc_limit = \
                    float(config['trispec evaluation']['lower_calc_limit'])
            else:
                self.lower_calc_limit = 1e-200
        else:
            self.matter_klim = 1e-3
            print("The minimum wavenumber that is considered to calculate " +
                  "the trispectrum [trispec evaluation]: 'matter_klim' is " +
                  "set to 1e-3.")
            self.matter_mulim = 1e-3
            print("The minimum angle between to k-vectors that is " +
                  "considered to calculate the trispectrum [trispec " +
                  "evaluation]: 'matter_mulim' is set to 1e-3.")
            self.tri_logk_bins = 60
            print("The number of wavenumber bins [trispec evaluation]: " +
                  "'log10k_bins' is set to " +
                  str(self.tri_logk_bins) + ".")
            self.tri_logk_min = -4
            if self.tri_logk_min < self.log10k_min:
                self.tri_logk_min = self.log10k_min
            print("The minimum logarithmic10 wavenumber [trispec " +
                  "evaluation]: 'log10k_min' is set to " +
                  str(self.tri_logk_min) + ".")
            self.tri_logk_max = 2
            if self.tri_logk_max > self.log10k_max:
                self.tri_logk_max = self.log10k_max
            print("The maximum logarithmic10 wavenumber [trispec " +
                  "evaluation]: 'log10k_max' is set to " +
                  str(self.tri_logk_max) + ".")
            self.tri_small_k_damping = 'damped'
            print("The suppression of power for the 1-halo term on large " +
                  "scales / small wavenumbers [trispec evaluation]: " +
                  "'small_k_damping_for1h' is set to 'damped'.")
            self.lower_calc_limit = 1e-200

        return True

    def __read_in_misc(self,
                       config):
        """
        Reads in miscellaneous information.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """

        if 'misc' in config:
            if 'num_cores' in config['misc']:
                self.num_cores = int(config['misc']['num_cores'])
            else:
                self.num_cores = 8
        else:
            self.num_cores = 8

        return True

    def __zip_to_dicts(self):
        """
        This method stores all private variables from this class, that 
        are needed to calculate the covariance, into dictionaries. It 
        creates a second dictionary '_abr' if the first can contain a 
        variable that is None or is an array. The second dictionary is 
        only needed to write the parameters back into a save_configs 
        file.

        """
        keys = ['gauss', 'split_gauss', 'nongauss', 'ssc']
        values = [self.gauss, self.split_gauss, self.nongauss, self.ssc]
        self.covterms = dict(zip(keys, values))

        keys = ['cosmic_shear', 'est_shear', 'ggl', 'est_ggl', 'clustering',
                'est_clust', 'cross_terms', 'clustering_z', 'unbiased_clustering', 'csmf', 'csmf_log10M_bins', "is_cell", "csmf_diagonal"]
        values = [self.cosmicshear, self.est_shear, self.ggl, self.est_ggl,
                  self.clustering, self.est_clust, self.cross_terms, self.clustering_z, self.unbiased_clustering,
                  self.cstellar_mf, self.csmf_log10M_bins, False, self.csmf_diagonal]
        self.observables = dict(zip(keys, values))
        self.observables_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        keys = ['directory', 'file', 'style', 'corrmatrix_plot',
                'save_configs', 'save_Cells', 'save_trispectra', 'save_alms', 'use_tex', 'list_style_spatial_first', 'save_as_binary']
        values = [self.output_dir, self.output_file, self.output_style,
                  self.make_plot, self.save_configs, self.save_Cells,
                  self.save_trispecs, self.save_alms, self.use_tex, self.list_style_spatial_first, self.save_as_binary]
        self.output_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        self.output_abr['file'] = \
            ', '.join(map(str, self.output_file))
        self.output_abr['style'] = \
            ', '.join(map(str, self.output_style))
        if self.make_plot and self.output_dir is not None:
            self.make_plot = path.join(self.output_dir, self.make_plot)
        if self.output_file and self.output_dir is not None:
            self.output_file = [path.join(self.output_dir, file)
                                for file in self.output_file]
        if self.save_Cells and self.output_dir is not None:
            self.save_Cells = path.join(self.output_dir, self.save_Cells)
        if self.save_trispecs and self.output_dir is not None:
            self.save_trispecs = path.join(self.output_dir, self.save_trispecs)
        keys = ['file', 'style', 'make_plot', 'Cell', 'trispec', 'save_alms', 'use_tex', 'list_style_spatial_first', 'save_as_binary']
        values = [self.output_file, self.output_style, self.make_plot,
                  self.save_Cells, self.save_trispecs, self.save_alms, self.use_tex, self.list_style_spatial_first,self.save_as_binary]
        self.output = dict(zip(keys, values))
        keys = ['limber','nglimber','pixelised_cell','pixel_Nside', 'ell_min', 'ell_max', 'ell_bins', 'ell_type', 'delta_z',
                'integration_steps', 'nz_polyorder', 'tri_delta_z', 'mult_shear_bias', 'n_spec',
                'ell_spec_min', 'ell_spec_max', 'ell_spec_bins', 'ell_spec_type', 'ell_photo_min', 'ell_photo_max', 'ell_photo_bins', 'ell_photo_type',
                'ell_min_clustering', 'ell_max_clustering', 'ell_bins_clustering', 'ell_type_clustering',
                'ell_min_lensing', 'ell_max_lensing', 'ell_bins_lensing', 'ell_type_lensing']
        values = [self.limber, self.nglimber, self.pixelised_cell, self.pixel_Nside, self.ell_min, self.ell_max, self.ell_bins, self.ell_type,
                  self.delta_z, self.integration_steps, self.nz_polyorder,
                  self.tri_delta_z, self.multiplicative_shear_bias_uncertainty, self.n_spec,
                  self.ell_spec_min, self.ell_spec_max, self.ell_spec_bins, self.ell_spec_type,
                  self.ell_photo_min, self.ell_photo_max, self.ell_photo_bins, self.ell_photo_type,
                  self.ell_min_clustering, self.ell_max_clustering, self.ell_bins_clustering, self.ell_type_clustering,
                  self.ell_min_lensing, self.ell_max_lensing, self.ell_bins_lensing, self.ell_type_lensing]
        self.covELLspace_settings = dict(zip(keys, values))
        self.covELLspace_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        if self.cosmicshear or self.ggl:
            self.covELLspace_settings_abr['mult_shear_bias'] = \
                    ', '.join(map(str, self.multiplicative_shear_bias_uncertainty))

        keys = ['theta_min', 'theta_max', 'theta_bins', 'theta_type',
                'theta_min_clustering', 'theta_max_clustering', 'theta_bins_clustering', 'theta_type_clustering',
                'theta_min_lensing', 'theta_max_lensing', 'theta_bins_lensing', 'theta_type_lensing', 'theta_list', 'xi_pp',
                'xi_pm', 'xi_mm', 'theta_acc', 'integration_intervals', 'mix_term_file_path_catalog',
                'mix_term_col_name_weight',
                'mix_term_col_name_pos1', 'mix_term_col_name_pos2', 'mix_term_col_name_zbin',
                'mix_term_isspherical', 'mix_term_target_patchsize', 'mix_term_do_overlap',
                'mix_term_do_mix_for', 'mix_term_nbins_phi', 'mix_term_nmax',
                'mix_term_do_ec', 'mix_term_subsample', 'mix_term_nsubr', 'mix_term_file_path_save_triplets',
                'mix_term_file_path_load_triplets']
        values = [self.theta_min, self.theta_max, self.theta_bins, self.theta_type,
                  self.theta_min_clustering, self.theta_max_clustering, self.theta_bins_clustering, self.theta_type_clustering,
                  self.theta_min_lensing, self.theta_max_lensing, self.theta_bins_lensing, self.theta_type_lensing,
                  self.theta_list, self.xi_pp, self.xi_pm, self.xi_mm, self.theta_acc,
                  self.integration_intervals, self.mix_term_file_path_catalog, self.mix_term_col_name_weight,
                  self.mix_term_col_name_pos1, self.mix_term_col_name_pos2, self.mix_term_col_name_zbin,
                  self.mix_term_isspherical, self.mix_term_target_patchsize, self.mix_term_do_overlap,
                  self.mix_term_do_mix_for, self.mix_term_nbins_phi, self.mix_term_nmax,
                  self.mix_term_do_ec, self.mix_term_subsample, self.mix_term_nsubr, 
                  self.mix_term_file_path_save_triplets, self.mix_term_file_path_load_triplets]
        self.covTHETAspace_settings = dict(zip(keys, values))
        self.covTHETAspace_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        keys = ['En_modes', 'theta_min', 'theta_max', 'En_acc', 'Wn_style',
                'En_modes_clustering', 'theta_min_clustering', 'theta_max_clustering',
                'En_modes_lensing', 'theta_min_lensing', 'theta_max_lensing',
                'Wn_acc', 'dimensionless_cosebi']
        values = [self.En_modes, self.theta_min_cosebi, self.theta_max_cosebi, self.En_acc, self.Wn_style,
                  self.En_modes_clustering, self.theta_min_cosebi_clustering, self.theta_max_cosebi_clustering,
                  self.En_modes_lensing, self.theta_min_cosebi_lensing, self.theta_max_cosebi_lensing,
                  self.Wn_acc, self.dimensionless_cosebi]
        self.covCOSEBI_settings = dict(zip(keys, values))
        keys = ['En_modes', 'theta_min', 'theta_max', 'En_accuracy', 'Wn_style',
                'En_modes_clustering', 'theta_min_clustering', 'theta_max_clustering',
                'En_modes_lensing', 'theta_min_lensing', 'theta_max_lensing',
                'Wn_accuracy', 'dimensionless_cosebi']
        self.covCOESBI_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        
        keys = ['apodisation_log_width_clustering', 'theta_lo_clustering', 'theta_up_clustering', 'ell_min_clustering',
                'ell_max_clustering', 'ell_bins_clustering', 'ell_type_clustering',
                'apodisation_log_width_lensing', 'theta_lo_lensing', 'theta_up_lensing', 'ell_min_lensing',
                'ell_max_lensing', 'ell_bins_lensing', 'ell_type_lensing',
                'bandpower_accuracy', 'theta_binning']
        values = [self.apodisation_log_width_clustering, self.theta_lo_clustering, self.theta_up_clustering,
                  self.bp_ell_min_clustering, self.bp_ell_max_clustering, self.bp_ell_bins_clustering, self.bp_ell_type_clustering,
                  self.apodisation_log_width_lensing, self.theta_lo_lensing, self.theta_up_lensing,
                  self.bp_ell_min_lensing, self.bp_ell_max_lensing, self.bp_ell_bins_lensing, self.bp_ell_type_lensing,
                  self.bandpower_accuracy, self.theta_binning]
        self.covbandpowers_settings = dict(zip(keys,values))
        self.covbandpowers_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        keys = ['do_arbitrary_summary', 'oscillations_straddle', 'arbitrary_accuracy']
        values = [self.do_arbitrary_obs, self.oscillations_straddle, self.arbitrary_accuracy]
        self.arbitrary_summary_settings = dict(zip(keys,values))
        self.arbitrary_summary_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        keys = ['radius_min', 'radius_max',
                'radius_bins', 'radius_type',
                'mean_redshift', 'clust_length']
        values = [self.projected_radius_min, self.projected_radius_max,
                  self.projected_radius_bins, self.projected_radius_type,
                  self.mean_redshift, self.projection_length_clustering]
        self.covRspace_settings = dict(zip(keys, values))
        keys = ['projected_radius_min', 'projected_radius_max',
                'projected_radius_bins', 'projected_radius_type',
                'mean_redshift', 'projection_length_clustering']
        self.covRspace_settings_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        if self.mean_redshift is not None:
            self.covRspace_settings_abr['mean_redshift'] = \
                ', '.join(map(str, self.mean_redshift))

        keys = ['sigma8', 'A_s', 'h', 'omega_m', 'omega_b', 'omega_de', 'w0',
                'wa', 'ns', 'neff', 'm_nu', 'Tcmb0']
        values = [self.sigma8, self.As, self.h, self.omegam, self.omegab, self.omegade,
                  self.w0, self.wa,  self.ns, self.neff, self.mnu, self.Tcmb0]
        self.cosmo = dict(zip(keys, values))
        self.cosmo_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        

        keys = ['model', 'bias_2h', 'Mc_relation_cen',
                'Mc_relation_sat', 'norm_Mc_relation_sat', 'norm_Mc_relation_cen', 'log10mass_bins', 'has_csmf']
        values = [self.bias_model, self.bias_2h, self.Mc_relation_cen,
                  self.Mc_relation_sat, self.norm_Mc_relation_sat, self.norm_Mc_relation_cen, self.logmass_bins, self.cstellar_mf]
        self.bias_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        if self.logmass_bins is not None:
            self.bias_abr['logmass_bins'] = \
                ', '.join(map(str, self.logmass_bins))
        else:
            self.logmass_bins = np.array([0, 0])
            values = [self.bias_model, self.bias_2h, self.Mc_relation_cen,
                      self.Mc_relation_sat, self.norm_Mc_relation_sat, self.norm_Mc_relation_cen, self.logmass_bins, self.cstellar_mf]
        keys = ['model', 'bias_2h', 'Mc_relation_cen',
                'Mc_relation_sat', 'norm_Mc_relation_sat', 'norm_Mc_relation_cen', 'logmass_bins', 'has_csmf']
        self.bias = dict(zip(keys, values))
        keys = ['A_IA', 'eta_IA', 'z_pivot_IA']
        values = [self.A_IA, self.eta_IA, self.z_pivot_IA]
        self.intrinsic_alignments = dict(zip(keys, values))
        self.intrinsic_alignments_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})

        keys = ['model_mor_cen', 'model_mor_sat', 'dpow_logM0_cen',
                'dpow_logM1_cen', 'dpow_a_cen', 'dpow_b_cen', 'dpow_norm_cen',
                'dpow_logM0_sat', 'dpow_logM1_sat', 'dpow_a_sat', 'dpow_b_sat',
                'dpow_norm_sat', 'model_scatter_cen', 'model_scatter_sat',
                'logn_sigma_c_cen', 'logn_sigma_c_sat', 'modsch_logMref_cen',
                'modsch_alpha_s_cen', 'modsch_b_cen', 'modsch_logMref_sat',
                'modsch_alpha_s_sat', 'modsch_b_sat']
        values = [self.hod_model_mor_cen, self.hod_model_mor_sat,
                  self.dpow_logM0_cen, self.dpow_logM1_cen, self.dpow_a_cen,
                  self.dpow_b_cen, self.dpow_norm_cen, self.dpow_logM0_sat,
                  self.dpow_logM1_sat, self.dpow_a_sat, self.dpow_b_sat,
                  self.dpow_norm_sat, self.hod_model_scatter_cen,
                  self.hod_model_scatter_sat, self.logn_sigma_c_cen,
                  self.logn_sigma_c_sat, self.modsch_logMref_cen,
                  self.modsch_alpha_s_cen, self.modsch_b_cen,
                  self.modsch_logMref_sat, self.modsch_alpha_s_sat,
                  self.modsch_b_sat]
        self.hod.update({k: v for k, v in zip(keys, values) if v is not None})
        values = [self.hod_model_mor_cen[5:], self.hod_model_mor_sat[5:],
                  self.dpow_logM0_cen, self.dpow_logM1_cen, self.dpow_a_cen,
                  self.dpow_b_cen, self.dpow_norm_cen, self.dpow_logM0_sat,
                  self.dpow_logM1_sat, self.dpow_a_sat, self.dpow_b_sat,
                  self.dpow_norm_sat, self.hod_model_scatter_cen[5:],
                  self.hod_model_scatter_sat[5:], self.logn_sigma_c_cen,
                  self.logn_sigma_c_sat, self.modsch_logMref_cen,
                  self.modsch_alpha_s_cen, self.modsch_b_cen,
                  self.modsch_logMref_sat, self.modsch_alpha_s_sat,
                  self.modsch_b_sat]
        self.hod_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        if self.modsch_b_cen is not None:
            self.hod_abr['modsch_b_cen'] = \
                ', '.join(map(str, self.modsch_b_cen))
        if self.modsch_b_sat is not None:
            self.hod_abr['modsch_b_sat'] = \
                ', '.join(map(str, self.modsch_b_sat))

        keys = ['mask_directory', 'mask_file_clust', 'alm_file_clust',
                'survey_area_clust_in_deg2', 'n_eff_clust', 'shot_noise_clust',
                'tomos_6x2pt_clust', 'mask_file_ggl', 'alm_file_ggl',
                'survey_area_ggl_in_deg2', 'n_eff_ggl', 'mask_file_lensing',
                'alm_file_lensing', 'survey_area_lensing_in_deg2',
                'n_eff_lensing', 'ellipticity_dispersion', 'shot_noise_gamma',
                'alm_file_clust_lensing', 'alm_file_clust_ggl',
                'alm_file_lensing_ggl']
        values = [self.mask_dir, self.mask_file_clust, self.alm_file_clust,
                  self.survey_area_clust, self.n_eff_clust,
                  self.shot_noise_clust, self.tomos_6x2pt_clust,
                  self.mask_file_ggl, self.alm_file_ggl, self.survey_area_ggl,
                  self.n_eff_ggl, self.mask_file_lens, self.alm_file_lens,
                  self.survey_area_lens, self.n_eff_lens, self.sigma_eps,
                  self.shot_noise_gamma, self.alm_file_clust_lens,
                  self.alm_file_clust_ggl, self.alm_file_lens_ggl]
        self.survey_params_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        if self.mask_file_clust is not None:
            self.survey_params_abr['mask_file_clust'] = \
                ', '.join(map(str, self.mask_file_clust))
        if self.alm_file_clust is not None:
            self.survey_params_abr['alm_file_clust'] = \
                ', '.join(map(str, self.alm_file_clust))
        if self.survey_area_clust is not None:
            self.survey_params_abr['survey_area_clust_in_deg2'] = \
                ', '.join(map(str, self.survey_area_clust))
        if self.n_eff_clust is not None:
            self.survey_params_abr['n_eff_clust'] = \
                ', '.join(map(str, self.n_eff_clust))
        if self.shot_noise_clust is not None:
            self.survey_params_abr['shot_noise_clust'] = \
                ', '.join(map(str, self.shot_noise_clust))
        if self.tomos_6x2pt_clust is not None:
            self.survey_params_abr['tomos_6x2pt_clust'] = \
                ', '.join(map(str, self.tomos_6x2pt_clust))
        if self.mask_file_ggl is not None:
            self.survey_params_abr['mask_file_ggl'] = \
                ', '.join(map(str, self.mask_file_ggl))
        if self.alm_file_ggl is not None:
            self.survey_params_abr['alm_file_ggl'] = \
                ', '.join(map(str, self.alm_file_ggl))
        if self.survey_area_ggl is not None:
            self.survey_params_abr['survey_area_ggl_in_deg2'] = \
                ', '.join(map(str, self.survey_area_ggl))
        if self.n_eff_ggl is not None:
            self.survey_params_abr['n_eff_ggl'] = \
                ', '.join(map(str, self.n_eff_ggl))
        if self.mask_file_lens is not None:
            self.survey_params_abr['mask_file_lensing'] = \
                ', '.join(map(str, self.mask_file_lens))
        if self.alm_file_lens is not None:
            self.survey_params_abr['alm_file_lensing'] = \
                ', '.join(map(str, self.alm_file_lens))
        if self.survey_area_lens is not None:
            self.survey_params_abr['survey_area_lensing_in_deg2'] = \
                ', '.join(map(str, self.survey_area_lens))
        if self.n_eff_lens is not None:
            self.survey_params_abr['n_eff_lensing'] = \
                ', '.join(map(str, self.n_eff_lens))
        self.survey_params_abr['ellipticity_dispersion'] = \
            ', '.join(map(str, self.sigma_eps))
        if self.shot_noise_gamma is not None:
            self.survey_params_abr['shot_noise_gamma'] = \
                ', '.join(map(str, self.shot_noise_gamma))
        if self.alm_file_clust_lens is not None:
            self.survey_params_abr['alm_file_clust_lensing'] = \
                ', '.join(map(str, self.alm_file_clust_lens))
        if self.alm_file_clust_ggl is not None:
            self.survey_params_abr['alm_file_clust_ggl'] = \
                ', '.join(map(str, self.alm_file_clust_ggl))
        if self.alm_file_lens_ggl is not None:
            self.survey_params_abr['alm_file_lensing_ggl'] = \
                ', '.join(map(str, self.alm_file_lens_ggl))
        keys = ['survey_area_clust', 'mask_file_clust_clust',
                'read_mask_clust_clust', 'alm_file_clust_clust',
                'read_alm_clust_clust', 'n_eff_clust', 'shot_noise_clust',
                'tomos_6x2pt_clust', 'survey_area_ggl', 'mask_file_ggl_ggl',
                'read_mask_ggl_ggl', 'alm_file_ggl_ggl', 'read_alm_ggl_ggl',
                'n_eff_ggl', 'survey_area_lens', 'mask_file_lens_lens',
                'read_mask_lens_lens', 'alm_file_lens_lens',
                'read_alm_lens_lens', 'n_eff_lens', 'ellipticity_dispersion',
                'shot_noise_gamma', 'alm_file_clust_lens',
                'read_mask_clust_lens', 'read_alm_clust_lens',
                'alm_file_clust_ggl', 'read_mask_clust_ggl',
                'read_alm_clust_ggl', 'alm_file_lens_ggl',
                'read_mask_lens_ggl', 'read_alm_lens_ggl', 'save_alms', 'use_tex']
        if self.mask_file_clust is not None:
            self.mask_file_clust = [path.join(self.mask_dir, mfile)
                                    for mfile in self.mask_file_clust]
        if self.alm_file_clust is not None:
            self.alm_file_clust = [path.join(self.mask_dir, afile)
                                   for afile in self.alm_file_clust]
        if self.mask_file_ggl is not None:
            self.mask_file_ggl = [path.join(self.mask_dir, mfile)
                                  for mfile in self.mask_file_ggl]
        if self.alm_file_ggl is not None:
            self.alm_file_ggl = [path.join(self.mask_dir, afile)
                                 for afile in self.alm_file_ggl]
        if self.mask_file_lens is not None:
            self.mask_file_lens = [path.join(self.mask_dir, mfile)
                                   for mfile in self.mask_file_lens]
        if self.alm_file_lens is not None:
            self.alm_file_lens = [path.join(self.mask_dir, afile)
                                  for afile in self.alm_file_lens]
        if self.alm_file_clust_lens is not None:
            self.alm_file_clust_lens = [path.join(self.mask_dir, afile)
                                        for afile in self.alm_file_clust_lens]
        if self.alm_file_clust_ggl is not None:
            self.alm_file_clust_ggl = [path.join(self.mask_dir, afile)
                                       for afile in self.alm_file_clust_ggl]
        if self.alm_file_lens_ggl is not None:
            self.alm_file_lens_ggl = [path.join(self.mask_dir, afile)
                                      for afile in self.alm_file_lens_ggl]
        if self.save_alms:
            self.save_alms = path.join(self.mask_dir, self.save_alms)
        values = [self.survey_area_clust, self.mask_file_clust,
                  self.read_mask_clust, self.alm_file_clust,
                  self.read_alm_clust, self.n_eff_clust, self.shot_noise_clust,
                  self.tomos_6x2pt_clust, self.survey_area_ggl,
                  self.mask_file_ggl, self.read_mask_ggl, self.alm_file_ggl,
                  self.read_alm_ggl, self.n_eff_ggl, self.survey_area_lens,
                  self.mask_file_lens, self.read_mask_lens, self.alm_file_lens,
                  self.read_alm_lens, self.n_eff_lens, self.sigma_eps,
                  self.shot_noise_gamma, self.alm_file_clust_lens,
                  self.read_mask_clust_lens, self.read_alm_clust_lens,
                  self.alm_file_clust_ggl, self.read_mask_clust_ggl,
                  self.read_alm_clust_ggl, self.alm_file_lens_ggl,
                  self.read_mask_lens_ggl, self.read_alm_lens_ggl,
                  self.save_alms, self.use_tex]
        self.survey_params = dict(zip(keys, values))

        keys = ['M_bins', 'log10M_min', 'log10M_max', 'hmf_model',
                'mdef_model', 'mdef_params', 'disable_mass_conversion',
                'delta_c', 'transfer_model', 'small_k_damping']
        values = [self.M_bins, self.log10M_min, self.log10M_max,
                  self.hmf_model, self.mdef_model, self.mdef_params,
                  self.disable_mass_conversion, self.delta_c,
                  self.transfer_model, self.small_k_damping]
        self.hm_prec = dict(zip(keys, values))
        keys = ['M_bins', 'log10M_min', 'log10M_max', 'hmf_model',
                'mdef_model', 'mdef_params', 'disable_mass_conversion',
                'delta_c', 'transfer_model', 'small_k_damping_for1h']
        self.hm_prec_abr.update(
            {k: v for k, v in zip(keys, values) if v is not None})
        if self.mdef_params is not None:
            self.hm_prec_abr['mdef_params'] = \
                ", ".join(", ".join((k, str(v)))
                          for k, v in self.mdef_params.items())

        keys = ['nl_model', 'log10k_bins', 'log10k_min', 'log10k_max', 'HMCode_A_baryon', 'HMCode_eta_baryon', 'HMCode_logT_AGN']
        values = [self.nl_model, self.log10k_bins, self.log10k_min,
                  self.log10k_max, self.HMCode_A_baryon, self.HMCode_eta_baryon, self.HMCode_logT_AGN]
        self.powspec_prec = dict(zip(keys, values))
        keys = ['non_linear_model', 'log10k_bins', 'log10k_min', 'log10k_max']
        self.powspec_prec_abr = dict(zip(keys, values))

        keys = ['log10k_bins', 'log10k_min', 'log10k_max', 'matter_klim',
                'matter_mulim', 'small_k_damping', 'lower_calc_limit']
        values = [self.tri_logk_bins, self.tri_logk_min, self.tri_logk_max,
                  self.matter_klim, self.matter_mulim,
                  self.tri_small_k_damping, self.lower_calc_limit]
        self.trispec_prec = dict(zip(keys, values))
        keys = ['log10k_bins', 'log10k_min', 'log10k_max', 'matter_klim',
                'matter_mulim', 'small_k_damping_for1h', 'lower_calc_limit']
        self.trispec_prec_abr = dict(zip(keys, values))

        keys = ['num_cores']
        values = [self.num_cores]
        self.misc = dict(zip(keys, values))

        return True

    def __write_save_configs_file(self, config_pars):
        """
        This methods creates a save_configs file which contains all the
        parameters that are not None, whether they have been explicitly
        set in the original configuration file or whether they have been
        implicitly set by this class.

        """
        if not self.save_configs:
            return False

        params_used = configparser.ConfigParser()
        params_used['covariance terms'] = self.covterms
        params_used['observables'] = self.observables_abr
        params_used['output settings'] = self.output_abr
        if len(self.covELLspace_settings_abr) > 0:
            params_used['covELLspace settings'] = self.covELLspace_settings_abr
        if len(self.covTHETAspace_settings_abr) > 0:
            params_used['covTHETAspace settings'] = self.covTHETAspace_settings_abr
        if len(self.covCOESBI_settings_abr) > 0:
            params_used['covCOSEBI settings'] = self.covCOESBI_settings_abr
        if len(self.covbandpowers_settings_abr) > 0:
            params_used['covbandpowers settings'] = self.covbandpowers_settings_abr
        if len(self.covRspace_settings_abr) > 0:
            params_used['covRspace settings'] = self.covRspace_settings_abr
        if len(self.arbitrary_summary_settings_abr) > 0:
            params_used['arbitrary_summary'] = self.arbitrary_summary_settings_abr
        params_used['cosmo'] = self.cosmo_abr
        params_used['bias'] = self.bias_abr
        params_used['IA'] = self.intrinsic_alignments_abr
        if len(self.hod_abr) > 0:
            params_used['hod'] = self.hod_abr
        params_used['survey specs'] = self.survey_params_abr
        params_used['halomodel evaluation'] = self.hm_prec_abr
        params_used['powspec evaluation'] = self.powspec_prec_abr
        if self.nongauss:
            params_used['trispec evaluation'] = self.trispec_prec_abr
        params_used['misc'] = self.misc
        #all_section_names: list[str] = config_pars.sections()
        #all_section_names.append("DEFAULT")
        #for section_name in all_section_names:
        #    for key, value in config_pars.items(section_name):
        #        print(key, value)
        if self.output_dir is None:
            self.output_dir = ''
        with open(
                path.join(self.output_dir, self.save_configs), 'w') as paramfile:
            paramfile.write("### Configuration file for the covariance " +
                            "code\n\n")
            paramfile.write("# Lists all input parameters of the covariance " +
                            "matrix for future reference. Can be read in again with " +
                            "'./bin/NAME -c " + self.output_abr['save_configs'] +
                            "'.\n\n\n")
            params_used.write(paramfile)
        if self.output_dir == '':
            self.output_dir = None

        return True

    def read_input(self,
                   config_name='config.ini'):
        """
        This method reads in all the parameters necessary to calculate
        the covariance matrix. It also checks whether compulsory
        parameters have been set. The parameters are then zipped into
        dictionaries. Finally, a file is produced that lists all
        explicitly and implicitly set parameters for future reference.

        Parameters
        ----------
        config_name : string
            default : 'config.ini'
            Name of the configuration file.

        Return
        ---------
        covterms : dictionary
            Specifies which terms of the covariance (Gaussian, 
            non-Gaussian, super-sample covariance) should be calculated.
            Possible keys: 'gauss', 'split_gauss', 'nongauss', 'ssc'.
        output : dictionary
            Specifies the full path and name, and the style(s) for the 
            output file(s).
            Possible keys: 'file', 'style', 'make_plot', 'Cell', 
                           'trispec'
        observables : dictionary
            with the following keys (also a dictionary each)
            'observables' : dictionary
                Specifies which observables (cosmic shear, galaxy-galaxy 
                lensing and/or clustering) should be calculated. Also, 
                indicates whether cross-terms are evaluated.
                Possible keys: 'cosmic_shear', 'est_shear', 'ggl', 
                               'est_ggl', 'clustering', 'est_clust', 
                               'cross_terms'
            'ELLspace' : dictionary
                Specifies information needed to calculate the covariance
                for the estimator 'C_ell'.
                Possible keys: 'ell_min', 'ell_max', 'ell_bins', 
                               'ell_type', 'delta_z', 
                               'integration_steps', 'nz_polyorder', 
                               'tri_delta_z'
            'THETAspace' : dictionary
                Specifies information needed to calculate the covariance
                for the correlation function estimator 'xi_pm'. Also,
                allows to turn calculation of xi_pp/pm/mm off.
                Possible keys: 'theta_min', 'theta_max', 'theta_bins', 
                               'theta_type', 'xi_pp', 'xi_pm', 'xi_mm'
            'COSEBIs' : dictionary
                Specifies information needed to calculate the covariance
                for the COSEBI estimator.
                Possible keys: ...
            'Rspace' : dictionary
                Specifies information needed to calculate the covariance
                for the galaxy-galaxy lensing estimator
                'projected_real'.
                Possible keys: 'projected_radius_min', 
                               'projected_radius_max', 
                               'projected_radius_bins', 
                               'projected_radius_type', 'mean_redshift', 
                               'projection_length_clustering'
        cosmo : dictionary
            Specifies all cosmological parameters.
            Possible keys: 'sigma8', 'A_s', 'h', 'omega_m', 'omega_b', 
                           'omega_de', 'w0', 'wa', 'ns', 'neff', 'm_nu', 
                           'Tcmb0'
        bias : dictionary
            Specifies all the information about the bias model.
            Possible keys: 'model', 'bias_2h', 'Mc_relation_cen', 
                           'Mc_relation_sat', 'logmass_bins'
        hod : dictionary
            Specifies all the information about the halo occupation 
            distribution. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations.
            Possible keys: 'model_mor_cen', 'model_mor_sat', 
                           'dpow_logM0_cen', 'dpow_logM1_cen', 
                           'dpow_a_cen', 'dpow_b_cen', 'dpow_norm_cen', 
                           'dpow_logM0_sat', 'dpow_logM1_sat', 
                           'dpow_a_sat', 'dpow_b_sat', 'dpow_norm_sat', 
                           'model_scatter_cen', 'model_scatter_sat', 
                           'logn_sigma_c_cen', 'logn_sigma_c_sat', 
                           'modsch_logMref_cen', 'modsch_alpha_s_cen', 
                           'modsch_b_cen', 'modsch_logMref_sat', 
                           'modsch_alpha_s_sat', 'modsch_b_sat'
        survey_params : dictionary
            Specifies all the information unique to a specific survey.
            Possible keys: 'survey_area_clust', 'mask_file_clust', 
                           'read_mask_clust', 'alm_file_clust', 
                           'read_alm_clust', 'n_eff_clust', 
                           'shot_noise_clust', 'survey_area_ggl', 
                           'mask_file_ggl', 'read_mask_ggl', 
                           'alm_file_ggl', 'read_alm_ggl', 'n_eff_ggl', 
                           'survey_area_lens', 'mask_file_lens', 
                           'read_mask_lens', 'alm_file_lens', 
                           'read_alm_lens', 'n_eff_lens', 
                           'ellipticity_dispersion', 'shot_noise_gamma',
                           'alm_file_clust_lens', 
                           'read_mask_clust_lens', 
                           'read_alm_clust_lens', 'alm_file_clust_ggl', 
                           'read_mask_clust_ggl', 'read_alm_clust_ggl', 
                           'alm_file_lens_ggl', 'read_mask_lens_ggl', 
                           'read_alm_lens_ggl', 'save_alms'
        prec : dictionary
            with the following keys (also a dictionary each)
            'hm' : dictionary
                Contains precision information about the HaloModel mass 
                function (also, see hmf documentation by Steven Murray).
                Possible keys: 'M_bins', 'log10M_min', 'log10M_max', 
                               'hmf_model', 'mdef_model', 'mdef_params', 
                               'disable_mass_conversion', 'delta_c', 
                               'transfer_model', 'small_k_damping'
            'powspec' : dictionary
                Contains precision information about the power spectra, 
                this includes k-range and spacing.
                Possible keys: 'nl_model', 'log10k_bins', 'log10k_min', 
                               'log10k_max'
            'trispec' : dictionary
                Contains precision information about the trispectra,
                this includes k-range and spacing and the desired  
                precisionlimits.
                Possible keys: 'matter_klim', 'matter_mulim', 
                               'log10k_bins', 'log10k_min', 
                               'log10k_max', 'small_k_damping'
            'misc' : dictionary
                Contains miscellaneous information.
                Possible keys: 'num_cores'

        """

        self.config_name = config_name

        config = configparser.ConfigParser()
        config.read(config_name)

        self.__read_in_cov_dict(config, config_name)
        self.__read_in_obs_dict(config, config_name)
        self.__read_in_covELLspace_settings(config, config_name)
        self.__read_in_covTHETAspace_settings(config, config_name)
        self.__read_in_covRspace_settings(config, config_name)
        self.__read_in_covCOSEBI_settings(config, config_name)
        self.__read_in_covbandpowers_settings(config, config_name)
        self.__read_in_arbitrary_summary_settings(config, config_name)
        self.__read_in_output_dict(config)
        self.__read_in_cosmo_dict(config)
        self.__read_in_bias_dict(config, config_name)
        self.__read_in_IA_dict(config, config_name)
        self.__read_in_hod_dict(config, config_name)
        self.__read_in_survey_params_dict(config, config_name)
        self.__read_in_hm_prec(config)
        self.__read_in_powspec_prec(config)
        self.__read_in_trispec_prec(config)
        self.__read_in_misc(config)
        self.__zip_to_dicts()
        
        self.__write_save_configs_file(config)

        observables = {'observables': self.observables,
                       'ELLspace': self.covELLspace_settings,
                       'THETAspace': self.covTHETAspace_settings,
                       'COSEBIs': self.covCOSEBI_settings,
                       'bandpowers': self.covbandpowers_settings,
                       'Rspace': self.covRspace_settings,
                       'arbitrary_summary': self.arbitrary_summary_settings}
        prec = {'hm': self.hm_prec,
                'powspec': self.powspec_prec,
                'trispec': self.trispec_prec,
                'misc': self.misc}

        return self.covterms, observables, self.output, self.cosmo, \
            self.bias, self.intrinsic_alignments, self.hod, self.survey_params, prec

# inp = Input()


class FileInput:
    """
    Provides methods to read-in look-up tables listed in a configuration 
    file. Apart from the redshift distribution(s), they are optional 
    additions that can speed up the covariance code or simply feed it a 
    model that is not configured. All input files will be checked for
    internal consistencies. In the Setup class more consistency tests
    across different variable and file inputs will be performed. The new 
    configuration file that the Input class can generate will be 
    appended with a list of all the files used.

    Attributes
    ----------
        too many to list but all relevant ones are put into
        dictionaries and explained in the method 'read_input'

    Example :
    ---------
    from cov_input import FileInput
    fileinp = FileInput(bias)
    read_in_tables = fileinp.read_input(bias)

    """

    def __init__(self,
                 bias_dict):

        self.bias_dict = bias_dict
        # clustering redshift bins
        self.zet_clust = dict()
        self.zet_clust_dir = None
        self.zet_clust_file = None
        self.tomos_6x2pt_clust = None
        self.zet_clust_ext = None
        self.value_loc_in_clustbin = None
        self.zet_clust_z = None
        self.zet_clust_nz = None
        self.n_tomo_clust = None

        # lensing redshift bins
        self.zet_lens = dict()
        self.zet_lens_dir = None
        self.zet_lens_file = None
        self.zet_lens_ext = None
        self.value_loc_in_lensbin = None
        self.zet_lens_z = None
        self.zet_lens_photoz = None
        self.n_tomo_lens = None
        
        self.zet_csmf = dict()
        self.zet_csmf_dir = None
        self.zet_csmf_file = None
        self.zet_csmf_ext = None
        self.zet_csmf_z = None
        self.zet_csmf_pz = None
        self.n_tomo_csmf = None
        
        self.csmf = dict()
        self.csmf_directory = None
        self.V_max_file = None
        self.f_tomo_file = None
        self.V_max = None
        self.f_tomo = None


        # number of galaxy pairs per angular bin
        self.npair = dict()
        self.npair_dir = None
        self.npair_gg_file = None
        self.theta_npair_gg = None
        self.npair_gg = None
        self.npair_gm_file = None
        self.theta_npair_gm = None
        self.npair_gm = None
        self.npair_mm_file = None
        self.theta_npair_mm = None
        self.npair_mm = None

        # power spectra from files
        self.Pxy_tab = dict()
        self.powspec_dir = None
        self.Pmm_file = None
        self.Pgm_file = None
        self.Pgg_file = None
        self.Pxy_k = None
        self.Pxy_z = None
        self.Pmm_tab = None
        self.Pgm_tab = None
        self.Pgg_tab = None

        # C_ell from files
        self.Cellxy_tab = dict()
        self.Cell_dir = None
        self.Cmm_file = None
        self.Cgm_file = None
        self.Cgg_file = None
        self.Cxy_ell = None
        self.Cxy_ell_clust = None
        self.Cxy_ell_lens = None
        self.Cxy_tomo_clust = None
        self.Cxy_tomo_lens = None
        self.Cmm_tab = None
        self.Cgm_tab = None
        self.Cgg_tab = None
        self.Txy_ell = None

        # effective bias from files
        self.effbias_tab = dict()
        self.effbias_dir = None
        self.effbias_z = None
        self.effbias_file = None
        self.effbias = None
        self.bias_files = None
        self.bias_bz = None
        self.bias_z = None
        self.unbiased_clustering = None

        # mor from files
        self.mor_tab = dict()
        self.mor_dir = None
        self.mor_cen_file = None
        self.mor_sat_file = None
        self.mor_M = None
        self.mor_cen = None
        self.mor_sat = None

        # hod probability from files
        self.occprob_tab = dict()
        self.occprob_dir = None
        self.occprob_cen_file = None
        self.occprob_sat_file = None
        self.occprob_M = None
        self.occprob_Mbins = None
        self.occprob_cen = None
        self.occprob_sat = None

        # hod number from files
        self.occnum_tab = dict()
        self.occnum_dir = None
        self.occnum_cen_file = None
        self.occnum_sat_file = None
        self.occnum_M = None
        self.occnum_cen = None
        self.occnum_sat = None

        # trispectra from files
        self.tri_tab = dict()
        self.tri_dir = None
        self.tri_file = None
        self.tri_z = None
        self.tri_log10k = None
        self.tri_mmmm = None
        self.tri_mmgm = None
        self.tri_gmgm = None
        self.tri_ggmm = None
        self.tri_gggm = None
        self.tri_gggg = None

        # auxillary files for cosebis
        self.cosebis = dict()
        self.cosebi_dir = None
        self.wn_log_file = None
        self.wn_lin_file = None
        self.wn_log_ell = None
        self.wn_log = None
        self.wn_lin_ell = None
        self.wn_lin = None
        self.norm_file = None
        self.root_file = None
        self.norms = None
        self.roots = None
        self.Tn_plus_file = None
        self.Tn_minus_file = None
        self.Tn_plus = None
        self.Tn_minus = None
        self.Tn_theta = None
        self.wn_gg_file = None
        self.wn_gg = None
        self.wn_gg_ell = None
        self.Qn_file = None
        self.Un_file = None
        self.Qn = None
        self.Un = None
        self.Qn_theta = None
        self.Un_theta = None

        self.arbitrary_summary = dict()
        self.arbitrary_summary_dir = None
        self.arb_fourier_filter_gg_file = None
        self.arb_real_filter_gg_file = None
        self.arb_fourier_filter_gm_file = None
        self.arb_real_filter_gm_file = None
        self.arb_fourier_filter_mmE_file = None
        self.arb_fourier_filter_mmB_file = None
        self.arb_real_filter_mm_p_file = None
        self.arb_real_filter_mm_m_file = None
        self.arb_number_summary_gg = None
        self.arb_number_summary_gm = None
        self.arb_number_summary_mm = None
        self.arb_number_first_summary_gg = None
        self.arb_number_first_summary_gm = None
        self.arb_number_first_summary_mm = None
        self.gg_summary_name = None
        self.gm_summary_name = None
        self.mmE_summary_name = None
        self.mmB_summary_name = None

        # for save_config.ini
        self.zet_input = dict()
        self.tab_input = dict()

        # helpy variables for consistency checks
        self.cosmicshear = None
        self.est_shear = None
        self.ggl = None
        self.cstellar_mf = None
        self.est_ggl = None
        self.clustering = None
        self.est_clust = None
        self.mean_redshift = None
        self.output_dir = None
        self.save_configs = True
        self.sampledim = 1
        self.hod_model_mor_cen = None
        self.hod_model_scatter_cen = None
        self.csmf_N_log10M_bin = None
        self.do_arbitrary_obs = None

    def __find_filename_two_inserts(self, fn, n_tomo1, n_tomo2):
        loc_pt1 = fn.find('?')
        fn_pt1 = fn[:loc_pt1]
        loc_pt2 = fn[loc_pt1+1:].find('?')
        if loc_pt2 == -1:
            raise Exception("ConfigError: Cannot auto-insert all " +
                            "combinations from  " + str(n_tomo1) + "x" + str(n_tomo2) +
                            " unique bin combinations into the filename " + fn + ". " +
                            "Auto-insertion requires a filename with two '?' which the " +
                            "code replaces with bin combinations (1,1), (1,2), ...")
        fn_pt2 = fn[loc_pt1+1:loc_pt1+loc_pt2+1]
        fn_pt3 = fn[loc_pt1+loc_pt2+2:]
        all_fn = []
        for bin1 in range(n_tomo1):
            start = bin1 if n_tomo1 == n_tomo2 else 0
            for bin2 in range(start, n_tomo1):
                all_fn.append(
                    fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3)
        return all_fn

    def __read_config_for_consistency_checks(self,
                                             config,
                                             config_name='config.ini'):
        """
        Reads in some variables from the configuration file that 
        specify how the covariance should be calculated. These are all
        used for internal consistency checks in this and subsequently 
        called methods.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        """
        self.config_name = config_name
        if 'covariance terms' in config:
            if 'ssc' in config['covariance terms']:
                self.ssc = config['covariance terms'].getboolean('ssc')
            else:
                self.ssc = True
        else:
            self.ssc = True

        if 'observables' in config:
            if 'cosmic_shear' in config['observables']:
                self.cosmicshear = \
                    config['observables'].getboolean('cosmic_shear')
            else:
                self.cosmicshear = False
            if 'est_shear' in config['observables']:
                self.est_shear = config['observables']['est_shear']
            else:
                self.est_shear = ''
            if 'ggl' in config['observables']:
                self.ggl = config['observables'].getboolean('ggl')
            else:
                self.ggl = False
            if 'est_ggl' in config['observables']:
                self.est_ggl = config['observables']['est_ggl']
            else:
                self.est_ggl = ''
            if 'clustering' in config['observables']:
                self.clustering = \
                    config['observables'].getboolean('clustering')
            else:
                self.clustering = False
            if 'est_clust' in config['observables']:
                self.est_clust = config['observables']['est_clust']
            else:
                self.est_clust = ''
            if 'cstellar_mf' in config['observables']:
                self.cstellar_mf = config['observables'].getboolean('cstellar_mf')
        else:
            self.cosmicshear = False
            self.est_shear = ''
            self.ggl = False
            self.est_ggl = ''
            self.cosmicshear = False
            self.est_clust = ''
            self.cstellar_mf = False

        if 'covRspace settings' in config:  # ask for tomo_clust
            if 'mean_redshift' in config['covRspace settings']:
                self.mean_redshift = np.array(
                    config['covRspace settings']['mean_redshift'].split(','))
                self.mean_redshift = (self.mean_redshift).astype(float)
        else:
            ...

        if 'output settings' in config:
            if 'directory' in config['output settings']:
                self.output_dir = config['output settings']['directory']

            self.save_configs = True
            if 'save_configs' in config['output settings']:
                if (config['output settings']
                        ['save_configs'].casefold() == 'true' or
                    config['output settings']
                        ['save_configs'].casefold() == 'false'):
                    self.save_configs = \
                        config['output settings'].getboolean(
                            'save_configs')
                else:
                    self.save_configs = \
                        config['output settings']['save_configs']
            if type(self.save_configs) is bool and self.save_configs:
                self.save_configs = 'save_configs.ini'
        else:
            self.save_configs = 'save_configs.ini'

        if 'bias' in config:
            if 'log10mass_bins' in config['bias']:
                logmass_bins = \
                    np.array(config['bias']['log10mass_bins'].split(','))
                logmass_bins = logmass_bins.astype(float)
                self.sampledim = len(logmass_bins) - 1
            else:
                self.sampledim = 1
        else:
            self.sampledim = 1

        if 'hod' in config:
            if 'model_mor_cen' in config['hod']:
                self.hod_model_mor_cen = config['hod']['model_mor_cen']
            if 'model_scatter_cen' in config['hod']:
                self.hod_model_scatter_cen = config['hod']['model_scatter_cen']
        else:
            ...
        
        if 'csmf settings' in config:
            if 'csmf_N_log10M_bin' in config['csmf settings']:
                self.csmf_N_log10M_bin = int(config['csmf settings']['csmf_N_log10M_bin'])
            if 'csmf_log10M_bins' in config['csmf settings']:
                self.csmf_N_log10M_bin = int(len(np.array(config['csmf settings']['csmf_log10M_bins'].split(',')).astype(float)) - 1)
        if 'arbitrary_summary' in config:
            if 'do_arbitrary_obs' in config['arbitrary_summary']:
                self.do_arbitrary_obs = config['arbitrary_summary'].getboolean('do_arbitrary_obs')
            else:
                self.do_arbitrary_obs = False


        return True

    def __read_in_z_files(self,
                          config,
                          config_name):
        """
        Reads in the redshift distributions for which the covariance 
        should be calculated. If no redshift distribution is given, an
        exception is raised.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        File structure :
        --------------
        # z     n(z)
        0.1     4.1e-4
        0.2     1.3e-3
        ...     ...
        1.1     0.0

        """
        if 'redshift' in config:
            if 'zclust_directory' in config['redshift']:
                self.zet_clust_dir = \
                    config['redshift']['zclust_directory']
            elif 'z_directory' in config['redshift']:
                self.zet_clust_dir = \
                    config['redshift']['z_directory']
            else:
                self.zet_clust_dir = ''

            if 'zclust_specz_file' in config['redshift'] and \
               'zclust_photz_file' in config['redshift']:
                self.zet_clust_file = \
                    (config['redshift']['zclust_specz_file'].replace(
                        " ", "")).split(',') \
                    + (config['redshift']['zclust_photz_file'].replace(
                        " ", "")).split(',')
                self.tomos_6x2pt_clust = np.array(
                    [len(config['redshift']['zclust_specz_file'].replace(
                        " ", "")).split(','),
                     len(config['redshift']['zclust_photz_file'].replace(
                         " ", "")).split(',')])
            elif ('zclust_specz_file' in config['redshift']) != \
                 ('zclust_photz_file' in config['redshift']):
                raise Exception("ConfigError: The redshift files for the " +
                                "extended 6x2pt analysis [redshift]: " +
                                "'zclust_specz_file' and 'zclust_photz_file' must " +
                                "always be passed together.")
            elif 'zclust_file' in config['redshift']:
                self.zet_clust_file = \
                    (config['redshift']['zclust_file'].replace(
                        " ", "")).split(',')
            if 'zclust_extension' in config['redshift']:
                self.zet_clust_ext = \
                    config['redshift']['zclust_extension'].casefold()
            if 'value_loc_in_clustbin' in config['redshift']:
                self.value_loc_in_clustbin = \
                    config['redshift']['value_loc_in_clustbin']
            elif 'value_loc_in_bin' in config['redshift']:
                self.value_loc_in_clustbin = \
                    config['redshift']['value_loc_in_bin']
            else:
                self.value_loc_in_clustbin = 'mid'

            if 'zlens_directory' in config['redshift']:
                self.zet_lens_dir = config['redshift']['zlens_directory']
            elif 'z_directory' in config['redshift']:
                self.zet_lens_dir = config['redshift']['z_directory']
            else:
                self.zet_lens_dir = ''
            if 'zlens_file' in config['redshift']:
                self.zet_lens_file = \
                    (config['redshift']['zlens_file'].replace(
                        " ", "")).split(',')
            if 'zlens_extension' in config['redshift']:
                self.zet_lens_ext = \
                    config['redshift']['zlens_extension'].casefold()
            if 'value_loc_in_lensbin' in config['redshift']:
                self.value_loc_in_lensbin = \
                    config['redshift']['value_loc_in_lensbin']
            elif 'value_loc_in_bin' in config['redshift']:
                self.value_loc_in_lensbin = \
                    config['redshift']['value_loc_in_bin']
            else:
                self.value_loc_in_lensbin = 'mid'
            if 'zcsmf_file' in config['redshift'] and self.cstellar_mf:
                self.zet_csmf_file =  \
                    (config['redshift']['zcsmf_file'].replace(
                        " ", "")).split(',')
            if 'zcsmf_extension' in config['redshift']:
                self.zet_csmf_ext = \
                    config['redshift']['zcsmf_extension'].casefold()
            if 'value_loc_in_csmfbin' in config['redshift']:
                self.value_loc_in_csmfbin = \
                    config['redshift']['value_loc_in_csmfbin']
            elif 'value_loc_in_bin' in config['redshift']:
                self.value_loc_in_csmfbin = \
                    config['redshift']['value_loc_in_bin']
            else:
                self.value_loc_in_csmfbin = 'mid'
            if 'zcsmf_directory' in config['redshift']:
                self.zet_csmf_dir = config['redshift']['zlens_directory']
            elif 'z_directory' in config['redshift']:
                self.zet_csmf_dir = config['redshift']['z_directory']
            

        else:
            ...

        if self.zet_clust_file is None:
            if (self.ggl and
                self.est_ggl != 'k_space' and
                self.est_ggl != 'projected_real') or \
               (self.clustering and
                self.est_clust != 'k_space' and
                    self.est_clust != 'projected_real'):
                raise Exception("ConfigError: No file(s) with redshift " +
                                "distributions for clustering have been specified. Must " +
                                "be adjusted in config file " + config_name + ", " +
                                "[redshift]: 'zclust_file = ...' (separated by comma/s).")
        else:
            if (self.clustering and self.est_clust == 'projected_real') or \
               (self.ggl and self.est_ggl == 'projected_real'):
                # self.zet_clust_file = None
                # do I need the lens files? I don't think so ??? work here
                print("The estimator 'projected_real' will be calculated " +
                      "with the mean redshifts only. In especially, the " +
                      "[redshift]: 'zclust_file = ...' will be ignored.")
            if '.fits' in self.zet_clust_file[0] and \
               self.zet_clust_ext is None:
                raise Exception("ConfigError: A fits zclust_file is " +
                                "specified for the redshift distribution this requires " +
                                "the name of the extension where to find the n(z). " +
                                "Please adjust '[redshift]: zclust_extension = ' to go " +
                                "on.")
        if (self.ggl and
            self.est_ggl != 'k_space' and
            self.est_ggl != 'projected_real') or \
           (self.clustering and
            self.est_clust != 'k_space' and
                self.est_clust != 'projected_real'):
            ...
        else:
            print("InputWarning: The files for the clustering redshift " +
                  "distribution will be ignored as no clustering estimator " +
                  "is calculated.")
            self.zet_clust_file = None

        if self.zet_lens_file is None:
            if (self.ggl and
                self.est_ggl != 'k_space' and
                self.est_ggl != 'projected_real') or \
               (self.cosmicshear and
                    self.est_shear != 'k_space'):
                raise Exception("ConfigError: No file(s) with redshift " +
                                "distributions for lensing have been specified. Must " +
                                "be adjusted in config file " + config_name + ", " +
                                "[redshift]: 'zlens_file = ...' (separated by comma/s).")
        else:
            if '.fits' in self.zet_lens_file and self.zet_lens_ext is None:
                raise Exception("ConfigError: A fits zlens_file is " +
                                "specified for the redshift distribution which requires " +
                                "the name of the extension where to find the n(z). " +
                                "Please adjust '[redshift]: zlens_extension = ' to go on.")
        if self.zet_csmf_file is None:
            if self.cstellar_mf:
                raise Exception("ConfigError: No file(s) with redshift " +
                                "distributions for the conditional stellar mass function have been specified. Must " +
                                "be adjusted in config file " + config_name + ", " +
                                "[redshift]: 'zcsmf_file = ...' (separated by comma/s).")


        if (self.ggl and
            self.est_ggl != 'k_space' and
            self.est_ggl != 'projected_real') or \
           (self.cosmicshear and
                self.est_shear != 'k_space'):
            ...
        else:
            print("InputWarning: The files for the lensing redshift " +
                  "distribution will be ignored as no lensing estimator is " +
                  "calculated.")
            self.zet_lens_file = None

        self.zet_clust_nz = np.array([])
        try:  # ascii
            save_zet_clust_z = []
            save_zet_clust_nz = []
            for fidx, file in enumerate(self.zet_clust_file):
                print("Reading in redshift distributions for clustering " +
                      "from file " + path.join(self.zet_clust_dir, file) + ".")
                data = ascii.read(path.join(self.zet_clust_dir, file))
                if len(data.colnames) < 2:
                    print("InputWarning: The file " + file + " in keyword " +
                          "'zclust_file' has less than 2 columns. The data " +
                          "file should provide the redshift on the first " +
                          "column and the redshift distribution in the " +
                          "second. This file will be ignored.")
                    continue
                different_redshifts = False
                if fidx == 0:
                    self.zet_clust_z = np.array(data[data.colnames[0]])
                    save_zet_clust_z.append(self.zet_clust_z)
                    self.zet_clust_nz = np.array(data[data.colnames[1]])
                    save_zet_clust_nz.append(self.zet_clust_nz)
                    for colname in data.colnames[2:]:
                        self.zet_clust_nz = \
                            np.vstack([self.zet_clust_nz, data[colname]])
                        save_zet_clust_nz.append(data[colname])
                else:
                    save_zet_clust_z.append(np.array(data[data.colnames[0]]))
                    if len(np.array(data[data.colnames[0]])) != len(self.zet_clust_z):
                        redshift_increment = min(self.zet_clust_z[1]- self.zet_clust_z[0], np.array(data[data.colnames[0]])[1] - np.array(data[data.colnames[0]][0]))
                        redshift_max = max(np.max(min(self.zet_clust_z)),np.max(np.array(data[data.colnames[0]])))
                        redshift_min = min(np.min(min(self.zet_clust_z)),np.min(np.array(data[data.colnames[0]])))
                        self.zet_clust_z = np.linspace(redshift_min,redshift_max,int((redshift_max -redshift_min)/redshift_increment))
                        different_redshifts = True         
                        print("ConfigWarning: Adjusting the redshift range in the zclust_files due to different redshift ranges in clustering redshift distribution")
                    if not different_redshifts:
                        for colname in data.colnames[1:]:
                            self.zet_clust_nz = \
                                np.vstack([self.zet_clust_nz, data[colname]])
                            save_zet_clust_nz.append(data[colname])
                    else:
                        for colname in data.colnames[1:]:
                            save_zet_clust_nz.append(data[colname])
            if different_redshifts:
                self.zet_clust_nz = np.array([])
                for i_z in range(len(save_zet_clust_nz)):
                    if i_z == 0:
                        self.zet_clust_nz = np.interp(self.zet_clust_z,
                                                        save_zet_clust_z[i_z],
                                                        save_zet_clust_nz[i_z],
                                                        left = 0,
                                                        right = 0)
                    else:
                        self.zet_clust_nz = np.vstack([self.zet_clust_nz, np.interp(self.zet_clust_z,
                                                                                    save_zet_clust_z[i_z],
                                                                                    save_zet_clust_nz[i_z],
                                                                                    left = 0,
                                                                                    right = 0)])
        except TypeError:
            self.zet_clust_nz = None
        except UnicodeDecodeError:  # fits
            hdul = fits.open(path.join(self.zet_clust_dir, file))
            ext = 1
            try:
                while self.zet_clust_ext != \
                        hdul[ext].header['EXTNAME'].casefold():
                    ext += 1
            except IndexError:
                raise Exception('ConfigError: The extension name ' +
                                self.zet_clust_ext + ' could not be found in the file ' +
                                path.join(self.zet_clust_dir, file) + '. Must be ' +
                                'adjusted to go on.')

            try:
                self.zet_clust_z = hdul[ext].data['Z_MID']
                self.value_loc_in_clustbin = 'mid'
            except KeyError:
                self.zet_clust_z = hdul[ext].data['Z_LOW']
                self.value_loc_in_clustbin = 'left'

            bin_idx = 1
            while 'BIN'+str(bin_idx) in hdul[ext].data.names:
                self.zet_clust_nz = np.concatenate((self.zet_clust_nz,
                                                    hdul[ext].data['BIN'+str(bin_idx)]))
                bin_idx += 1
            self.zet_clust_nz = self.zet_clust_nz.reshape((bin_idx-1,
                                                           hdul[ext].data['BIN'+str(bin_idx-1)].shape[0]))

        self.zet_lens_photoz = np.array([])
        try:
            save_zet_lens_z = []
            save_zet_lens_nz = []
            for fidx, file in enumerate(self.zet_lens_file):
                print("Reading in redshift distributions for lensing from " +
                      "file " + path.join(self.zet_lens_dir, file) + ".")
                data = ascii.read(path.join(self.zet_lens_dir, file))
                if len(data.colnames) < 2:
                    print("InputWarning: The file " + file + " in keyword " +
                          "'zlens_file' has less than 2 columns. The data " +
                          "file should provide the redshift on the first " +
                          "column and the redshift distribution in the " +
                          "second. This file will be ignored.")
                    continue
                different_redshifts = False
                if fidx == 0:
                    self.zet_lens_z = np.array(data[data.colnames[0]])
                    save_zet_lens_z.append(self.zet_lens_z)
                    self.zet_lens_photoz = np.array(data[data.colnames[1]])
                    save_zet_lens_nz.append(self.zet_lens_photoz)
                    for colname in data.colnames[2:]:
                        self.zet_lens_photoz = \
                            np.vstack([self.zet_lens_photoz, data[colname]])
                        save_zet_lens_nz.append(data[colname])
                else:
                    save_zet_lens_z.append(np.array(data[data.colnames[0]]))
                    if len(np.array(data[data.colnames[0]])) != len(self.zet_lens_z):
                        redshift_increment = min(self.zet_lens_z[1]- self.zet_lens_z[0], np.array(data[data.colnames[0]])[1] - np.array(data[data.colnames[0]][0]))
                        redshift_max = max(np.max(min(self.zet_lens_z)),np.max(np.array(data[data.colnames[0]])))
                        redshift_min = min(np.min(min(self.zet_lens_z)),np.min(np.array(data[data.colnames[0]])))
                        self.zet_lens_z = np.linspace(redshift_min,redshift_max,int((redshift_max -redshift_min)/redshift_increment))
                        different_redshifts = True         
                        print("ConfigWarning: Adjusting the redshift range in the zlens_files due to different redshift ranges in lensing redshift distribution")
                    if not different_redshifts:
                        for colname in data.colnames[1:]:
                            self.zet_lens_photoz = \
                                np.vstack([self.zet_lens_photoz, data[colname]])
                            save_zet_lens_nz.append(data[colname])
                    else:
                        for colname in data.colnames[1:]:
                            save_zet_lens_nz.append(data[colname])
            if different_redshifts:
                self.zet_lens_photoz = np.array([])
                for i_z in range(len(save_zet_lens_nz)):
                    if i_z == 0:
                        self.zet_lens_photoz = np.interp(self.zet_lens_z,
                                                        save_zet_lens_z[i_z],
                                                        save_zet_lens_nz[i_z],
                                                        left = 0,
                                                        right = 0)
                    else:
                        self.zet_lens_photoz = np.vstack([self.zet_lens_photoz, np.interp(self.zet_lens_z,
                                                                                    save_zet_lens_z[i_z],
                                                                                    save_zet_lens_nz[i_z],
                                                                                    left = 0,
                                                                                    right = 0)])
        except TypeError:
            self.zet_lens_photoz = None
        except UnicodeDecodeError:  # fits
            hdul = fits.open(path.join(self.zet_lens_dir, file))
            ext = 1
            try:
                while self.zet_lens_ext != \
                        hdul[ext].header['EXTNAME'].casefold():
                    ext += 1
            except IndexError:
                raise Exception('ConfigError: The extension name ' +
                                self.zet_lens_ext + ' could not be found in the file ' +
                                path.join(self.zet_lens_dir, file) + '. Must be adjusted ' +
                                'to go on.')

            try:
                self.zet_lens_z = hdul[ext].data['Z_MID']
                self.value_loc_in_lensbin = 'mid'
            except KeyError:
                self.zet_lens_z = hdul[ext].data['Z_LOW']
                self.value_loc_in_lensbin = 'left'

            bin_idx = 1
            while 'BIN'+str(bin_idx) in hdul[ext].data.names:
                self.zet_lens_photoz = np.concatenate((self.zet_lens_photoz,
                                                       hdul[ext].data['BIN'+str(bin_idx)]))
                bin_idx += 1
            self.zet_lens_photoz = self.zet_lens_photoz.reshape((bin_idx-1,
                                                                 hdul[ext].data['BIN'+str(bin_idx-1)].shape[0]))
        
        if self.zet_clust_z is not None:
            if self.zet_clust_z[0] < 1e-2 and self.value_loc_in_clustbin != 'left':
                self.zet_clust_z = self.zet_clust_z[1:]
                if len(self.zet_clust_nz.shape) == 1:
                    self.zet_clust_nz = self.zet_clust_nz[1:]
                else:
                    self.zet_clust_nz = self.zet_clust_nz[:, 1:]
            if len(self.zet_clust_nz.shape) == 1:
                self.zet_clust_nz = np.array([self.zet_clust_nz])
            self.n_tomo_clust = len(self.zet_clust_nz)
        if self.zet_lens_z is not None:
            if self.zet_lens_z[0] < 1e-2 and self.value_loc_in_lensbin != 'left':
                try:
                    self.zet_lens_photoz = self.zet_lens_photoz[:, 1:]
                    self.zet_lens_z = self.zet_lens_z[1:]
                except:
                    self.zet_lens_z = self.zet_lens_z[1:]
                    self.zet_lens_photoz = self.zet_lens_photoz[1:]
            if len(self.zet_lens_photoz.shape) == 1:
                self.zet_lens_photoz = np.array([self.zet_lens_photoz])
            self.n_tomo_lens = len(self.zet_lens_photoz)

        self.zet_csmf_pz = np.array([])
        try:
            save_zet_csmf_z = []
            save_zet_csmf_nz = []
            for fidx, file in enumerate(self.zet_csmf_file):
                print("Reading in redshift distributions for csmf from " +
                      "file " + path.join(self.zet_csmf_dir, file) + ".")
                data = ascii.read(path.join(self.zet_csmf_dir, file))
                if len(data.colnames) < 2:
                    print("InputWarning: The file " + file + " in keyword " +
                          "'zcsmf_file' has less than 2 columns. The data " +
                          "file should provide the redshift on the first " +
                          "column and the redshift distribution in the " +
                          "second. This file will be ignored.")
                    continue
                different_redshifts = False
                if fidx == 0:
                    self.zet_csmf_z = np.array(data[data.colnames[0]])
                    save_zet_csmf_z.append(self.zet_csmf_z)
                    self.zet_csmf_pz = np.array(data[data.colnames[1]])
                    save_zet_csmf_nz.append(self.zet_csmf_pz)
                    for colname in data.colnames[2:]:
                        self.zet_csmf_pz = \
                            np.vstack([self.zet_csmf_pz, data[colname]])
                        save_zet_csmf_nz.append(data[colname])
                else:
                    save_zet_csmf_z.append(np.array(data[data.colnames[0]]))
                    if len(np.array(data[data.colnames[0]])) != len(self.zet_csmf_z):
                        redshift_increment = min(self.zet_csmf_z[1]- self.zet_csmf_z[0], np.array(data[data.colnames[0]])[1] - np.array(data[data.colnames[0]][0]))
                        redshift_max = max(np.max(min(self.zet_csmf_z)),np.max(np.array(data[data.colnames[0]])))
                        redshift_min = min(np.min(min(self.zet_csmf_z)),np.min(np.array(data[data.colnames[0]])))
                        self.zet_csmf_z = np.linspace(redshift_min,redshift_max,int((redshift_max -redshift_min)/redshift_increment))
                        different_redshifts = True         
                        print("ConfigWarning: Adjusting the redshift range in the zcsmf_files due to different redshift ranges in csmf redshift distribution")
                    if not different_redshifts:
                        for colname in data.colnames[1:]:
                            self.zet_csmf_pz = \
                                np.vstack([self.zet_csmf_pz, data[colname]])
                            save_zet_csmf_nz.append(data[colname])
                    else:
                        for colname in data.colnames[1:]:
                            save_zet_csmf_nz.append(data[colname])
            if different_redshifts:
                self.zet_csmf_pz = np.array([])
                for i_z in range(len(save_zet_csmf_nz)):
                    if i_z == 0:
                        self.zet_csmf_pz = np.interp(self.zet_csmf_z,
                                                        save_zet_csmf_z[i_z],
                                                        save_zet_csmf_nz[i_z],
                                                        left = 0,
                                                        right = 0)
                    else:
                        self.zet_csmf_pz = np.vstack([self.zet_csmf_pz, np.interp(self.zet_csmf_z,
                                                                                    save_zet_csmf_z[i_z],
                                                                                    save_zet_csmf_nz[i_z],
                                                                                    left = 0,
                                                                                    right = 0)])
        except TypeError:
            self.zet_csmf_pz = None
        except UnicodeDecodeError:  # fits
            hdul = fits.open(path.join(self.zet_csmf_dir, file))
            ext = 1
            try:
                while self.zet_csmf_ext != \
                        hdul[ext].header['EXTNAME'].casefold():
                    ext += 1
            except IndexError:
                raise Exception('ConfigError: The extension name ' +
                                self.zet_csmf_ext + ' could not be found in the file ' +
                                path.join(self.zet_csmf_dir, file) + '. Must be adjusted ' +
                                'to go on.')

            try:
                self.zet_csmf_z = hdul[ext].data['Z_MID']
                self.value_loc_in_csmfbin = 'mid'
            except KeyError:
                self.zet_csmf_z = hdul[ext].data['Z_LOW']
                self.value_loc_in_csmfbin = 'left'

            bin_idx = 1
            while 'BIN'+str(bin_idx) in hdul[ext].data.names:
                self.zet_csmf_pz = np.concatenate((self.zet_csmf_pz,
                                                       hdul[ext].data['BIN'+str(bin_idx)]))
                bin_idx += 1
            self.zet_csmf_pz = self.zet_csmf_pz.reshape((bin_idx-1,
                                                                 hdul[ext].data['BIN'+str(bin_idx-1)].shape[0]))

        
        if self.zet_clust_z is not None:
            if self.zet_clust_z[0] < 1e-2 and self.value_loc_in_clustbin != 'left':
                self.zet_clust_z = self.zet_clust_z[1:]
                if len(self.zet_clust_nz.shape) == 1:
                    self.zet_clust_nz = self.zet_clust_nz[1:]
                else:
                    self.zet_clust_nz = self.zet_clust_nz[:, 1:]
            if len(self.zet_clust_nz.shape) == 1:
                self.zet_clust_nz = np.array([self.zet_clust_nz])
            self.n_tomo_clust = len(self.zet_clust_nz)
        if self.zet_csmf_z is not None:
            if self.zet_csmf_z[0] < 1e-2 and self.value_loc_in_csmfbin != 'left':
                self.zet_csmf_z = self.zet_csmf_z[1:]
                self.zet_csmf_pz = self.zet_csmf_pz[:, 1:]
            if len(self.zet_csmf_pz.shape) == 1:
                self.zet_csmf_pz = np.array([self.zet_csmf_pz])
            self.n_tomo_csmf = len(self.zet_csmf_pz)
        if self.zet_lens_z is not None:
            if self.zet_lens_z[0] < 1e-2 and self.value_loc_in_lensbin != 'left':
                self.zet_lens_z = self.zet_lens_z[1:]
                self.zet_lens_photoz = self.zet_lens_photoz[:, 1:]
            if len(self.zet_lens_photoz.shape) == 1:
                self.zet_lens_photoz = np.array([self.zet_lens_photoz])
            self.n_tomo_lens = len(self.zet_lens_photoz)
        return True
    
    def __read_in_csmf_files(self, config):
        """
        Reads in the files for the conditional stellar mass function
        """
        if self.cstellar_mf:
            if 'csmf settings' in config:
                if 'csmf_directory' in config['csmf settings']:
                    self.csmf_directory = \
                        config['csmf settings']['csmf_directory']
                else:
                    self.csmf_directory = ''
            self.V_max = np.zeros((self.csmf_N_log10M_bin,self.n_tomo_csmf))
            if self.csmf_directory != '':
                if 'V_max_file' in config['csmf settings']:
                    self.V_max_file = config['csmf settings']['V_max_file']
                    data = ascii.read(path.join(self.csmf_directory, self.V_max_file))
                    if len(data[data.colnames[0]]) != int(self.n_tomo_csmf*self.csmf_N_log10M_bin):
                        raise Exception("ConfigError: The Vmax file needs to have the dimensions fitting to the number of stellar mass bins "+
                                        "times the number of tomographic bins for the csmf. Please check those in the csmf and redshift section respectively.")
                    for tomo in range(self.n_tomo_csmf):
                        V_max = np.array(data[data.colnames[1]][tomo*self.csmf_N_log10M_bin:(tomo+1)*self.csmf_N_log10M_bin])
                        V_max[V_max == 0.0] = np.nan
                        self.V_max[:, tomo] = self.__fill_nan(V_max)
                else:
                    raise Exception("ConfigError: We require a file for the Vmax estimator if the csmf is to be calculated. "+
                                    "Please specify it in the config vile in the [csmf settings] section via V_max_file = ...." )
                if 'f_tomo_file' in config['csmf settings']:
                    self.f_tomo_file = config['csmf settings']['f_tomo_file']
                    data = np.array(np.loadtxt(path.join(self.csmf_directory, self.f_tomo_file)))
                    if self.n_tomo_csmf != 1:
                        if len(data) != int(self.n_tomo_csmf):
                            raise Exception("ConfigError: The f_tomo file needs to have the dimensions fitting to the "+
                                            "number of tomographic bins for the csmf. Please check those in the csmf and redshift section respectively.")
                        self.f_tomo = data
                    else:
                        self.f_tomo = data
                else:
                    raise Exception("ConfigError: We require a file for the Vmax estimator if the csmf is to be calculated. "+
                                    "Please specify it in the config vile in the [csmf settings] section via V_max_file = ...." )
            else:
                raise Exception("ConfigError: We require a files for the Vmax estimator if the csmf is to be calculated. "+
                                "Please specify it in the config vile in the [csmf settings] section via V_max_file = ...." +
                                "f_tomo_file = ....")
        return True


    def __fill_nan(self, a):
        not_nan = np.isfinite(a)
        indices = np.arange(len(a))
        if not_nan.sum() == 0:
            return a
        else:
            func = interp1d(indices[not_nan], a[not_nan], bounds_error=False, fill_value='extrapolate')
            return func(indices)


    def __read_in_npair_files(self,
                              nfile):
        """
        Reads in the number of galaxy pairs per angular bin for an 
        accurate calculation of the shot noise. If no npair file is  
        given, an approximation of the shot noise contribution will be 
        used.

        Parameters
        ----------
        nfile : string
            Name of the number of galaxy pairs file.

        File structure :
        --------------
        N_pairs must be the last column (as in the treecorr output, or 
                                         just a 2 column file)
        # theta  ...  N_pairs
        0.5      ...  100
        0.6      ...  300
        ...      ...  ...
        100      ...  2000000

        """
        print("Reading in tabulated number of galaxy pairs from file " +
              path.join(self.npair_dir, nfile) + ".")
        try:
            data = ascii.read(path.join(self.npair_dir, nfile))
        except:
            print("InputWarning: The file " +
                  path.join(self.npair_dir, nfile) + " in keyword " +
                  "'npair_XX_file' was not found. The data file " +
                  "should provide the angular bins in the first column, and " +
                  "the next (last) column should hold the number of galaxy " +
                  "pairs. All npair files for XX will be ignored.")
            return False, False
        if len(data.colnames) < 2:
            print("InputWarning: The file " +
                  path.join(self.npair_dir, nfile) + " in keyword " +
                  "'npair_XX_file' has less than 2 columns. The data file " +
                  "should provide the angular bins in the first column, and " +
                  "the next (last) column should hold the number of galaxy " +
                  "pairs. All npair files for XX will be ignored.")
            return False, False

        theta_npair = np.array(data[data.colnames[0]])
        npair = np.array(data[data.colnames[-1]])

        return theta_npair, npair

    def __get_npair_tabs(self,
                         config):
        """
        Reads in the number of galaxy pairs per angular bin for an 
        accurate calculation of the shot noise. If no npair file is  
        given, an approximation of the shot noise contribution will be 
        used. Allows for an auto-generation of filenames if all files
        are named in the same way and only the numbers for the
        tomographic bin combination is changed. In such a case, replace
        the two bin number with a '?' each.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """
        if 'tabulated inputs files' in config:
            if 'npair_directory' in config['tabulated inputs files']:
                self.npair_dir = \
                    config['tabulated inputs files']['npair_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.npair_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.npair_dir = ''
            if 'npair_gg_file' in config['tabulated inputs files']:
                self.npair_gg_file = (config['tabulated inputs files']
                                      ['npair_gg_file'].replace(" ", "")).split(',')
                if self.npair_gg_file[0] == '':
                    self.npair_gg_file = None
            if 'npair_gm_file' in config['tabulated inputs files']:
                self.npair_gm_file = (config['tabulated inputs files']
                                      ['npair_gm_file'].replace(" ", "")).split(',')
                if self.npair_gm_file[0] == '':
                    self.npair_gm_file = None
            if 'npair_mm_file' in config['tabulated inputs files']:
                self.npair_mm_file = (config['tabulated inputs files']
                                      ['npair_mm_file'].replace(" ", "")).split(',')
                if self.npair_mm_file[0] == '':
                    self.npair_mm_file = None
        else:
            ...

        if self.clustering and self.npair_gg_file is not None:
            if len(self.bias_dict['logmass_bins']) <= 2:
                if '?' in self.npair_gg_file[0]:
                    loc_pt1 = self.npair_gg_file[0].find('?')
                    fn_pt1 = self.npair_gg_file[0][:loc_pt1]
                    loc_pt2 = self.npair_gg_file[0][loc_pt1+1:].find('?')
                    if loc_pt2 == -1:
                        raise Exception("ConfigError: Cannot auto-insert the " +
                                        str(self.n_tomo_clust) + "*(" + str(self.n_tomo_clust) +
                                        "+1) bin combinations into the filename " +
                                        self.npair_gg_file[0] + ". Auto-insertion requires " +
                                        "a filename with two '?' which the code replaces " +
                                        "with bin combinations (1,1), (1,2), ...")
                    fn_pt2 = self.npair_gg_file[0][loc_pt1+1:loc_pt1+loc_pt2+1]
                    fn_pt3 = self.npair_gg_file[0][loc_pt1+loc_pt2+2:]
                    self.npair_gg_file = []
                    for bin1 in range(self.n_tomo_clust):
                        for bin2 in range(bin1, self.n_tomo_clust):
                            self.npair_gg_file.append(
                                fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3)
                fidx = 0
                for bin1 in range(self.n_tomo_clust):
                    for bin2 in range(bin1, self.n_tomo_clust):
                        nfile = self.npair_gg_file[fidx]
                        theta_npair_gg, npair_gg = self.__read_in_npair_files(
                            nfile)
                        if type(npair_gg) == bool:
                            self.theta_npair_gg = None
                            self.npair_gg = None
                            break
                        if self.theta_npair_gg is not None:
                            if all(abs(theta_npair_gg - self.theta_npair_gg) > 1e-4):
                                raise Exception("ConfigError: The angular bins " +
                                                "in file " + path.join(self.npair_dir, nfile) +
                                                " don't match the angular bins of the other " +
                                                "files.")
                        if self.npair_gg is None:
                            self.npair_gg = np.zeros((len(npair_gg),
                                                    self.n_tomo_clust,
                                                    self.n_tomo_clust))
                        self.theta_npair_gg = theta_npair_gg
                        self.npair_gg[:, bin1, bin2] = npair_gg
                        self.npair_gg[:, bin2, bin1] = npair_gg
                        self.npair_gg = self.npair_gg[:,:,:,None]
                        fidx += 1
            else:
                if '?' in self.npair_gg_file[0]:
                    loc_pt1 = self.npair_gg_file[0].find('?')
                    fn_pt1 = self.npair_gg_file[0][:loc_pt1]
                    loc_pt2 = self.npair_gg_file[0][loc_pt1+1:].find('?')
                    fn_pt2 = self.npair_gg_file[0][:loc_pt2]
                    loc_pt3 = self.npair_gg_file[0][loc_pt2+1:].find('?')
                    if loc_pt3 == -1:
                        raise Exception("ConfigError: Cannot auto-insert the " +
                                        str(self.n_tomo_clust) + "*(" + str(self.n_tomo_clust) +
                                        "+1)/2*" + str(len(self.bias_dict['logmass_bins']))+ " bin combinations into the filename " +
                                        self.npair_gg_file[0] + ". Auto-insertion requires " +
                                        "a filename with three '?' which the code replaces " +
                                        "with bin combinations (1,1,1), (1,2,1), ...,("+str(self.n_tomo_clust+1) + "," + str(self.n_tomo_clust + 1)  + "),...")
                    fn_pt2 = self.npair_gg_file[0][loc_pt1+1:loc_pt1+loc_pt2+1]
                    fn_pt3 = self.npair_gg_file[0][loc_pt1+loc_pt2+2:loc_pt3+1]
                    fn_pt4 = self.npair_gg_file[0][loc_pt1+loc_pt3+3:]
                    
                    self.npair_gg_file = []
                    for bin_sample in range(len(self.bias_dict['logmass_bins']) - 1):
                        for bin1 in range(self.n_tomo_clust):
                            for bin2 in range(bin1, self.n_tomo_clust):
                                self.npair_gg_file.append(
                                    fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3 + str(bin_sample + 1) + fn_pt4)
                fidx = 0
                for bin_sample in range(len(self.bias_dict['logmass_bins']) - 1):
                    for bin1 in range(self.n_tomo_clust):
                        for bin2 in range(bin1, self.n_tomo_clust):
                            nfile = self.npair_gg_file[fidx]
                            theta_npair_gg, npair_gg = self.__read_in_npair_files(
                                nfile)
                            if type(npair_gg) == bool:
                                self.theta_npair_gg = None
                                self.npair_gg = None
                                break
                            if self.theta_npair_gg is not None:
                                if all(abs(theta_npair_gg - self.theta_npair_gg) > 1e-4):
                                    raise Exception("ConfigError: The angular bins " +
                                                    "in file " + path.join(self.npair_dir, nfile) +
                                                    " don't match the angular bins of the other " +
                                                    "files.")
                            if self.npair_gg is None:
                                self.npair_gg = np.zeros(len(npair_gg),
                                                        self.n_tomo_clust,
                                                        self.n_tomo_clust,
                                                        len(self.bias_dict['logmass_bins']) - 1)
                            self.theta_npair_gg = theta_npair_gg
                            self.npair_gg[:, bin1, bin2, bin_sample] = npair_gg
                            self.npair_gg[:, bin2, bin1, bin_sample] = npair_gg

                            fidx += 1

        if self.ggl and self.npair_gm_file is not None:
            if len(self.bias_dict['logmass_bins']) <= 2:
                if '?' in self.npair_gm_file[0]:
                    loc_pt1 = self.npair_gm_file[0].find('?')
                    fn_pt1 = self.npair_gm_file[0][:loc_pt1]
                    loc_pt2 = self.npair_gm_file[0][loc_pt1+1:].find('?')
                    if loc_pt2 == -1:
                        raise Exception("ConfigError: Cannot auto-insert the " +
                                        str(self.n_tomo_clust) + "*" + str(self.n_tomo_lens) +
                                        ") bin combinations into the filename " +
                                        self.npair_gm_file[0] + ". Auto-insertion requires " +
                                        "a filename with two '?' which the code replaces " +
                                        "with bin combinations (1,1), (1,2), ...")
                    fn_pt2 = self.npair_gm_file[0][loc_pt1+1:loc_pt1+loc_pt2+1]
                    fn_pt3 = self.npair_gm_file[0][loc_pt1+loc_pt2+2:]
                    self.npair_gm_file = []
                    for bin1 in range(self.n_tomo_clust):
                        for bin2 in range(self.n_tomo_lens):
                            self.npair_gm_file.append(
                                fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3)
                fidx = 0
                for bin1 in range(self.n_tomo_clust):
                    for bin2 in range(self.n_tomo_lens):
                        nfile = self.npair_gm_file[fidx]
                        theta_npair_gm, npair_gm = self.__read_in_npair_files(
                            nfile)
                        if type(npair_gm) == bool:
                            self.theta_npair_gm = None
                            self.npair_gm = None
                            break
                        if self.theta_npair_gm is not None:
                            if all(abs(theta_npair_gm - self.theta_npair_gm) > 1e-4):
                                raise Exception("ConfigError: The angular bins " +
                                                "in file " + path.join(self.npair_dir, nfile) +
                                                " don't match the angular bins of the other " +
                                                "files.")
                        if self.npair_gm is None:
                            self.npair_gm = np.zeros((len(npair_gm),
                                                    self.n_tomo_clust,
                                                    self.n_tomo_lens))
                        self.theta_npair_gm = theta_npair_gm
                        self.npair_gm[:, bin1, bin2] = npair_gm
                        self.npair_gm = self.npair_gm[:,:,:,None]
                        fidx += 1
            else: 
                if '?' in self.npair_gm_file[0]:
                    loc_pt1 = self.npair_gm_file[0].find('?')
                    fn_pt1 = self.npair_gm_file[0][:loc_pt1]
                    loc_pt2 = self.npair_gm_file[0][loc_pt1+1:].find('?')
                    fn_pt2 = self.npair_gm_file[0][:loc_pt2]
                    loc_pt3 = self.npair_gm_file[0][loc_pt2+1:].find('?')
                    if loc_pt3 == -1:
                        raise Exception("ConfigError: Cannot auto-insert the " +
                                        str(self.n_tomo_clust) + "*(" + str(self.n_tomo_lens) +
                                        "+1)/2*" + str(len(self.bias_dict['logmass_bins']))+ " bin combinations into the filename " +
                                        self.npair_gm_file[0] + ". Auto-insertion requires " +
                                        "a filename with three '?' which the code replaces " +
                                        "with bin combinations (1,1,1), (1,2,1), ...,(" +str(self.n_tomo_clust + 1) + "," + str(self.n_tomo_lens + 1)  + ",...")
                    fn_pt2 = self.npair_gm_file[0][loc_pt1+1:loc_pt1+loc_pt2+1]
                    fn_pt3 = self.npair_gm_file[0][loc_pt1+loc_pt2+2:loc_pt3+1]
                    fn_pt4 = self.npair_gm_file[0][loc_pt1+loc_pt3+3:]
                    
                    self.npair_gm_file = []
                    for bin_sample in range(len(self.bias_dict['logmass_bins']) - 1):
                        for bin1 in range(self.n_tomo_clust):
                            for bin2 in range(self.n_tomo_lens):
                                self.npair_gm_file.append(
                                    fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3 + str(bin_sample + 1) + fn_pt4)
                fidx = 0
                for bin_sample in range(len(self.bias_dict['logmass_bins']) - 1):
                    for bin1 in range(self.n_tomo_clust):
                        for bin2 in range(self.n_tomo_lens):
                            nfile = self.npair_gm_file[fidx]
                            theta_npair_gm, npair_gm = self.__read_in_npair_files(
                                nfile)
                            if type(npair_gm) == bool:
                                self.theta_npair_gm = None
                                self.npair_gm = None
                                break
                            if self.theta_npair_gm is not None:
                                if all(abs(theta_npair_gm - self.theta_npair_gm) > 1e-4):
                                    raise Exception("ConfigError: The angular bins " +
                                                    "in file " + path.join(self.npair_dir, nfile) +
                                                    " don't match the angular bins of the other " +
                                                    "files.")
                            if self.npair_gm is None:
                                self.npair_gm = np.zeros(len(npair_gm),
                                                        self.n_tomo_clust,
                                                        self.n_tomo_lens,
                                                        len(self.bias_dict['logmass_bins']) - 1)
                            self.theta_npair_gm = theta_npair_gm
                            self.npair_gm[:, bin1, bin2, bin_sample] = npair_gm
                            fidx += 1

        if self.cosmicshear and self.npair_mm_file is not None:
            if '?' in self.npair_mm_file[0]:
                loc_pt1 = self.npair_mm_file[0].find('?')
                fn_pt1 = self.npair_mm_file[0][:loc_pt1]
                loc_pt2 = self.npair_mm_file[0][loc_pt1+1:].find('?')
                if loc_pt2 == -1:
                    raise Exception("ConfigError: Cannot auto-insert the " +
                                    str(self.n_tomo_lens) + "*(" + str(self.n_tomo_lens) +
                                    "+1) bin combinations into the filename " +
                                    self.npair_mm_file[0] + ". Auto-insertion requires " +
                                    "a filename with two '?' which the code replaces " +
                                    "with bin combinations (1,1), (1,2), ...")
                fn_pt2 = self.npair_mm_file[0][loc_pt1+1:loc_pt1+loc_pt2+1]
                fn_pt3 = self.npair_mm_file[0][loc_pt1+loc_pt2+2:]
                self.npair_mm_file = []
                try:
                    ascii.read(path.join(self.npair_dir, fn_pt1 + str(1) + fn_pt2 + str(self.n_tomo_lens) + fn_pt3))
                    for bin1 in range(self.n_tomo_lens):
                        for bin2 in range(bin1, self.n_tomo_lens):
                            fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3
                            self.npair_mm_file.append(
                                fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3)
                except:
                    for bin1 in range(self.n_tomo_lens):
                        for bin2 in range(bin1, self.n_tomo_lens):
                            fn_pt1 + str(bin1+1) + fn_pt2 + str(bin2+1) + fn_pt3
                            self.npair_mm_file.append(
                                fn_pt1 + str(bin2+1) + fn_pt2 + str(bin1+1) + fn_pt3)
            fidx = 0
            for bin1 in range(self.n_tomo_lens):
                for bin2 in range(bin1, self.n_tomo_lens):
                    nfile = self.npair_mm_file[fidx]
                    theta_npair_mm, npair_mm = self.__read_in_npair_files(
                        nfile)
                    if type(npair_mm) == bool:
                        self.theta_npair_mm = None
                        self.npair_mm = None
                        break
                    if self.theta_npair_mm is not None:
                        if all(abs(theta_npair_mm - self.theta_npair_mm) > 1e-4):
                            raise Exception("ConfigError: The angular bins " +
                                            "in file " + path.join(self.npair_dir, nfile) +
                                            " don't match the angular bins of the other " +
                                            "files.")
                    if self.npair_mm is None:
                        self.npair_mm = np.zeros((len(npair_mm),
                                                  self.n_tomo_lens,
                                                  self.n_tomo_lens))
                    self.theta_npair_mm = theta_npair_mm
                    self.npair_mm[:, bin1, bin2] = npair_mm
                    self.npair_mm[:, bin2, bin1] = npair_mm
                    fidx += 1
            if self.npair_mm is not None:
                self.npair_mm = self.npair_mm[:,:,:,None]
          
    def __read_in_powspec_files(self,
                                Pfile):
        """
        Reads in one file with a tabulated power spectrum which is used
        to calculate the covariance.

        Parameters
        ----------
        Pfile : string
            Name of the power spectrum file.

        File structure :
        --------------
        # z     k           P(z,k)[sample 1] ...  P(z,k)[sample N] 
        0.1     4.0e-5      265.1            ...  555.0
        0.2     3.9e-5      286.4            ...  599.7
        ...     ...         ...              ...  ...
        1.1     3.0e+2      0.02             ...  0.07

        """
        try:
            print("Reading in tabulated power spectra from file " +
                  path.join(self.powspec_dir, Pfile) + ".")
            data = ascii.read(path.join(self.powspec_dir, Pfile))
            if len(data.colnames) < 3:
                print("InputWarning: The file " +
                      path.join(self.powspec_dir, Pfile) + " in keyword " +
                      "'Pxy_file' (gg, gm, mm) has less than 3 columns. The " +
                      "data file should provide the redshift on the first " +
                      "column, the wavenumber in the second column, and the " +
                      "next column(s) should hold the P_xy(z,k) values. One " +
                      "column per (e.g.) stellar mass sample. This file " +
                      "will be ignored.")
                return None, None, None, None

            kdim = np.min(np.where(
                data[data.colnames[0]] > data[data.colnames[0]][0]))
            sampledim = len(data.colnames[2:])
            self.sampledim = max(self.sampledim, sampledim)
            zdim = int(len(data)/kdim)

            self.Pxy_z = np.array(data[data.colnames[0]][::kdim])
            self.Pxy_k = np.array(data[data.colnames[1]][:kdim])
            Ptab = np.array(data[data.colnames[2]])
            for cidx, colname in enumerate(data.colnames[3:]):
                Ptab = np.vstack([Ptab, data[colname]])
            Ptab = Ptab.reshape((sampledim, zdim, kdim))
            Ptab = np.swapaxes(Ptab, 1, 2)
            Ptab = np.swapaxes(Ptab, 0, 1)
            return Ptab, kdim, sampledim, zdim
        except TypeError:
            return None, None, None, None

    def __get_powspec_tabs(self,
                           config):
        """
        Calls the read-in method for all tabulated power spectra which 
        are used to calculate the covariance matrix. It then performs
        internal consistency checks and gives appropriate warnings or
        raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # z     k           P(z,k)[sample 1] ...  P(z,k)[sample N] 
        0.1     4.0e-5      265.1            ...  555.0
        0.2     3.9e-5      286.4            ...  599.7
        ...     ...         ...              ...  ...
        1.1     3.0e+2      0.02             ...  0.07

        """

        if 'tabulated inputs files' in config:
            if 'powspec_directory' in config['tabulated inputs files']:
                self.powspec_dir = \
                    config['tabulated inputs files']['powspec_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.powspec_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.powspec_dir = ''
            if 'Pgg_file' in config['tabulated inputs files']:
                self.Pgg_file = \
                    config['tabulated inputs files']['Pgg_file']
            if 'Pgm_file' in config['tabulated inputs files']:
                self.Pgm_file = \
                    config['tabulated inputs files']['Pgm_file']
            if 'Pmm_file' in config['tabulated inputs files']:
                self.Pmm_file = \
                    config['tabulated inputs files']['Pmm_file']
        else:
            ...

        self.Pgg_tab, kdim_gg, sampledim_gg, zdim_gg = \
            self.__read_in_powspec_files(self.Pgg_file)
        self.Pgm_tab, kdim_gm, sampledim_gm, zdim_gm = \
            self.__read_in_powspec_files(self.Pgm_file)
        self.Pmm_tab, kdim_mm, sampledim_mm, zdim_mm = \
            self.__read_in_powspec_files(self.Pmm_file)

        if kdim_gg is not None and kdim_gm is not None:
            if kdim_gg != kdim_gm:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")
            if sampledim_gg != sampledim_gm:
                raise Exception("FileInputError: The number of columns in " +
                                "the files " + self.Pgg_file + " and " + self.Pgm_file +
                                " does not match. Must be adjusted to go on.")
            if zdim_gg != zdim_gm:
                raise Exception("FileInputError: The number of redshift " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")

        if kdim_mm is not None and kdim_gg is not None:
            if kdim_mm != kdim_gg:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")
            if sampledim_mm > sampledim_gg:
                self.Pmm_tab = np.delete(self.Pmm_tab,
                                         (sampledim_gg, sampledim_mm),
                                         axis=1)
                sampledim_mm = sampledim_gg
                self.sampledim = sampledim_gg
                print("FileInputWarning: The number of columns in the file " +
                      self.Pmm_file + " is larger than in " + self.Pgg_file +
                      ". The matter power spectrum will be truncated to the " +
                      "same number of columns.")
            if zdim_mm != zdim_gg:
                raise Exception("FileInputError: The number of redshift " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")

        if kdim_mm is not None and kdim_gm is not None:
            if kdim_mm != kdim_gm:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")
            if sampledim_mm > sampledim_gm:
                self.Pmm_tab = np.delete(self.Pmm_tab,
                                         (sampledim_gm, sampledim_mm),
                                         axis=1)
                sampledim_mm = sampledim_gm
                self.sampledim = sampledim_gm
                print("FileInputWarning: The number of columns in the file " +
                      self.Pmm_file + " is larger than in " + self.Pgm_file +
                      ". The matter power spectrum will be truncated to the " +
                      "same number of columns.")
            if zdim_mm != zdim_gm:
                raise Exception("FileInputError: The number of redshift " +
                                "steps in the files " + self.Pgg_file + " and " +
                                self.Pgm_file + " does not match. Must be adjusted to " +
                                "go on.")

        if kdim_mm is not None and sampledim_mm < self.sampledim:
            self.Pmm_tab = np.insert(
                self.Pmm_tab, [0, 1], self.Pmm_tab, axis=1)

        return True

    def __read_in_bias_files(self,
                             config,
                             config_name):
        """
        Reads in the redshift dependent but scale-independent bias for the clustering.
        If no bias is given the HoD or unbiased clustering will be assumed.
        
        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.
        config_name : string
            Name of the config file. Needed for giving meaningful
            exception texts.

        File structure :
        --------------
        # z     n(z)
        0.1     4.1e-4
        0.2     1.3e-3
        ...     ...
        1.1     0.0

        """
        if 'unbiased_clustering' in config['observables']:
            self.unbiased_clustering = config['observables'].getboolean('unbiased_clustering')
            
        if 'bias' in config:
            if 'bias_files' in config['bias']:
                self.bias_files = \
                    list((config['bias']['bias_files'].replace(
                        " ", "")).split(','))
        if self.bias_files is not None and (self.clustering or self.ggl):
            try:
                save_zet_bias_z = []
                save_zet_bias_bz = []
                for fidx, file in enumerate(self.bias_files):
                    data = ascii.read(file)
                    if len(data.colnames) < 2:
                        print("InputWarning: The file " + file + " in keyword " +
                            "'bias_files' has less than 2 columns. The data " +
                            "file should provide the redshift on the first " +
                            "column and the bias in the " +
                            "second. This file will be ignored.")
                        continue
                    different_redshifts = False
                    if fidx == 0:
                        self.bias_z = np.array(data[data.colnames[0]])
                        save_zet_bias_z.append(self.bias_z)
                        self.bias_bz = np.array(data[data.colnames[1]])
                        save_zet_bias_bz.append(self.bias_bz)
                        for colname in data.colnames[2:]:
                            self.bias_bz = \
                                np.vstack([self.bias_bz, data[colname]])
                            save_zet_bias_bz.append(data[colname])
                    else:
                        save_zet_bias_z.append(np.array(data[data.colnames[0]]))
                        if len(np.array(data[data.colnames[0]])) != len(self.bias_z):
                            redshift_increment = min(self.bias_z[1]- self.bias_z[0], np.array(data[data.colnames[0]])[1] - np.array(data[data.colnames[0]][0]))
                            redshift_max = max(np.max(min(self.bias_z)),np.max(np.array(data[data.colnames[0]])))
                            redshift_min = min(np.min(min(self.bias_z)),np.min(np.array(data[data.colnames[0]])))
                            self.bias_z = np.linspace(redshift_min,redshift_max,int((redshift_max -redshift_min)/redshift_increment))
                            different_redshifts = True         
                            print("ConfigWarning: Adjusting the redshift range in the bias files due to different redshift ranges in bias files")
                        if not different_redshifts:
                            for colname in data.colnames[1:]:
                                self.bias_bz = \
                                    np.vstack([self.bias_bz, data[colname]])
                                save_zet_bias_bz.append(data[colname])
                        else:
                            for colname in data.colnames[1:]:
                                save_zet_bias_bz.append(data[colname])
                if different_redshifts:
                    self.bias_bz = np.array([])
                    for i_z in range(len(save_zet_bias_bz)):
                        if i_z == 0:
                            self.bias_bz = np.interp(self.bias_z,
                                                            save_zet_bias_z[i_z],
                                                            save_zet_bias_bz[i_z],
                                                            left = 0,
                                                            right = 0)
                        else:
                            self.bias_bz = np.vstack([self.bias_bz, np.interp(self.bias_z,
                                                                                        save_zet_bias_z[i_z],
                                                                                        save_zet_bias_bz[i_z],
                                                                                        left = 0,
                                                                                        right = 0)])
                
                self.bias_bz = np.array([])
                for i_z in range(len(save_zet_bias_bz)):
                    if i_z == 0:
                        interp = interp1d(self.bias_z, save_zet_bias_bz[i_z], fill_value = "extrapolate")
                        self.bias_bz = interp(self.zet_clust_z)
                    else:
                        interp = interp1d(self.bias_z, save_zet_bias_bz[i_z], fill_value = "extrapolate")
                        
                        self.bias_bz = np.vstack([self.bias_bz, interp(self.zet_clust_z)])
                if not self.unbiased_clustering:
                    print("Using redshft dependent bias and NOT HoD for galaxy count modelling from file " + self.bias_files[0] + "...")
                else:
                    print("Using unbiased clustering for galaxy count modelling.")
                if(len(save_zet_bias_bz) != self.n_tomo_clust):
                    raise Exception("InputError: From the redshift files, " + self.n_tomo_clust + " bias functions are required. You have only specified " + len(save_zet_bias_bz) + ". Pleas change this in the config file." )

            except:
                print("InputWarning: The bias files " + self.bias_files[0] + "... where not found, will procede with HoD or unbiased clustering")
        return True

    def __read_in_Cell_manyfiles(self,
                                 Cfile,
                                 type):
        """
        Reads in one file with a tabulated projected power spectrum 
        which is used to calculate the covariance.

        Parameters
        ----------
        Cfile : string
            Name of the projected power spectrum file.
        type : string
            Name of the type to read in

        File structure :
        --------------
        For gg
        # ell    C(ell)[s1,s1] ...  C(ell)[sN,sN]
        2        xxe-x              ...  xxe-x
        2        xxe-x              ...  xxe-x
        2        xxe-x              ...  xxe-x 
        2        xxe-x              ...  xxe-x
        3        xxe-x              ...  xxe-x
        ...                     ...  ...
        1.0e5    xxe-x              ...  xxe-x

        For gkappa
        # ell    C(ell)[s1] ...  C(ell)[sN]
        2        xxe-x              ...  xxe-x
        2        xxe-x              ...  xxe-x
        2        xxe-x              ...  xxe-x 
        2        xxe-x              ...  xxe-x
        3        xxe-x              ...  xxe-x
        ...              ...  ...
        1.0e5    xxe-x              ...  xxe-x
        
        For kappakappa
        # ell    C(ell)[s1,s1] 
        2        xxe-x             
        2        xxe-x              
        2        xxe-x             
        2        xxe-x              
        3        xxe-x             
        ...       ...  
        1.0e5    xxe-x     

        s -> galaxy sample bin number (out of N)
             all(!) sample bin combinations required
        t -> all(!) tomographic bin combination (out of T1*T2)
             for kappakappa: M = tomo_lens**2
             for gkappa:     M = tomo_lens*tomo_clust
             for gg:         M = tomo_clust**2
        """
        try:
            print("Reading in tabulated C_ells from file " +
                  path.join(self.Cell_dir, Cfile) + ".")
            data = ascii.read(path.join(self.Cell_dir, Cfile))
            if len(data.colnames) < 2:
                print("InputWarning: The file " +
                      path.join(self.Cell_dir, Cfile) + " in keyword " +
                      "'Cxy_file' (gg, gm, mm) has less than 3 columns. The " +
                      "data file should provide the ell modes in the second " +
                      "column, and the next column(s) should hold the " +
                      "C_xy(ell) values. One column per (e.g.) stellar mass " +
                      "sample. This file will be ignored.")
                return None
            if type == "gg":
                sampledim = int(np.sqrt(len(data.colnames[1:])))
                self.sampledim = max(self.sampledim, sampledim)
                self.Cxy_ell_clust = np.array(data[data.colnames[0]])
            if type == "gm":
                sampledim = len(data.colnames[1:])
                self.sampledim = max(self.sampledim, sampledim)
                if self.Cxy_ell_clust is not None:
                    if len(self.Cxy_ell_clust) != len(np.array(data[data.colnames[0]])):
                        raise Exception("GGL and clustering C_ell files do not have the same support")
                self.Cxy_ell_clust = np.array(data[data.colnames[0]])
            if type == "mm":
                sampledim = 1
                self.Cxy_ell_lens = np.array(data[data.colnames[0]])
            self.Cxy_ell = np.array(data[data.colnames[0]])
            Ctab = np.array(data[data.colnames[1]])
            for colname in data.colnames[3:]:
                Ctab = np.vstack([Ctab, data[colname]])
            if type == "gg":
                Ctab = Ctab.reshape((sampledim, sampledim, len(self.Cxy_ell)))
                Ctab = np.transpose(Ctab, (2, 0,1))
            if type == "gm":
                Ctab = Ctab.reshape((sampledim, len(self.Cxy_ell)))
                Ctab = np.transpose(Ctab, (1,0))
            return Ctab
        except TypeError:
            return None

    def __read_in_Cell_files(self,
                             Cfiles,
                             type):
        """
        Reads in one file with a tabulated projected power spectrum 
        which is used to calculate the covariance.

        Parameters
        ----------
        Cfiles : list
            List of names of the projected power spectrum file.
        type : string
            Name of the type to read in

        File structure :
        --------------
        For gg
        # ell   t1  t2    C(ell)[t1, t2, s1,s1] ...  C(ell)[t1, t2, sN,sN]
        2       1   1     xxe-x              ...  xxe-x
        2       1   ...   xxe-x              ...  xxe-x
        2       ... ...   xxe-x              ...  xxe-x 
        2       max max   xxe-x              ...  xxe-x
        3       1   1     xxe-x              ...  xxe-x
        ...     ...       ...                ...  ...
        1.0e5   max max   xxe-x              ...  xxe-x

        For gkappa
        # ell   t1  t2    C(ell)[t1, t2, s1] ...  C(ell)[t1, t2, sN]
        2       1   1     xxe-x              ...  xxe-x
        2       1   ...   xxe-x              ...  xxe-x
        2       ... ...   xxe-x              ...  xxe-x 
        2       max max   xxe-x              ...  xxe-x
        3       1   1     xxe-x              ...  xxe-x
        ...     ...       ...                ...  ...
        1.0e5   max max   xxe-x              ...  xxe-x
        
        For kappakappa
        # ell   t1  t2    C(ell)[t1, t2]
        2       1   1     xxe-x              
        2       1   ...   xxe-x           
        2       ... ...   xxe-x        
        2       max max   xxe-x             
        3       1   1     xxe-x             
        ...     ...       ...                
        1.0e5   max max   xxe-x              

        s -> galaxy sample bin number (out of N)
             all(!) sample bin combinations required
        t -> all(!) tomographic bin combination (out of T1*T2)
             for kappakappa: M = tomo_lens**2
             for gkappa:     M = tomo_lens*tomo_clust
             for gg:         M = tomo_clust**2
        """
        if Cfiles is None:
            return None, None, None, None, None
        elif len(Cfiles) == 1:
            try:
                print("Reading in tabulated C_ells from file " +
                      path.join(self.Cell_dir, Cfiles[0]) + ".")
                if (path.exists(path.join(self.Cell_dir, Cfiles[0]))):
                    data = ascii.read(path.join(self.Cell_dir, Cfiles[0]))
                    if len(data.colnames) == 2:
                        Ctab = self.__read_in_Cell_files(Cfiles[0], type)
                        Ctab = Ctab[:, :, None, None]
                        return Ctab, len(self.Cxy_ell), len(Ctab[0]), 1, 1
                    if len(data.colnames) < 4:
                        print("InputWarning: The file " +
                              path.join(self.Cell_dir, Cfiles[0]) + " in keyword " +
                              "'Cxy_file' (gg, gkappa, kappakappa) has less than 4 " +
                              "columns. The data file should provide the wavenumber " +
                              "in the first column, and the next two columns should " +
                              "the tomographic bin combination and the next " +
                              "column(s) should hold the C_xy(ell) values for all " +
                              "(e.g.) stellar mass samples. This file will be " +
                              "ignored.")
                        return None, None, None, None, None
                    n_tomo_1 = int(max(data[data.colnames[1]]))
                    n_tomo_2 = int(max(data[data.colnames[2]]))
                    if(min(data[data.colnames[1]]) == 0):
                        n_tomo_1 += 1
                    if(min(data[data.colnames[2]]) == 0):
                        n_tomo_2 += 1
                    elldim = int(len(data) / n_tomo_1 / n_tomo_2)
                    if type == "gg":
                        sampledim = int(np.sqrt(len(data.colnames[3:])))
                        if sampledim > self.sampledim:
                            self.sampledim = sampledim
                        else:
                            sampledim = self.sampledim
                    if type == "gm":
                        sampledim = len(data.colnames[3:])
                        if sampledim > self.sampledim:
                            self.sampledim = sampledim
                        else:
                            sampledim = self.sampledim
                    if type == "mm":
                        sampledim = 1
                        if sampledim > self.sampledim:
                            self.sampledim = sampledim
                        else:
                            sampledim = self.sampledim
                    self.Cxy_ell = np.array(
                        data[data.colnames[0]][::n_tomo_1*n_tomo_2])
                    Ctab = np.array(data[data.colnames[3]].reshape((len(self.Cxy_ell),n_tomo_1,n_tomo_2)))
                    if type == "gg":
                        self.Cxy_ell_clust = np.array(
                            data[data.colnames[0]][::n_tomo_1*n_tomo_2])
                        Ctab_aux = np.zeros((len(self.Cxy_ell),sampledim,sampledim,n_tomo_1,n_tomo_2))
                        Ctab_aux[:,0,0,:,:] = Ctab
                        counter = 4
                        for i_sample in range(sampledim):
                            for j_sample in range(sampledim):
                                if i_sample == 0 and j_sample == 0:
                                    continue
                                else:
                                    Ctab_aux[:,i_sample, j_sample,:,:] = data[data.colnames[counter]].reshape((len(self.Cxy_ell),n_tomo_1,n_tomo_2))
                                    counter +=1
                    if type == "gm":
                        self.Cxy_ell_clust = np.array(
                            data[data.colnames[0]][::n_tomo_1*n_tomo_2])
                        if self.Cxy_ell_clust is not None:
                            if len(self.Cxy_ell_clust) != len(np.array(data[data.colnames[0]][::n_tomo_1*n_tomo_2])):
                                raise Exception("GGL and clustering C_ell files do not have the same support")
                        Ctab_aux = np.zeros((len(self.Cxy_ell),sampledim,n_tomo_1,n_tomo_2))
                        Ctab_aux[:,0,:,:] = Ctab
                        counter = 4
                        for i_sample in range(sampledim):
                            if i_sample == 0:
                                continue
                            else:
                                Ctab_aux[:,i_sample,:,:] = data[data.colnames[counter]].reshape((len(self.Cxy_ell),n_tomo_1,n_tomo_2))
                                counter +=1
                    if type == "gg":
                        Ctab = Ctab_aux
                    if type == "gm":
                        Ctab = Ctab_aux
                    if type == "mm":
                        self.Cxy_ell_lens = np.array(
                            data[data.colnames[0]][::n_tomo_1*n_tomo_2])
                        Ctab = Ctab.reshape((len(self.Cxy_ell),n_tomo_1,n_tomo_2))[:,None, :,:]*np.ones(sampledim)[None, :, None, None]
                    return Ctab, elldim, sampledim, n_tomo_1, n_tomo_2
                else:
                    print("InputWarning: The file " +
                          path.join(self.Cell_dir, Cfiles[0]) + " in keyword "
                          "'Cxy_file' (gg, gkappa, kappakappa) or directory does not "
                          "exist. Will proced calculating the Cells.")
                    return None, None, None, None, None
            except TypeError:
                return None, None, None, None, None
        else:
            Ctabs = []
            for Cfile in Cfiles:
                if 'gg' in Cfile:
                    type = 'gg'
                if 'gm' in Cfile:
                    type = 'gm'
                if 'mm' in Cfile:
                    type = 'mm'
                Ctab = self.__read_in_Cell_manyfiles(Cfile,type)
                if Ctab is not None:
                    Ctabs.append(Ctab)
                else:
                    raise Exception("Something happened with z-files.")
            Ctabs = np.moveaxis(np.array(Ctabs), 0, -1)
            return Ctabs, len(self.Cxy_ell), len(Ctab[0]), -1, -1

    def __get_Cell_tabs(self,
                        config):
        """
        Calls the read-in method for all tabulated projected power 
        spectra which are used to calculate the covariance matrix. It 
        then performs internal consistency checks and gives appropriate 
        warnings or raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # ell   t1  t2    C(ell)[t1, t2, s1] ...  C(ell)[t1, t2, sN]
        2       1   1     xxe-x              ...  xxe-x
        2       1   ...   xxe-x              ...  xxe-x
        2       ... ...   xxe-x              ...  xxe-x 
        2       max max   xxe-x              ...  xxe-x
        3       1   1     xxe-x              ...  xxe-x
        ...     ...       ...                ...  ...
        1.0e5   max max   xxe-x              ...  xxe-x

        or N_unique_tomo_combinations files with one ell column and 
        N_sampledims Cell columns
        # ell   C(ell)[s1] ...  C(ell)[sN]
        2       xxe-x      ...  xxe-x
        3       xxe-x      ...  xxe-x
        ...     ...        ...  ...
        1.0e5   xxe-x      ...  xxe-x

        s -> galaxy sample bin number (out of N)
             [currently only one supported]
        t -> tomographic bin combination (out of M)
             for kappakappa: M = tomo_lens**2
             for gkappa:     M = tomo_lens*tomo_clust
             for gg:         M = tomo_clust**2

        """

        if 'tabulated inputs files' in config:
            if 'Cell_directory' in config['tabulated inputs files']:
                self.Cell_dir = \
                    config['tabulated inputs files']['Cell_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.Cell_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.Cell_dir = ''
            if 'Cgg_file' in config['tabulated inputs files']:
                self.Cgg_file = \
                    (config['tabulated inputs files']
                        ['Cgg_file'].replace(" ", "")).split(',')
            if 'Cgm_file' in config['tabulated inputs files']:
                self.Cgm_file = \
                    (config['tabulated inputs files']
                        ['Cgm_file'].replace(" ", "")).split(',')
            if 'Cmm_file' in config['tabulated inputs files']:
                self.Cmm_file = \
                    (config['tabulated inputs files']
                        ['Cmm_file'].replace(" ", "")).split(',')
        else:
            ...

        if self.Cgg_file is not None:
            if '?' in self.Cgg_file[0]:
                self.Cgg_file = self.__find_filename_two_inserts(
                    self.Cgg_file[0], self.n_tomo_clust, self.n_tomo_clust)
        if self.Cgm_file is not None:
            if '?' in self.Cgm_file[0]:
                self.Cgm_file = self.__find_filename_two_inserts(
                    self.Cgm_file[0], self.n_tomo_clust, self.n_tomo_lens)
        if self.Cmm_file is not None:
            if '?' in self.Cmm_file[0]:
                self.Cmm_file = self.__find_filename_two_inserts(
                    self.Cmm_file[0], self.n_tomo_lens, self.n_tomo_lens)

        self.Cgg_tab, elldim_gg, sampledim_gg, n_tomo_clust_gg, _ = \
            self.__read_in_Cell_files(self.Cgg_file, 'gg')
        if n_tomo_clust_gg == -1:
            Cgg_reshape = np.zeros((elldim_gg, sampledim_gg,
                                    self.n_tomo_clust, self.n_tomo_clust))
            tidx = 0
            for t1 in range(self.n_tomo_clust):
                for t2 in range(t1, self.n_tomo_clust):
                    Cgg_reshape[:, :, t1, t2] = self.Cgg_tab[:, :, tidx]
                    Cgg_reshape[:, :, t2, t1] = self.Cgg_tab[:, :, tidx]
                    tidx += 1
            self.Cgg_tab = Cgg_reshape
            n_tomo_clust_gg = self.n_tomo_clust
        self.Cgm_tab, elldim_gm, sampledim_gm, n_tomo_clust_gm, \
            n_tomo_lens_gm = self.__read_in_Cell_files(self.Cgm_file, 'gm')
        if n_tomo_clust_gm == -1:
            Cgm_reshape = np.zeros((elldim_gm, sampledim_gm,
                                    self.n_tomo_clust, self.n_tomo_lens))
            tidx = 0
            for t1 in range(self.n_tomo_clust):
                for t2 in range(self.n_tomo_lens):
                    Cgm_reshape[:, :, t1, t2] = self.Cgm_tab[:, :, tidx]
                    tidx += 1
            self.Cgm_tab = Cgm_reshape
            n_tomo_clust_gm = self.n_tomo_clust
            n_tomo_lens_gm = self.n_tomo_lens
        self.Cmm_tab, elldim_mm, sampledim_mm, n_tomo_lens_mm, _ = \
            self.__read_in_Cell_files(self.Cmm_file, 'mm')
        if n_tomo_lens_mm == -1:
            Cmm_reshape = np.zeros((elldim_mm, sampledim_mm,
                                    self.n_tomo_lens, self.n_tomo_lens))
            tidx = 0
            for t1 in range(self.n_tomo_lens):
                for t2 in range(t1, self.n_tomo_lens):
                    Cmm_reshape[:, :, t1, t2] = self.Cmm_tab[:, :, tidx]
                    Cmm_reshape[:, :, t2, t1] = self.Cmm_tab[:, :, tidx]
                    tidx += 1
            self.Cmm_tab = Cmm_reshape
            n_tomo_lens_mm = self.n_tomo_lens
        if elldim_gg is not None and elldim_gm is not None:
            if elldim_gg != elldim_gm:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + str(self.Cgg_file) + " and " +
                                str(self.Cgm_file) + " does not match. Must be adjusted to " +
                                "go on.")
            self.sampledim = max(self.sampledim, sampledim_gg, sampledim_gm)
            if sampledim_gg != sampledim_gm:
                raise Exception("FileInputError: The number of columns in " +
                                "the files " + str(self.Cgg_file) + " and " + str(self.Cgm_file) +
                                " does not match. Must be adjusted to go on.")
            if n_tomo_clust_gg != n_tomo_clust_gm:
                raise Exception("FileInputError: The number of tomographic " +
                                "clustering bins in the files " + str(self.Cgg_file) + " and " +
                                str(self.Cgm_file) + " does not match. Must be adjusted to " +
                                "go on.")
        if elldim_mm is not None and elldim_gm is not None:
            if elldim_mm != elldim_gm:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + str(self.Cmm_file) + " and " +
                                str(self.Cgm_file) + " does not match. Must be adjusted to " +
                                "go on.")
            self.sampledim = max(self.sampledim, sampledim_mm, sampledim_gm)
            if sampledim_mm != sampledim_gm:
                raise Exception("FileInputError: The number of columns in " +
                                "the files " + str(self.Cmm_file) + " and " + str(self.Cgm_file) +
                                " does not match. Must be adjusted to go on.")
            if n_tomo_lens_mm != n_tomo_lens_gm:
                raise Exception("FileInputError: The number of tomographic " +
                                "lensing bins in the files " + str(self.Cmm_file) + " and " +
                                str(self.Cgm_file) + " does not match. Must be adjusted to " +
                                "go on.")
        if elldim_mm is not None and elldim_gg is not None:
            if elldim_mm != elldim_gg:
                raise Exception("FileInputError: The number of wavenumber " +
                                "steps in the files " + str(self.Cmm_file) + " and " +
                                str(self.Cgg_file) + " does not match. Must be adjusted to " +
                                "go on.")
            self.sampledim = max(self.sampledim, sampledim_mm, sampledim_gg)
            if sampledim_mm != sampledim_gg:
                raise Exception("FileInputError: The number of columns in " +
                                "the files " + str(self.Cmm_file) + " and " + str(self.Cgg_file) +
                                " does not match. Must be adjusted to go on.")

        if n_tomo_clust_gg is not None:
            self.Cxy_tomo_clust = n_tomo_clust_gg
        else:
            self.Cxy_tomo_clust = n_tomo_clust_gm
        if n_tomo_lens_mm is not None:
            self.Cxy_tomo_lens = n_tomo_lens_mm
        else:
            self.Cxy_tomo_lens = n_tomo_lens_gm

        return True
    
    def __read_in_Tell_files(self,
                             Tfiles):
        """
        Reads in one file with a tabulated projected trispectrum 
        which is used to calculate the covariance.

        Parameters
        ----------
        Tfiles : list
            List of names of the projected trispectrum file.

        File structure :
        --------------
        # ell1 ell2   T(ell)[s1] ...  T(ell)[sN]
        2      2      xxe-x      ...  xxe-x
        2      ...    xxe-x      ...  xxe-x
        2      ellmax xxe-x      ...  xxe-x
        3      2      xxe-x      ...  xxe-x
        ...    ...    ...        ...  ...
        ellmax ellmax xxe-x      ...  xxe-x

        s -> galaxy sample bin number (out of N)
             [currently only one supported]
        N_unique_tomo_combinations files with one ell column and 
        N_sampledims Cell columns
             for kappakappa: M = tomo_lens**2
             for gkappa:     M = tomo_lens*tomo_clust
             for gg:         M = tomo_clust**2
        """
        if Tfiles is None:
            return None, None, None, None, None
        elif len(Tfiles) == 1:
            try:
                print("Reading in tabulated T_ells from file " +
                      path.join(self.Cell_dir, Tfiles[0]) + ".")
                if (path.exists(path.join(self.Cell_dir, Tfiles[0]))):
                    data = ascii.read(path.join(self.Cell_dir, Tfiles[0]))
                    if len(data.colnames) == 3:
                        Ttab = self.__read_in_Tell_files(Tfiles[0])
                        Ttab = Ttab[:, :, None, None]
                        return Ttab, len(self.Txy_ell), len(Ttab[0]), 1, 1
                    if len(data.colnames) < 5:
                        print("InputWarning: The file " +
                              path.join(self.Cell_dir, Tfiles[0]) + " in keyword " +
                              "'Cxy_file' (gg, gkappa, kappakappa) has less than 4 " +
                              "columns. The data file should provide the wavenumbers " +
                              "in the first two columns, and the next four columns should " +
                              "the tomographic bin combination and the next " +
                              "column(s) should hold the T_xyuv(ell) values for all " +
                              "(e.g.) stellar mass samples. This file will be " +
                              "ignored.")
                        return None, None, None, None, None
                    n_tomo_1 = int(max(data[data.colnames[1]]))
                    n_tomo_2 = int(max(data[data.colnames[2]]))
                    n_tomo_3 = int(max(data[data.colnames[3]]))
                    n_tomo_4 = int(max(data[data.colnames[4]]))
                    if(min(data[data.colnames[1]]) == 0):
                        n_tomo_1 += 1
                    if(min(data[data.colnames[2]]) == 0):
                        n_tomo_2 += 1
                    if(min(data[data.colnames[3]]) == 0):
                        n_tomo_3 += 1
                    if(min(data[data.colnames[4]]) == 0):
                        n_tomo_4 += 1
                    elldim = int(len(data) / n_tomo_1 / n_tomo_2)
                    sampledim = len(data.colnames[3:])
                    self.Cxy_ell = np.array(
                        data[data.colnames[0]][::n_tomo_1*n_tomo_2])

                    Ctab = np.array(data[data.colnames[3]])
                    for colname in data.colnames[4:]:
                        Ctab = np.vstack([Ctab, data[colname]])
                    Ctab = Ctab.reshape(sampledim, elldim,
                                        n_tomo_1, n_tomo_2).swapaxes(0, 1)

                    return Ctab, elldim, sampledim, n_tomo_1, n_tomo_2
                else:
                    print("InputWarning: The file " +
                          path.join(self.Cell_dir, Tfiles[0]) + " in keyword "
                          "'Cxy_file' (gg, gkappa, kappakappa) or directory does not "
                          "exist. Will proced calculating the Cells.")
                    return None, None, None, None, None
            except TypeError:
                return None, None, None, None, None
        else:
            Ctabs = []
            for Cfile in Tfiles:
                if 'gg' in Cfile:
                    type = 'gg'
                if 'gm' in Cfile or 'gkappa' in Cfile:
                    type = 'gm'
                if 'mm' in Cfile or 'kappakappa' in Cfile:
                    type = 'mm'
                Ctab = self.__read_in_Cell_manyfiles(Cfile,type)
                if Ctab is not None:
                    Ctabs.append(Ctab)
                else:
                    raise Exception("Something happened with z-files.")
            Ctabs = np.moveaxis(np.array(Ctabs), 0, -1)
            return Ctabs, len(self.Cxy_ell), len(Ctab[0]), -1, -1

    def __read_in_effbias_files(self,
                                bfile):
        """
        Reads in one file with a tabulated effective bias which is used
        to calculate the covariance.

        Parameters
        ----------
        bfile : string
            Name of the power spectrum file.

        File structure :
        --------------
        # z     bias(z)[sample 1] ...  bias(z)[sample N] 
        0.1     1.2               ...  1.7
        0.2     1.3               ...  1.9
        ...     ...               ...  ...
        1.1     1.9               ...  3.2

        """

        try:
            print("Reading in tabulated effective bias from file " +
                  path.join(self.effbias_dir, bfile) + ".")
            data = ascii.read(path.join(self.effbias_dir, bfile))
            if len(data.colnames) < 2:
                print("InputWarning: The file " +
                      path.join(self.effbias_dir, bfile) + " in keyword " +
                      "'effbias_file' has less than 2 columns. The data " +
                      "file should provide the redshift on the first " +
                      "column, and the next column(s) should hold the " +
                      "effective bias values. One column per (e.g.) stellar " +
                      "mass sample. This file will be ignored.")
                return None, None, None

            zdim = len(data)
            sampledim = len(data.colnames[1:])
            self.sampledim = max(self.sampledim, sampledim)

            self.effbias_z = np.array(data[data.colnames[0]])
            effbiastab = np.array(data[data.colnames[1]])
            for cidx, colname in enumerate(data.colnames[2:]):
                effbiastab = np.vstack([effbiastab, data[colname]])
            effbiastab = effbiastab.reshape((sampledim, zdim))
            return effbiastab, sampledim, zdim
        except TypeError:
            return None, None, None

    def __get_effbias_tabs(self,
                           config):
        """
        Calls the read-in method for the tabulated effective bias which 
        is used to calculate the covariance matrix. It then performs
        internal consistency checks and gives appropriate warnings or 
        raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # z     bias(z)[sample 1] ...  bias(z)[sample N] 
        0.1     1.2               ...  1.7
        0.2     1.3               ...  1.9
        ...     ...               ...  ...
        1.1     1.9               ...  3.2

        """

        if 'tabulated inputs files' in config:
            if 'effbias_directory' in config['tabulated inputs files']:
                self.effbias_dir = \
                    config['tabulated inputs files']['effbias_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.effbias_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.effbias_dir = ''
            if 'effbias_file' in config['tabulated inputs files']:
                self.effbias_file = \
                    config['tabulated inputs files']['effbias_file']
        else:
            ...

        self.effbias, sampledim, zdim = \
            self.__read_in_effbias_files(self.effbias_file)

        if zdim is not None and sampledim < self.sampledim:
            raise Exception("FileInputError: The number of columns in the " +
                            "file " + self.effbias_file + " does not match the number " +
                            "of galaxy sample bins. Must be adjusted to go on.")

        return True

    def __read_in_mor_files(self,
                            mfile):
        """
        Reads in one file with a tabulated mass-observable relation 
        which is used to calculate the covariance.

        Parameters
        ----------
        mfile : string
            Name of the mass-observable relation file.

        File structure :
        --------------
        # log10(M)     mor(M) 
         9.0           8.3e+2
         9.1           1.9e+3
        ...            ...
        18.0           1.8e+17

        """

        try:
            print("Reading in tabulated mass-observable relation from file " +
                  path.join(self.mor_dir, mfile) + ".")
            data = ascii.read(path.join(self.mor_dir, mfile))
            if len(data.colnames) != 2:
                print("InputWarning: The file " +
                      path.join(self.mor_dir, mfile) + " in keyword " +
                      "'mor_file' does not have 2 columns. The data file " +
                      "should provide the log10(mass) in the first column, " +
                      "and the result of the mass-observable relation in the " +
                      "second. This file will be ignored.")
                return None, None

            Mdim = len(data)
            self.mor_M = 10**np.array(data[data.colnames[0]])
            mor = np.array(data[data.colnames[1]])
            return mor, Mdim
        except TypeError:
            return None, None

    def __get_mor_tabs(self,
                       config):
        """
        Calls the read-in method for the tabulated mass-observable 
        relation which is used to calculate the covariance matrix. It 
        then performs internal consistency checks and gives appropriate 
        warnings or raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # log10(M)     mor(M) 
         9.0           8.3e+2
         9.1           1.9e+3
        ...            ...
        18.0           1.8e+17

        """

        if 'tabulated inputs files' in config:
            if 'mor_directory' in config['tabulated inputs files']:
                self.mor_dir = \
                    config['tabulated inputs files']['mor_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.mor_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.mor_dir = ''
            if 'mor_cen_file' in config['tabulated inputs files']:
                self.mor_cen_file = \
                    config['tabulated inputs files']['mor_cen_file']
            if 'mor_sat_file' in config['tabulated inputs files']:
                self.mor_sat_file = \
                    config['tabulated inputs files']['mor_sat_file']
        else:
            ...

        self.mor_cen, Mdim_cen = self.__read_in_mor_files(self.mor_cen_file)
        self.mor_sat, Mdim_sat = self.__read_in_mor_files(self.mor_sat_file)

        if Mdim_cen is not None and Mdim_sat is not None:
            if Mdim_cen != Mdim_sat:
                raise Exception("FileInputError: The number of rows in the " +
                                "files " + self.mor_cen_file + " and " +
                                self.mor_sat_file + " does not match. Must be adjusted " +
                                "to go on.")
        if self.mor_sat is None:
            self.mor_sat = self.mor_cen

        return True

    def __read_in_occprob_files(self,
                                ofiles,
                                sampledim):
        """
        Reads in one file with a tabulated occupation distribution which 
        is used to calculate the covariance.

        Parameters
        ----------
        ofiles : string
            Name of the occupation distribution files.
        sampledim : int
            Number of galaxy samples (e.g., in stellar mass bins).

        File structure :
        --------------
        # the following should be provided per galaxy sample bin
        # log10(M)     log10(M for sample)  occprob(M, M for sample) 
         9.0           10.0                 0.0
         9.0           10.1                 0.0
        ...            ...                  ...
         9.0           11.0                 0.0
         9.1           10.0                 0.0
        ...            ...                  ...
        18.0           11.0                 0.0

        # of all galaxies should be considered (sampledim = 1)
        # log10(M)     occprob(M, M for sample) 
         9.0           0.0
         9.1           0.0
        ...            ...
        18.0           0.0

        (The example above lists extreme values for M and sample bin M,
        therefore, most values will just be 0, of course not all...)

        """

        try:
            Mbins = np.array([])
            probtab = np.array([])
            for ofile in ofiles:
                print("Reading in tabulated halo occupation probability " +
                      "from file " + path.join(self.occprob_dir, ofile) + ".")
                data = ascii.read(path.join(self.occprob_dir, ofile))
                if len(data.colnames) < 2:
                    print("InputWarning: The file " +
                          path.join(self.occprob_dir, ofile) + " in keyword " +
                          "'occprob_file' has less than 2 columns. The data " +
                          "file should provide the log10(mass) in the first " +
                          "column, if the galaxy population is split into " +
                          "several samples, the corresponding mass bins " +
                          "should be in the second column, and the " +
                          "respective next column should hold the result of " +
                          "the halo occupation probability values. This " +
                          "file will be ignored.")
                    continue

                Mbindim = np.min(np.where(
                    data[data.colnames[0]] > data[data.colnames[0]][0]))
                self.sampledim = max(self.sampledim, sampledim)
                Mdim = int(len(data)/Mbindim)

                self.occprob_M = \
                    10**np.array(data[data.colnames[0]][::Mbindim])
                if len(data.colnames) == 2:
                    probtab = np.array(data[data.colnames[1]])
                else:
                    Mbins = np.concatenate((Mbins,
                                            np.array(data[data.colnames[1]][:Mbindim])))
                    probtab = np.concatenate((probtab,
                                              np.array(data[data.colnames[2]])))
            try:
                Mbins = 10**Mbins.reshape((sampledim, Mbindim))
            except ValueError:
                Mbins = self.occprob_M.reshape((sampledim, Mdim))
            probtab = probtab.reshape((sampledim, Mdim, Mbindim))
            probtab = np.swapaxes(probtab, 1, 2)
            return probtab, Mbins, Mdim, Mbindim
        except TypeError:
            return None, None, None, None

    def __get_occprob_tabs(self,
                           config):
        """
        Calls the read-in method for the tabulated occupation 
        probability which is used to calculate the covariance matrix. It
        then performs internal consistency checks and gives appropriate 
        warnings or raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # the following should be provided per galaxy sample bin
        # log10(M)     log10(M for sample)  occprob(M, M for sample) 
         9.0           10.0                 0.0
         9.0           10.1                 0.0
        ...            ...                  ...
         9.0           11.0                 0.0
         9.1           10.0                 0.0
        ...            ...                  ...
        18.0           11.0                 0.0

        # of all galaxies should be considered (sampledim = 1)
        # log10(M)     occprob(M, M for sample) 
         9.0           0.0
         9.1           0.0
        ...            ...
        18.0           0.0

        (The example above lists extreme values for M and sample bin M,
        therefore, most values will just be 0, of course not all...)

        """

        if 'tabulated inputs files' in config:
            if 'occprob_directory' in config['tabulated inputs files']:
                self.occprob_dir = \
                    config['tabulated inputs files']['occprob_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.occprob_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.occprob_dir = ''
            if 'occprob_cen_file' in config['tabulated inputs files']:
                self.occprob_cen_file = (config['tabulated inputs files']
                                         ['occprob_cen_file'].replace(" ", "")).split(',')
            if 'occprob_sat_file' in config['tabulated inputs files']:
                self.occprob_sat_file = (config['tabulated inputs files']
                                         ['occprob_sat_file'].replace(" ", "")).split(',')
        else:
            ...

        try:
            sampledim_cen = len(self.occprob_cen_file)
        except TypeError:
            sampledim_cen = None
        try:
            sampledim_sat = len(self.occprob_sat_file)
        except TypeError:
            sampledim_sat = None
        if sampledim_cen is not None and sampledim_sat is not None:
            if sampledim_cen != sampledim_sat:
                raise Exception("FileInputError: The number of files listed " +
                                "for 'occprob_cen_file' and 'occprob_sat_file' does not " +
                                "match. Must be adjusted to go on.")

        self.occprob_cen, Mbins_cen, Mdim_cen, Mbindim_cen = \
            self.__read_in_occprob_files(self.occprob_cen_file, sampledim_cen)
        self.occprob_sat, Mbins_sat, Mdim_sat, Mbindim_sat = \
            self.__read_in_occprob_files(self.occprob_sat_file, sampledim_sat)

        if self.occprob_M is not None and self.mor_M is not None:
            if np.any(self.occprob_M != self.mor_M):
                print("ConfigWarning: The masses in the first column listed " +
                      "in the 'mor_cen/sat_file' are not the exact same as in " +
                      "the first column of the files listed in " +
                      "'occprob_cen/sat_file'. This might cause a problem " +
                      "later on.")
        if Mdim_cen is not None and Mdim_sat is not None:
            if Mdim_cen != Mdim_sat:
                raise Exception("FileInputError: The number of mass steps " +
                                "on the first columns in the files listed for " +
                                "'occprob_cen_file' and 'occprob_sat_file' does not " +
                                "match. Must be adjusted to go on.")
        if Mbindim_cen is not None and Mbindim_sat is not None:
            if Mbindim_cen != Mbindim_sat:
                raise Exception("FileInputError: The number of mass steps " +
                                "on the second columns in the files listed for " +
                                "'occprob_cen_file' and 'occprob_sat_file' does not " +
                                "match. Must be adjusted to go on.")
        if self.occprob_sat is None:
            self.occprob_sat = self.occprob_cen

        if Mbins_cen is not None:
            self.occprob_Mbins = Mbins_cen
        elif Mbins_sat is not None:
            self.occprob_Mbins = Mbins_sat
        if Mbins_cen is not None and Mbins_sat is not None:
            if np.any(Mbins_cen != Mbins_sat):
                raise Exception("FileInputError: The masses in the second " +
                                "columns of the 'occprob_cen' files are not the exact " +
                                "same as in the second columns of the 'occprob_sat' " +
                                "files. Must be adjusted to go on.")

        return True

    def __read_in_occnum_files(self,
                               nfile):
        """
        Reads in one file with a tabulated occupation number which is 
        used to calculate the covariance.

        Parameters
        ----------
        nfile : string
            Name of the occupation number file.

        File structure :
        --------------
        # log10(M)     occnum(M)[sample 1]  ...  occnum(M)[sample N] 
         9.0           0.0                  ...  0.0
         9.1           0.0                  ...  0.0
        ...            ...                  ...  ...
        18.0           0.0                  ...  1.0e-34

        (The example above lists extreme values for M, therefore, most 
        values will just be 0, of course not all...)

        """

        try:
            print("Reading in tabulated halo occupation number from file " +
                  path.join(self.occnum_dir, nfile) + ".")
            data = ascii.read(path.join(self.occnum_dir, nfile))
            if len(data.colnames) < 2:
                print("InputWarning: The file " +
                      path.join(self.occnum_dir, nfile) + " in keyword " +
                      "'occnum_file' has less than 2 columns. The data file " +
                      "should provide the log10(mass) in the first column, " +
                      "and the next column(s) should hold the occuption " +
                      "number as given by the chosen hod. One column per " +
                      "(e.g.) stellar mass sample. This file will be ignored.")
                return None, None

            Mdim = len(data)
            sampledim = len(data.colnames[1:])
            self.sampledim = max(self.sampledim, sampledim)

            self.occnum_M = 10**np.array(data[data.colnames[0]])
            num = np.array(data[data.colnames[1]])
            for cidx, colname in enumerate(data.colnames[2:]):
                num = np.vstack([num, data[colname]])
            num = num.reshape((sampledim, Mdim))
            return num, Mdim
        except TypeError:
            return None, None

    def __get_occnum_tabs(self,
                          config):
        """
        Calls the read-in method for the tabulated occupation number 
        which is used to calculate the covariance matrix. It then 
        performs internal consistency checks and gives appropriate
        warnings or raises exceptions.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # log10(M)     occnum(M)[sample 1]  ...  occnum(M)[sample N] 
         9.0           0.0                  ...  0.0
         9.1           0.0                  ...  0.0
        ...            ...                  ...  ...
        18.0           0.0                  ...  1.0e-34

        (The example above lists extreme values for M, therefore, most 
        values will just be 0, of course not all...)

        """

        if 'tabulated inputs files' in config:
            if 'occnum_directory' in config['tabulated inputs files']:
                self.occnum_dir = \
                    config['tabulated inputs files']['occnum_directory']
            if 'input_directory' in config['tabulated inputs files']:
                self.occnum_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.occnum_dir = ''
            if 'occnum_cen_file' in config['tabulated inputs files']:
                self.occnum_cen_file = \
                    config['tabulated inputs files']['occnum_cen_file']
            if 'occnum_sat_file' in config['tabulated inputs files']:
                self.occnum_sat_file = \
                    config['tabulated inputs files']['occnum_sat_file']
        else:
            ...

        self.occnum_cen, Mdim_cen = \
            self.__read_in_occnum_files(self.occnum_cen_file)
        self.occnum_sat, Mdim_sat = \
            self.__read_in_occnum_files(self.occnum_sat_file)

        if Mdim_cen is not None and Mdim_sat is not None:
            if Mdim_cen != Mdim_sat:
                raise Exception("FileInputError: The number of rows in the " +
                                "files " + self.occnum_cen_file + " and " +
                                self.occnum_sat_file + " does not match. Must be " +
                                "adjusted to go on.")

        if self.occnum_M is not None and self.mor_M is not None:
            if np.any(self.occnum_M != self.mor_M):
                print("ConfigWarning: The masses in the first column listed " +
                      "in the 'mor_cen/sat_file' are not the exact same as in " +
                      "the first column of the files listed in " +
                      "'occnum_cen/sat_file'. The masses for from the " +
                      "'occprob_cen/sat_file' will be ignored.")
        if self.occnum_M is not None and self.occprob_M is not None:
            if np.any(self.occnum_M != self.occprob_M):
                print("ConfigWarning: The masses in the first column listed " +
                      "in the 'occprob_cen/sat_file' are not the exact same " +
                      "as in the first column of the files listed in " +
                      "'occnum_cen/sat_file'. The masses for from the " +
                      "'occprob_cen/sat_file' will be ignored.")

        if self.occnum_sat is None:
            self.occnum_sat = self.occnum_cen

        if self.occnum_cen is None and self.occprob_cen is None:
            if self.mor_cen is None and self.hod_model_mor_cen is None:
                raise Exception("ConfigError: Neither a mass-observable " +
                                "relation is given to calculate the halo occupation " +
                                "distribution, nor a look-up table is provided. Must be " +
                                "adjusted to go on.")
            if self.hod_model_scatter_cen is None:
                raise Exception("ConfigError: Neither a scattering relation " +
                                "is given to calculate the halo occupation " +
                                "distribution, nor a look-up table is provided. Must be " +
                                "adjusted to go on.")

        return True

    def __read_in_trispec_files(self,
                                Tfiles):
        """
        Reads in all files with a tabulated trispectrum which is used to
        calculate the covariance.

        Parameters
        ----------
        Tfiles : string
            Name of the trispectrum files.

        File structure :
        --------------
        # log10k1  log10k2  m   n   gggg gggm ggmm gmgm mmgm mmmm 
          -1e1     -1e1     0   0   .... .... .... .... .... ....
          ...      ...      ... ... .... .... .... .... .... ....
          -1e1     +2e1     0   n   .... .... .... .... .... ....
          +2e1     -1e1     1   1   .... .... .... .... .... ....
          ...      ...      ... ... .... .... .... .... .... ....
          +2e1     +2e1     m   n   .... .... .... .... .... ....

        (m,n: optional galaxy sample bins, if only one bin set to 0  0)

        """

        print("Reading in tabulated trispectra from files " +
              path.join(self.tri_dir, self.tri_file) + ".")
        for zidx, Tfile in enumerate(Tfiles):
            data = ascii.read(path.join(self.tri_dir, Tfile))
            if len(data.colnames) != 10:
                print("InputWarning: The file " +
                      path.join(self.tri_dir, Tfile) + " in keyword " +
                      "'trispec_file' does not have the 10 columns " +
                      "'log10ki', 'log10kj', 'm', 'n', 'gggg', 'gggm', " +
                      "'ggmm', 'gmgm', 'mmgm', 'mmmm', where 'm' and 'n' " +
                      "are galaxy sample bins. All look-up tables will be " +
                      "ignored and the trispectra will be calculate again.")
                self.tri_dir, self.tri_file = None, None
                return None, None, None, None, None, None, None

            if zidx == 0:
                sampledim = np.max(data[data.colnames[2]]) + 1
                kdim = np.min(np.where(
                    data[data.colnames[0]] > data[data.colnames[0]][0]))
                kdim = int(kdim/sampledim/(sampledim-1))
                zdim = len(self.tri_z)
                self.tri_log10k = \
                    np.array(data[data.colnames[1]][
                        :kdim*sampledim*(sampledim-1):sampledim*(sampledim-1)])

                xxxx = np.zeros((6, kdim, kdim, sampledim, sampledim, zdim))

            idx = 0
            for m in range(sampledim):
                for n in range(m, sampledim):
                    for tri in range(6):
                        xxxx[tri, :, :, m, n, zidx] = np.array(
                            data[data.colnames[4+tri]][idx:idx+kdim**2]
                        ).reshape((kdim, kdim))
                        xxxx[tri, :, :, n, m, zidx] = np.array(
                            data[data.colnames[4+tri]][idx:idx+kdim**2]
                        ).reshape((kdim, kdim))
                    idx += kdim**2

        return xxxx[5], xxxx[4], xxxx[3], xxxx[2], xxxx[1], xxxx[0], sampledim

    def __get_trispec_tabs(self, config):
        """
        Calls the read-in method for all tabulated trispectra which are 
        used to calculate the covariance matrix. It then performs
        internal consistency checks and gives appropriate warnings or
        raises exceptions. The list of available redshifts are defined 
        in the filename, e.g., trispectra_0.1.ascii for redshift z=0.1.
        The code will search for extensions '_redshift' with a given
        filename.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        File structure :
        --------------
        # log10k1  log10k2  m   n   gggg gggm ggmm gmgm mmgm mmmm 
          -1e1     -1e1     0   0   .... .... .... .... .... ....
          ...      ...      ... ... .... .... .... .... .... ....
          -1e1     +2e1     0   n   .... .... .... .... .... ....
          +2e1     -1e1     1   1   .... .... .... .... .... ....
          ...      ...      ... ... .... .... .... .... .... ....
          +2e1     +2e1     m   n   .... .... .... .... .... ....

        (m,n: optional galaxy sample bins, if only one bin set to 0  0)

        """
        if 'tabulated inputs files' in config:
            if 'trispec_directory' in config['tabulated inputs files']:
                self.tri_dir = \
                    config['tabulated inputs files']['trispec_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.tri_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.tri_dir = ''

            if 'trispec_file' in config['tabulated inputs files']:
                self.tri_file = \
                    config['tabulated inputs files']['trispec_file']
        else:
            self.tri_dir = ''

        if self.tri_file is not None:
            dotloc_b = self.tri_file[::-1].find('.')
            if dotloc_b == -1 or dotloc_b == 0:
                dotloc_b = 1
            dotloc_f = len(self.tri_file) - dotloc_b - 1

            _, _, filenames = next(walk(self.tri_dir))
            tri_files = [fstr for fstr in filenames
                         if self.tri_file[:dotloc_f] in fstr]

            self.tri_z = []
            for file in tri_files:
                self.tri_z.append(file[dotloc_f+1:-dotloc_b-1+len(file)])
            try:
                self.tri_z = np.array(self.tri_z).astype(float)
            except ValueError:
                self.tri_dir, self.tri_file = None, None
                self.tri_z = None
                print("FileInputWarning: One or more of the  saved " +
                      "trispectra files " +
                      path.join(self.tri_dir, self.tri_file) + " do not " +
                      "follow the correct naming convention. To circumvent " +
                      "this warning in the future it is advised to delete " +
                      "all files '" + self.tri_file + "' and produce new " +
                      "ones.")

            self.tri_mmmm, self.tri_mmgm, self.tri_gmgm, self.tri_ggmm, \
                self.tri_gggm, self.tri_gggg, sampledim = \
                self.__read_in_trispec_files(tri_files)

            if sampledim is not None and self.sampledim is not None:
                if sampledim != self.sampledim:
                    print("FileInputWarning: The number of galaxy sample bins " +
                          "in the files " + self.tri_file + " do not match " +
                          "other inputs. The tabulated files will be omitted in " +
                          "this case and the trispectra computed later on.")
                    self.tri_log10k, self.tri_z = None, None
                    self.tri_mmmm, self.tri_mmgm, self.tri_gmgm,
                    self.tri_ggmm, self.tri_gggm, self.tri_gggg = \
                        None, None, None, None, None, None
                    self.tri_dir, self.tri_file = None, None
        else:
            ...

        return True

    def __read_in_fourier_filter_files(self,
                                       wfile):
        """
        Reads in ...

        Parameters
        ----------
        wfile : string
            Name of the filter file.

        File structure :
        --------------
        # ell   Wn_log/lin
        2       0.123456789
        3       0.234567891
        ...         ...
        100     0.912345678

        """
        print("Reading in tabulated kernels for arbitrary summary statistics from file " +
              path.join(self.arbitrary_summary_dir, wfile) + ".")
        data = np.loadtxt(path.join(self.arbitrary_summary_dir, wfile))
        if len(data[0]) != 2:
            raise Exception("FileInputError: The file " +
                            path.join(self.arbitrary_summary_dir, wfile) +
                            " has not exactly 2 columns. The data file " +
                            "should provide the angular modes in the first column, and " +
                            "the second column should hold the Fourier filter value.")
        np.seterr(over='ignore')
        if np.exp(data[-1, 0]) > 1e7:
            wn_ell = data[:, 0]
            wn = data[:, 1]
        else:
            wn_ell = np.exp(data[:, 0])
            wn = data[:, 1]
        np.seterr(over='warn')
        return wn_ell, wn
    
    def __read_in_real_filter_files(self,
                                    wfile):
        """
        Reads in ...

        Parameters
        ----------
        wfile : string
            Name of the filter file.

        File structure :
        --------------
        # theta   R(theta)
        2       0.123456789
        3       0.234567891
        ...         ...
        100     0.912345678

        """
        print("Reading in tabulated kernels for arbitrary summary statistics from file " +
              path.join(self.arbitrary_summary_dir, wfile) + ".")
        data = np.loadtxt(path.join(self.arbitrary_summary_dir, wfile))
        if len(data[0]) != 2:
            raise Exception("FileInputError: The file " +
                            path.join(self.arbitrary_summary_dir, wfile) +
                            " has not exactly 2 columns. The data file " +
                            "should provide the angular modes in the first column, and " +
                            "the second column should hold the Real filter value.")

        wn_ell = data[:, 0]
        wn = data[:, 1]

        return wn_ell, wn

    def __read_in_Wn_files(self,
                           wfile):
        """
        Reads in ...

        Parameters
        ----------
        wfile : string
            Name of the COSEBI kernel file.

        File structure :
        --------------
        # ln(ell)   Wn_log/lin
        2           0.123456789
        3           0.234567891
        ...         ...
        100         0.912345678

        """
        print("Reading in tabulated kernels for COSEBIs from file " +
              path.join(self.cosebi_dir, wfile) + ".")
        data = np.loadtxt(path.join(self.cosebi_dir, wfile))
        if len(data[0]) != 2:
            raise Exception("FileInputError: The file " +
                            path.join(self.cosebi_dir, wfile) + " in keyword " +
                            "'wn_log/lin_file' has not exactly 2 columns. The data file " +
                            "should provide the angular modes in the first column, and " +
                            "the second column should hold the kernel value.")

        np.seterr(over='ignore')
        if np.exp(data[-1, 0]) > 1e7:
            wn_ell = data[:, 0]
            wn = data[:, 1]
        else:
            wn_ell = np.exp(data[:, 0])
            wn = data[:, 1]
        np.seterr(over='warn')
        wn = data[:, 1]

        return wn_ell, wn

    def __read_in_Tn_pm_files(self,
                              Tfile):
        """
        Reads in ...

        Parameters
        ----------
        Tfile : string
            Name of the COSEBI Tn_pm kernel file.

        File structure :
        --------------
        # theta   Tn_pm
        
        """
        print("Reading in tabulated real space kernels for COSEBIs from file " +
              path.join(self.cosebi_dir, Tfile) + ".")
        data = np.loadtxt(path.join(self.cosebi_dir, Tfile))
        if len(data[0]) != 2:
            raise Exception("FileInputError: The file " +
                            path.join(self.cosebi_dir, Tfile) + " in keyword " +
                            "'Tn_pm' has not exactly 2 columns. The data file " +
                            "should provide the angle in the first column, and " +
                            "the second column should hold the Tn_pm kernel value.")

        Tn_theta = data[:, 0]
        Tn = data[:, 1]

        return Tn_theta, Tn

    def __get_arbitrary_filter_tabs(self,
                                    config):
        """
        Reads in the ... Allows for an auto-generation of filenames if all files
        are named in the same way and only the numbers for the
        tomographic bin combination is changed. In such a case, replace
        the two bin number with a '?' each.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """
        if not self.do_arbitrary_obs:
            return False
        else:
            if 'tabulated inputs files' in config:
                if 'arb_summary_directory' in config['tabulated inputs files']:
                    self.arbitrary_summary_dir = \
                        config['tabulated inputs files']['arb_summary_directory']
                elif 'input_directory' in config['tabulated inputs files']:
                    self.arbitrary_summary_dir = \
                        config['tabulated inputs files']['input_directory']
                else:
                    self.arbitrary_summary_dir = ''
                if self.clustering:
                    if 'arb_fourier_filter_gg_file' in config['tabulated inputs files']:
                        self.arb_fourier_filter_gg_file =(config['tabulated inputs files']
                                                            ['arb_fourier_filter_gg_file'].replace(" ", "")).split(',')
                        self.arb_fourier_filter_gg_file_save = np.copy(self.arb_fourier_filter_gg_file)
                        self.arb_number_summary_gg = len(self.arb_fourier_filter_gg_file)
                        if len(self.arb_fourier_filter_gg_file) > 2:
                            raise Exception("ConfigError: You are passing more than two arbitrary summary statistics for clustering " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gg_file")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for clustering, " +
                                        "files for the corresponding fourier filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gg_file")
                    if 'arb_real_filter_gg_file' in config['tabulated inputs files']:
                        self.arb_real_filter_gg_file =(config['tabulated inputs files']
                                                            ['arb_real_filter_gg_file'].replace(" ", "")).split(',')
                        self.arb_real_filter_gg_file_save = np.copy(self.arb_real_filter_gg_file)
                        if self.arb_number_summary_gg != len(self.arb_real_filter_gg_file):
                            raise Exception("ConfigError: You are passing more real space filters than Fourier filters for arbitrary summary statistics of clustering " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gg_file and arb_real_filter_gg_file.")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for clustering, " +
                                        "files for the corresponding Real space filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_real_filter_gg_file") 
                if self.ggl:
                    if 'arb_fourier_filter_gm_file' in config['tabulated inputs files']:
                        self.arb_fourier_filter_gm_file =(config['tabulated inputs files']
                                                            ['arb_fourier_filter_gm_file'].replace(" ", "")).split(',')
                        self.arb_fourier_filter_gm_file_save = np.copy(self.arb_fourier_filter_gm_file)
                        self.arb_number_summary_gm = len(self.arb_fourier_filter_gm_file)
                        if len(self.arb_fourier_filter_gm_file) > 2:
                            raise Exception("ConfigError: You are passing more than two arbitrary summary statistics for GGL " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gm_file")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for GGL, " +
                                        "files for the corresponding fourier filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gm_file")
                    if 'arb_real_filter_gm_file' in config['tabulated inputs files']:
                        self.arb_real_filter_gm_file =(config['tabulated inputs files']
                                                            ['arb_real_filter_gm_file'].replace(" ", "")).split(',')
                        self.arb_real_filter_gm_file_save = np.copy(self.arb_real_filter_gm_file)
                        if self.arb_number_summary_gm != len(self.arb_real_filter_gm_file):
                            raise Exception("ConfigError: You are passing more real space filters than Fourier filters for arbitrary summary statistics of GGL " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_gm_file and arb_real_filter_gm_file.")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for GGL, " +
                                        "files for the corresponding Real space filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_real_filter_gm_file")
                if self.cosmicshear:
                    if 'arb_fourier_filter_mmE_file' in config['tabulated inputs files']:
                        self.arb_fourier_filter_mmE_file =(config['tabulated inputs files']
                                                            ['arb_fourier_filter_mmE_file'].replace(" ", "")).split(',')
                        self.arb_fourier_filter_mmE_file_save = np.copy(self.arb_fourier_filter_mmE_file)
                        self.arb_number_summary_mm = len(self.arb_fourier_filter_mmE_file)
                        if len(self.arb_fourier_filter_mmE_file) > 2:
                            raise Exception("ConfigError: You are passing more than two arbitrary summary statistics for GGL " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_mmE_file and/or arb_fourier_filter_mmB_file")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for lensing, " +
                                        "files for the corresponding fourier filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_mmE_file")
                    if 'arb_fourier_filter_mmB_file' in config['tabulated inputs files']:
                        self.arb_fourier_filter_mmB_file =(config['tabulated inputs files']
                                                            ['arb_fourier_filter_mmB_file'].replace(" ", "")).split(',')
                        self.arb_fourier_filter_mmB_file_save = np.copy(self.arb_fourier_filter_mmB_file)
                        self.arb_fourier_filter_no_B = [False, False]
                        for i in range(self.arb_number_summary_mm):
                            if self.arb_fourier_filter_mmB_file[i] == self.arb_fourier_filter_mmE_file[i]:
                                self.arb_fourier_filter_no_B[i] = True
                        if self.arb_number_summary_mm != len(self.arb_fourier_filter_mmB_file):
                            raise Exception("ConfigError: You are passing more B mode filters than E mode filters to the arbitrary summary statistics for lensing " +
                                            "Please adjust in" +
                                            "the config file under [tabulated inputs files] and arb_fourier_filter_mmE_file and/or arb_fourier_filter_mmB_file")
                    else:
                        self.arb_fourier_filter_mmB_file = self.arb_fourier_filter_mmE_file
                        print("ConfigWarning: No B-mode Fourier filter file has been passed for lensing, setting this partto zero")
                    
                    if 'arb_real_filter_mm_p_file' in config['tabulated inputs files']:
                        self.arb_real_filter_mm_p_file =(config['tabulated inputs files']
                                                            ['arb_real_filter_mm_p_file'].replace(" ", "")).split(',')
                        self.arb_real_filter_mm_p_file_save = np.copy(self.arb_real_filter_mm_p_file)
                        if self.arb_number_summary_mm != len(self.arb_real_filter_mm_p_file):
                            raise Exception("ConfigError: You are passing more real space filters than Fourier filters for arbitrary summary statistics of lensing " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_mmE_file and arb_real_filter_mm_p_file.")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for lensing, " +
                                        "files for the corresponding Real space filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_real_filter_mm_p_file")
                    if 'arb_real_filter_mm_m_file' in config['tabulated inputs files']:
                        self.arb_real_filter_mm_m_file =(config['tabulated inputs files']
                                                            ['arb_real_filter_mm_m_file'].replace(" ", "")).split(',')
                        self.arb_real_filter_mm_m_file_save = np.copy(self.arb_real_filter_mm_m_file)
                        if self.arb_number_summary_mm != len(self.arb_real_filter_mm_m_file):
                            raise Exception("ConfigError: You are passing more real space filters than Fourier filters for arbitrary summary statistics of lensing " +
                                        "Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_fourier_filter_mmE_file and arb_real_filter_mm_m_file.")
                    else:
                        raise Exception("ConfigError: To calculate the arbitrary summary statistics for lensing, " +
                                        "files for the corresponding Real space filter must be provided. Please adjust in" +
                                        "the config file under [tabulated inputs files] and arb_real_filter_mm_m_file") 
                if self.clustering:
                    self.gg_summary_name = []
                    if '?' in self.arb_fourier_filter_gg_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_gg):
                            last_slash_index = self.arb_fourier_filter_gg_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_fourier_filter_gg_file[i][:last_slash_index + 1]))
                            file_id = self.arb_fourier_filter_gg_file[i][:self.arb_fourier_filter_gg_file[i].find('?')]
                            aux_dir = self.arb_fourier_filter_gg_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_gg_file[i][:last_slash_index + 1] + fstr]))
                            if i == 0:
                                self.arb_number_first_summary_gg = number_files
                            self.gg_summary_name.append(file_id)
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1
                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_gg_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_fourier_filter_gg_file = aux_arb_file
                        self.WL_gg, self.WL_ell_gg = [], []
                        if len(self.arb_fourier_filter_gg_file) == 0:
                            raise Exception("ConfigError: galaxy clustering requested but the Fourier Filter files for the E-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_fourier_filter_gg_file:
                            wn_ell, wn = self.__read_in_fourier_filter_files(wfile)
                            self.WL_ell_gg.append(wn_ell)
                            self.WL_gg.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary Fourier filters for clustering in the desired format")
                else:
                    self.WL_gg, self.WL_ell_gg = None, None
                if self.clustering:
                    if '?' in self.arb_real_filter_gg_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_gg):
                            last_slash_index = self.arb_real_filter_gg_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_real_filter_gg_file[i][:last_slash_index + 1]))
                            file_id = self.arb_real_filter_gg_file[i][:self.arb_real_filter_gg_file[i].find('?')]
                            aux_dir = self.arb_real_filter_gg_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_gg_file[i][:last_slash_index + 1] + fstr]))
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_gg_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_real_filter_gg_file = aux_arb_file
                        self.RL_gg, self.RL_theta_gg = [], []
                        if len(self.arb_real_filter_gg_file) == 0:
                            raise Exception("ConfigError: galaxy clustering requested but the Real Filter files for the w-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_real_filter_gg_file:
                            wn_ell, wn = self.__read_in_real_filter_files(wfile)
                            self.RL_theta_gg.append(wn_ell)
                            self.RL_gg.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary real space filters for clustering in the desired format")
                else:
                    self.RL_gg, self.RL_theta_gg = None, None
                if self.ggl:
                    self.gm_summary_name = []
                    if '?' in self.arb_fourier_filter_gm_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_gm):
                            last_slash_index = self.arb_fourier_filter_gm_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_fourier_filter_gm_file[i][:last_slash_index + 1]))
                            file_id = self.arb_fourier_filter_gm_file[i][:self.arb_fourier_filter_gm_file[i].find('?')]
                            aux_dir = self.arb_fourier_filter_gm_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_gm_file[i][:last_slash_index + 1] + fstr]))
                            self.gm_summary_name.append(file_id)
                            if i == 0:
                                self.arb_number_first_summary_gm = number_files
                            
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_gm_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_fourier_filter_gm_file = aux_arb_file
                        self.WL_gm, self.WL_ell_gm = [], []
                        if len(self.arb_fourier_filter_gm_file) == 0:
                            raise Exception("ConfigError: GGL requested but the Fourier Filter files for the E-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_fourier_filter_gm_file:
                            wn_ell, wn = self.__read_in_fourier_filter_files(wfile)
                            self.WL_ell_gm.append(wn_ell)
                            self.WL_gm.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary Fourier filters for GGL in the desired format")
                else:
                    self.WL_gm, self.WL_ell_gm = None, None
                if self.ggl:
                    if '?' in self.arb_real_filter_gm_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_gm):
                            last_slash_index = self.arb_real_filter_gm_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_real_filter_gm_file[i][:last_slash_index + 1]))
                            file_id = self.arb_real_filter_gm_file[i][:self.arb_real_filter_gm_file[i].find('?')]
                            aux_dir = self.arb_real_filter_gm_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_gm_file[i][:last_slash_index + 1] + fstr]))
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_gm_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_real_filter_gm_file = aux_arb_file
                        self.RL_gm, self.RL_theta_gm = [], []
                        if len(self.arb_real_filter_gm_file) == 0:
                            raise Exception("ConfigError: GGL requested but the Real Filter files for the gt-mode have not been found, please check the path in " + str(self.config_name))   
                        for wfile in self.arb_real_filter_gm_file:
                            wn_ell, wn = self.__read_in_real_filter_files(wfile)
                            self.RL_theta_gm.append(wn_ell)
                            self.RL_gm.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary real space filters for GGL in the desired format")
                else:
                    self.RL_gm, self.RL_theta_gm = None, None
                if self.cosmicshear:
                    self.mmE_summary_name = []
                    self.mmB_summary_name = []
                    if '?' in self.arb_fourier_filter_mmE_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_mm):
                            last_slash_index = self.arb_fourier_filter_mmE_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_fourier_filter_mmE_file[i][:last_slash_index + 1]))
                            file_id = self.arb_fourier_filter_mmE_file[i][:self.arb_fourier_filter_mmE_file[i].find('?')]
                            aux_dir = self.arb_fourier_filter_mmE_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_mmE_file[i][:last_slash_index + 1] + fstr]))          
                            self.mmE_summary_name.append(file_id)
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_mmE_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_fourier_filter_mmE_file = aux_arb_file
                        self.WL_mmE, self.WL_ell_mmE = [], []
                        if len(self.arb_fourier_filter_mmE_file) == 0:
                            raise Exception("ConfigError: Cosmic Shear requested but the Fourier Filter files for the E-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_fourier_filter_mmE_file:
                            wn_ell, wn = self.__read_in_fourier_filter_files(wfile)
                            self.WL_ell_mmE.append(wn_ell)
                            self.WL_mmE.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary Fourier filters for lensing in the desired format")
                else:
                    self.WL_mmE, self.WL_ell_mmE = None, None
                if self.cosmicshear:
                    if '?' in self.arb_fourier_filter_mmB_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_mm):
                            last_slash_index = self.arb_fourier_filter_mmB_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_fourier_filter_mmB_file[i][:last_slash_index + 1]))
                            file_id = self.arb_fourier_filter_mmB_file[i][:self.arb_fourier_filter_mmB_file[i].find('?')]
                            aux_dir = self.arb_fourier_filter_mmB_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_mmB_file[i][:last_slash_index + 1] + fstr]))
                            if i == 0:
                                self.arb_number_first_summary_mm = number_files
                            self.mmB_summary_name.append(file_id)
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_fourier_filter_mmB_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_fourier_filter_mmB_file = aux_arb_file
                        self.WL_mmB, self.WL_ell_mmB = [], []
                        i_counter = 0
                        if len(self.arb_fourier_filter_mmB_file) == 0:
                            raise Exception("ConfigError: Cosmic Shear requested but the Fourier Filter files for the B-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_fourier_filter_mmB_file:
                            if i_counter < self.arb_number_first_summary_mm:
                                i = 0
                            else:
                                i = 1
                            wn_ell, wn = self.__read_in_fourier_filter_files(wfile)
                            self.WL_ell_mmB.append(wn_ell)
                            if self.arb_fourier_filter_no_B[i]:
                                self.WL_mmB.append(np.zeros_like(wn))
                            else:
                                self.WL_mmB.append(wn)
                            i_counter += 1
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary Fourier filters for lensing in the desired format")
                else:
                    self.WL_mmB, self.WL_ell_mmB = None, None
                if self.cosmicshear:
                    if '?' in self.arb_real_filter_mm_p_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_mm):
                            last_slash_index = self.arb_real_filter_mm_p_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_real_filter_mm_p_file[i][:last_slash_index + 1]))
                            file_id = self.arb_real_filter_mm_p_file[i][:self.arb_real_filter_mm_p_file[i].find('?')]
                            aux_dir = self.arb_real_filter_mm_p_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_mm_p_file[i][:last_slash_index + 1] + fstr]))
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_mm_p_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]
                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_real_filter_mm_p_file = aux_arb_file
                        self.RL_mm_p, self.RL_theta_mm_p = [], []
                        if len(self.arb_real_filter_mm_p_file) == 0:
                            raise Exception("ConfigError: Cosmic Shear requested but the Real Filter files for the +-mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_real_filter_mm_p_file:
                            wn_ell, wn = self.__read_in_real_filter_files(wfile)
                            self.RL_theta_mm_p.append(wn_ell)
                            self.RL_mm_p.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary real space filters for lensing in the desired format")
                else:
                    self.RL_mm_p, self.RL_theta_mm_p = None, None
                if self.cosmicshear:
                    if '?' in self.arb_real_filter_mm_m_file[0]:
                        aux_arb_file = []
                        start_index = 0
                        end_index = 0
                        for i in range(self.arb_number_summary_mm):
                            last_slash_index = self.arb_real_filter_mm_m_file[i].rfind('/')
                            _, _, filenames = next(walk(self.arbitrary_summary_dir + self.arb_real_filter_mm_m_file[i][:last_slash_index + 1]))
                            file_id = self.arb_real_filter_mm_m_file[i][:self.arb_real_filter_mm_m_file[i].find('?')]
                            aux_dir = self.arb_real_filter_mm_m_file[i][:last_slash_index + 1]
                            number_files = len(sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_mm_m_file[i][:last_slash_index + 1] + fstr]))
                            for i_files in range(number_files):
                                aux_arb_file.append(None)
                                end_index += 1

                            aux_arb_file[start_index:end_index] = sorted([fstr for fstr in filenames
                                                                        if file_id in self.arb_real_filter_mm_m_file[i][:last_slash_index + 1] + fstr])
                            for j, wnlogfile in enumerate(aux_arb_file[start_index:end_index]):
                                aux_arb_file[j + start_index] = aux_dir + aux_arb_file[start_index:end_index][j]

                            for i_files in range(number_files):
                                start_index += 1
                        self.arb_real_filter_mm_m_file = aux_arb_file
                        self.RL_mm_m, self.RL_theta_mm_m = [], []
                        if len(self.arb_real_filter_mm_m_file) == 0:
                            raise Exception("ConfigError: Cosmic Shear requested but the Real Filter files for the --mode have not been found, please check the path in " + str(self.config_name))
                        for wfile in self.arb_real_filter_mm_m_file:
                            wn_ell, wn = self.__read_in_real_filter_files(wfile)
                            self.RL_theta_mm_m.append(wn_ell)
                            self.RL_mm_m.append(wn)
                    else:
                        raise Exception("ConfigError: Please pass the arbitrary real space filters for lensing in the desired format")
                else:
                    self.RL_mm_m, self.RL_theta_mm_m = None, None
            else:
                raise Exception("ConfigError: To calculate the arbitrary summary statistics " +
                                "covariance for the Filter functions in fourier and real space must be provided in " +
                                "external table. Must be included in [tabulated inputs " +
                                "files] as 'arb_fourier_filter_gg_file' etc. to go on.")



    def __get_cosebi_tabs(self,
                          config):
        """
        Reads in the ... Allows for an auto-generation of filenames if all files
        are named in the same way and only the numbers for the
        tomographic bin combination is changed. In such a case, replace
        the two bin number with a '?' each.

        Parameters
        ----------
        config : class
            This class holds all the information specified the config 
            file. It originates from the configparser module.

        """
        if 'covCOSEBI settings' in config:
            if 'En_modes' in config['covCOSEBI settings']:
                En_modes = int(config['covCOSEBI settings']['En_modes'])
        else:
            return False
        if 'tabulated inputs files' in config:
            if 'cosebi_directory' in config['tabulated inputs files']:
                self.cosebi_dir = \
                    config['tabulated inputs files']['cosebi_directory']
            elif 'input_directory' in config['tabulated inputs files']:
                self.cosebi_dir = \
                    config['tabulated inputs files']['input_directory']
            else:
                self.cosebi_dir = ''
            if 'wn_log_file' in config['tabulated inputs files']:
                self.wn_log_file = (config['tabulated inputs files']
                                    ['wn_log_file'].replace(" ", "")).split(',')
            if 'wn_lin_file' in config['tabulated inputs files']:
                self.wn_lin_file = (config['tabulated inputs files']
                                    ['wn_lin_file'].replace(" ", "")).split(',')
            if 'Tn_plus_file' in config['tabulated inputs files']:
                self.Tn_plus_file = (config['tabulated inputs files']
                                    ['Tn_plus_file'].replace(" ", "")).split(',')
            if 'Tn_minus_file' in config['tabulated inputs files']:
                self.Tn_minus_file = (config['tabulated inputs files']
                                    ['Tn_minus_file'].replace(" ", "")).split(',')
            if 'Qn_file' in config['tabulated inputs files']:
                self.Qn_file = (config['tabulated inputs files']
                                    ['Qn_file'].replace(" ", "")).split(',')
            if 'Un_file' in config['tabulated inputs files']:
                self.Un_file = (config['tabulated inputs files']
                                    ['Un_file'].replace(" ", "")).split(',')
            if 'wn_gg_file' in config['tabulated inputs files']:
                self.wn_gg_file = (config['tabulated inputs files']
                                    ['wn_gg_file'].replace(" ", "")).split(',')
        else:
            ...

        if self.est_ggl == 'cosebi' or self.est_clust == 'cosebi' and En_modes > 0:
            if self.wn_gg_file is None:
                raise Exception("ConfigError: To calculate the COSEBI for clustering or ggl" +
                                "covariance the W_n_gg kernels must be provided as an " +
                                "external table. Must be included in [tabulated inputs " +
                                "files] as 'wn_gg_file' to go on.")
            if self.est_ggl == 'cosebi':
                if self.Qn_file is None:
                    raise Exception("ConfigError: To calculate the COSEBI " +
                                    "covariance for GGL the Qn kernels must be provided as an " +
                                    "external table. Must be included in [tabulated inputs " +
                                    "files] as 'Qn_file' to go on.")
            if self.est_clust == 'cosebi':
                if self.Un_file is None:
                    raise Exception("ConfigError: To calculate the COSEBI " +
                                    "covariance for GGL the Un kernels must be provided as an " +
                                    "external table. Must be included in [tabulated inputs " +
                                    "files] as 'Un_file' to go on.")
            
        if self.est_shear == 'cosebi' and En_modes > 0:
            if self.wn_log_file is None and \
               self.wn_lin_file is None:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the W_n kernels must be provided as an " +
                                "external table. Must be included in [tabulated inputs " +
                                "files] as 'wn_log_file' or 'wn_lin_file' to go on.")
            if self.Tn_plus_file is None or self.Tn_minus_file is None:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the Tn_pm kernels must be provided as an " +
                                "external table. Must be included in [tabulated inputs " +
                                "files] as 'Tn_plus_file' and 'Tn_minus_file' to go on.")
        
        if self.wn_log_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True) or (self.est_shear == 'cosebi' and self.cosmicshear == True)) and not self.do_arbitrary_obs:
            if '?' in self.wn_log_file[0]:
                last_slash_index = self.wn_log_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.wn_log_file[0][:last_slash_index + 1]) )
                file_id = self.wn_log_file[0][:self.wn_log_file[0].find('?')]
                aux_dir = self.wn_log_file[0][:last_slash_index + 1]
                self.wn_log_file = sorted([fstr for fstr in filenames
                                           if file_id in self.wn_log_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.wn_log_file):
                    self.wn_log_file[i] = aux_dir + self.wn_log_file[i]
            if len(self.wn_log_file) >= En_modes:
                self.wn_log_file = self.wn_log_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the W_n kernels must be provided as an " +
                                "external table. Currently " + str(len(self.wn_log_file)) +
                                " (" + str(self.wn_log_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'wn_log_file' " +
                                "to go on.")
            self.wn_log, wn_ell = [], None
            for wfile in self.wn_log_file:
                wn_ell, wn = self.__read_in_Wn_files(wfile)
                if self.wn_log_ell is not None:
                    if any(abs(wn_ell - self.wn_log_ell) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angular modes of the other "
                                        "files.")
                self.wn_log_ell = wn_ell
                self.wn_log.append(wn)
            self.wn_log = np.array(self.wn_log)
        
        if self.wn_gg_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True)):
            if '?' in self.wn_gg_file[0]:
                last_slash_index = self.wn_gg_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.wn_gg_file[0][:last_slash_index + 1]) )
                file_id = self.wn_gg_file[0][:self.wn_gg_file[0].find('?')]
                aux_dir = self.wn_gg_file[0][:last_slash_index + 1]
                self.wn_gg_file = sorted([fstr for fstr in filenames
                                           if file_id in self.wn_gg_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.wn_gg_file):
                    self.wn_gg_file[i] = aux_dir + self.wn_gg_file[i]
            
            if len(self.wn_gg_file) >= En_modes:
                self.wn_gg_file = self.wn_gg_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the wn_gg kernels must be provided as an " +
                                "external table. Currently " + str(len(self.wn_gg_file)) +
                                " (" + str(self.wn_gg_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'wn_gg_file' " +
                                "to go on.")
            self.wn_gg, wn_gg_ell = [], None
            for wfile in self.wn_gg_file:
                wn_gg_ell, wn_gg = self.__read_in_Wn_files(wfile)
                if self.wn_gg_ell is not None:
                    if any(abs(wn_gg_ell - self.wn_gg_ell) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angular modes of the other "
                                        "files.")
                self.wn_gg_ell = wn_gg_ell
                self.wn_gg.append(wn_gg)
            self.wn_gg = np.array(self.wn_gg)

        if self.Tn_plus_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True) or (self.est_shear == 'cosebi' and self.cosmicshear == True)) and not self.do_arbitrary_obs:
            if '?' in self.Tn_plus_file[0]:
                last_slash_index = self.Tn_plus_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.Tn_plus_file[0][:last_slash_index + 1]) )
                file_id = self.Tn_plus_file[0][:self.Tn_plus_file[0].find('?')]
                aux_dir = self.Tn_plus_file[0][:last_slash_index + 1]
                self.Tn_plus_file = sorted([fstr for fstr in filenames
                                           if file_id in self.Tn_plus_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.Tn_plus_file):
                    self.Tn_plus_file[i] = aux_dir + self.Tn_plus_file[i]
            if len(self.Tn_plus_file) >= En_modes:
                self.Tn_plus_file = self.Tn_plus_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the Tn_plus kernels must be provided as an " +
                                "external table. Currently " + str(len(self.Tn_plus_file)) +
                                " (" + str(self.Tn_plus_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'Tn_plus_file' " +
                                "to go on.")
            self.Tn_plus, Tn_theta = [], None
            for wfile in self.Tn_plus_file:
                Tn_theta, Tn = self.__read_in_Tn_pm_files(wfile)
                if self.Tn_theta is not None:
                    if any(abs(Tn_theta - self.Tn_theta) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angles of the other "
                                        "files.")
                self.Tn_theta = Tn_theta
                self.Tn_plus.append(Tn)
            self.Tn_plus = np.array(self.Tn_plus)

        if self.Tn_minus_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True) or (self.est_shear == 'cosebi' and self.cosmicshear == True)) and not self.do_arbitrary_obs:
            if '?' in self.Tn_minus_file[0]:
                last_slash_index = self.Tn_minus_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.Tn_minus_file[0][:last_slash_index + 1]) )
                file_id = self.Tn_minus_file[0][:self.Tn_minus_file[0].find('?')]
                aux_dir = self.Tn_minus_file[0][:last_slash_index + 1]
                self.Tn_minus_file = sorted([fstr for fstr in filenames
                                           if file_id in self.Tn_minus_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.Tn_minus_file):
                    self.Tn_minus_file[i] = aux_dir + self.Tn_minus_file[i]
            if len(self.Tn_minus_file) >= En_modes:
                self.Tn_minus_file = self.Tn_minus_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the Tn_minus kernels must be provided as an " +
                                "external table. Currently " + str(len(self.Tn_minus_file)) +
                                " (" + str(self.Tn_minus_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'Tn_minus_file' " +
                                "to go on.")
            self.Tn_minus, Tn_theta = [], None
            for wfile in self.Tn_minus_file:
                Tn_theta, Tn = self.__read_in_Tn_pm_files(wfile)
                if self.Tn_theta is not None:
                    if any(abs(Tn_theta - self.Tn_theta) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angles of the other "
                                        "files.")
                self.Tn_theta = Tn_theta
                self.Tn_minus.append(Tn)
            self.Tn_minus = np.array(self.Tn_minus)

        if self.Qn_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True)) and not self.do_arbitrary_obs:
            if '?' in self.Qn_file[0]:
                last_slash_index = self.Qn_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.Qn_file[0][:last_slash_index + 1]) )
                file_id = self.Qn_file[0][:self.Qn_file[0].find('?')]
                aux_dir = self.Qn_file[0][:last_slash_index + 1]
                self.Qn_file = sorted([fstr for fstr in filenames
                                           if file_id in self.Qn_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.Qn_file):
                    self.Qn_file[i] = aux_dir + self.Qn_file[i]
            if len(self.Qn_file) >= En_modes:
                self.Qn_file = self.Qn_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance for ggl the Q_n kernels must be provided as an " +
                                "external table. Currently " + str(len(self.Qn_file)) +
                                " (" + str(self.Qn_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'Qn_file' " +
                                "to go on.")
            self.Qn, Qn_theta = [], None
            for wfile in self.Qn_file:
                Qn_theta, Qn = self.__read_in_Tn_pm_files(wfile)
                if self.Qn_theta is not None:
                    if any(abs(Qn_theta - self.Qn_theta) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angles of the other "
                                        "files.")
                self.Qn_theta = Qn_theta
                self.Qn.append(Qn)
            self.Qn = np.array(self.Qn)
        
        if self.Un_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True)) and not self.do_arbitrary_obs:
            if '?' in self.Un_file[0]:
                last_slash_index = self.Un_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.Un_file[0][:last_slash_index + 1]) )
                file_id = self.Un_file[0][:self.Un_file[0].find('?')]
                aux_dir = self.Un_file[0][:last_slash_index + 1]
                self.Un_file = sorted([fstr for fstr in filenames
                                           if file_id in self.Un_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.Un_file):
                    self.Un_file[i] = aux_dir + self.Un_file[i]
            if len(self.Un_file) >= En_modes:
                self.Un_file = self.Un_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance for ggl the U_n kernels must be provided as an " +
                                "external table. Currently " + str(len(self.Un_file)) +
                                " (" + str(self.Un_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'Un_file' " +
                                "to go on.")
            self.Un, Un_theta = [], None
            for wfile in self.Un_file:
                Un_theta, Un = self.__read_in_Tn_pm_files(wfile)
                if self.Un_theta is not None:
                    if any(abs(Un_theta - self.Un_theta) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes " +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angles of the other "
                                        "files.")
                self.Un_theta = Un_theta
                self.Un.append(Un)
            self.Un = np.array(self.Un)

        if self.wn_lin_file is not None and ((self.est_ggl == 'cosebi' and self.ggl == True) or (self.est_clust == 'cosebi' and self.clustering == True) or (self.est_shear == 'cosebi' and self.cosmicshear == True)) and not self.do_arbitrary_obs:
            if '?' in self.wn_lin_file[0]:
                last_slash_index = self.wn_lin_file[0].rfind('/')
                _, _, filenames = next(walk(self.cosebi_dir + self.wn_lin_file[0][:last_slash_index + 1]) )
                file_id = self.wn_lin_file[0][:self.wn_lin_file[0].find('?')]
                aux_dir = self.wn_lin_file[0][:last_slash_index + 1]
                self.wn_lin_file = sorted([fstr for fstr in filenames
                                           if file_id in self.wn_lin_file[0][:last_slash_index + 1] +fstr])
                for i, wnlogfile in enumerate(self.wn_lin_file):
                    self.wn_lin_file[i] = aux_dir + self.wn_lin_file[i]
            if len(self.wn_lin_file) >= En_modes:
                self.wn_lin_file = self.wn_lin_file[:En_modes]
            else:
                raise Exception("ConfigError: To calculate the COSEBI " +
                                "covariance the W_n kernels must be provided as an " +
                                "external table. Currently " + str(len(self.wn_lin_file)) +
                                " (" + str(self.wn_lin_file) + ") files are given, but " +
                                str(En_modes) + " E_n modes are requested. Must be " +
                                "included in [tabulated inputs files] as 'wn_lin_file' " +
                                "to go on.")
            self.wn_lin, wn_ell = [], None
            for wfile in self.wn_lin_file:
                wn_ell, wn = self.__read_in_Wn_files(wfile)
                if self.wn_lin_ell is not None:
                    if any(abs(wn_ell - self.wn_lin_ell) > 1e-4):
                        raise Exception("ConfigError: The angular ell modes" +
                                        "in file " + path.join(self.cosebi_dir, wfile) +
                                        " don't match the angular modes of the other "
                                        "files.")
                self.wn_lin_ell = wn_ell
                self.wn_lin.append(wn)

    def __zip_to_dicts(self):
        """
        This method stores all private variables from this class, that 
        are needed to calculate the covariance, into dictionaries. It 
        creates a separate dictionary needed to write the parameters 
        back into a save_configs file.

        """

        keys = ['z', 'nz', 'value_loc_in_bin', 'tomos_6x2pt']
        values = [self.zet_clust_z, self.zet_clust_nz,
                  self.value_loc_in_clustbin, self.tomos_6x2pt_clust]
        self.zet_clust = dict(zip(keys, values))

        keys = ['z', 'photoz', 'value_loc_in_bin']
        values = [self.zet_lens_z, self.zet_lens_photoz,
                  self.value_loc_in_lensbin]
        self.zet_lens = dict(zip(keys, values))

        keys = ['z', 'pz']
        values = [self.zet_csmf_z, self.zet_csmf_pz]
        self.zet_csmf = dict(zip(keys, values))

        keys = ['V_max', 'f_tomo']
        values = [self.V_max, self.f_tomo]
        self.csmf = dict(zip(keys, values))


        keys = []
        values = []
        if self.zet_clust_file is not None:
            keys.extend(['zclust_directory', 'zclust_file',
                         'value_loc_in_clustbin'])
            values.extend([self.zet_clust_dir, self.zet_clust_file,
                           self.value_loc_in_clustbin])
            if self.tomos_6x2pt_clust is not None:
                keys.extend('tomos_6x2pt_clust')
                values.extend(self.tomos_6x2pt_clust)
        if self.zet_lens_file is not None:
            keys.extend(['zlens_directory', 'zlens_file',
                         'value_loc_in_lensbin'])
            values.extend([self.zet_lens_dir, self.zet_lens_file,
                           self.value_loc_in_lensbin])
        if self.zet_csmf_file is not None:
            keys.extend(['zcsmf_directory', 'zcsmf_file'])
            values.extend([self.zet_csmf_dir, self.zet_csmf_file])
        self.zet_input = dict(zip(keys, values))
        if self.zet_clust_file is not None:
            self.zet_input['zclust_file'] = \
                ', '.join(map(str, self.zet_clust_file))
        if self.zet_lens_file is not None:
            self.zet_input['zlens_file'] = \
                ', '.join(map(str, self.zet_lens_file))
        if self.zet_csmf_file is not None:
            self.zet_input['zcsmf_file'] = \
                ', '.join(map(str, self.zet_csmf_file))

        keys = ['theta_mm', 'npair_mm', 'theta_gm', 'npair_gm', 'theta_gg',
                'npair_gg']
        values = [self.theta_npair_mm, self.npair_mm, self.theta_npair_gm,
                  self.npair_gm, self.theta_npair_gg, self.npair_gg, ]
        self.npair = dict(zip(keys, values))

        keys = ['k', 'z', 'mm', 'gm', 'gg']
        values = [self.Pxy_k, self.Pxy_z, self.Pmm_tab, self.Pgm_tab,
                  self.Pgg_tab, ]
        self.Pxy_tab = dict(zip(keys, values))

        keys = ['ell','ell_clust', 'ell_lens', 'tomo_clust', 'tomo_lens', 'mm', 'gm', 'gg']
        values = [self.Cxy_ell, self.Cxy_ell_clust, self.Cxy_ell_lens, self.Cxy_tomo_clust, self.Cxy_tomo_lens,
                  self.Cmm_tab, self.Cgm_tab, self.Cgg_tab, ]
        self.Cxy_tab = dict(zip(keys, values))

        keys = ['z', 'bias']
        values = [self.effbias_z, self.effbias]
        self.effbias_tab = dict(zip(keys, values))

        keys = ['M', 'cen', 'sat']
        values = [self.mor_M, self.mor_cen, self.mor_sat]
        self.mor_tab = dict(zip(keys, values))

        keys = ['M', 'Mbins', 'cen', 'sat']
        values = [self.occprob_M, self.occprob_Mbins, self.occprob_cen,
                  self.occprob_sat]
        self.occprob_tab = dict(zip(keys, values))

        keys = ['M', 'cen', 'sat']
        values = [self.occnum_M, self.occnum_cen, self.occnum_sat]
        self.occnum_tab = dict(zip(keys, values))

        keys = ['log10k', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm', 'gggm', 'gggg']
        values = [self.tri_log10k, self.tri_z, self.tri_mmmm, self.tri_mmgm,
                  self.tri_gmgm, self.tri_ggmm, self.tri_gggm, self.tri_gggg]
        self.tri_tab = dict(zip(keys, values))

        keys = ['wn_log_ell', 'wn_log', 'wn_lin_ell', 'wn_lin', 'norms',
                'roots','Tn_pm_theta','Tn_p', 'Tn_m', 'wn_gg_ell', 'wn_gg', 'Qn_theta', 'Un_theta', 'Qn', 'Un']
        values = [self.wn_log_ell, self.wn_log, self.wn_lin_ell, self.wn_lin,
                  self.norms, self.roots, self.Tn_theta, self.Tn_plus, self.Tn_minus, self.wn_gg_ell, self.wn_gg, self.Qn_theta, self.Un_theta,
                  self.Qn, self.Un]
        self.cosebis = dict(zip(keys, values))
        if self.do_arbitrary_obs:
            keys = ['ell_gg', 'WL_gg', 'theta_gg', 'RL_gg', 'ell_gm', 'WL_gm', 'theta_gm', 'RL_gm',
                    'ell_mmE', 'WL_mmE', 'ell_mmB', 'WL_mmB', 'theta_mm_p', 'RL_mm_p', 'theta_mm_m', 'RL_mm_m',
                    'number_summary_gg', 'number_summary_gm', 'number_summary_mm',
                    'arb_number_first_summary_gg', 'arb_number_first_summary_gm', 'arb_number_first_summary_mm',
                    'gg_summary_name', 'gm_summary_name','mmE_summary_name','mmB_summary_name',]
            values = [self.WL_ell_gg, self.WL_gg, self.RL_theta_gg, self.RL_gg, self.WL_ell_gm, self.WL_gm, self.RL_theta_gm, self.RL_gm,
                    self.WL_ell_mmE, self.WL_mmE, self.WL_ell_mmB, self.WL_mmB, self.RL_theta_mm_p, self.RL_mm_p, self.RL_theta_mm_m, self.RL_mm_m,
                    self.arb_number_summary_gg, self.arb_number_summary_gm, self.arb_number_summary_mm, 
                    self.arb_number_first_summary_gg, self.arb_number_first_summary_gm, self.arb_number_first_summary_mm,
                    self.gg_summary_name, self.gm_summary_name, self.mmE_summary_name, self.mmB_summary_name]
            self.arbitrary_summary = dict(zip(keys, values))
        else:
            self.arbitrary_summary = dict([])
        keys = []
        values = []
        if self.npair_gg_file is not None or \
           self.npair_gm_file is not None or \
           self.npair_mm_file is not None:
            keys.append('npair_directory')
            values.append(self.npair_dir)
        if self.npair_mm_file is not None:
            keys.append('npair_mm_file')
            values.append(self.npair_mm_file)
        if self.npair_gm_file is not None:
            keys.append('npair_gm_file')
            values.append(self.npair_gm_file)
        if self.npair_gg_file is not None:
            keys.append('npair_gg_file')
            values.append(self.npair_gg_file)
        if self.Pmm_file is not None or self.Pgm_file is not None or \
           self.Pgg_file is not None:
            keys.append('powspec_directory')
            values.append(self.powspec_dir)
        if self.Pmm_file is not None:
            keys.append('Pmm_file')
            values.append(self.Pmm_file)
        if self.Pgm_file is not None:
            keys.append('Pgm_file')
            values.append(self.Pgm_file)
        if self.Pgg_file is not None:
            keys.append('Pgg_file')
            values.append(self.Pgg_file)
        if self.Cmm_file is not None or self.Cgm_file is not None or \
           self.Cgg_file is not None:
            keys.append('Cell_directory')
            values.append(self.Cell_dir)
        if self.Cmm_file is not None:
            keys.append('Cmm_file')
            values.append(', '.join(map(str, self.Cmm_file)))
        if self.Cgm_file is not None:
            keys.append('Cgm_file')
            values.append(', '.join(map(str, self.Cgm_file)))
        if self.Cgg_file is not None:
            keys.append('Cgg_file')
            values.append(', '.join(map(str, self.Cgg_file)))
        if self.effbias_file is not None:
            keys.extend(['effbias_directory', 'effbias_file'])
            values.extend([self.effbias_dir, self.effbias_file])
        if self.mor_cen_file is not None or self.mor_sat_file is not None:
            keys.append('mor_directory')
            values.append(self.mor_dir)
        if self.mor_cen_file is not None:
            keys.append('mor_cen_file')
            values.append(self.mor_cen_file)
        if self.mor_sat_file is not None:
            keys.append('mor_sat_file')
            values.append(self.mor_sat_file)
        if self.occprob_cen_file is not None or \
           self.occprob_sat_file is not None:
            keys.append('occprob_directory')
            values.append(self.occprob_dir)
        if self.occprob_cen_file is not None:
            keys.append('occprob_cen_file')
            values.append(self.occprob_cen_file)
        if self.occprob_sat_file is not None:
            keys.append('occprob_sat_file')
            values.append(self.occprob_sat_file)
        if self.occnum_cen_file is not None or \
           self.occnum_sat_file is not None:
            keys.append('occnum_directory')
            values.append(self.occnum_dir)
        if self.occnum_cen_file is not None:
            keys.append('occnum_cen_file')
            values.append(self.occnum_cen_file)
        if self.occnum_sat_file is not None:
            keys.append('occnum_sat_file')
            values.append(self.occnum_sat_file)
        if self.tri_file is not None:
            keys.extend(['trispec_directory', 'trispec_file'])
            values.extend([self.tri_dir, self.tri_file])
        if self.wn_log_file is not None or \
           self.wn_lin_file is not None or \
           self.norm_file is not None or \
           self.root_file is not None:
            keys.append('cosebi_directory')
            values.append(self.cosebi_dir)
        if self.wn_log_file is not None:
            keys.append('wn_log_file')
            values.append(self.wn_log_file)
        if self.wn_lin_file is not None:
            keys.append('wn_lin_file')
            values.append(self.wn_lin_file)
        if self.norm_file is not None:
            keys.append('norm_file')
            values.append(self.norm_file)
        if self.root_file is not None:
            keys.append('root_file')
            values.append(self.root_file)
        if self.Tn_plus_file is not None:
            keys.append('Tn_plus_file')
            values.append(self.Tn_plus_file)
        if self.Tn_minus_file is not None:
            keys.append('Tn_minus_file')
            values.append(self.Tn_minus_file)
        if self.wn_gg_file is not None:
            keys.append('wn_gg_file')
            values.append(', '.join(map(str, self.wn_gg_file)))
        if self.Qn_file is not None:
            keys.append('Qn_file')
            values.append(', '.join(map(str, self.Qn_file)))
        if self.Un_file is not None:
            keys.append('Un_file')
            values.append(', '.join(map(str, self.Un_file)))
        
        #', '.join(map(str, self.zet_lens_file
        
        if self.arbitrary_summary_dir is not None:
            keys.append('arbitrary_summary_dir')
            values.append(self.arbitrary_summary_dir)
        if self.arb_fourier_filter_gg_file is not None:
            keys.append('arb_fourier_filter_gg_file')
            values.append(', '.join(map(str, self.arb_fourier_filter_gg_file_save)))
        if self.arb_real_filter_gg_file is not None:
            keys.append('arb_real_filter_gg_file')
            values.append(', '.join(map(str, self.arb_real_filter_gg_file_save)))
        if self.arb_fourier_filter_gm_file is not None:
            keys.append('arb_fourier_filter_gm_file')
            values.append(', '.join(map(str, self.arb_fourier_filter_gm_file_save)))
        if self.arb_real_filter_gm_file is not None:
            keys.append('arb_real_filter_gm_file')
            values.append(', '.join(map(str, self.arb_real_filter_gm_file_save)))
        if self.arb_fourier_filter_mmE_file is not None:
            keys.append('arb_fourier_filter_mmE_file')
            values.append(', '.join(map(str, self.arb_fourier_filter_mmE_file_save)))
        if self.arb_fourier_filter_mmE_file is not None:
            keys.append('arb_fourier_filter_mmB_file')
            values.append(', '.join(map(str, self.arb_fourier_filter_mmB_file_save)))
        if self.arb_real_filter_mm_p_file is not None:
            keys.append('arb_real_filter_mm_p_file')
            values.append(', '.join(map(str, self.arb_real_filter_mm_p_file_save)))
        if self.arb_real_filter_mm_m_file is not None:
            keys.append('arb_real_filter_mm_m_file')
            values.append(', '.join(map(str, self.arb_real_filter_mm_m_file_save)))


        self.tab_input = dict(zip(keys, values))
        if self.npair_mm_file is not None:
            self.tab_input['npair_mm_file'] = \
                ', '.join(map(str, self.npair_mm_file))
        if self.npair_gm_file is not None:
            self.tab_input['npair_gm_file'] = \
                ', '.join(map(str, self.npair_gm_file))
        if self.npair_gg_file is not None:
            self.tab_input['npair_gg_file'] = \
                ', '.join(map(str, self.npair_gg_file))
        if self.occprob_cen_file is not None:
            self.tab_input['occprob_cen_file'] = \
                ', '.join(map(str, self.occprob_cen_file))
        if self.occprob_sat_file is not None:
            self.tab_input['occprob_sat_file'] = \
                ', '.join(map(str, self.occprob_sat_file))
        if self.wn_log_file is not None:
            self.tab_input['wn_log_file'] = \
                ', '.join(map(str, self.wn_log_file))
        if self.wn_lin_file is not None:
            self.tab_input['wn_lin_file'] = \
                ', '.join(map(str, self.wn_lin_file))
        if self.Tn_plus_file is not None:
            self.tab_input['Tn_plus_file'] = \
                ', '.join(map(str, self.Tn_plus_file))
        if self.Tn_minus_file is not None:
            self.tab_input['Tn_minus_file'] = \
                ', '.join(map(str, self.Tn_minus_file))
        return True

    def __write_params(self):
        """
        This methods appends to the save_configs file created in the 
        Input class which contains all the parameters that are not None, 
        whether they have been explicitly set in the original 
        configuration file or whether they have been implicitly set by 
        this class.

        """
        if not self.save_configs:
            return False

        params_used = configparser.ConfigParser()
        if len(self.zet_input) > 0:
            params_used['redshift'] = self.zet_input
        if len(self.tab_input) > 0:
            params_used['tabulated inputs files'] = self.tab_input

        if self.output_dir is None:
            self.output_dir = ''
        with open(
                path.join(self.output_dir, self.save_configs), 'a') as paramfile:
            params_used.write(paramfile)
        if self.output_dir == '':
            self.output_dir = None

        return True

    def read_input(self,
                   config_name='config.ini'):
        """
        This method reads in all the look-up tables necessary and 
        optional to calculate the covariance matrix. It also checks 
        whether compulsory parameters have been set. The parameters are 
        then zipped into dictionaries. Finally, a file is produced that 
        lists all explicitly and implicitly set parameters for future 
        reference.

        Parameters
        ----------
        config_name : string
            default : 'config.ini'
            Name of the configuration file.

        Return
        ---------
        a dictionary with the following keys (also a dictionary each)
        'zclust' : dictionary
            Look-up table for the clustering redshifts and their number 
            of tomographic bins. Relevant for clustering and galaxy-
            galaxy lensing estimators.
            Possible keys: 'z', 'nz', 'value_loc_in_bin'
        'zlens' : dictionary
            Look-up table for the lensing redshifts and their number of 
            tomographic bins. Relevant for cosmic shear and galaxy-
            galaxy lensing estimators.
            Possible keys: 'z', 'photoz', 'value_loc_in_bin'
        'npair' : dictionary
            Look-up table for the number of galaxy bins per angular bin.
            Can have an arbitray number of angular bins that will be
            rebinned if necessary.
            Possible keys: 'theta_mm', 'npair_mm', 'theta_gm', 
                           'npair_gm', 'theta_gg', 'npair_gg'
        'Pxy' : dictionary
            Look-up table for the power spectra (matter-matter, tracer-
            tracer, matter-tracer, optional).
            Possible keys: 'k', 'z', 'mm', 'gm', 'gg'
        'Cxy' : dictionary
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer, optional).
            Possible keys: 'ell', 'tomo', 'mm', 'gm', 'gg'
        'effbias' : dictionary
            Look-up table for the effective bias as a function of 
            redshift (optional).
            Possible keys: 'z', 'bias'
        'mor' : dictionary
            Look-up table for the mass-observable relation (optional). 
            Possible keys: 'M', 'cen', 'sat'
        'occprob' : dictionary
            Look-up table for the occupation probability as a function 
            of halo mass per galaxy sample (optional).
            Possible keys: 'M', 'Mbins', 'cen', 'sat'
        'occnum' : dictionary
            Look-up table for the occupation number as a function of 
            halo mass per galaxy sample (optional).
            Possible keys: 'M', 'cen', 'sat'
        'tri': dictionary
            Look-up table for the trispectra (for all combinations of 
            matter 'm' and tracer 'g', optional).
            Possible keys: 'log10k', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
                           'gggm', 'gggg'
        'COSEBIs' : dictionary
            Look-up tables for calculating COSEBIs. Currently the code 
            cannot generate the roots and normalizations for the kernel
            functions itself, so either the kernel's roots and 
            normalizations or the kernel (W_n) itself must be given.
            Possible keys: 'wn_log_ell', 'wn_log', 'wn_lin_ell', 
                           'wn_lin', 'norms', 'roots'

        """

        config = configparser.ConfigParser()
        config.read(config_name)
        self.__read_config_for_consistency_checks(config, config_name)
        self.__read_in_z_files(config, config_name)
        self.__read_in_csmf_files(config)
        self.__read_in_bias_files(config, config_name)
        self.__get_npair_tabs(config)
        self.__get_powspec_tabs(config)
        self.__get_Cell_tabs(config)  
        self.__get_effbias_tabs(config)
        self.__get_mor_tabs(config)
        self.__get_occprob_tabs(config)
        self.__get_occnum_tabs(config)
        self.__get_trispec_tabs(config)
        #self.__get_cosebi_tabs(config)
        self.__get_arbitrary_filter_tabs(config)
        self.__zip_to_dicts()
        self.__write_params()

        return {'zclust': self.zet_clust,
                'zlens': self.zet_lens,
                'zcsmf': self.zet_csmf,
                'csmf': self.csmf,
                'npair': self.npair,
                'Pxy': self.Pxy_tab,
                'Cxy': self.Cxy_tab,
                'effbias': self.effbias_tab,
                'zet_dep_bias' : self.bias_bz,
                'mor': self.mor_tab,
                'occprob': self.occprob_tab,
                'occnum': self.occnum_tab,
                'tri': self.tri_tab,
                'COSEBIs': self.cosebis,
                'arb_summary': self.arbitrary_summary}

