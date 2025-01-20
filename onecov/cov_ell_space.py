import numpy as np
import time
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import simpson
from scipy.special import j1
import multiprocessing as mp
import healpy as hp
import levin
import camb
from camb import model
from scipy.signal import argrelextrema


from scipy.interpolate import RegularGridInterpolator


try:
    from onecov.cov_output import Output
    from onecov.cov_polyspectra import PolySpectra
except:
    from cov_output import Output
    from cov_polyspectra import PolySpectra


            


class CovELLSpace(PolySpectra):
    """
    This class calculates the ell-space covariance for angular power
    spectra estimators (kappa-kappa, tracer-tracer and tracer-kappa).
    Inherits the functionality of the PolySpectra class.

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
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
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

    Attributes
    ----------
    see CovKSpace class
    ellrange : array
        Multipoles at which the covariance is calculated
    deg2torad2 : float
        Conversion factor from degrees to radian squared
    arcmin2torad2 : float
        Conversion factor from arcminutes to radian squared
    spline_zclust : list
        List of spline objects for the redshift distributuon for each
        tomographic bin of the clustering analysis
    spline_zlens : list
        List of spline objects for the redshift distributuon for each
        tomographic bin of the lensing analysis
    los_integration_chi : array
        Values of the comoving distance at which the line of sight integration
        is carriend out.
    spline_z_of_chi : spline_object
        ...
    spline_lensweight : list
        List of spline objects for the lensing weight for each
        tomographic bin of the lensing analysis
    Cell_gg : array
        Array storing the C_ell of the clustering analysis with shape
        (# ell modes, # sample bins, # tomographic bins,
        # tomographic bins)
    Cell_gm, Cell_gkappa : array
        Array storing the C_ell of the galaxy-galaxy lensing analysis
        with shape (# ell modes, # sample bins, # tomographic bins,
        # tomographic bins)
    Cell_mm, Cell_kappakappa : array
        Array storing the C_ell of the cosmic shear analysis with shape
        (# ell modes, # tomographic bins, # tomographic bins)

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_ell_space import CovELLSpace
    inp = Input()
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    covell = CovELLSpace(covterms, observables, output, cosmo, bias,
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
        PolySpectra.__init__(self,
                             0,
                             cov_dict,
                             obs_dict,
                             cosmo_dict,
                             bias_dict,
                             hod_dict,
                             survey_params_dict,
                             prec,
                             read_in_tables)
        self.do_not_update_ell = False
        self.checked_input_cells = False
        self.ellrange_spec = None
        self.ellrange_photo = None
        self.est_shear = obs_dict['observables']['est_shear']
        self.est_ggl = obs_dict['observables']['est_ggl']
        self.est_clust = obs_dict['observables']['est_clust']
        self.clustering_z = obs_dict['observables']['clustering_z']
        self.csmf = obs_dict['observables']['csmf']
        if self.csmf:
            self.log10csmf_mass_bins = obs_dict['observables']['csmf_log10M_bins'][:-1] + (obs_dict['observables']['csmf_log10M_bins'][1:] - obs_dict['observables']['csmf_log10M_bins'][:-1])/2
            self.deltaM_csmf = 10**obs_dict['observables']['csmf_log10M_bins'][1:] - 10**obs_dict['observables']['csmf_log10M_bins'][:-1]
            self.Vmax = read_in_tables['csmf']['V_max']
            self.f_tomo = np.zeros(self.n_tomo_csmf)
            self.f_tomo[:] = read_in_tables['csmf']['f_tomo']
            self.csmf_number_mass_bins = len(self.log10csmf_mass_bins)
            self.csmf_diagonal = obs_dict['observables']['csmf_diagonal']
            if self.csmf_diagonal and self.n_tomo_csmf != self.csmf_number_mass_bins:
                raise Exception("ConfigError: You ask for the diagonal-only mode in the stellar mass function, i.e. considering only bins i with tomographic bin j. However, "
                                "the number of tomographic bins is " + str(self.n_tomo_csmf) + ", while the number of mass bins is " + str(self.csmf_number_mass_bins) +
                                ". Please fix this in the config so that they are equal, or deactivate 'csmf_diagonal'.") 
        self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
        self.Cells, self.tab_bools = self.__check_for_tabulated_Cells(read_in_tables['Cxy'])
        
        if (self.mm and self.est_shear == 'C_ell') and ((self.gm and self.est_ggl == 'C_ell') or (self.gg and self.est_clust == 'C_ell')):
            if (self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is None) or (self.ellrange_clustering_ul is None and self.ellrange_lensing_ul is not None):
                raise Exception("ConfigError: You require the C_ell covariance for lensing and clustering or ggl. However, you only specified the ellrange for lensing and not for clustering/GGL or vice versa. Please fix this in the config file under covELLspace settings.")     
        self.integration_intervals = obs_dict['THETAspace']['integration_intervals']
        self.deg2torad2 = 180 / np.pi * 180 / np.pi
        self.arcmin2torad2 = 60*60 * self.deg2torad2
        if self.redshift_dep_bias:
            self.bias_of_zet = [] 
            for i_tomo in range(self.n_tomo_clust):
                self.bias_of_zet.append(UnivariateSpline(self.cosmology.comoving_distance(read_in_tables['zclust']['z']).value * self.cosmology.h,read_in_tables['zet_dep_bias'][i_tomo,:],k=3,s=0,ext=0))
        self.__set_redshift_distribution_splines(obs_dict['ELLspace'], read_in_tables)
        self.__check_krange_support(
            obs_dict, cosmo_dict, bias_dict, hod_dict, prec)
        if obs_dict['ELLspace']['pixelised_cell']:
            obs_dict['ELLspace']['ellmax'] = int(3*obs_dict['ELLspace']['pixel_Nside'] + 1)
            self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
            integer_ell = np.copy(self.ellrange.astype(int))
            self.pixel_weight = (hp.sphtfunc.pixwin(obs_dict['ELLspace']['pixel_Nside']))[integer_ell]
            self.pixelweight_matrix = (self.pixel_weight**2)[:,None]*(self.pixel_weight**2)[None,:]
            self.Cells, self.tab_bools = None, [False, False, False]
        else:
            self.pixel_weight = np.ones_like(self.ellrange)
        self.camb_pars.set_matter_power(kmax=self.mass_func.k[-1], redshifts = [0])
        results = camb.get_results(self.camb_pars_new)
        self.camb_pars_new.InitPower.set_params(ns=cosmo_dict['ns'],
                                As = 1.8e-9/results.get_sigma8()**2*cosmo_dict['sigma8']**2)
        self.num_cores_save = self.num_cores       
        self.calc_survey_area(survey_params_dict)
        self.get_Cells(obs_dict, output_dict,
                       bias_dict, iA_dict, hod_dict, prec, read_in_tables)
        if not self.cov_dict['gauss']:
            self.__set_lensweight_splines(obs_dict['ELLspace'], iA_dict)
        
    
    def __check_krange_support(self,
                               obs_dict,
                               cosmo_dict,
                               bias_dict,
                               hod_dict,
                               prec):
        r"""
        Performs consistency checks whether the k-range support is big enough
        to house a given ell_range calculations for the Cells. If not, the ranges
        are updated accordingly and the user is notified.

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
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.
        cosmo_dict : dictionary
            Specifies all cosmological parameters. To be passed from the
            read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
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

        Returns:
        ---------
        True

        """
        update_massfunc, update_ellrange, kmin, kmax, ellmin, ellmax, ellbins = \
            self.consistency_checks_for_Cell_calculation(
                obs_dict, cosmo_dict, prec['powspec'],
                self.ellrange, self.los_integration_chi, self.do_not_update_ell)
        if update_massfunc:
            self.mass_func.update(lnk_min=np.log(10**kmin),
                                  lnk_max=np.log(10**kmax))
            self.update_mass_func(
                self.mass_func.z, bias_dict, hod_dict, prec)

        if update_ellrange:
            obs_dict['ELLspace']['ell_min'] = ellmin
            obs_dict['ELLspace']['ell_max'] = ellmax
            obs_dict['ELLspace']['ell_bins'] = ellbins
            obs_dict['ELLspace']['ell_type'] = 'log'
            self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
        return True

    def __set_multipoles(self,
                         covELLspacesettings):
        r"""
        Calculates the multioples at which the covariance is calculated.
        Also sets the multipoles for the binned ell-space covariance for
        the spectroscopic and photometric samples.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the multipoles at which the covariance is
            evaluated.

        Returns
        -------
        ellrange : array
            with shape (ell bins)

        """
        self.ellrange_clustering_ul = None
        self.ellrange_clustering = None
        if covELLspacesettings['ell_type_clustering'] is not None and covELLspacesettings['ell_bins_clustering'] is not None and covELLspacesettings['ell_max_clustering'] is not None and covELLspacesettings['ell_min_clustering'] is not None:
            if (self.est_clust == "C_ell" and self.gg) or (self.est_ggl == "C_ell" and self.gm): 
                if covELLspacesettings['ell_type_clustering'] == 'lin':
                    self.ellrange_clustering_ul = np.linspace(
                        covELLspacesettings['ell_min_clustering'],
                        covELLspacesettings['ell_max_clustering'],
                        covELLspacesettings['ell_bins_clustering'] + 1).astype(int)
                    self.ellrange_clustering = .5 * (self.ellrange_clustering_ul[1:] + self.ellrange_clustering_ul[:-1])
                else:
                    self.ellrange_clustering_ul = np.unique(np.geomspace(covELLspacesettings['ell_min_clustering'], covELLspacesettings['ell_max_clustering'], covELLspacesettings['ell_bins_clustering'] + 1).astype(int))
                    if len(self.ellrange_clustering_ul) != covELLspacesettings['ell_bins_clustering'] + 1:
                        print("InputWarning: you required", covELLspacesettings['ell_bins_clustering'], "logarithmically-spaced clustering bins.",
                              "However, the specified clustering ell-range from ", covELLspacesettings['ell_min_clustering'],  "to", covELLspacesettings['ell_max_clustering'], "does not support that many unique bins.",
                              "Adjusting the number of bins to", len(self.ellrange_clustering_ul) - 1, ". The bin boundaries are:")
                        print(self.ellrange_clustering_ul) 
                    self.ellrange_clustering = np.exp(.5 * (np.log(self.ellrange_clustering_ul[1:])
                                        + np.log(self.ellrange_clustering_ul[:-1])))
        self.ellrange_lensing_ul = None
        self.ellrange_lensing = None
        if covELLspacesettings['ell_type_lensing'] is not None and covELLspacesettings['ell_bins_lensing'] is not None and covELLspacesettings['ell_max_lensing'] is not None and covELLspacesettings['ell_min_lensing'] is not None:
            if (self.est_shear == "C_ell" and self.mm): 
                if covELLspacesettings['ell_type_lensing'] == 'lin':
                    self.ellrange_lensing_ul = np.linspace(
                        covELLspacesettings['ell_min_lensing'],
                        covELLspacesettings['ell_max_lensing'],
                        covELLspacesettings['ell_bins_lensing'] + 1).astype(int)
                    self.ellrange_lensing = .5 * (self.ellrange_lensing_ul[1:] + self.ellrange_lensing_ul[:-1])
                else:
                    self.ellrange_lensing_ul = np.unique(np.geomspace(covELLspacesettings['ell_min_lensing'], covELLspacesettings['ell_max_lensing'], covELLspacesettings['ell_bins_lensing'] + 1).astype(int))
                    if len(self.ellrange_lensing_ul) != covELLspacesettings['ell_bins_lensing'] + 1:
                        print("InputWarning: you required", covELLspacesettings['ell_bins_lensing'],"logarithmically-spaced lensing bins.",
                              "However, the specified clustering ell-range from", covELLspacesettings['ell_min_lensing'],  "to", covELLspacesettings['ell_max_lensing'], "does not support that many unique bins.", 
                              "Adjusting the number of bins to", len(self.ellrange_lensing_ul) - 1, ". The bin boundaries are:") 
                        print(self.ellrange_lensing_ul)
                    self.ellrange_lensing = np.exp(.5 * (np.log(self.ellrange_lensing_ul[1:])
                                        + np.log(self.ellrange_lensing_ul[:-1])))


        if covELLspacesettings['n_spec'] is not None and covELLspacesettings['n_spec'] != 0:
            if covELLspacesettings['ell_spec_type'] == 'lin':
                self.ellrange_spec_ul = np.linspace(
                    covELLspacesettings['ell_spec_min'],
                    covELLspacesettings['ell_spec_max'],
                    covELLspacesettings['ell_spec_bins'] + 1)
                self.ellrange_spec = .5 * (self.ellrange_spec_ul[1:] + self.ellrange_spec_ul[:-1])
            else:
                self.ellrange_spec_ul = np.geomspace(covELLspacesettings['ell_spec_min'], covELLspacesettings['ell_spec_max'], covELLspacesettings['ell_spec_bins'] + 1)
                self.ellrange_spec = np.exp(.5 * (np.log(self.ellrange_spec_ul[1:])
                                    + np.log(self.ellrange_spec_ul[:-1])))
            if covELLspacesettings['ell_photo_type'] == 'lin':
                self.ellrange_photo_ul = np.linspace(
                    covELLspacesettings['ell_photo_min'],
                    covELLspacesettings['ell_photo_max'],
                    covELLspacesettings['ell_photo_bins'] + 1)
                self.ellrange_photo = .5 * (self.ellrange_photo_ul[1:] + self.ellrange_photo_ul[:-1])
            else:
                self.ellrange_photo_ul = np.geomspace(covELLspacesettings['ell_photo_min'], covELLspacesettings['ell_photo_max'], covELLspacesettings['ell_photo_bins'] + 1)
                self.ellrange_photo = np.exp(.5 * (np.log(self.ellrange_photo_ul[1:])
                                    + np.log(self.ellrange_photo_ul[:-1])))

        if covELLspacesettings['ell_type'] == 'lin':
            return np.linspace(
                covELLspacesettings['ell_min'],
                covELLspacesettings['ell_max'],
                covELLspacesettings['ell_bins'])
        else:
            if covELLspacesettings['limber']:
                return np.geomspace(covELLspacesettings['ell_min'], covELLspacesettings['ell_max'], covELLspacesettings['ell_bins'])
            else:
                return np.unique(np.geomspace(covELLspacesettings['ell_min'], covELLspacesettings['ell_max'], covELLspacesettings['ell_bins']).astype(int)).astype(float)
        
        

    def __set_redshift_distribution_splines(self,
                                            covELLspacesettings,
                                            read_in_tables):
        r"""
        Sets the splines the redshift distributions for later integration along
        the line-of-sight.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the order of the polynomial for the interpolation
            of redshift distributions

        """
        los_integration_z = np.linspace(
            0, self.zet_max, covELLspacesettings['integration_steps'])
        self.spline_z_of_chi = UnivariateSpline(
            self.cosmology.comoving_distance(los_integration_z).value
            * self.cosmology.h,
            los_integration_z,
            k=1, s=0, ext=0)

        self.chimin = 10
        self.chimax = self.cosmology.comoving_distance(self.zet_max).value \
            * self.cosmology.h
        self.los_integration_chi = np.geomspace(
            self.chimin, self.chimax, covELLspacesettings['integration_steps'])

        self.spline_zclust = []
        self.spline_zlens = []
        self.spline_zcsmf = []
        self.chi_min_clust = np.zeros(self.n_tomo_clust)
        self.chi_max_clust = np.zeros(self.n_tomo_clust)
        if self.gm or self.gg:
            dzdchi_clust = self.cosmology.efunc(self.zet_clust['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        if self.gm or self.mm:
            dzdchi_lens = self.cosmology.efunc(self.zet_lens['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        if self.csmf:
            dzdchi_csmf= self.cosmology.efunc(self.zet_csmf['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        for tomo in range(self.n_tomo_clust):
            norm = simpson(self.zet_clust['nz'][tomo], x = self.zet_clust['z'])
            self.zet_clust['nz'][tomo] /= norm
            if covELLspacesettings['nz_polyorder'] == 0:
                self.spline_zclust.append(interp1d(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust,
                    kind='nearest-up', fill_value=0))
            else:
                self.spline_zclust.append(UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust,
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
            min_check = 0
            for i_z, zet in enumerate(self.los_integration_chi, start = 1):     
                cdf = simpson(
                    self.spline_zclust[tomo](self.los_integration_chi[:i_z]),  x = self.los_integration_chi[:i_z])
                if cdf > 0. and min_check == 0:
                    if i_z > 1:
                        self.chi_min_clust[tomo] = self.los_integration_chi[i_z -2]
                    else:
                        self.chi_min_clust[tomo] = self.los_integration_chi[0]
                    min_check = 1
                if cdf >= 0.9999 or i_z == len(self.los_integration_chi) - 1:
                    if i_z < len(self.los_integration_chi) - 1:
                        self.chi_max_clust[tomo] = self.los_integration_chi[i_z +1]
                    else:
                        self.chi_max_clust[tomo] = self.los_integration_chi[-1]
                    break
            if self.redshift_dep_bias:
                self.spline_zclust[tomo] = UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust*read_in_tables['zet_dep_bias'][tomo,:],
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1)

        if self.csmf:
            for tomo in range(self.n_tomo_csmf):
                norm = simpson(self.zet_csmf['pz'][tomo], x = self.zet_csmf['z'])
                self.zet_csmf['pz'][tomo] /= norm
                self.spline_zcsmf.append(UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_csmf['z']).value
                    * self.cosmology.h,
                    self.zet_csmf['pz'][tomo]*dzdchi_csmf,
                    s=0, ext=1))
                norm = simpson(self.spline_zcsmf[tomo](self.los_integration_chi), x = self.los_integration_chi)
                self.spline_zcsmf[tomo] = UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_csmf['z']).value
                    * self.cosmology.h,
                    self.zet_csmf['pz'][tomo]*dzdchi_csmf/norm,
                    s=0, ext=1)
                
            aux_zet_total = np.zeros_like(self.los_integration_chi)
            for tomo in range(self.n_tomo_csmf):
                aux_zet_total += self.spline_zcsmf[tomo](self.los_integration_chi)
            norm = simpson(aux_zet_total, x = self.los_integration_chi)
            self.spline_zcsmf_total = UnivariateSpline(self.los_integration_chi, aux_zet_total/norm, s=0, ext=1)
            
        for tomo in range(self.n_tomo_lens):
            if covELLspacesettings['nz_polyorder'] == 0:
                self.spline_zlens.append(interp1d(self.cosmology.comoving_distance(
                    self.zet_lens['z']).value*self.cosmology.h,
                    self.zet_lens['photoz'][tomo]*dzdchi_lens,
                    kind='nearest-up', fill_value=0))
            else:
                self.spline_zlens.append(UnivariateSpline(
                    self.cosmology.comoving_distance(
                        self.zet_lens['z']).value*self.cosmology.h,
                    self.zet_lens['photoz'][tomo]*dzdchi_lens,
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))

        
        
        
    def __set_lensweight_splines(self,
                                 covELLspacesettings,
                                 iA_dict):
        r"""
        Calculates and splines the lensing weight for later integration
        along the line-of-sight.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        """
        linear_growth_factor = np.interp(self.los_integration_chi,self.los_chi,(self.power_mm_lin_z[:,int(len(self.mass_func.k)/2)]/self.power_mm_lin_z[0,int(len(self.mass_func.k)/2)])**0.5)
        self.spline_lensweight = []
        for tomo in range(self.n_tomo_lens):
            norm = simpson(self.spline_zlens[tomo](
                self.los_integration_chi), x = self.los_integration_chi)
            aux_integral = np.zeros_like(self.los_integration_chi)
            for zetidx in range(covELLspacesettings['integration_steps']):
                aux_chi = self.los_integration_chi[zetidx]
                chi = np.geomspace(
                    aux_chi, self.chimax, covELLspacesettings['integration_steps'])
                aux_integral[zetidx] = simpson(self.spline_zlens[tomo](chi)
                                                * (chi - aux_chi)/chi,
                                                x = chi)/norm
            aux_integral = aux_integral * 3/2 \
                / self.cosmology.hubble_distance.value**2 \
                / self.cosmology.h**2 \
                * self.cosmology.Om0 \
                * self.los_integration_chi \
                / self.cosmology.scale_factor(
                    self.spline_z_of_chi(self.los_integration_chi))
            aux_W_iA = -iA_dict['A_IA']*((1.0 + self.spline_z_of_chi(self.los_integration_chi))/(1+iA_dict['z_pivot_IA']))**(iA_dict['eta_IA'])*0.0134*self.cosmology.Om0/linear_growth_factor*self.spline_zlens[tomo](
                self.los_integration_chi)/norm
            self.spline_lensweight.append(UnivariateSpline(self.los_integration_chi,
                                                           aux_integral + aux_W_iA,
                                                           k=1, s=0, ext=0))
            
        
    def __check_for_tabulated_Cells(self,
                                    Cxy_tab):
        r"""
        Checks wether previously caclulated Cells are stored in read in files
        as provided in the input

        Parameters
        ----------
        Cxy_tab : dictionary
            default : None
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer, optional).

        Returns
        ----------
        [Cgg, Cgm, Cmm]: list of arrays
            Contain the read in files for the Cells
            as previously specified. If none are specified, returns 'None'
        [gg_tab_bool, gm_tab_bool, mm_tab_bool] : boolean
            Flags whether the input files have been specified and read

        """
        if not self.checked_input_cells:
            ellrange_clustering = None
            ellrange_lensing = None
            if Cxy_tab['ell'] is None:
                self.do_not_update_ell = False 
                return None, [False, False, False]
            self.do_not_update_ell = True
            if abs(self.ellrange[0] - Cxy_tab['ell'][0]) > 1e-3 or \
            abs(self.ellrange[-1] - Cxy_tab['ell'][-1]) > 1e-3 or \
            len(self.ellrange) != len(Cxy_tab['ell']):
                print("FileInputWarning: The tabulated angular modes (ells) from " +
                    "the angular power spectra file(s) do not match the ones " +
                    "currently in use, the ells will now be overwritten to " +
                    "the tabulated ones.")
              
            self.ellrange = Cxy_tab['ell']
            gg_tab_bool, gm_tab_bool, mm_tab_bool = False, False, False
            if self.gg or self.gm and Cxy_tab['gg'] is not None:
                gg_tab_bool = True
                ellrange_clustering = Cxy_tab['ell_clust']
            if (self.mm or self.gm) and Cxy_tab['mm'] is not None:
                mm_tab_bool = True
                ellrange_lensing = Cxy_tab['ell_lens']
            if (self.gm or (self.gg and self.mm and self.cross_terms)) and \
            Cxy_tab['gm'] is not None:
                ellrange_clustering = Cxy_tab['ell_clust']
                gm_tab_bool = True
                if not gg_tab_bool or not mm_tab_bool:
                    print("FileInputWarning: To calculate the cross " +
                        "galaxy-matter covariance the three angular power " +
                        "spectra 'galaxy-galaxy', 'galaxy-kappa' and " +
                        "'kappa-kappa' are needed. The missing C_ells will be " +
                        "calculated which may cause a biased result.")
            Cgg, Cgm, Cmm = None, None, None
            if gg_tab_bool:
                Cgg = Cxy_tab['gg']
            if mm_tab_bool:
                Cmm = Cxy_tab['mm']
            if gm_tab_bool:
                Cgm = Cxy_tab['gm']
                if gg_tab_bool:
                    Cgg = Cxy_tab['gg']
                if mm_tab_bool:
                    Cmm = Cxy_tab['mm']
            if ellrange_clustering is not None and ellrange_lensing is not None:
                if not np.array_equal(ellrange_clustering, ellrange_lensing):
                    ellmin = ellrange_clustering[0]
                    ellmax = ellrange_clustering[-1]
                    if ellrange_lensing[0] < ellmin and ellrange_lensing[-1] > ellmax :
                        ellmin = ellrange_lensing[0]
                        ellmax = ellrange_lensing[-1]
                        Cgg_aux = np.zeros((len(Cmm[:,0,0,0]), len(Cgg[0,:,0,0,0]), len(Cgg[0,0, :,0,0]), len(Cgg[0,0, 0, :,0]), len(Cgg[0,0,0,0,:])))
                        for i_sample in range(len(Cgg[0,:,0,0,0])):
                            for j_sample in range(len(Cgg[0,0, :, 0,0])):
                                for i_tomo in range(len(Cgg[0,0,0,:,0])):
                                    for j_tomo in range(len(Cgg[0,0,0,0,:])):
                                        Cggspline = UnivariateSpline(ellrange_clustering, Cgg[:,i_sample, j_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                        Cgg_aux[i_sample, j_sample, i_tomo,j_tomo] = Cggspline(ellrange_lensing)
                        Cgg = Cgg_aux
                        Cgm_aux = np.zeros((len(Cmm[:,0,0,0]), len(Cgm[0, :,0,0]), len(Cgm[0, 0, :,0]), len(Cgm[0,0,0,:])))
                        for i_sample in range(len(Cgm[0,:,0,0])):
                            for i_tomo in range(len(Cgm[0,0,:,0])):
                                for j_tomo in range(len(Cgm[0,0,0,:])):
                                        Cgmspline = UnivariateSpline(ellrange_clustering, Cgm[:,i_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                        Cmm_aux[i_sample, i_tomo,j_tomo] = Cgmspline(ellrange_lensing)
                        Cgm = Cgm_aux
                        self.ellrange = ellrange_lensing
                    else:
                        Cmm_aux = np.zeros((len(Cmm[:,0,0,0]), len(Cmm[0, :,0,0]), len(Cmm[0, 0, :,0]), len(Cmm[0,0,0,:])))
                        for i_sample in range(len(Cmm[0,:,0,0])):
                            for i_tomo in range(len(Cmm[0,0,:,0])):
                                for j_tomo in range(len(Cmm[0,0,0,:])):
                                        Cmmspline = UnivariateSpline(ellrange_lensing, Cmm[:,i_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                        Cmm_aux[i_sample, i_tomo,j_tomo] = Cgmspline(ellrange_clustering)
                        Cmm = Cmm_aux
                        self.ellrange = ellrange_clustering
                else:
                    self.ellrange = ellrange_lensing
            else:
                if ellrange_clustering is None:
                    self.ellrange = ellrange_lensing
                else:
                    self.ellrange = ellrange_clustering
            self.checked_input_cells = True
            return [Cgg, Cgm, Cmm], [gg_tab_bool, gm_tab_bool, mm_tab_bool]
        else:
            self.checked_input_cells = True
        
    def __check_for_tabulated_Tells(self,
                                    Tuvxy_tab):
        r"""
        Checks wether previously caclulated trispectra are stored in read in files
        as provided in the input

        Parameters
        ----------
        Tuvxy_tab : dictionary
            default : None
            Look-up table for the T_ell projected tri-spectra (gggg, gggm, ggmm, gmgm , mmgm, mmmm, optional).

        Returns
        ----------
        [Tgggg, Tgggm, Tggmm, Tgmgm, Tmmgm, Tmmmm]: list of arrays
            Contain the read in files for the Cells
            as previously specified. If none are specified, returns 'None'
        """
        if Tuvxy_tab['ell'] is None:
            return None
        if abs(self.ellrange[0] - Tuvxy_tab['ell'][0]) > 1e-3 or \
           abs(self.ellrange[-1] - Tuvxy_tab['ell'][-1]) > 1e-3 or \
           len(self.ellrange) != len(Tuvxy_tab['ell']):
            if np.any(self.tab_bools):
                raise Exception("InputError: The tabulated angular modes (ells) from the trispectrum files does not match the ones provided in the tabulated C_ells")
            else:
                print("FileInputWarning: The tabulated angular modes (ells) from " +
                  "the angular power spectra file(s) do not match the ones " +
                  "currently in use, the ells will now be overwritten to " +
                  "the tabulated ones.")


    def __update_los_integration_chi(self, chi_min, chi_max, covELLspacesettings):
        r"""
        Function to update the line-of-sight integration range.

        Parameters
        ----------
        chimin, chimax : float
            minimum and maximum comoving distance of the integration range
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.

        Returns:
        ------------
        Updates the private variable 'self.los_integration_chi'

        """
        self.los_integration_chi = np.geomspace(
            chi_min, chi_max, covELLspacesettings['integration_steps'])

    def __get_updated_los_integration_chi(self, chi_min, chi_max, covELLspacesettings):
        r"""
        Returns an updated version of the line-of-sight integration range.

        Parameters
        ----------
        chimin, chimax : float
            minimum and maximum comoving distance of the integration range
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.

        Returns:
        ------------
        self.los_integration_chi : array
            Integration range for line-of-sight integration

        """
        return np.geomspace(chi_min, chi_max, covELLspacesettings['integration_steps'])

    def get_Cells(self,
                  obs_dict,
                  output_dict,
                  bias_dict,
                  iA_dict,
                  hod_dict,
                  prec, read_in_tables):
        r"""
        Calculates the angular power spectra either with the Limber
        or full non-Limber projection.

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
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
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

        Returns:
        ------------
        None :  sets private variables 'self.Cell_gg', 'self.Cell_gm' and 'self.Cell_mm'
            Used later to calculate the Gaussian covariance matrix. See class documentation.

        """
        if self.cov_dict['gauss']:
            self.Cell_gg, self.Cell_gm, self.Cell_mm = \
                self.calc_Cells_Limber(obs_dict['ELLspace'],
                                    bias_dict,
                                    iA_dict,
                                    hod_dict, prec,
                                    read_in_tables['Cxy'])
            if not obs_dict['ELLspace']['limber']:
                self.Cell_gg_limber = np.copy(self.Cell_gg)
                self.calc_Cells_nonLimber(obs_dict['ELLspace'],
                                        read_in_tables['Cxy'])
            if output_dict['Cell']:
                out = Output(output_dict)
                Cells = [self.Cell_gg, self.Cell_gm, self.Cell_mm]
                out.write_Cells(self.ellrange, self.n_tomo_clust, self.n_tomo_lens,
                                Cells)
                
    def calc_Cells_nonLimber(self,
                             covELLspacesettings,
                             Cxy_tab):
        r"""
        Calculates the non-Limber projection of the power spectrum for all
        tracers and tomographic bin combinations over the ell range
        specified in covELLspacesettings

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        Cxy_tab : dictionary
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer).

        Returns
        ----------
        None :
            Sets the private variables self.Cell_xx to the non-Limber version
            for multipoles below 2000
        """

        if self.Cells is not None:
            if ((self.gg and self.tab_bools[0]) or not self.gg) and \
               ((self.mm and self.tab_bools[2]) or not self.mm) and \
               ((self.gm and np.all(self.tab_bools)) or not self.gm):
                return self.Cells[0], self.Cells[1], self.Cells[2]

        print("Calculating non-limber angular power spectra (C_ell's).")
        n_tomo_clust_copy = np.copy(self.n_tomo_clust)
        ellmax = 200
        if self.clustering_z:
            self.n_tomo_clust = 1
            ellmax = 1000
        
                    
        if (self.gg or self.gm) and not self.tab_bools[0]:
            for eidx, ell in enumerate(self.ellrange):
                if (int(ell) < ellmax):
                    if ell < 100:
                        kmax = 50*(ell + 0.5)/np.min(self.chi_min_clust)
                        kmin = 0.05*(ell + 0.5)/np.max(self.chi_min_clust)
                    if ell < 10:
                        kmax = 100*(ell + 0.5)/np.min(self.chi_min_clust)
                        kmin = 0.05*(ell + 0.5)/np.max(self.chi_min_clust)
                    if ell >100:
                        kmax = 3*(ell + 0.5)/np.min(self.chi_min_clust)
                        kmin = 0.3*(ell + 0.5)/np.max(self.chi_min_clust)
                    if ell >200:
                        kmax = 2*(ell + 0.5)/np.min(self.chi_min_clust)
                        kmin = 0.3*(ell + 0.5)/np.max(self.chi_min_clust)
                    k_nonlimber = np.geomspace(max(kmin,self.mass_func.k[0]), min(kmax,self.mass_func.k[-1]), 1000)
                    for i_sample in range(self.sample_dim):
                        for j_sample in range(i_sample,self.sample_dim): 
                            global non_limber_k_integral

                            def non_limber_k_integral(k_integral):
                                lev = levin.Levin(1, 16, 32, 1e-6, self.integration_intervals)
                                result = np.zeros(self.n_tomo_clust)
                                for tomo_i in range(self.n_tomo_clust):
                                    chi_low = self.chi_min_clust[tomo_i]
                                    chi_high = self.chi_max_clust[tomo_i]
                                    los_chi = self.__get_updated_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    integrand = np.sqrt((10**self.spline_Pgg[i_sample*self.sample_dim +j_sample](
                                        np.log10(k_integral), los_chi))[:, 0])*self.spline_zclust[tomo_i](
                                        los_chi)
                                    lev.init_integral(
                                        los_chi, integrand[:, None], True, True)
                                    result[tomo_i] = np.array(lev.single_bessel(
                                            k_integral, int(ell), los_chi[0], los_chi[-1]))*k_integral
                                return result
                            pool = mp.Pool(self.num_cores)
                            inner_integral_gg = np.array(pool.map(
                                non_limber_k_integral, k_nonlimber)).T
                            pool.close()
                            pool.terminate()
                            
                            
                            
                            if ell > 80:
                                inner_integral_gg[0,np.where(np.abs(inner_integral_gg[0,:]) < 1e-90)[0]] = 0
                                index_first_max =argrelextrema(inner_integral_gg[0,:], np.greater)[0][0]
                                inner_integral_gg[0,np.where(np.abs(inner_integral_gg[0,:]/inner_integral_gg[0,index_first_max])> 1.)[0]] = 0.0
                                
                            
                            self.Cell_gg[eidx, i_sample, j_sample, :self.n_tomo_clust, :self.n_tomo_clust] = \
                                simpson(
                                    inner_integral_gg[:, None, :]*inner_integral_gg[None,:, :], x = k_nonlimber, axis = -1)*2.0/np.pi
                            self.Cell_gg[eidx, j_sample, i_sample, :self.n_tomo_clust, :self.n_tomo_clust] = self.Cell_gg[eidx, i_sample, j_sample, :self.n_tomo_clust, :self.n_tomo_clust]
                            
                            

        
        
        if self.clustering_z:
            self.n_tomo_clust = n_tomo_clust_copy
        elif self.tab_bools[0]:
            self.Cell_gg = self.Cells[0]
        else:
            self.Cell_gg = 0

    def calc_Cells_Limber(self,
                          covELLspacesettings,
                          bias_dict,
                          iA_dict,
                          hod_dict,
                          prec,
                          Cxy_tab):
        r"""
        Calculates the Limber projection of the power spectrum for all
        tracers and tomographic bin combinations over the ell range
        specified in covELLspacesettings

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be
            passedfrom the read_input method of the Input class.
        iA_dict: dictionary
            Specifies all the information about the intrinsic alignment model.
            To be passed from the read_input method in the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the
            differentgalaxy populations. To be passed from the
            read_input method of the Input class.
        prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        Cxy_tab : dictionary
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer).

        Returns
        ----------
        Cell_gg, Cell_gm, Cell_mm : list of arrays
            Angular power spectra for the different tracers. They each
            have the shape (len(self.ellrange), self.sample_dim,
            self.n_tomo_clust, self.n_tomo_clust)

        """
        
        self.los_interpolation_sampling = int(
            (self.zet_max - 0) / covELLspacesettings['delta_z'])
        if (self.los_interpolation_sampling < 3):
            self.los_interpolation_sampling = 3
        aux_gg = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k),
                           self.sample_dim, self.sample_dim))
        aux_gm = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k),
                           self.sample_dim))
        aux_mm = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k)))
        self.power_mm_lin_z = np.zeros_like(aux_mm)
        self.los_z = np.linspace(
            0, self.zet_max, self.los_interpolation_sampling)
        self.los_chi = self.cosmology.comoving_distance(
            self.los_z).value * self.cosmology.h
        aux_ngal = np.zeros((self.los_interpolation_sampling,self.sample_dim))
        if (self.mm or self.gm or self.unbiased_clustering) and prec['hm']['transfer_model'] == 'CAMB' and self.Pxy_tab['mm'] is None:
            self.camb_pars_new.set_matter_power(redshifts = self.los_z[::-1], kmax=self.mass_func.k[-1])
            results = camb.get_results(self.camb_pars_new)
            self.camb_pars_new.NonLinear = model.NonLinear_pk
            if prec['powspec']['nl_model'] == "mead2020_feedback":
                self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'], HMCode_logT_AGN = prec['powspec']['HMCode_logT_AGN'])
            else:
                if prec['powspec']['nl_model'] == "mead2015" or prec['powspec']['nl_model'] == "mead2016":
                    self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'], HMCode_A_baryon = prec['powspec']['HMCode_A_baryon'], HMCode_eta_baryon = prec['powspec']['HMCode_eta_baryon'])
                else:
                    self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'])
            results.calc_power_spectra(self.camb_pars_new)
            _,_, aux_mm = results.get_matter_power_spectrum(minkh=self.mass_func.k[0],
                                                            maxkh=self.mass_func.k[-1],
                                                            npoints = len(self.mass_func.k))
        t0 = time.time()
        
        if self.csmf:
            aux_stellar_mass_func = np.zeros((self.los_interpolation_sampling, len(self.log10csmf_mass_bins)))
            aux_stellar_mass_func_bias = np.zeros((self.los_interpolation_sampling, len(self.log10csmf_mass_bins)))
            self.csmf_count_matter_bispectrum = np.zeros((len(self.mass_func.k), self.los_interpolation_sampling,len(self.log10csmf_mass_bins)))
            aux_response_mm = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            aux_response_gg = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            aux_response_gm = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            
            save_mm = False
            if not self.mm:
                save_mm = True
                self.mm = True
        for zet in range(self.los_interpolation_sampling):
            self.update_mass_func(self.los_z[zet], bias_dict, hod_dict, prec)
            aux_ngal[zet, :] = self.ngal
            if self.gg or self.gm:
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        if self.unbiased_clustering:
                            aux_gg[zet, :, i_sample, j_sample] = aux_mm[zet,:]
                        else:
                            aux_gg[zet, :, i_sample, j_sample] = self.Pgg[:, i_sample, j_sample]
                    if self.unbiased_clustering:
                        aux_gm[zet, :, i_sample] = aux_mm[zet,:]
                    else:
                        if self.gm or (self.gg and self.mm):
                            aux_gm[zet, :, i_sample] = self.Pgm[:, i_sample]
            if prec['hm']['transfer_model'] != 'CAMB' or self.Pxy_tab['mm'] is not None :
                aux_mm[zet, :] = self.Pmm[:, 0]
            self.power_mm_lin_z[zet, :] = self.mass_func.power[:]
            eta = (time.time()-t0) * \
                (self.los_interpolation_sampling/(zet+1)-1)
            if self.csmf:
                aux_stellar_mass_func[zet,:] = self.galaxy_smf_c(10**self.log10csmf_mass_bins)  + self.galaxy_smf_s(10**self.log10csmf_mass_bins)
                aux_stellar_mass_func_bias[zet,:] = self.galaxy_smf_bias_c(10**self.log10csmf_mass_bins)  + self.galaxy_smf_bias_s(10**self.log10csmf_mass_bins)
                self.csmf_count_matter_bispectrum[:, zet,:] = self.get_count_matter_bispectrum(bias_dict, hod_dict, prec['hm'], self.log10csmf_mass_bins)
                aux_response_gg[zet, :, :], aux_response_gm[zet, :, :], aux_response_mm[zet, :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
            print('\rPreparations for C_ell calculation at '
                    + str(round((zet+1)/self.los_interpolation_sampling*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)
                                    ) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
        print(" ")
        if self.csmf and save_mm:
            self.mm = False
            if self.mm:
                self.spline_responsePmm = RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), np.log(aux_response_mm[:, :, 0]).T,bounds_error= False, fill_value = None)
            if self.gm:
                self.spline_responsePgm = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgm.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gm[:, :, i_sample]).T,bounds_error= False, fill_value = None))
            if self.gg:
                self.spline_responsePgg = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgg.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gg[:, :, i_sample]).T,bounds_error= False, fill_value = None))
        else:
            if self.csmf:
                if self.mm:
                    self.spline_responsePmm = RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), np.log(aux_response_mm[:, :, 0]).T,bounds_error= False, fill_value = None)
                if self.gm:
                    self.spline_responsePgm = []
                    for i_sample in range(self.sample_dim):
                        self.spline_responsePgm.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gm[:, :, i_sample]).T,bounds_error= False, fill_value = None))
                if self.gg:
                    self.spline_responsePgg = []
                    for i_sample in range(self.sample_dim):
                        self.spline_responsePgg.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gg[:, :, i_sample]).T,bounds_error= False, fill_value = None))

        self.power_mm_lin_spline = RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),np.log(self.power_mm_lin_z),bounds_error= False, fill_value = None)
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        if self.csmf:
            self.csmf_at_tomo_and_mass = np.zeros((len(self.log10csmf_mass_bins),self.n_tomo_csmf))
            self.phi_tilde_spline = []
            for i_tomo in range(self.n_tomo_csmf):
                for i_smf_m_bins in range(len(self.log10csmf_mass_bins)):
                    aux_csmf_spline = UnivariateSpline(self.los_chi, aux_stellar_mass_func[:, i_smf_m_bins], k=2, s=0, ext=0)
                    self.csmf_at_tomo_and_mass[i_smf_m_bins, i_tomo] = simpson(self.spline_zcsmf[i_tomo](self.los_integration_chi)*aux_csmf_spline(self.los_integration_chi)*self.f_tomo[i_tomo], x = self.los_integration_chi)
                    if i_tomo == 0:
                        self.phi_tilde_spline.append(UnivariateSpline(self.los_chi, aux_stellar_mass_func_bias[:, i_smf_m_bins], k=2, s=0, ext=0))
        self.Ngal = np.zeros((self.sample_dim, self.n_tomo_clust))
        for i_sample in range(self.sample_dim):
            spline_nbar = UnivariateSpline(self.los_chi, aux_ngal[:, i_sample], k=2, s=0, ext=0)
            for tomo_i in range(self.n_tomo_clust):
                prob = self.spline_zclust[tomo_i](self.los_integration_chi)*np.append((self.los_integration_chi[1:] -self.los_integration_chi[:-1]),0)
                self.Ngal[i_sample,tomo_i] = simpson(prob*self.los_integration_chi**2*spline_nbar(self.los_integration_chi), x = self.los_integration_chi)
        if self.gg or self.gm:
            spline_Pgg, spline_Pgm = [], []
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    spline_Pgg.append(RegularGridInterpolator((self.los_chi, np.log10(self.mass_func.k)),
                                            np.log10(aux_gg[:, :, i_sample, j_sample]),bounds_error= False, fill_value = None))
                if self.gm or self.mm:
                    spline_Pgm.append(RegularGridInterpolator((self.los_chi,np.log10(self.mass_func.k)), np.log10(aux_gm[:, :, i_sample]),bounds_error= False, fill_value = None))
            if self.gm or self.mm:
                self.spline_Pgg = spline_Pgg
                self.spline_Pgm = spline_Pgm
    
        if (self.mm or self.gm) and not self.tab_bools[2]:
            spline_Pmm = RegularGridInterpolator((self.los_chi,np.log10(self.mass_func.k)), np.log10(aux_mm[:, :]),bounds_error= False, fill_value = None)
            self.spline_Pmm = spline_Pmm

        self.__set_lensweight_splines(covELLspacesettings, iA_dict)
        if self.Cells is not None:
            if ((self.gg and self.tab_bools[0]) or not self.gg) and ((self.mm and self.tab_bools[2]) or not self.mm) and ((self.gm and np.all(self.tab_bools)) or not self.gm):
                return self.Cells[0], self.Cells[1], self.Cells[2]

        fil = np.sqrt((self.ellrange + 2)*(self.ellrange + 1)*(self.ellrange)*(self.ellrange -1))
        fijl = fil**2*(self.ellrange+0.5)**(-4)
        print("Calculating angular power spectra (C_ell's).")

        if (self.gg or self.gm) and not self.tab_bools[0]:
            Cell_gg = np.zeros((len(self.ellrange), self.sample_dim, self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_clust))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for tomo_i in range(self.n_tomo_clust):
                        for tomo_j in range(tomo_i, self.n_tomo_clust):
                            chi_low = max(
                                self.chi_min_clust[tomo_i], self.chi_min_clust[tomo_j])
                            chi_high = min(
                                self.chi_max_clust[tomo_i], self.chi_max_clust[tomo_j])
                            if chi_low >= chi_high:
                                continue
                            self.__update_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                            flat_idx = 0
                            for i_chi in range(len(self.los_integration_chi)):
                                 for i_ell in range(len(self.ellrange)):
                                    ki = np.log10((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                    x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                    x_values[1,flat_idx] = ki
                                    flat_idx +=1
                            integrand = spline_Pgg[i_sample*self.sample_dim + j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                            Cell_gg[:, i_sample, j_sample, tomo_i, tomo_j] = simpson(10**integrand*(self.spline_zclust[tomo_i](self.los_integration_chi)*self.spline_zclust[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x = self.los_integration_chi, axis = 0)
                            Cell_gg[:, i_sample, j_sample, tomo_j, tomo_i] = \
                                Cell_gg[:, i_sample, j_sample,  tomo_i, tomo_j]
                            self.__update_los_integration_chi(
                                self.chimin, self.chimax, covELLspacesettings)
        elif self.tab_bools[0]:
            Cell_gg = self.Cells[0]
        else:
            Cell_gg = 0

        if (self.gm or (self.gg and self.mm and self.cross_terms)) and \
           not self.tab_bools[1]:
            Cell_gm = np.zeros((len(self.ellrange), self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_lens))
            for i_sample in range(self.sample_dim):
                for tomo_i in range(self.n_tomo_clust):
                    for tomo_j in range(self.n_tomo_lens):
                        chi_low = self.chi_min_clust[tomo_i]
                        chi_high = self.chi_max_clust[tomo_i]
                        self.__update_los_integration_chi(
                            chi_low, chi_high, covELLspacesettings)

                        x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                        flat_idx = 0
                        for i_chi in range(len(self.los_integration_chi)):
                            for i_ell in range(len(self.ellrange)):
                                ki = np.log10((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                x_values[1,flat_idx] = ki
                                flat_idx +=1
                        integrand = spline_Pgm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        Cell_gm[:, i_sample, tomo_i, tomo_j] = np.sqrt(fijl)*simpson(10**integrand*(self.spline_zclust[tomo_i](self.los_integration_chi)*self.spline_lensweight[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x = (self.los_integration_chi), axis = 0)
                        self.__update_los_integration_chi(
                            self.chimin, self.chimax, covELLspacesettings)
        elif self.tab_bools[1]:
            Cell_gm = self.Cells[1]
        else:
            Cell_gm = 0
        if (self.mm or self.gm) and not self.tab_bools[2]:
            Cell_mm = np.zeros((len(self.ellrange),
                                self.n_tomo_lens, self.n_tomo_lens))
            x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
            flat_idx = 0
            for i_chi in range(len(self.los_integration_chi)):
                for i_ell in range(len(self.ellrange)):
                    ki = np.log10((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                    x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                    x_values[1,flat_idx] = ki
                    flat_idx +=1
            for tomo_i in range(self.n_tomo_lens):
                for tomo_j in range(tomo_i, self.n_tomo_lens):
                    integrand = spline_Pmm((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                    Cell_mm[:, tomo_i, tomo_j] = fijl*simpson(10**integrand*(self.spline_lensweight[tomo_i](self.los_integration_chi)*self.spline_lensweight[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x =(self.los_integration_chi), axis = 0)
                    Cell_mm[:, tomo_j, tomo_i] = Cell_mm[:, tomo_i, tomo_j]
            Cell_mm = Cell_mm[:, None, :, :] \
                * np.ones(self.sample_dim)[None, :, None, None]
            
        elif self.tab_bools[2]:
            Cell_mm = self.Cells[2]
        else:
            Cell_mm = 0

        return Cell_gg, Cell_gm, Cell_mm

    def calc_covELL(self,
                    obs_dict,
                    output_dict,
                    bias_dict,
                    hod_dict,
                    survey_params_dict,
                    prec,
                    read_in_tables):
        r"""
        Calculates the full covariance between the specified
        observables in ell-space config file.

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
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
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
            each with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        print("Calculating covariance for angular power spectra (C_ell's).")
        if not self.cov_dict['split_gauss']:
            if self.ellrange_photo is not None:
                gaussELLgggg_ssss_new, gaussELLgggg_sssp_new, gaussELLgggg_sspp_new, \
                gaussELLgggg_spsp_new, gaussELLgggg_ppsp_new, gaussELLgggg_pppp_new, \
                gaussELLgggm_sssm_new, gaussELLgggm_sspm_new, gaussELLgggm_spsm_new, \
                gaussELLgggm_sppm_new, gaussELLgggm_ppsm_new, gaussELLgggm_pppm_new, \
                gaussELLggmm_ssmm_new, gaussELLggmm_spmm_new, gaussELLggmm_ppmm_new, \
                gaussELLgmgm_smsm_new, gaussELLgmgm_smpm_new, gaussELLgmgm_pmsm_new, \
                gaussELLgmgm_pmpm_new, gaussELLmmgm_mmsm_new, gaussELLmmgm_mmpm_new, \
                gaussELLmmmm_mmmm_new, gaussELLgggg_ssss_new_sn, gaussELLgggg_sssp_new_sn, \
                gaussELLgggg_sspp_new_sn, gaussELLgggg_spsp_new_sn, gaussELLgggg_ppsp_new_sn, \
                gaussELLgggg_pppp_new_sn, gaussELLgmgm_smsm_new_sn, gaussELLgmgm_smpm_new_sn, \
                gaussELLgmgm_pmsm_new_sn, gaussELLgmgm_pmpm_new_sn, gaussELLmmmm_mmmm_new_sn = \
                self.covELL_gaussian(obs_dict['ELLspace'],
                                     survey_params_dict, True)
                gauss = [gaussELLgggg_ssss_new + gaussELLgggg_ssss_new_sn,
                         gaussELLgggg_sssp_new + gaussELLgggg_sssp_new_sn,
                         gaussELLgggg_sspp_new + gaussELLgggg_sspp_new_sn,
                         gaussELLgggg_spsp_new + gaussELLgggg_spsp_new_sn,
                         gaussELLgggg_ppsp_new + gaussELLgggg_ppsp_new_sn,
                         gaussELLgggg_pppp_new + gaussELLgggg_pppp_new_sn,
                         gaussELLgggm_sssm_new, gaussELLgggm_sspm_new, gaussELLgggm_spsm_new,
                         gaussELLgggm_sppm_new, gaussELLgggm_ppsm_new, gaussELLgggm_pppm_new,
                         gaussELLggmm_ssmm_new, gaussELLggmm_spmm_new, gaussELLggmm_ppmm_new, 
                         gaussELLgmgm_smsm_new + gaussELLgmgm_smsm_new_sn,
                         gaussELLgmgm_smpm_new + gaussELLgmgm_smpm_new_sn,
                         gaussELLgmgm_pmsm_new + gaussELLgmgm_pmsm_new_sn,
                         gaussELLgmgm_pmpm_new + gaussELLgmgm_pmpm_new_sn,
                         gaussELLmmgm_mmsm_new, gaussELLmmgm_mmpm_new,
                         gaussELLmmmm_mmmm_new + gaussELLmmmm_mmmm_new_sn]
            else:
                if self.csmf:
                    gaussgggg, gaussgggm, gaussggmm, \
                    gaussgmgm, gaussmmgm, gaussmmmm, \
                    gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn, \
                    csmf_auto, csmf_gg, csmf_gm, csmf_mm = \
                    self.covELL_gaussian(obs_dict['ELLspace'],
                                        survey_params_dict, True)
                    gauss = [gaussgggg + gaussgggg_sn, gaussgggm,
                            gaussggmm, gaussgmgm + gaussgmgm_sn,
                            gaussmmgm, gaussmmmm + gaussmmmm_sn,
                            csmf_auto, csmf_gg, csmf_gm, csmf_mm]
                else:
                    gaussgggg, gaussgggm, gaussggmm, \
                    gaussgmgm, gaussmmgm, gaussmmmm, \
                    gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn = \
                    self.covELL_gaussian(obs_dict['ELLspace'],
                                        survey_params_dict, True)
                    gauss = [gaussgggg + gaussgggg_sn, gaussgggm,
                            gaussggmm, gaussgmgm + gaussgmgm_sn,
                            gaussmmgm, gaussmmmm + gaussmmmm_sn]
        else:
            gauss = self.covELL_gaussian(obs_dict['ELLspace'],
                                         survey_params_dict, True)
        nongauss = self.covELL_non_gaussian(obs_dict['ELLspace'],
                                            output_dict,
                                            bias_dict,
                                            hod_dict,
                                            prec,
                                            read_in_tables['tri'])
                    
        if self.cov_dict['nongauss'] and self.ellrange_photo is None:
            nongauss_new = []
            if self.gg:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gggg contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_clust'],
                                                              nongauss[0],
                                                              True,
                                                              True,
                                                              True))
                else:
                    nongauss[0][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[0][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_clust']/self.deg2torad2)
                    nongauss_new.append(nongauss[0][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gg and self.gm and self.cross_terms:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gggm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[1],
                                                              True,
                                                              True,
                                                              False))
                else:
                    nongauss[1][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[1][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2)
                    nongauss_new.append(nongauss[1][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gg and self.mm and self.cross_terms:
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian ggmm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_lens'],
                                                              nongauss[2],
                                                              True,
                                                              True,
                                                              True))
                else:
                    nongauss[2][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[2][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens'])/self.deg2torad2)
                    nongauss_new.append(nongauss[2][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gm:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gmgm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_ggl'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[3],
                                                              True,
                                                              False,
                                                              False))
                else:
                    nongauss[3][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[3][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_ggl']/self.deg2torad2)
                    nongauss_new.append(nongauss[3][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.mm and self.gm and self.cross_terms:           
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian mmgm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[4],
                                                              True,
                                                              True,
                                                              False))
                else:
                    nongauss[4][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[4][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2)
                    nongauss_new.append(nongauss[4][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.mm:
                if self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian mmmm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_lens'],
                                                              nongauss[5],
                                                              True,
                                                              True,
                                                              True))
                else:
                    nongauss[5][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[5][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_lens']/self.deg2torad2)
                    nongauss_new.append(nongauss[5][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            nongauss = nongauss_new
        if self.ellrange_photo is not None:
            nongaussELLgggg_ssss_new = 0
            nongaussELLgggg_sssp_new = 0
            nongaussELLgggg_sspp_new = 0
            nongaussELLgggg_spsp_new = 0
            nongaussELLgggg_ppsp_new = 0
            nongaussELLgggg_pppp_new = 0

            nongaussELLgggm_sssm_new = 0
            nongaussELLgggm_sspm_new = 0
            nongaussELLgggm_spsm_new = 0
            nongaussELLgggm_sppm_new = 0
            nongaussELLgggm_ppsm_new = 0
            nongaussELLgggm_pppm_new = 0

            nongaussELLggmm_ssmm_new = 0
            nongaussELLggmm_spmm_new = 0
            nongaussELLggmm_ppmm_new = 0

            nongaussELLgmgm_smsm_new = 0
            nongaussELLgmgm_smpm_new = 0
            nongaussELLgmgm_pmsm_new = 0
            nongaussELLgmgm_pmpm_new = 0

            nongaussELLmmgm_mmsm_new = 0
            nongaussELLmmgm_mmpm_new = 0

            nongaussELLmmmm_mmmm_new = 0
            if self.cov_dict['nongauss']:
                if self.gg:
                    nongaussELLgggg_ssss_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    nongaussELLgggg_sssp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    nongaussELLgggg_sspp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    nongaussELLgggg_spsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    nongaussELLgggg_ppsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    nongaussELLgggg_pppp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',True
                                                                )
                if self.gm and self.gg:
                    nongaussELLgggm_sssm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgggm_sspm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgggm_spsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgggm_sppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgggm_ppsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgggm_pppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                if self.mm and self.gg:
                    nongaussELLggmm_ssmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                    nongaussELLggmm_spmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                    nongaussELLggmm_ppmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                if self.gm:
                    nongaussELLgmgm_smsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgmgm_smpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgmgm_pmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLgmgm_pmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                if self.mm and self.gm:
                    nongaussELLmmgm_mmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[4],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                True
                                                                )
                    nongaussELLmmgm_mmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[4],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                True
                                                                )
                if self.mm:
                    nongaussELLmmmm_mmmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[5],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens',
                                                                True
                                                                )
            nongauss = nongaussELLgggg_ssss_new, nongaussELLgggg_sssp_new, nongaussELLgggg_sspp_new, \
                nongaussELLgggg_spsp_new, nongaussELLgggg_ppsp_new, nongaussELLgggg_pppp_new, \
                nongaussELLgggm_sssm_new, nongaussELLgggm_sspm_new, nongaussELLgggm_spsm_new, \
                nongaussELLgggm_sppm_new, nongaussELLgggm_ppsm_new, nongaussELLgggm_pppm_new, \
                nongaussELLggmm_ssmm_new, nongaussELLggmm_spmm_new, nongaussELLggmm_ppmm_new, \
                nongaussELLgmgm_smsm_new, nongaussELLgmgm_smpm_new, nongaussELLgmgm_pmsm_new, \
                nongaussELLgmgm_pmpm_new, nongaussELLmmgm_mmsm_new, nongaussELLmmgm_mmpm_new, \
                nongaussELLmmmm_mmmm_new
        
        ssc = self.covELL_ssc(bias_dict,
                              hod_dict,
                              prec,
                              survey_params_dict,
                              obs_dict['ELLspace'])
        if self.ellrange_photo is None and self.cov_dict['ssc']:
            ssc_new = []    
            if self.gg:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gggg contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_clust'],
                                                              ssc[0],
                                                              False,
                                                              True,
                                                              True))
                else:
                    ssc_new.append(ssc[0])
            else:
                ssc_new.append(0)
            if self.gg and self.gm and self.cross_terms:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gggm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[1],
                                                              False,
                                                              True,
                                                              False))
                else:
                    ssc_new.append(ssc[1])
            else:
                ssc_new.append(0)
            if self.gg and self.mm and self.cross_terms:
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning SSC ggmm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_lens'],
                                                              ssc[2],
                                                              False,
                                                              True,
                                                              True))
                else:
                    ssc_new.append(ssc[2])
            else:
                ssc_new.append(0)
            if self.gm:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gmgm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_ggl'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[3],
                                                              False,
                                                              False,
                                                              False))
                else:
                    ssc_new.append(ssc[3])
            else:
                ssc_new.append(0)
            if self.mm and self.gm and self.cross_terms:           
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning SSC mmgm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[4],
                                                              False,
                                                              True,
                                                              False))
                else:
                    ssc_new.append(ssc[4])
            else:
                ssc_new.append(0)
            if self.mm:
                if self.ellrange_lensing_ul is not None:
                    print("Binning SSC mmmm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_lens'],
                                                              ssc[5],
                                                              False,
                                                              True,
                                                              True))
                else:
                    ssc_new.append(ssc[5])
            else:
                ssc_new.append(0)
            ssc = ssc_new

        if self.ellrange_photo is not None:
            sscELLgggg_ssss_new = 0
            sscELLgggg_sssp_new = 0
            sscELLgggg_sspp_new = 0
            sscELLgggg_spsp_new = 0
            sscELLgggg_ppsp_new = 0
            sscELLgggg_pppp_new = 0

            sscELLgggm_sssm_new = 0
            sscELLgggm_sspm_new = 0
            sscELLgggm_spsm_new = 0
            sscELLgggm_sppm_new = 0
            sscELLgggm_ppsm_new = 0
            sscELLgggm_pppm_new = 0

            sscELLggmm_ssmm_new = 0
            sscELLggmm_spmm_new = 0
            sscELLggmm_ppmm_new = 0

            sscELLgmgm_smsm_new = 0
            sscELLgmgm_smpm_new = 0
            sscELLgmgm_pmsm_new = 0
            sscELLgmgm_pmpm_new = 0

            sscELLmmgm_mmsm_new = 0
            sscELLmmgm_mmpm_new = 0

            sscELLmmmm_mmmm_new = 0
            if self.cov_dict['ssc']:
                if self.gg:
                    sscELLgggg_ssss_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    sscELLgggg_sssp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    sscELLgggg_sspp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    sscELLgggg_spsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    sscELLgggg_ppsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    sscELLgggg_pppp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                if self.gm and self.gg:
                    sscELLgggm_sssm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgggm_sspm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgggm_spsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgggm_sppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgggm_ppsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgggm_pppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                if self.mm and self.gg:
                    sscELLggmm_ssmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                    sscELLggmm_spmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                    sscELLggmm_ppmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                if self.gm:
                    sscELLgmgm_smsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgmgm_smpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgmgm_pmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLgmgm_pmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                if self.mm and self.gm:
                    sscELLmmgm_mmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[4],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                False
                                                                )
                    sscELLmmgm_mmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[4],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                False
                                                                )
                if self.mm:
                    sscELLmmmm_mmmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[5],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens',
                                                                False
                                                                )
            ssc = sscELLgggg_ssss_new, sscELLgggg_sssp_new, sscELLgggg_sspp_new, \
                sscELLgggg_spsp_new, sscELLgggg_ppsp_new, sscELLgggg_pppp_new, \
                sscELLgggm_sssm_new, sscELLgggm_sspm_new, sscELLgggm_spsm_new, \
                sscELLgggm_sppm_new, sscELLgggm_ppsm_new, sscELLgggm_pppm_new, \
                sscELLggmm_ssmm_new, sscELLggmm_spmm_new, sscELLggmm_ppmm_new, \
                sscELLgmgm_smsm_new, sscELLgmgm_smpm_new, sscELLgmgm_pmsm_new, \
                sscELLgmgm_pmpm_new, sscELLmmgm_mmsm_new, sscELLmmgm_mmpm_new, \
                sscELLmmmm_mmmm_new
        
        return list(gauss), list(nongauss), list(ssc)
    
    def calc_covELL_csmf(self,
                         covELLspacesettings,
                         survey_params_dict):
        r"""
        Calculates the covariance of the conditional stellar mass function and its cross-correlation
        with the tracers defined in the config file.
        
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
        
        Returns
        -------

        """
        csmf_gg_sva,csmf_gm_sva,csmf_mm_sva = self.covELL_csmf_cross_LSS_sva(covELLspacesettings)
        csmf_gg_ssc,csmf_gm_ssc,csmf_mm_ssc = self.covELL_csmf_cross_LSS_ssc(covELLspacesettings, survey_params_dict)
        cov_csmf = self.covELL_csmf_SN() + self.covELL_csmf_SSC(survey_params_dict)
        return cov_csmf, csmf_gg_sva + csmf_gg_ssc, csmf_gm_sva + csmf_gm_ssc, csmf_mm_sva + csmf_mm_ssc
    
    def __bin_cov_ell_csmf(self,
                           ellrange_12_ul,
                           cov,
                           unique_12):
        r"""
        Auxillary function binning the cross-covariance matrix 
        between the SMF and the LSS in ell-bins.

        """
        
        if not isinstance(cov, np.ndarray):
            return 0
        binned_covariance = np.zeros(np.zeros(len(ellrange_12_ul) - 1).shape + cov[0].shape)
        #np.zeros((len(ellrange_12_ul) - 1, len(cov[0,:,0,0,0,0], len(cov[0,0,:,0,0,0], len(cov[0,0,])))
        for i_ell in range(len(ellrange_12_ul) - 1):
            area12_ell = np.pi*((ellrange_12_ul[i_ell +1])**2 - (ellrange_12_ul[i_ell])**2)                                            
            integration_ell = np.arange(ellrange_12_ul[i_ell],ellrange_12_ul[i_ell +1])
            for i_csmf in range(len(cov[0,0,:,0,0,0])):
                for i_sample in range(len(cov[0,0,:,0,0,0])):
                    for csmf_tomo in range(len(cov[0,0,0,:,0,0])):
                        for i_tomo in range(len(cov[0,0,0,0,:,0])):
                            j_tomo_start = 0
                            if unique_12:
                                j_tomo_start = i_tomo
                            for j_tomo in range(j_tomo_start, len(cov[0,0,0,0,0,:])):
                                if len(np.where(cov[:, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo])[0]):
                                    spline = UnivariateSpline(self.ellrange, self.ellrange*cov[:, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo], k=2, s=0, ext=1)   
                                    binned_covariance[i_ell, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo] = simpson(spline(integration_ell),integration_ell)*2.*np.pi/area12_ell
        return binned_covariance

    
    def __bin_Gaussian(self,
                       covELLspacesettings,
                       survey_params_dict,
                       covariance,
                       field1_spec,
                       field2_spec,
                       field3_spec,
                       field4_spec,
                       probe12,
                       probe34):
        r"""
        Auxillary function binning the Gaussian covariance matrix in ell-bins

        """
        if not isinstance(covariance, np.ndarray):
            return 0
        full_sky_angle = 4*np.pi * self.deg2torad2

        if probe12 == "clust":
            area_12 = survey_params_dict['survey_area_clust'][1]
            if field1_spec == field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][0]
            else:
                area_12 = max(survey_params_dict['survey_area_clust'][0], survey_params_dict['survey_area_clust'][1])
        if probe12 == "ggl":
            area_12 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_12 = survey_params_dict['survey_area_ggl'][0]
        if probe12 == "lens":
            area_12 = survey_params_dict['survey_area_lens']

        if probe34 == "clust":
            area_34 = survey_params_dict['survey_area_clust'][1]
            if field1_spec == field2_spec:
                area_34 = survey_params_dict['survey_area_clust'][0]
            else:
                area_34 = max(survey_params_dict['survey_area_clust'][0], survey_params_dict['survey_area_clust'][1])
        if probe34 == "ggl":
            area_34 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_34 = survey_params_dict['survey_area_ggl'][0]
        if probe34 == "lens":
            area_34 = survey_params_dict['survey_area_lens']

        if probe12 == "clust" and probe34 == "clust":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = covELLspacesettings['n_spec']
            up_limit_4 = self.n_tomo_clust
            if field4_spec:
                lo_limit_4 = 0
                up_limit_4 = covELLspacesettings['n_spec']
        
        if probe12 == "clust" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "clust" and probe34 == "lens":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "ggl" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "ggl":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "lens":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
    
        covariance_aux = covariance[:, :, :, :, lo_limit_1 : up_limit_1, lo_limit_2 : up_limit_2, lo_limit_3 : up_limit_3, lo_limit_4 : up_limit_4]
        ellrange_12 = self.ellrange_photo
        ellrange_12_ul = self.ellrange_photo_ul
        if field1_spec or field2_spec:
            ellrange_12 = self.ellrange_spec
            ellrange_12_ul = self.ellrange_spec_ul
        ellrange_34 = self.ellrange_photo
        ellrange_34_ul = self.ellrange_photo_ul
        if field3_spec or field4_spec:
            ellrange_34 = self.ellrange_spec
            ellrange_34_ul = self.ellrange_spec_ul
        
        binned_covariance = np.zeros((len(ellrange_12), len(ellrange_34), self.sample_dim, self.sample_dim, up_limit_1 - lo_limit_1, up_limit_2 - lo_limit_2, up_limit_3 - lo_limit_3, up_limit_4 - lo_limit_4))
            
        for i_ell in range(len(ellrange_12)):
            for j_ell in range(len(ellrange_34)):
                integration_ell_12 = np.arange(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1]).astype(int)
                N_ell_12 = len(integration_ell_12)
                integration_ell_34 = np.arange(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1]).astype(int)
                N_ell_34 = len(integration_ell_34)
                overlapping_elements = np.array(list(set(integration_ell_12).intersection(set(integration_ell_34))))
                if len(overlapping_elements) == 0:
                    continue
                else:
                    for i_tomo in range(up_limit_1 - lo_limit_1):
                        for j_tomo in range(up_limit_2 - lo_limit_2):
                            for k_tomo in range(up_limit_3 - lo_limit_3):
                                for l_tomo in range(up_limit_4 - lo_limit_4):
                                    if len(np.where(np.diagonal(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo]))[0]):
                                        spline = UnivariateSpline(self.ellrange,np.log(np.diagonal(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo])), k=2, s=0, ext=1)
                                        result = full_sky_angle / max(area_12,area_34)/np.sum((2.*overlapping_elements + 1)/np.exp(spline(overlapping_elements)))
                                        binned_covariance[i_ell, j_ell, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo] = result
        return binned_covariance
    

    def __bin_cov_ell_gauss(self,
                            ellrange_12_ul,
                            ellrange_34_ul,
                            area_12,
                            area_34,
                            cov,
                            unique_12,
                            unique_34):
        if not isinstance(cov, np.ndarray):
            return 0
        fsky = max(area_12,area_34)/(4.0*np.pi * self.deg2torad2)

        binned_covariance = np.zeros((len(ellrange_12_ul) - 1, len(ellrange_34_ul) - 1, len(cov[0,0,:,0,0,0,0,0]), len(cov[0,0,0,:,0,0,0,0]), len(cov[0,0,0,0,:,0,0,0]), len(cov[0,0,0,0,0,:,0,0]), len(cov[0,0,0,0,0,0,:,0]), len(cov[0,0,0,0,0,0,0,:])))
        for i_ell in range(len(ellrange_12_ul) - 1):
            for j_ell in range(len(ellrange_34_ul) - 1):
                integration_ell_12 = np.arange(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1]).astype(int)
                N_ell_12 = np.sum(2*integration_ell_12 + 1)
                integration_ell_34 = np.arange(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1]).astype(int)
                N_ell_34 = np.sum(2*integration_ell_34 + 1)
                overlapping_elements = np.array(list(set(integration_ell_12).intersection(set(integration_ell_34))))
                if len(overlapping_elements) == 0:
                    continue
                else:
                    for i_sample in range(len(cov[0,0,:,0,0,0,0,0])):
                        for j_sample in range(len(cov[0,0,0,:,0,0,0,0])):
                            for i_tomo in range(len(cov[0,0,0,0,:,0,0,0])):
                                j_tomo_start = 0
                                if unique_12:
                                    j_tomo_start = i_tomo
                                for j_tomo in range(j_tomo_start, len(cov[0,0,0,0,0,:,0,0])):
                                    for k_tomo in range(len(cov[0,0,0,0,0,0,:,0])):
                                        l_tomo_start = 0
                                        if unique_34:
                                            l_tomo_start = k_tomo
                                        for l_tomo in range(l_tomo_start, len(cov[0,0,0,0,0,0,0,:])):
                                            if len(np.where(np.diagonal(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]))[0]):
                                                spline = UnivariateSpline(self.ellrange, np.diagonal(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]), k=1, s=0, ext=1)
                                                binned_covariance[i_ell, j_ell, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = 1/fsky/np.sum((2.*overlapping_elements + 1)/spline(overlapping_elements))
        return binned_covariance            

    def __bin_cov_ell_nongauss(self,
                               ellrange_12_ul,
                               ellrange_34_ul,
                               ellrange_12,
                               ellrange_34,
                               area_12,
                               area_34,
                               cov,
                               connected,
                               unique_12,
                               unique_34):
        if not isinstance(cov, np.ndarray):
            return 0 
        binned_covariance = np.zeros((len(ellrange_12_ul) - 1, len(ellrange_34_ul) - 1, len(cov[0,0,:,0,0,0,0,0]), len(cov[0,0,0,:,0,0,0,0]), len(cov[0,0,0,0,:,0,0,0]), len(cov[0,0,0,0,0,:,0,0]), len(cov[0,0,0,0,0,0,:,0]), len(cov[0,0,0,0,0,0,0,:])))
        for i_sample in range(len(cov[0,0,:,0,0,0,0,0])):
            for j_sample in range(len(cov[0,0,0,:,0,0,0,0])):        
                for i_tomo in range(len(cov[0,0,0,0,:,0,0,0])):
                    j_tomo_start = 0
                    if unique_12:
                        j_tomo_start = i_tomo
                    for j_tomo in range(j_tomo_start, len(cov[0,0,0,0,0,:,0,0])):
                        for k_tomo in range(len(cov[0,0,0,0,0,0,:,0])):
                            l_tomo_start = 0
                            if unique_34:
                                l_tomo_start = k_tomo
                            for l_tomo in range(l_tomo_start, len(cov[0,0,0,0,0,0,0,:])):
                                if len(np.where(np.diagonal(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]))[0]):
                                    islog = True
                                    if(np.all(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] > 0)):
                                        spline = RegularGridInterpolator((self.ellrange,self.ellrange), np.log(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]),bounds_error= False, fill_value = None)
                                    else:
                                        islog = False
                                        spline = RegularGridInterpolator((self.ellrange,self.ellrange), cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo],bounds_error= False, fill_value = None)
                                    for i_ell in range(len(ellrange_12_ul) - 1):
                                        area12_ell = np.pi*((ellrange_12_ul[i_ell +1])**2 - (ellrange_12_ul[i_ell])**2)                                            
                                        Numberi = int((np.abs(self.ellrange - ellrange_12_ul[i_ell+1])).argmin() - (np.abs(self.ellrange - ellrange_12_ul[i_ell])).argmin())
                                        if Numberi < 3:
                                                Numberi = 3
                                        if Numberi > 10:
                                            Numberi = 10
                                        integration_ell_12 = np.geomspace(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1],Numberi)
                                        for j_ell in range(len(ellrange_34_ul) - 1):
                                            area34_ell = np.pi*((ellrange_34_ul[j_ell +1])**2 - (ellrange_34_ul[j_ell])**2)                                            
                                            Numberj = int((np.abs(self.ellrange - ellrange_34_ul[j_ell+1])).argmin() - (np.abs(self.ellrange - ellrange_34_ul[j_ell])).argmin())                
                                            if Numberj < 3:
                                                Numberj = 3
                                            if Numberj > 10:
                                                Numberj = 10
                                            integration_ell_34 = np.geomspace(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1],Numberj)
                                            ell1, ell2 = np.meshgrid(integration_ell_12, integration_ell_34,indexing='ij')
                                            if islog:
                                                result = simpson(simpson(np.exp(spline((ell1, ell2)))*integration_ell_12[:,None]*integration_ell_34[None, :], x = integration_ell_34), x = integration_ell_12)
                                            else:
                                                result = simpson(simpson((spline((ell1, ell2)))*integration_ell_12[:,None]*integration_ell_34[None, :], x = integration_ell_34), x = integration_ell_12)
                                            result /= (area12_ell*area34_ell)
                                            result *= 4*np.pi**2
                                            if connected:
                                                result *=  self.deg2torad2/max(area_12,area_34)
                                            binned_covariance[i_ell, j_ell, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = result    
        return binned_covariance


    def __bin_non_Gaussian(self,
                            covELLspacesettings,
                            survey_params_dict,
                            covariance,
                            field1_spec,
                            field2_spec,
                            field3_spec,
                            field4_spec,
                            probe12,
                            probe34,
                            connected):
        r"""
        Auxillary function binning the Non-Gaussian covariance matrix in ell-bins

        """
        if not isinstance(covariance, np.ndarray):
            return 0
        full_sky_angle = 1 * self.deg2torad2

        if probe12 == "clust":
            area_12 = survey_params_dict['survey_area_clust'][1]
            if field1_spec == field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][0]
            else:
                area_12 = max(survey_params_dict['survey_area_clust'][0], survey_params_dict['survey_area_clust'][1])
        if probe12 == "ggl":
            area_12 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_12 = survey_params_dict['survey_area_ggl'][0]
        if probe12 == "lens":
            area_12 = survey_params_dict['survey_area_lens']

        if probe34 == "clust":
            area_34 = survey_params_dict['survey_area_clust'][1]
            if field1_spec == field2_spec:
                area_34 = survey_params_dict['survey_area_clust'][0]
            else:
                area_34 = max(survey_params_dict['survey_area_clust'][0], survey_params_dict['survey_area_clust'][1])
        if probe34 == "ggl":
            area_34 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_34 = survey_params_dict['survey_area_ggl'][0]
        if probe34 == "lens":
            area_34 = survey_params_dict['survey_area_lens']

        if probe12 == "clust" and probe34 == "clust":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = covELLspacesettings['n_spec']
            up_limit_4 = self.n_tomo_clust
            if field4_spec:
                lo_limit_4 = 0
                up_limit_4 = covELLspacesettings['n_spec']
        
        if probe12 == "clust" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "clust" and probe34 == "lens":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "ggl" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "ggl":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "lens":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
    
        covariance_aux = covariance[:, :, :, :, lo_limit_1 : up_limit_1, lo_limit_2 : up_limit_2, lo_limit_3 : up_limit_3, lo_limit_4 : up_limit_4]
        ellrange_12 = self.ellrange_photo
        ellrange_12_ul = self.ellrange_photo_ul
        if field1_spec or field2_spec:
            ellrange_12 = self.ellrange_spec
            ellrange_12_ul = self.ellrange_spec_ul
        ellrange_34 = self.ellrange_photo
        ellrange_34_ul = self.ellrange_photo_ul
        if field3_spec or field4_spec:
            ellrange_34 = self.ellrange_spec
            ellrange_34_ul = self.ellrange_spec_ul
        
        binned_covariance = np.zeros((len(ellrange_12), len(ellrange_34), self.sample_dim, self.sample_dim, up_limit_1 - lo_limit_1, up_limit_2 - lo_limit_2, up_limit_3 - lo_limit_3, up_limit_4 - lo_limit_4))
            
        global aux__bin_non_Gaussian

        def aux__bin_non_Gaussian(i_ell):
            aux_result = np.zeros((len(ellrange_34), self.sample_dim, self.sample_dim, up_limit_1 - lo_limit_1, up_limit_2 - lo_limit_2, up_limit_3 - lo_limit_3, up_limit_4 - lo_limit_4))
            for j_ell in range(len(ellrange_34)):
                area12_ell = (ellrange_12_ul[i_ell +1] - ellrange_12_ul[i_ell])*ellrange_12[i_ell]
                area34_ell = (ellrange_34_ul[j_ell +1] - ellrange_34_ul[j_ell])*ellrange_34[j_ell]
                Numberi = int((np.abs(self.ellrange - ellrange_12_ul[i_ell+1])).argmin() - (np.abs(self.ellrange - ellrange_12_ul[i_ell])).argmin())
                Numberj = int((np.abs(self.ellrange - ellrange_34_ul[j_ell+1])).argmin() - (np.abs(self.ellrange - ellrange_34_ul[j_ell])).argmin())
                integration_ell_12 = np.geomspace(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1],Numberi)
                integration_ell_34 = np.geomspace(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1],Numberj)
                ell1, ell2 = np.meshgrid(integration_ell_12, integration_ell_34, indexing='ij')
                for i_tomo in range(up_limit_1 - lo_limit_1):
                    for j_tomo in range(up_limit_2 - lo_limit_2):
                        for k_tomo in range(up_limit_3 - lo_limit_3):
                            for l_tomo in range(up_limit_4 - lo_limit_4):
                                if len(np.where(np.diagonal(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo]))[0]):
                                    if(np.all(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo] > 0)):
                                        spline = RegularGridInterpolator((self.ellrange,self.ellrange), np.log(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo]),bounds_error= False, fill_value = None)
                                        result = simpson(simpson(np.exp(spline((ell1, ell2)))*integration_ell_12[:,None]*integration_ell_34[None, :], x = integration_ell_34), x = integration_ell_12)
                                    else:
                                        spline = RegularGridInterpolator(self.ellrange,self.ellrange, covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo],bounds_error= False, fill_value = None)
                                        result = simpson(simpson((spline((ell1, ell2)))*integration_ell_12[:,None]*integration_ell_34[:, None], x = integration_ell_34), x = integration_ell_12)
                                    result /= (area12_ell*area34_ell)
                                    if connected:
                                        result *= full_sky_angle / max(area_12,area_34)
                                    aux_result[ j_ell, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo] = result
            return aux_result
        
        pool = mp.Pool(self.num_cores)
        binned_covariance = np.array(pool.map(
            aux__bin_non_Gaussian, [i_ell for i_ell in range(len(ellrange_12))]))
        pool.close()
        pool.terminate()
        return binned_covariance
                                    

    def covELL_gaussian(self,
                        covELLspacesettings,
                        survey_params_dict,
                        calc_prefac=True):
        r"""
        Calculates the Gaussian (disconnected) covariance in ell space
        between two observables.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : bool
            default : True
            If True, returns the full Cell-covariance. If False, it
            omits the prefactor 1/(2*ell+1)/f_sky, where f_sky is the
            observed fraction on the sky.

        Returns
        -------
        gaussELLgggg, gaussELLgggm, gaussELLggmm, \
        gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
        gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn : list of
                                                            arrays
            with shape (ell bins, ell bins,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the pure kappa-kappa ('mmmm'),
        tracer-tracer ('gggg'), and kappa-tracer ('gmgm') terms, one
        needs to add gaussELLxxyy + gaussELLxxyy_sn. They are kept
        separate for a later numerical integration.

        """

        print("Calculating gaussian covariance for angular power spectra " +
              "(C_ell's).")

        if not calc_prefac or self.ellrange_spec is not None:
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                self.__covELL_split_gaussian(covELLspacesettings,
                                            survey_params_dict,
                                            False)
        else:
            if calc_prefac and ((self.mm and self.ellrange_lensing_ul is None) or (self.gm and self.ellrange_clustering_ul is None) or (self.gg and self.ellrange_clustering_ul is None)):
                gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                    gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                    gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                    gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                    gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                    gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                    self.__covELL_split_gaussian(covELLspacesettings,
                                                survey_params_dict,
                                                True)
            else:
                gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                    gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                    gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                    gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                    gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                    gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                    self.__covELL_split_gaussian(covELLspacesettings,
                                                survey_params_dict,
                                                False)
        if self.csmf:
            cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm = self.calc_covELL_csmf(covELLspacesettings, survey_params_dict)
            if self.gg and self.ellrange_clustering_ul is not None:
                cov_csmf_gg = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cov_csmf_gg,True)
            if self.gm and self.ellrange_clustering_ul is not None:
                cov_csmf_gm = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cov_csmf_gm,False)
            if self.mm and self.ellrange_lensing_ul is not None:
                cov_csmf_mm = self.__bin_cov_ell_csmf(self.ellrange_lensing_ul,cov_csmf_mm,True)
        if covELLspacesettings['pixelised_cell']:
            
            gaussELLgggg_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgggg_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgggg_sn *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgggm_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgggm_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]  
            gaussELLgggm_sn *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLggmm_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLggmm_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLggmm_sn *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgmgm_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgmgm_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLgmgm_sn *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmgm_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmgm_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmgm_sn *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmmm_sva *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmmm_mix *= self.pixelweight_matrix[:,:, None, None, None, None, None, None] 
            gaussELLmmmm_sn  *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]

        if calc_prefac or self.ellrange_spec is None:
            if self.gg and self.ellrange_clustering_ul is not None:
                gaussELLgggg_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_sva,
                                                            True,
                                                            True)
                gaussELLgggg_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_mix,
                                                            True,
                                                            True)
                gaussELLgggg_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True)
            if self.gg and self.gm and self.ellrange_clustering_ul is not None:
                gaussELLgggm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_sva,
                                                            True,
                                                            False)
                gaussELLgggm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_mix,
                                                            True,
                                                            False)
                gaussELLgggm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_sn,
                                                            True,
                                                            False)
            if self.gg and self.mm and self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                gaussELLggmm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_sva,
                                                            True,
                                                            True)
                gaussELLggmm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_mix,
                                                            True,
                                                            True)
                gaussELLggmm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_sn,
                                                            True,
                                                            True)
            if self.gm and self.ellrange_clustering_ul is not None:
                gaussELLgmgm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_sva,
                                                            False,
                                                            False)
                gaussELLgmgm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_mix,
                                                            False,
                                                            False)
                gaussELLgmgm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False)
            if self.gm and self.mm and self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:  
                gaussELLmmgm_sva = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmgm_sva,
                                                            True,
                                                            False)
                gaussELLmmgm_mix = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmgm_mix,
                                                            True,
                                                            False)
                gaussELLmmgm_sn = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmgm_sn,
                                                            True,
                                                            False)
            
            if self.mm and self.ellrange_lensing_ul is not None:
            
                gaussELLmmmm_sva = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_sva,
                                                            True,
                                                            True)
                gaussELLmmmm_mix = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_mix,
                                                            True,
                                                            True)
                gaussELLmmmm_sn = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_sn,
                                                            True,
                                                            True)

        if not self.cov_dict['split_gauss'] and self.ellrange_spec is not None and calc_prefac:
            gaussELLgggg = gaussELLgggg_sva + gaussELLgggg_mix
            gaussELLgggm = gaussELLgggm_sva + gaussELLgggm_mix
            gaussELLgmgm = gaussELLgmgm_sva + gaussELLgmgm_mix
            gaussELLmmgm = gaussELLmmgm_sva + gaussELLmmgm_mix
            gaussELLmmmm = gaussELLmmmm_sva + gaussELLmmmm_mix
            gaussELLggmm = gaussELLggmm_sva + gaussELLggmm_mix    
            
            gaussELLgggg_ssss_new = 0
            gaussELLgggg_sssp_new = 0
            gaussELLgggg_sspp_new = 0
            gaussELLgggg_spsp_new = 0
            gaussELLgggg_ppsp_new = 0
            gaussELLgggg_pppp_new = 0

            gaussELLgggm_sssm_new = 0
            gaussELLgggm_sspm_new = 0
            gaussELLgggm_spsm_new = 0
            gaussELLgggm_sppm_new = 0
            gaussELLgggm_ppsm_new = 0
            gaussELLgggm_pppm_new = 0

            gaussELLggmm_ssmm_new = 0
            gaussELLggmm_spmm_new = 0
            gaussELLggmm_ppmm_new = 0

            gaussELLgmgm_smsm_new = 0
            gaussELLgmgm_smpm_new = 0
            gaussELLgmgm_pmsm_new = 0
            gaussELLgmgm_pmpm_new = 0

            gaussELLmmgm_mmsm_new = 0
            gaussELLmmgm_mmpm_new = 0

            gaussELLmmmm_mmmm_new = 0


            gaussELLgggg_ssss_new_sn = 0
            gaussELLgggg_sssp_new_sn = 0
            gaussELLgggg_sspp_new_sn = 0
            gaussELLgggg_spsp_new_sn = 0
            gaussELLgggg_ppsp_new_sn = 0
            gaussELLgggg_pppp_new_sn = 0

            gaussELLgmgm_smsm_new_sn = 0
            gaussELLgmgm_smpm_new_sn = 0
            gaussELLgmgm_pmsm_new_sn = 0
            gaussELLgmgm_pmpm_new_sn = 0

            gaussELLmmmm_mmmm_new_sn = 0
            
            if self.gg:
                gaussELLgggg_ssss_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ssss_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )

            if self.gm and self.gg:
                gaussELLgggm_sssm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sspm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_spsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sppm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_ppsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_pppm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
            if self.mm and self.gg:
                gaussELLggmm_ssmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_spmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_ppmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
            if self.gm:
                gaussELLgmgm_smsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
            if self.mm and self.gm:
                gaussELLmmgm_mmsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                gaussELLmmgm_mmpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
            if self.mm:
                gaussELLmmmm_mmmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                gaussELLmmmm_mmmm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
            return gaussELLgggg_ssss_new, \
                    gaussELLgggg_sssp_new, \
                    gaussELLgggg_sspp_new, \
                    gaussELLgggg_spsp_new, \
                    gaussELLgggg_ppsp_new, \
                    gaussELLgggg_pppp_new, \
                    gaussELLgggm_sssm_new, \
                    gaussELLgggm_sspm_new, \
                    gaussELLgggm_spsm_new, \
                    gaussELLgggm_sppm_new, \
                    gaussELLgggm_ppsm_new, \
                    gaussELLgggm_pppm_new, \
                    gaussELLggmm_ssmm_new, \
                    gaussELLggmm_spmm_new, \
                    gaussELLggmm_ppmm_new, \
                    gaussELLgmgm_smsm_new, \
                    gaussELLgmgm_smpm_new, \
                    gaussELLgmgm_pmsm_new, \
                    gaussELLgmgm_pmpm_new, \
                    gaussELLmmgm_mmsm_new, \
                    gaussELLmmgm_mmpm_new, \
                    gaussELLmmmm_mmmm_new, \
                    gaussELLgggg_ssss_new_sn, \
                    gaussELLgggg_sssp_new_sn, \
                    gaussELLgggg_sspp_new_sn, \
                    gaussELLgggg_spsp_new_sn, \
                    gaussELLgggg_ppsp_new_sn, \
                    gaussELLgggg_pppp_new_sn, \
                    gaussELLgmgm_smsm_new_sn, \
                    gaussELLgmgm_smpm_new_sn, \
                    gaussELLgmgm_pmsm_new_sn, \
                    gaussELLgmgm_pmpm_new_sn, \
                    gaussELLmmmm_mmmm_new_sn

                                       
        if not self.cov_dict['split_gauss']:
            gaussELLgggg = gaussELLgggg_sva + gaussELLgggg_mix
            gaussELLgggm = gaussELLgggm_sva + gaussELLgggm_mix
            gaussELLgmgm = gaussELLgmgm_sva + gaussELLgmgm_mix
            gaussELLmmgm = gaussELLmmgm_sva + gaussELLmmgm_mix
            gaussELLmmmm = gaussELLmmmm_sva + gaussELLmmmm_mix
            gaussELLggmm = gaussELLggmm_sva + gaussELLggmm_mix
            if self.csmf:
                return gaussELLgggg, gaussELLgggm, gaussELLggmm, \
                    gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
                    gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn, \
                    cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm
            else:
                return gaussELLgggg, gaussELLgggm, gaussELLggmm, \
                    gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
                    gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn
        else:
            if self.ellrange_spec is not None and calc_prefac:
                gaussELLgggg_ssss_new_sva = 0
                gaussELLgggg_sssp_new_sva = 0
                gaussELLgggg_sspp_new_sva = 0
                gaussELLgggg_spsp_new_sva = 0
                gaussELLgggg_ppsp_new_sva = 0
                gaussELLgggg_pppp_new_sva = 0

                gaussELLgggm_sssm_new_sva = 0
                gaussELLgggm_sspm_new_sva = 0
                gaussELLgggm_spsm_new_sva = 0
                gaussELLgggm_sppm_new_sva = 0
                gaussELLgggm_ppsm_new_sva = 0
                gaussELLgggm_pppm_new_sva = 0

                gaussELLggmm_ssmm_new_sva = 0
                gaussELLggmm_spmm_new_sva = 0
                gaussELLggmm_ppmm_new_sva = 0

                gaussELLgmgm_smsm_new_sva = 0
                gaussELLgmgm_smpm_new_sva = 0
                gaussELLgmgm_pmsm_new_sva = 0
                gaussELLgmgm_pmpm_new_sva = 0

                gaussELLmmgm_mmsm_new_sva = 0
                gaussELLmmgm_mmpm_new_sva = 0

                gaussELLmmmm_mmmm_new_sva = 0


                gaussELLgggg_ssss_new_mix = 0
                gaussELLgggg_sssp_new_mix = 0
                gaussELLgggg_sspp_new_mix = 0
                gaussELLgggg_spsp_new_mix = 0
                gaussELLgggg_ppsp_new_mix = 0
                gaussELLgggg_pppp_new_mix = 0

                gaussELLgggm_sssm_new_mix = 0
                gaussELLgggm_sspm_new_mix = 0
                gaussELLgggm_spsm_new_mix = 0
                gaussELLgggm_sppm_new_mix = 0
                gaussELLgggm_ppsm_new_mix = 0
                gaussELLgggm_pppm_new_mix = 0

                gaussELLggmm_ssmm_new_mix = 0
                gaussELLggmm_spmm_new_mix = 0
                gaussELLggmm_ppmm_new_mix = 0

                gaussELLgmgm_smsm_new_mix = 0
                gaussELLgmgm_smpm_new_mix = 0
                gaussELLgmgm_pmsm_new_mix = 0
                gaussELLgmgm_pmpm_new_mix = 0

                gaussELLmmgm_mmsm_new_mix = 0
                gaussELLmmgm_mmpm_new_mix = 0

                gaussELLmmmm_mmmm_new_mix = 0


                gaussELLgggg_ssss_new_sn = 0
                gaussELLgggg_sssp_new_sn = 0
                gaussELLgggg_sspp_new_sn = 0
                gaussELLgggg_spsp_new_sn = 0
                gaussELLgggg_ppsp_new_sn = 0
                gaussELLgggg_pppp_new_sn = 0

                gaussELLgggm_sssm_new_sn = 0
                gaussELLgggm_sspm_new_sn = 0
                gaussELLgggm_spsm_new_sn = 0
                gaussELLgggm_sppm_new_sn = 0
                gaussELLgggm_ppsm_new_sn = 0
                gaussELLgggm_pppm_new_sn = 0

                gaussELLggmm_ssmm_new_sn = 0
                gaussELLggmm_spmm_new_sn = 0
                gaussELLggmm_ppmm_new_sn = 0

                gaussELLmmgm_mmsm_new_sn = 0
                gaussELLmmgm_mmpm_new_sn = 0


                gaussELLgmgm_smsm_new_sn = 0
                gaussELLgmgm_smpm_new_sn = 0
                gaussELLgmgm_pmsm_new_sn = 0
                gaussELLgmgm_pmpm_new_sn = 0

                gaussELLmmmm_mmmm_new_sn = 0
                
                if self.gg:
                    gaussELLgggg_ssss_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sssp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sspp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_spsp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_ppsp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_pppp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_ssss_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sssp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sspp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_spsp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_ppsp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_pppp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_ssss_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sssp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_sspp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_spsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_ppsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )
                    gaussELLgggg_pppp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggg_sn,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust'
                                                                )

                if self.gm and self.gg:
                    gaussELLgggm_sssm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_sspm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_spsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_sppm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_ppsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_pppm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_sssm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_sspm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_spsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_sppm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_ppsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                    gaussELLgggm_pppm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgggm_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl'
                                                                )
                if self.mm and self.gg:
                    gaussELLggmm_ssmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_sva,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                    gaussELLggmm_spmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_sva,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                    gaussELLggmm_ppmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                    gaussELLggmm_ssmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_mix,
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                    gaussELLggmm_spmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_mix,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                    gaussELLggmm_ppmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLggmm_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens'
                                                                )
                if self.gm:
                    gaussELLgmgm_smsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sva,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_smpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sva,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sva,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_smsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_mix,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_smpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_mix,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_mix,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_smsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sn,
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_smpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sn,
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sn,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                    gaussELLgmgm_pmpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLgmgm_sn,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl'
                                                                )
                if self.mm and self.gm:
                    gaussELLmmgm_mmsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmgm_sva,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl'
                                                                )
                    gaussELLmmgm_mmpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmgm_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl'
                                                                )
                    gaussELLmmgm_mmsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmgm_mix,
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl'
                                                                )
                    gaussELLmmgm_mmpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmgm_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl'
                                                                )
                if self.mm:
                    gaussELLmmmm_mmmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmmm_sva,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens'
                                                                )
                    gaussELLmmmm_mmmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmmm_mix,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens'
                                                                )
                    gaussELLmmmm_mmmm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                                gaussELLmmmm_sn,
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens'
                                                                )
                return gaussELLgggg_ssss_new_sva, gaussELLgggg_ssss_new_mix, gaussELLgggg_ssss_new_sn, \
                       gaussELLgggg_sssp_new_sva, gaussELLgggg_sssp_new_mix, gaussELLgggg_sssp_new_sn, \
                       gaussELLgggg_sspp_new_sva, gaussELLgggg_sspp_new_mix, gaussELLgggg_sspp_new_sn, \
                       gaussELLgggg_spsp_new_sva, gaussELLgggg_spsp_new_mix, gaussELLgggg_spsp_new_sn, \
                       gaussELLgggg_ppsp_new_sva, gaussELLgggg_ppsp_new_mix, gaussELLgggg_ppsp_new_sn, \
                       gaussELLgggg_pppp_new_sva, gaussELLgggg_pppp_new_mix, gaussELLgggg_pppp_new_sn, \
                       gaussELLgggm_sssm_new_sva, gaussELLgggm_sssm_new_mix, gaussELLgggm_sssm_new_sn, \
                       gaussELLgggm_sspm_new_sva, gaussELLgggm_sspm_new_mix, gaussELLgggm_sspm_new_sn, \
                       gaussELLgggm_spsm_new_sva, gaussELLgggm_spsm_new_mix, gaussELLgggm_spsm_new_sn, \
                       gaussELLgggm_sppm_new_sva, gaussELLgggm_sppm_new_mix, gaussELLgggm_sppm_new_sn, \
                       gaussELLgggm_ppsm_new_sva, gaussELLgggm_ppsm_new_mix, gaussELLgggm_ppsm_new_sn, \
                       gaussELLgggm_pppm_new_sva, gaussELLgggm_pppm_new_mix, gaussELLgggm_pppm_new_sn, \
                       gaussELLggmm_ssmm_new_sva, gaussELLggmm_ssmm_new_mix, gaussELLggmm_ssmm_new_sn, \
                       gaussELLggmm_spmm_new_sva, gaussELLggmm_spmm_new_mix, gaussELLggmm_spmm_new_sn, \
                       gaussELLggmm_ppmm_new_sva, gaussELLggmm_ppmm_new_mix, gaussELLggmm_ppmm_new_sn, \
                       gaussELLgmgm_smsm_new_sva, gaussELLgmgm_smsm_new_mix, gaussELLgmgm_smsm_new_sn, \
                       gaussELLgmgm_smpm_new_sva, gaussELLgmgm_smpm_new_mix, gaussELLgmgm_smpm_new_sn, \
                       gaussELLgmgm_pmsm_new_sva, gaussELLgmgm_pmsm_new_mix, gaussELLgmgm_pmsm_new_sn, \
                       gaussELLgmgm_pmpm_new_sva, gaussELLgmgm_pmpm_new_mix, gaussELLgmgm_pmpm_new_sn, \
                       gaussELLmmgm_mmsm_new_sva, gaussELLmmgm_mmsm_new_mix, gaussELLmmgm_mmsm_new_sn,\
                       gaussELLmmgm_mmpm_new_sva, gaussELLmmgm_mmpm_new_mix, gaussELLmmgm_mmpm_new_sn,\
                       gaussELLmmmm_mmmm_new_sva, gaussELLmmmm_mmmm_new_mix, gaussELLmmmm_mmmm_new_sn
            else:
                if self.csmf:
                    return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                        gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                        gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                        gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                        gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                        gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn, \
                        cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm
                else:
                    return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                        gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                        gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                        gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                        gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                        gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn

    def __covELL_split_gaussian(self,
                                covELLspacesettings,
                                survey_params_dict,
                                calc_prefac):
        r"""
        Calculates the Gaussian (disconnected) covariance in ell space
        for the specified observables and splits it into sample-variance
        (SVA), shot noise (SN) and SNxSVA(mix) terms.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : bool
            default : True
            If True, returns the full Cell-covariance. If False, it
            omits the prefactor 1/(2*ell+1)/f_sky, where f_sky is the
            observed fraction on the sky. Alternatively, all prefactors
            are unity.

        Returns
        -------
        gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn,
        gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn,
        gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn,
        gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn,
        gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn,
        gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn: list of
                                                             arrays
            with shape (ell bins, ell bins,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        To get the full covariance contribution, one needs to add
        gaussELLwxyz_sva + gaussELLwxyz_mix + gaussELLwxyz_sn. Some
        terms are zero by definition (e.g., gaussgggm_sn).

        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if self.mm or self.gm:
            if survey_params_dict['n_eff_lens'] is not None:
                noise_kappa = \
                    np.diag(survey_params_dict['ellipticity_dispersion']**2 \
                    / survey_params_dict['n_eff_lens'] / self.arcmin2torad2)
            else:
                raise Exception("WildError: This shouldn't happen...")


        
        if self.gg or self.gm:
            if survey_params_dict['n_eff_clust'] is not None and self.sample_dim <= 1:
                noise_g =np.zeros((self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust))
                for i_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        noise_g[i_sample, i_sample, i_tomo, i_tomo] = 1 / survey_params_dict['n_eff_clust'][i_tomo] / self.arcmin2torad2
            else:
                noise_g =np.zeros((self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust))
                for i_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        noise_g[i_sample, i_sample, i_tomo, i_tomo] = 1.0/self.Ngal[i_sample,i_tomo]
       
        reshape_mat = np.eye(len(self.ellrange))[
            :, :, None, None, None, None, None, None]*np.ones((self.sample_dim,self.sample_dim))[None, None, :, :, None, None, None, None]
        if self.gg:
            # Cov_sva(gg^ij gg^kl) = Cgg^ik Cgg^jl + Cgg^il Cgg^jk
            gaussELLgggg_sva = self.Cell_gg[:, :, :, :, None, :, None] \
                * self.Cell_gg[:, :, :, None, :, None, :] \
                + self.Cell_gg[:, :, :, :, None, None, :] \
                * self.Cell_gg[:, :, :, None, :, :, None]
            
            # Cov_mix(gg^ij gg^kl) = Cgg^ik noise_g^jl + Cgg^jl noise_g^ik
            #                      + Cgg^il noise_g^jk + Cgg^jk noise_g^il
            gaussELLgggg_mix = self.Cell_gg[:, :, :, :, None, :, None] \
                * noise_g[None, :, :, None, :, None, :] \
                + self.Cell_gg[:, :, :, None, :, None, :] \
                * noise_g[None, :, :, :, None, :, None] \
                + self.Cell_gg[:, :, :, :, None, None, :]  \
                * noise_g[None, :, :, None, :, :, None] \
                + self.Cell_gg[:, :, :, None, :, :, None]\
                * noise_g[None, :, :, :, None, None, :]
            # Cov_sn(gg^ij gg^kl) = noise_g^ik noise_g^jl
            #                     + noise_g^il noise_g^jk
            gaussELLgggg_sn = \
                noise_g[None, :, :, :, None, :, None] \
                * noise_g[None, :, :, None, :, None, :]  \
                + noise_g[None, :, :, :, None, None, :] \
                * noise_g[None, :, :, None, :, :, None]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_clust, self.n_tomo_clust]
                prefac_gggg = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'])
                gaussELLgggg_sva = prefac_gggg * gaussELLgggg_sva
                gaussELLgggg_mix = prefac_gggg * gaussELLgggg_mix
                gaussELLgggg_sn = prefac_gggg * \
                    gaussELLgggg_sn
                
            gaussELLgggg_sva = gaussELLgggg_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgggg_mix = gaussELLgggg_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
            gaussELLgggg_sn = gaussELLgggg_sn[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn = 0, 0, 0

        if self.gg and self.gm and self.cross_terms:
            # Cov_sva(gg^ij gm^kl) = Cgg^ik Cgm^jl + Cgg^jk Cgm^il
            gaussELLgggm_sva = self.Cell_gg[:, :, :, :, None, :, None] \
                * self.Cell_gm[:, :, None, None, :, None, :] \
                + self.Cell_gg[:, :, :, None, :, :, None] \
                * self.Cell_gm[:, :, None, :, None, None, :]
            # Cov_mix(gg^ij gm^kl) = Cgm^jl noise_g^ik + Cgm^il noise_g^jk
            gaussELLgggm_mix = self.Cell_gm[:, :, None, None, :, None, :] \
                * noise_g[None,:, :, :, None, :, None] \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * noise_g[None, :, :, None, :, :, None]
            gaussELLgggm_sn = 0

            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim, 
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_gggm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'],
                    survey_params_dict['survey_area_ggl'])
                gaussELLgggm_sva = prefac_gggm * gaussELLgggm_sva
                gaussELLgggm_mix = prefac_gggm * gaussELLgggm_mix

            gaussELLgggm_sva = gaussELLgggm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgggm_mix = gaussELLgggm_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
        else:
            gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn = 0, 0, 0

        if self.gg and self.mm and self.cross_terms:
            # Cov_sva(gg^ij mm^kl) = Cgm^ik Cgm^jl + Cgm^il Cgm^jk
            gaussELLggmm_sva = self.Cell_gm[:, :, None ,:, None, :, None] \
                * self.Cell_gm[:, :, None, None, :, None, :] \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * self.Cell_gm[:, :, None, None, :, :, None]
            gaussELLggmm_mix = 0
            gaussELLggmm_sn = 0
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_lens, self.n_tomo_lens]
                prefac_ggmm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'],
                    survey_params_dict['survey_area_lens'])
                gaussELLggmm_sva = prefac_ggmm * gaussELLggmm_sva
            gaussELLggmm_sva = gaussELLggmm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLggmm_sva = gaussELLggmm_sva[:, :, :, :1, :, :, :, :]
        else:
            gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn = 0, 0, 0

        if self.gm:
            # Cov_sva(gm^ij gm^kl) = Cgg^ik Cmm^jl + Cgm^il Cgm^kj
            gaussELLgmgm_sva = (self.Cell_gg[:, :, :, :, None, :, None]
                                * self.Cell_mm[:, :, None, None, :, None, :]) \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None]
            # Cov_mix(gm^ij gm^kl) = Cgg^ik noise_m^jl + Cmm^jl noise_g^ik
            gaussELLgmgm_mix = self.Cell_gg[:, :, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :] \
                + self.Cell_mm[:, :, None, None, :, None, :] \
                * noise_g[None, :, :, :, None, :, None]
            # Cov_sn(gm^ij gm^kl) = noise_g^ik noise_m^jl
            gaussELLgmgm_sn = noise_g[None, :, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_lens,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_gmgm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_ggl'])
                gaussELLgmgm_sva = prefac_gmgm * gaussELLgmgm_sva
                gaussELLgmgm_mix = prefac_gmgm * gaussELLgmgm_mix
                gaussELLgmgm_sn = prefac_gmgm * \
                    gaussELLgmgm_sn
            gaussELLgmgm_sva = gaussELLgmgm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
                
            gaussELLgmgm_mix = gaussELLgmgm_mix[:, None, :, :, :, :, :, :]\
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgmgm_sn = gaussELLgmgm_sn[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
        else:
            gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn = 0, 0, 0

        if self.mm and self.gm and self.cross_terms:
            # Cov_sva(mm^ij gm^kl) = Cmm^lj Cgm^ki + Cmm^li Cgm^kj
            gaussELLmmgm_sva = self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, :, None, :, None] \
                * self.Cell_mm[:, :, None, None, :, None, :] \
                + self.Cell_mm[:, :, None, :, None, None, :] \
                * self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None]
            # Cov_mix(mm^ij gm^kl) = Cgm^ki noise_m^lj + Cgm^kj noise_m^li
            gaussELLmmgm_mix = self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :] \
                + self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None] \
                * noise_kappa[None, None, None, :, None, None, :]
            gaussELLmmgm_sn = 0

            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_mmgm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_lens'],
                    survey_params_dict['survey_area_ggl'])
                gaussELLmmgm_sva = prefac_mmgm * gaussELLmmgm_sva
                gaussELLmmgm_mix = prefac_mmgm * gaussELLmmgm_mix

            gaussELLmmgm_sva = gaussELLmmgm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLmmgm_mix = gaussELLmmgm_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLmmgm_sva = gaussELLmmgm_sva[:, :, :1, :, :, :, :, :]
            gaussELLmmgm_mix = gaussELLmmgm_mix[:, :, :1, :, :, :, :, :]    
        else:
            gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn = 0, 0, 0

        if self.mm:
            # Cov_sva(mm^ij mm^kl) = Cmm^ik Cmm^jl + Cmm^il Cmm^jk
            gaussELLmmmm_sva = self.Cell_mm[:, :, None, :, None, :, None] \
                * self.Cell_mm[:, None, :, None, :, None, :] \
                + self.Cell_mm[:, :, None, :, None, None, :] \
                * self.Cell_mm[:, None, :, None, :, :, None]
            # Cov_mix(mm^ij mm^kl) = Cmm^ik noise_m^jl + Cmm^jl noise_m^ik
            #                      + Cmm^il noise_m^jk + Cmm^jk noise_m^il
            gaussELLmmmm_mix = self.Cell_mm[:, :, :, None, :, None] \
                * noise_kappa[None, None, None, :, None, :] \
                + self.Cell_mm[:, :, None, :, None, :] \
                * noise_kappa[None, None, :, None, :, None] \
                + self.Cell_mm[:, :, :, None, None, :] \
                * noise_kappa[None, None, None, :, :, None] \
                + self.Cell_mm[:, :, None, :, :, None] \
                * noise_kappa[None, None :, None, None, :]
            # Cov_sn(mm^ij mm^kl) = noise_m^ik noise_m^jl
            #                     + noise_m^il noise_m^jk
            gaussELLmmmm_sn = \
                noise_kappa[:, None, :, None] \
                * noise_kappa[None, :, None, :] \
                + noise_kappa[:, None, None, :] \
                * noise_kappa[None, :, :, None]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens,
                              self.n_tomo_lens, self.n_tomo_lens]
                prefac_mmmm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_lens'])
                gaussELLmmmm_sva = prefac_mmmm * gaussELLmmmm_sva
                gaussELLmmmm_mix = prefac_mmmm * gaussELLmmmm_mix[:, None, :, :, :, :, :]
                gaussELLmmmm_sn = prefac_mmmm * \
                    gaussELLmmmm_sn[None, None, None, :, :, :, :]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, :, :1, :1, :, :, :, :]
            else:
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, None, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, None, None, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sn = gaussELLmmmm_sn[None, None, None, None, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, :, :1, :1, :, :, :, :]
                   
        else:
            gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = 0, 0, 0

        if self.mm or self.gm:
            if len(covELLspacesettings['mult_shear_bias']) < self.n_tomo_lens:
                covELLspacesettings['mult_shear_bias'] = np.zeros(self.n_tomo_lens)
                print("Multiplicative shear bias needs to be given for every tomographic bin. Is set to zero")
            if self.gm:
                self.gaussELLgmgm_sva_mult_shear_bias = np.zeros_like(gaussELLgmgm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_clust):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLgmgm_sva_mult_shear_bias[i_sample, j_sample, i_tomo,j_tomo,k_tomo,l_tomo] = covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
            if self.gm and self.mm and self.cross_terms:
                self.gaussELLmmgm_sva_mult_shear_bias = np.zeros_like(gaussELLmmgm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(1):
                    for j_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_clust):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLmmgm_sva_mult_shear_bias[i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = (covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo])
            if self.mm:
                self.gaussELLmmmm_sva_mult_shear_bias = np.zeros_like(gaussELLmmmm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(1):
                    for j_sample in range(1):
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_lens):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLmmmm_sva_mult_shear_bias[i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] =  (covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][k_tomo] 
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][k_tomo])

        if self.est_shear != "C_ell" and self.mm:
            for i_sample in range(1):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLmmmm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLmmmm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and self.gg:
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_clust):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLgggg_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLgggg_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and self.gg and self.mm:
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLggmm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and (self.mm and self.gm and self.cross_terms):
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo:
                                        gaussELLmmgm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLmmgm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and (self.gg and self.gm and self.cross_terms):
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo:
                                        gaussELLgggm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLgggm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0

        return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
            gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
            gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
            gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
            gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
            gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn

    def __calc_prefac_covELL(self,
                             covELLspacesettings,
                             tomo_shape,
                             survey_area,
                             survey_area2=None):
        r"""
        Calculates the prefactor,  1/(2*ell+1)/f_sky, which is used
        if the pure ell-space covariance should be used for inference.


        """
        if covELLspacesettings['ell_type'] == 'lin':
            delta_ellrange = self.ellrange[1]-self.ellrange[0]
        elif covELLspacesettings['ell_type'] == 'log':
            logdelta = np.log10(self.ellrange[1]/self.ellrange[0])/2
            ell_ul_range = \
                np.append(np.log10(self.ellrange) - logdelta,
                          np.log10(self.ellrange[-1]) + logdelta)
            delta_ellrange = 10**ell_ul_range[1:] - 10**ell_ul_range[:-1]
        full_sky_angle = 4*np.pi * self.deg2torad2
        prefac_noarea = full_sky_angle / (2*self.ellrange + 1)/ delta_ellrange
        prefac = None
        if len(survey_area) == 1:
            if survey_area2 is None:
                prefac = prefac_noarea / survey_area[0]
            elif len(survey_area2) == 1:
                prefac = prefac_noarea / max(survey_area[0], survey_area2[0])
        if prefac is not None:
            prefac = prefac[:, None, None, None, None, None, None] \
                * np.ones(([len(self.ellrange)] + tomo_shape))
        else:
            prefac = self.__calc_prefac6x2pt_covELL(prefac_noarea,
                                                    tomo_shape,
                                                    survey_area,
                                                    survey_area2)
        return prefac

    def __calc_prefac6x2pt_covELL(self,
                                  prefac_noarea,
                                  tomo_shape,
                                  survey_area,
                                  survey_area2):
        prefac = np.zeros(([len(self.ellrange)] + tomo_shape))
        prefac_noarea = prefac_noarea[:, None, None, None, None]
        t_sz = self.tomos_6x2pt_clust
        if survey_area2 is None:
            # gggg combs from ssss to pppp (nothing fixed)
            if tomo_shape[0] == tomo_shape[1]:
                # ssss
                prefac_ssss = prefac_noarea / survey_area[0]
                prefac[:, :t_sz, :t_sz, :t_sz, :t_sz] = prefac_ssss
                # pppp
                prefac_pppp = prefac_noarea / survey_area[1]
                prefac[:, t_sz:, t_sz:, t_sz:, t_sz:] = prefac_pppp
                # spsp + spps + pssp + psps
                prefac_spsp = prefac_noarea / survey_area[2]
                prefac[:, :t_sz, t_sz:, :t_sz, t_sz:] = prefac_spsp
                prefac[:, :t_sz, t_sz:, t_sz:, :t_sz] = prefac_spsp
                prefac[:, t_sz:, :t_sz, :t_sz, t_sz:] = prefac_spsp
                prefac[:, t_sz:, :t_sz, t_sz:, :t_sz] = prefac_spsp
                # sspp + ppss
                prefac_sspp = \
                    prefac_noarea / max(survey_area[0], survey_area[1])
                prefac[:, :t_sz, :t_sz, t_sz:, t_sz:] = prefac_sspp
                prefac[:, t_sz:, t_sz:, :t_sz, :t_sz] = prefac_sspp
                # sssp + ssps + spss + psss
                prefac_sssp = \
                    prefac_noarea / max(survey_area[0], survey_area[2])
                prefac[:, :t_sz, :t_sz, :t_sz, t_sz:] = prefac_sssp
                prefac[:, :t_sz, :t_sz, t_sz:, :t_sz] = prefac_sssp
                prefac[:, :t_sz, t_sz:, :t_sz, :t_sz] = prefac_sssp
                prefac[:, t_sz:, :t_sz, :t_sz, :t_sz] = prefac_sssp
                # sppp + pspp + ppsp + ppps
                prefac_ppps = \
                    prefac_noarea / max(survey_area[1], survey_area[2])
                prefac[:, :t_sz, t_sz:, t_sz:, t_sz:] = prefac_ppps
                prefac[:, t_sz:, :t_sz, t_sz:, t_sz:] = prefac_ppps
                prefac[:, t_sz:, t_sz:, :t_sz, t_sz:] = prefac_ppps
                prefac[:, t_sz:, t_sz:, t_sz:, :t_sz] = prefac_ppps
            # gmgm combs from sPsP to pPpP (P fixed)
            elif tomo_shape[0] != tomo_shape[1]:
                # sPsP
                prefac_sPsP = prefac_noarea / survey_area[0]
                prefac[:, :t_sz, :, :t_sz, :] = prefac_sPsP
                # pPpP
                prefac_pPpP = prefac_noarea / survey_area[1]
                prefac[:, t_sz:, :, t_sz:, :] = prefac_pPpP
                # sPpP + pPsP
                prefac_sPpP = \
                    prefac_noarea / max(survey_area[0], survey_area[1])
                prefac[:, :t_sz, :, t_sz:, :] = prefac_sPpP
                prefac[:, t_sz:, :, :t_sz, :] = prefac_sPpP
            else:
                raise Exception("too stupid for 6x2pt, diagonal case")
        else:
            # gggm combs from sssP to pppP (P fixed)
            if tomo_shape[0] == tomo_shape[1] and \
               tomo_shape[0] == tomo_shape[2]:
                # sssP
                prefac_sssP = \
                    prefac_noarea / max(survey_area[0], survey_area2[0])
                prefac[:, :t_sz, :t_sz, :t_sz, :] = prefac_sssP
                # sspP
                prefac_sspP = \
                    prefac_noarea / max(survey_area[0], survey_area2[1])
                prefac[:, :t_sz, :t_sz, t_sz:, :] = prefac_sspP
                # ppsP
                prefac_ppsP = \
                    prefac_noarea / max(survey_area[1], survey_area2[0])
                prefac[:, t_sz:, t_sz:, :t_sz, :] = prefac_ppsP
                # pppP
                prefac_pppP = \
                    prefac_noarea / max(survey_area[1], survey_area2[1])
                prefac[:, t_sz:, t_sz:, t_sz:, :] = prefac_pppP
                # spsP + spsP
                prefac_spsP = \
                    prefac_noarea / max(survey_area[2], survey_area2[0])
                prefac[:, :t_sz, t_sz:, :t_sz, :] = prefac_spsP
                prefac[:, t_sz:, :t_sz, :t_sz, :] = prefac_spsP
                # sppP + pspP
                prefac_sppP = \
                    prefac_noarea / max(survey_area[2], survey_area2[1])
                prefac[:, :t_sz, t_sz:, t_sz:, :] = prefac_sppP
                prefac[:, t_sz:, :t_sz, t_sz:, :] = prefac_sppP
            # ggmm combs from ssPP to ppPP (P fixed)
            elif tomo_shape[0] == tomo_shape[1] and \
                    tomo_shape[2] == tomo_shape[3]:
                # ssPP + PPss
                prefac_ssPP = \
                    prefac_noarea / max(survey_area[0], survey_area2[0])
                prefac[:, :t_sz, :t_sz, :, :] = prefac_ssPP
                # ppPP + PPpp
                prefac_ppPP = \
                    prefac_noarea / max(survey_area[1], survey_area2[0])
                prefac[:, t_sz:, t_sz:, :, :] = prefac_ppPP
                # spPP + psPP
                prefac_spPP = \
                    prefac_noarea / max(survey_area[2], survey_area2[0])
                prefac[:, :t_sz, t_sz:, :, :] = prefac_spPP
                prefac[:, t_sz:, :t_sz, :, :] = prefac_spPP
            # mmgm combs from PPsP to PPpP (P fixed)
            elif tomo_shape[0] == tomo_shape[1] and \
                    tomo_shape[0] == tomo_shape[3]:
                # PPsP
                prefac_sPPP = \
                    prefac_noarea / max(survey_area[0], survey_area2[0])
                prefac[:, :, :, :t_sz, :] = prefac_sPPP
                # PPpP
                prefac_pPPP = \
                    prefac_noarea / max(survey_area[0], survey_area2[1])
                prefac[:, :, :, t_sz:, :] = prefac_pPPP
            else:
                raise Exception("too stupid for 6x2pt, diagonal case")

        return prefac

    def covELL_non_gaussian(self,
                            covELLspacesettings,
                            output_dict,
                            bias_dict,
                            hod_dict,
                            prec,
                            tri_tab):
        r"""
        Calculates the non-Gaussian part of the covariance using
        Limber's approximation for all specified observables in
        ell-space config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
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
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'


        Returns
        -------
        nongauss : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """
        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating non-Gaussian covariance in ell space")
        splines_gggg = [[]for _ in range(self.sample_dim)]
        splines_gggm = [[]for _ in range(self.sample_dim)]
        splines_ggmm = [[]for _ in range(self.sample_dim)]
        splines_gmgm = [[]for _ in range(self.sample_dim)]
        splines_mmgm = [[]for _ in range(self.sample_dim)]
        splines_mmmm = [[]for _ in range(self.sample_dim)]
        
        '''splines_gggm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_ggmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]'''

        zet = self.zet_min
        # We want the trispectrum at least at two zet
        if (self.zet_min + covELLspacesettings['tri_delta_z']) >= self.zet_max:
            covELLspacesettings['tri_delta_z'] = self.zet_max - self.zet_min
        chi_list, idx_z =  [], 0
        
        gggg_z = []
        gggm_z = []
        ggmm_z = []
        gmgm_z = []
        mmgm_z = []
        mmmm_z = []

        zet_list = []
        if self.num_cores > 16:
            self.num_cores = 16
        while zet < self.zet_max:
            zet = self.zet_min + \
                covELLspacesettings['tri_delta_z']*idx_z
            if zet > self.zet_max:
                zet = self.zet_max
            if (idx_z == 0):
                chi_list.append(self.los_integration_chi[0])
            else:
                chi_list.append(self.cosmology.comoving_distance(
                    zet).value * self.cosmology.h)
            idx_z += 1
            self.update_mass_func(
                zet, bias_dict, hod_dict, prec)
            aux_tri = self.trispectra(
                output_dict, bias_dict, hod_dict, prec['hm'], tri_tab)
            gggg_z.append(aux_tri[0])
            gggm_z.append(aux_tri[1])
            ggmm_z.append(aux_tri[2])
            gmgm_z.append(aux_tri[3])
            mmgm_z.append(aux_tri[4])
            mmmm_z.append(aux_tri[5])
            zet_list.append(zet)
        zet += covELLspacesettings['tri_delta_z']
        chi_list.append(self.cosmology.comoving_distance(
            zet).value * self.cosmology.h)
        zet_list.append(zet)

        self.update_mass_func(
            zet, bias_dict, hod_dict, prec)
        aux_tri = self.trispectra(
            output_dict, bias_dict, hod_dict, prec['hm'], tri_tab)
        gggg_z.append(aux_tri[0])
        gggm_z.append(aux_tri[1])
        ggmm_z.append(aux_tri[2])
        gmgm_z.append(aux_tri[3])
        mmgm_z.append(aux_tri[4])
        mmmm_z.append(aux_tri[5])
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        print('Producing splines for non-Gaussian computation')
        gggg_z = np.array(gggg_z)
        gggm_z = np.array(gggm_z)
        ggmm_z = np.array(ggmm_z)
        gmgm_z = np.array(gmgm_z)
        mmgm_z = np.array(mmgm_z)
        mmmm_z = np.array(mmmm_z)
        for i_sample in range(self.sample_dim):
            for j_sample in range(self.sample_dim):
                if self.gg:
                    trispec_array = np.log(gggg_z[:,:,:,i_sample,j_sample])
                    splines_gggg[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))
                if self.gg and self.gm and self.cross_terms:
                    trispec_array = np.log(gggm_z[:,:,:,i_sample,j_sample])
                    splines_gggm[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))
                if self.gg and self.mm and self.cross_terms:
                    trispec_array = np.log(ggmm_z[:,:,:,i_sample,j_sample])
                    splines_ggmm[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))
                if self.gm:
                    trispec_array = np.log(gmgm_z[:,:,:,i_sample,j_sample])
                    splines_gmgm[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))        
                if self.gm and self.mm and self.cross_terms:
                    trispec_array = np.log(mmgm_z[:,:,:,i_sample,j_sample])
                    splines_mmgm[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))     
                if self.mm:
                    trispec_array = np.log(mmmm_z[:,:,:,i_sample,j_sample])
                    splines_mmmm[i_sample].append(RegularGridInterpolator(
                                (chi_list,np.log(self.mass_func.k), np.log(self.mass_func.k)), trispec_array,bounds_error= False, fill_value = None))
        flat_idx = 0  
        mesh_k = np.zeros((3,len(self.ellrange)**2*len(self.los_integration_chi)))
        for i_chi in range(len(self.los_integration_chi)):
            for i_ell in range(len(self.ellrange)):
                ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                for j_ell in range(len(self.ellrange)):
                    kj = np.log((self.ellrange[j_ell] + 0.5)/self.los_integration_chi[i_chi])
                    mesh_k[0,flat_idx] = self.los_integration_chi[i_chi]
                    mesh_k[1,flat_idx] = ki
                    mesh_k[2,flat_idx] = kj
                    flat_idx +=1
        if self.gg:
            trispec_integrand_gggg = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), self.sample_dim, self.sample_dim))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    trispec_integrand_gggg[:,:,:,i_sample,j_sample] = np.exp(splines_gggg[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        if self.gg and self.gm and self.cross_terms:
            trispec_integrand_gggm = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), self.sample_dim, self.sample_dim))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    trispec_integrand_gggm[:,:,:,i_sample,j_sample] = np.exp(splines_gggm[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        if self.gg and self.mm and self.cross_terms:
            trispec_integrand_ggmm = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), self.sample_dim, 1))
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    trispec_integrand_ggmm[:,:,:,i_sample,j_sample] = np.exp(splines_ggmm[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        if self.gm:
            trispec_integrand_gmgm = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), self.sample_dim, self.sample_dim))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    trispec_integrand_gmgm[:,:,:,i_sample,j_sample] = np.exp(splines_gmgm[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        if self.gm and self.mm and self.cross_terms:
            trispec_integrand_mmgm = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), 1, self.sample_dim))
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    trispec_integrand_mmgm[:,:,:,i_sample,j_sample] = np.exp(splines_mmgm[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        if self.mm:
            trispec_integrand_mmmm = np.zeros((len(self.los_integration_chi), len(self.ellrange),  len(self.ellrange), 1, 1))
            for i_sample in range(1):
                for j_sample in range(1):
                    trispec_integrand_mmmm[:,:,:,i_sample,j_sample] = np.exp(splines_mmmm[i_sample][j_sample]((mesh_k[0,:],mesh_k[1,:],mesh_k[2,:])).reshape((len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))))
        self.num_cores = self.num_cores_save
        nongaussELLgggg = None
        nongaussELLgggm = None
        nongaussELLggmm = None
        nongaussELLgmgm = None
        nongaussELLmmgm = None
        nongaussELLmmmm = None
        print('Line-of-sight integration for non-Gaussian components')
        if self.gg:
            nongaussELLgggg = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                        self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                l_tomo_start = k_tomo
                                if self.ellrange_photo is not None:
                                    l_tomo_start = 0
                                for l_tomo in range(l_tomo_start, self.n_tomo_clust):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo], self.chi_min_clust[l_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo], self.chi_max_clust[l_tomo])
                                    if chi_low >= chi_high or (self.clustering_z and (i_tomo >= 2 or k_tomo >= 2)):
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*self.spline_zclust[i_tomo](self.los_integration_chi) * self.spline_zclust[j_tomo](self.los_integration_chi)*self.spline_zclust[k_tomo](self.los_integration_chi)*self.spline_zclust[l_tomo](self.los_integration_chi)
                                    nongaussELLgggg[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_gggg[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)        
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gggg at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgggg *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            print("")
        else:
            nongaussELLgggg = 0

        if self.gg and self.gm and self.cross_terms:
            nongaussELLgggm = np.zeros((len(self.ellrange), len(self.ellrange),  self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                        self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*self.spline_zclust[i_tomo](self.los_integration_chi) * self.spline_zclust[j_tomo](self.los_integration_chi)*self.spline_zclust[k_tomo](self.los_integration_chi)*self.spline_lensweight[l_tomo](self.los_integration_chi)
                                    nongaussELLgggm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_gggm[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)                                    
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gggm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
                
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgggm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            print("")
        else:
            nongaussELLgggm = 0

        if self.gg and self.mm and self.cross_terms:
            nongaussELLggmm = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, 1, self.n_tomo_clust,
                                        self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*self.spline_zclust[i_tomo](self.los_integration_chi) * self.spline_zclust[j_tomo](self.los_integration_chi)*self.spline_lensweight[k_tomo](self.los_integration_chi)*self.spline_lensweight[l_tomo](self.los_integration_chi)
                                    nongaussELLggmm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_ggmm[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance ggmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            if covELLspacesettings['pixelised_cell']:
                nongaussELLggmm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            print("")
        else:
            nongaussELLggmm = 0

        if self.gm:
            nongaussELLgmgm = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                        self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_clust*(self.n_tomo_lens)*self.sample_dim**2
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_lens*self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*self.spline_zclust[i_tomo](self.los_integration_chi) * self.spline_lensweight[j_tomo](self.los_integration_chi)*self.spline_zclust[k_tomo](self.los_integration_chi)*self.spline_lensweight[l_tomo](self.los_integration_chi)
                                    nongaussELLgmgm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_gmgm[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgmgm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            if self.est_shear == "C_ell":
                adding = self.gaussELLgmgm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_gm[:, None, :, None, :,:, None, None]*self.Cell_gm[None, :, None, :, None, None, :, :])
                nongaussELLgmgm[:, :, :, :, :, :, :, :] = nongaussELLgmgm[:, :, :, :, :, :, :, :] + adding[:, :, :, :, :, :, :, :]
            print("")
        else:
            nongaussELLgmgm = 0

        if self.gm and self.mm and self.cross_terms:
            nongaussELLmmgm = np.zeros((len(self.ellrange), len(self.ellrange), 1, self.sample_dim, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_lens*(self.n_tomo_lens + 1)/2*self.sample_dim
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    weight = 1/self.los_integration_chi**6.0*self.spline_lensweight[i_tomo](self.los_integration_chi) * self.spline_lensweight[j_tomo](self.los_integration_chi)*self.spline_zclust[k_tomo](self.los_integration_chi)*self.spline_lensweight[l_tomo](self.los_integration_chi)
                                    nongaussELLmmgm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_mmgm[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance mmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            if covELLspacesettings['pixelised_cell']:
                nongaussELLmmgm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            if self.est_shear == "C_ell" and self.est_ggl == "C_ell":
                adding = self.gaussELLmmgm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_mm[:, None, :, None, :,:, None, None]*self.Cell_gm[None, :, None, :, None, None, :, :])
                nongaussELLmmgm[:, :, 0, :, :, :, :, :] = nongaussELLmmgm[:, :, 0, :, :, :, :, :] + adding[:, :, 0, :, :, :, :, :]
            print("")
        else:
            nongaussELLmmgm = 0

        if self.mm:
            nongaussELLmmmm = np.zeros((len(self.ellrange), len(self.ellrange), 1, 1, self.n_tomo_lens,
                                        self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_lens*(self.n_tomo_lens + 1)/2*1
            for i_sample in range(1):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    weight = 1./self.los_integration_chi**6.0*self.spline_lensweight[i_tomo](self.los_integration_chi) * self.spline_lensweight[j_tomo](self.los_integration_chi)*self.spline_lensweight[k_tomo](self.los_integration_chi)*self.spline_lensweight[l_tomo](self.los_integration_chi)
                                    nongaussELLmmmm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(trispec_integrand_mmmm[:, :, :, i_sample, j_sample]*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance mmmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            if covELLspacesettings['pixelised_cell']:
                nongaussELLmmmm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            print("")
            if self.est_shear == "C_ell":
                adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_mm[:, None, :, None, :,:, None, None]*self.Cell_mm[None, :, None, :, None, None, :, :])
                nongaussELLmmmm[:, :, 0, 0, :, :, :, :] = nongaussELLmmmm[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
        else:
            nongaussELLmmmm = 0
        if not covELLspacesettings['nglimber']:
            nongaussELLgggg = self.covELL_non_gaussian_non_Limber(
                covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab, nongaussELLgggg)
        return nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm

    def covELL_ssc(self,
                   bias_dict,
                   hod_dict,
                   prec,
                   survey_params_dict,
                   covELLspacesettings):
        r"""
        Calculates the super-sample part of the covariance using
        Limber's approximation for all specified observables in
        ell-space config file.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
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
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.


        Returns
        -------
        ssc : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating the SSC terms in ell space.")
        SSCELLgggg = None
        SSCELLgggm = None
        SSCELLggmm = None
        SSCELLgmgm = None
        SSCELLmmgm = None
        SSCELLmmmm = None
        spline_responsePgg = []
        spline_responsePgm = []
        spline_responsePmm = []

        survey_variance_gggg = np.ones_like(self.los_integration_chi)
        survey_variance_mmmm = np.ones_like(self.los_integration_chi)
        survey_variance_gmgm = np.ones_like(self.los_integration_chi)
        survey_variance_gggm = np.ones_like(self.los_integration_chi)
        survey_variance_mmgm = np.ones_like(self.los_integration_chi)
        survey_variance_ggmm = np.ones_like(self.los_integration_chi)

        

        if self.gg:
            print("Getting survey modes for gggg")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'gg', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_gggg = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust']**2/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    survey_params_dict['survey_area_clust']/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_gggg = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_gggg_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_gggg, k=1, s=0, ext=0)
        if self.mm:
            print("Getting survey modes for mmmm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('mm', 'mm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_mmmm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_lens']**2/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    survey_params_dict['survey_area_lens']/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_mmmm = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_mmmm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_mmmm, k=1, s=0, ext=0)

        if self.gm:
            print("Getting survey modes for gmgm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gm', 'gm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_gmgm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_ggl']**2/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    survey_params_dict['survey_area_ggl']/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_gmgm = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_gmgm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_gmgm, k=1, s=0, ext=0)

        if self.gg and self.gm and self.cross_terms:
            print("Getting survey modes for gggm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'gm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_gggm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust']*survey_params_dict['survey_area_ggl']/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    min(survey_params_dict['survey_area_clust'], survey_params_dict['survey_area_ggl'])/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_gggm = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_gggm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_gggm, k=1, s=0, ext=0)
            
        if self.gg and self.mm and self.cross_terms:
            print("Getting survey modes for ggmm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'mm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_ggmm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust']*survey_params_dict['survey_area_lens']/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    min(survey_params_dict['survey_area_clust'], survey_params_dict['survey_area_lens'])/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_ggmm = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_ggmm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_ggmm, k=1, s=0, ext=0)
        if self.gm and self.mm and self.cross_terms:
            print("Getting survey modes for mmgm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('mm', 'gm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance_mmgm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_lens']*survey_params_dict['survey_area_ggl']/self.deg2torad2**2)
            else:
                angular_scale_of_circular_survey_in_rad = np.sqrt(
                    min(survey_params_dict['survey_area_ggl'], survey_params_dict['survey_area_lens'])/self.deg2torad2/np.pi)
                y_aux = np.zeros_like(self.los_chi)
                for i_chi in range(len(self.los_chi)):
                    if i_chi == 0:
                        weight_function_squared = 1
                    else:
                        weight_function_squared = (2.0*j1(self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]) /
                                                (self.mass_func.k*angular_scale_of_circular_survey_in_rad*self.los_chi[i_chi]))**2
                    y_aux[i_chi] = simpson(weight_function_squared*self.power_mm_lin_z[i_chi,:]*self.mass_func.k, x = self.mass_func.k)/(2.0*np.pi)
                survey_variance_mmgm = np.interp(self.los_integration_chi, self.los_chi, y_aux)*self.los_integration_chi**2
            self.survey_variance_mmgm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_mmgm, k=1, s=0, ext=0)

        if not self.redshift_dep_bias:  
            self.aux_response_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            self.aux_response_gm = np.zeros_like(self.aux_response_gg)
            self.aux_response_mm = np.zeros_like(self.aux_response_gg)
            t0 = time.time()
            for i_chi in range(self.los_interpolation_sampling):
                self.update_mass_func(self.los_z[i_chi], bias_dict, hod_dict, prec)
                self.aux_response_gg[i_chi, :, :], self.aux_response_gm[i_chi, :, :], self.aux_response_mm[i_chi,
                                                                                            :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
                eta = (time.time()-t0) * \
                    (len(self.los_z)/(i_chi+1)-1)
                print('\rPreparations for SSC calculation at '
                    + str(round((i_chi+1)/len(self.los_z)*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
            
            for i_sample in range(self.sample_dim):
                spline_responsePgg.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gg[:, :, i_sample]),bounds_error= False, fill_value = None))
                spline_responsePgm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gm[:, :, i_sample]),bounds_error= False, fill_value = None))
                spline_responsePmm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_mm[:, :, i_sample]),bounds_error= False, fill_value = None))
        else:
            aux_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            aux_gm = np.zeros_like(aux_gg)
            self.aux_response_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust,
                                        self.n_tomo_clust))
            self.aux_response_gm = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust))
            self.aux_response_mm = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            t0 = time.time()
            for i_chi in range(self.los_interpolation_sampling):
                self.update_mass_func(self.los_z[i_chi], bias_dict, hod_dict, prec)
                aux_gg[i_chi, :, :], aux_gm[i_chi, :, :], self.aux_response_mm[i_chi, :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
                if self.gm:
                    for i_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            bias_i_tomo = self.bias_of_zet[i_tomo](self.los_z[i_chi])
                            self.aux_response_gm[i_chi, :, i_sample, i_tomo] = self.aux_response_mm[i_chi, :, i_sample]*bias_i_tomo
                            self.aux_response_gm[i_chi, :, i_sample, i_tomo] -= self.Pgm[:, i_sample]*bias_i_tomo**2
                            self.aux_response_gm[i_chi, :, i_sample, i_tomo] /= bias_i_tomo
                if self.gg:
                    for i_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            bias_i_tomo = self.bias_of_zet[i_tomo](self.los_z[i_chi])
                            for j_tomo in range(self.n_tomo_clust):
                                bias_j_tomo = self.bias_of_zet[j_tomo](self.los_z[i_chi])
                                self.aux_response_gg[i_chi, :, i_sample, i_tomo, j_tomo] = self.aux_response_mm[i_chi, :, i_sample]*bias_i_tomo*bias_j_tomo
                                self.aux_response_gg[i_chi, :, i_sample, i_tomo, j_tomo] -= (bias_i_tomo + bias_j_tomo)*np.diagonal(self.Pgg, axis1 = -2, axis2 =-1)[:,i_sample]*bias_i_tomo*bias_j_tomo
                                self.aux_response_gg[i_chi, :, i_sample, i_tomo, j_tomo] /= (bias_i_tomo*bias_j_tomo)
                eta = (time.time()-t0) * \
                    (len(self.los_z)/(i_chi+1)-1)
                print('\rPreparations for SSC calculation at '
                    + str(round((i_chi+1)/len(self.los_z)*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
            
            for i_sample in range(self.sample_dim):
                spline_responsePmm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_mm[:, :, i_sample]),bounds_error= False, fill_value = None))
                for i_tomo in range(self.n_tomo_clust):
                    spline_responsePgm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gm[:, :, i_sample, i_tomo]),bounds_error= False, fill_value = None))
                    for j_tomo in range(self.n_tomo_clust):        
                        spline_responsePgg.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                        (self.aux_response_gg[:, :, i_sample, i_tomo, j_tomo]),bounds_error= False, fill_value = None))
            self.spline_responsePgg = spline_responsePgg    
            self.spline_responsePgm = spline_responsePgm   
                    

        print("")
        print("Calculating SSC contribution in ell space")
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        if self.gg:
            print("At gggg contribution")
            SSCELLgggg = np.zeros((len(self.ellrange), len(self.ellrange),  self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                   self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                l_tomo_start = k_tomo
                                if self.ellrange_photo is not None:
                                    l_tomo_start = 0
                                for l_tomo in range(l_tomo_start, self.n_tomo_clust):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo], self.chi_min_clust[l_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo], self.chi_max_clust[l_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    if chi_low < self.los_integration_chi[0]:
                                        chi_low = self.los_integration_chi[0]
                                    if chi_high > self.los_integration_chi[-1]:
                                        chi_high = self.los_integration_chi[-1]
                                    
                                    
                                    self.__update_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    survey_variance = self.survey_variance_gggg_spline(self.los_integration_chi)
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_zclust[i_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[j_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[k_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[l_tomo](
                                            self.los_integration_chi)
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    if not self.redshift_dep_bias:
                                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgg[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    else:
                                        P1_response = spline_responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgg[j_sample*self.n_tomo_clust**2 + k_tomo*self.n_tomo_clust + l_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = P1_response[:, :, None]*P2_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLgggg[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLgggg *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
        else:
            SSCELLgggg = 0

        if self.gg and self.gm and self.cross_terms:
            print("At gggm contribution")
            SSCELLgggm = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                   self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    if chi_low < self.los_integration_chi[0]:
                                        chi_low = self.los_integration_chi[0]
                                    if chi_high > self.los_integration_chi[-1]:
                                        chi_high = self.los_integration_chi[-1]
                                    
                                    self.__update_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    survey_variance = self.survey_variance_gggm_spline(self.los_integration_chi)
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_zclust[i_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[j_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[k_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[l_tomo](
                                            self.los_integration_chi)
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    if not self.redshift_dep_bias:
                                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    else:
                                        P1_response = spline_responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample*self.n_tomo_clust + k_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = P1_response[:, :, None]*P2_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLgggm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLgggm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
        else:
            SSCELLgggm = 0

        if self.gg and self.mm and self.cross_terms:
            print("At ggmm contribution")
            SSCELLggmm = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, 1, self.n_tomo_clust,
                                   self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    if chi_low < self.los_integration_chi[0]:
                                        chi_low = self.los_integration_chi[0]
                                    if chi_high > self.los_integration_chi[-1]:
                                        chi_high = self.los_integration_chi[-1]
                                    
                                    self.__update_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    survey_variance = self.survey_variance_ggmm_spline(self.los_integration_chi)
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_zclust[i_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[j_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[k_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[l_tomo](
                                            self.los_integration_chi)
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    if not self.redshift_dep_bias:
                                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePmm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    else:
                                        P1_response = spline_responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePmm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = P1_response[:, :, None]*P2_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLggmm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLggmm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
        else:
            SSCELLggmm = 0

        if self.gm:
            print("At gmgm contribution")
            SSCELLgmgm = np.zeros((len(self.ellrange), len(self.ellrange), self.sample_dim, self.sample_dim, self.n_tomo_clust,
                                   self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    if chi_low < self.los_integration_chi[0]:
                                        chi_low = self.los_integration_chi[0]
                                    if chi_high > self.los_integration_chi[-1]:
                                        chi_high = self.los_integration_chi[-1]
                                    
                                    self.__update_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    survey_variance = self.survey_variance_gmgm_spline(self.los_integration_chi)
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_zclust[i_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[k_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[j_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[l_tomo](
                                            self.los_integration_chi)
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    if not self.redshift_dep_bias:
                                        P1_response = spline_responsePgm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    else:
                                        P1_response = spline_responsePgm[i_sample*self.n_tomo_clust + i_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample*self.n_tomo_clust + k_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = P1_response[:, :, None]*P2_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLgmgm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None],  x = self.los_integration_chi, axis = 0)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLgmgm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
        else:
            SSCELLgmgm = 0

        if self.mm:
            print("At mmmm contribution")
            SSCELLmmmm = np.zeros((len(self.ellrange), len(self.ellrange), 1, 1, self.n_tomo_lens,
                                  self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            
            survey_variance = survey_variance_mmmm
                                              

            t0, flat_tomo = time.time(), 0
            tomo_comb = (self.n_tomo_lens*(self.n_tomo_lens+1)/2)**2
            for i_sample in range(1):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_lensweight[i_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[k_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[j_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[l_tomo](
                                            self.los_integration_chi)                                   
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    Pmm_response = spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = Pmm_response[:, :, None]*Pmm_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLmmmm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    flat_tomo += 1
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLmmmm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
        else:
            SSCELLmmmm = 0

        if self.gm and self.mm and self.cross_terms:
            print("At mmgm contribution")
            SSCELLmmgm = np.zeros((len(self.ellrange), len(self.ellrange), 1, self.sample_dim, self.n_tomo_lens,
                                   self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = self.chi_min_clust[k_tomo]
                                    chi_high = self.chi_max_clust[k_tomo]
                                    if chi_low < self.los_integration_chi[0]:
                                        chi_low = self.los_integration_chi[0]
                                    if chi_high > self.los_integration_chi[-1]:
                                        chi_high = self.los_integration_chi[-1]
                                    
                                    if chi_low >= chi_high:
                                        continue
                                    self.__update_los_integration_chi(
                                        chi_low, chi_high, covELLspacesettings)
                                    survey_variance = self.survey_variance_mmgm_spline(self.los_integration_chi)
                                    
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        self.spline_lensweight[i_tomo](self.los_integration_chi) * \
                                        self.spline_zclust[k_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[j_tomo](self.los_integration_chi) * \
                                        self.spline_lensweight[l_tomo](
                                            self.los_integration_chi)
                                    x_values = np.zeros((2,len(self.ellrange)*len(self.los_integration_chi)))
                                    flat_idx = 0
                                    for i_chi in range(len(self.los_integration_chi)):
                                        for i_ell in range(len(self.ellrange)):
                                            ki = np.log((self.ellrange[i_ell] + 0.5)/self.los_integration_chi[i_chi])
                                            x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                                            x_values[1,flat_idx] = ki
                                            flat_idx +=1
                                    if not self.redshift_dep_bias:
                                        P1_response = spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    else:
                                        P1_response = spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                        P2_response = spline_responsePgm[j_sample*self.n_tomo_clust + k_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                                    integrand = P1_response[:, :, None]*P2_response[:, None, :]*survey_variance[:, None, None]
                                    SSCELLmmgm[:,  :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = simpson(integrand*weight[:, None, None], x = self.los_integration_chi, axis = 0)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                                    if covELLspacesettings['pixelised_cell']:
                                        SSCELLmmgm *= self.pixelweight_matrix[:,:, None, None, None, None, None, None]
            print("")
        else:
            SSCELLmmgm = 0
        return SSCELLgggg, SSCELLgggm, SSCELLggmm, SSCELLgmgm, SSCELLmmgm, SSCELLmmmm

    def covELL_csmf_SN(self):
        r"""
        Calculates the shot noise component of the stellar mass function covariance matrix
        
        Returns
        -------
        smf_sn : array
            with shape (csmf_mass_bins, csmf_mass_bins, csmf_tomo_bins, csmf_tomo_bins)
        """
        amplitude = self.csmf_at_tomo_and_mass/self.deltaM_csmf[:, None]/self.Vmax
        return np.eye(len(self.log10csmf_mass_bins))[:,:, None, None]*np.eye(self.n_tomo_csmf)[None, None, :, :]*amplitude[:, None, :, None]

    def covELL_csmf_SSC(self,
                        survey_params_dict):
        r"""
        Calculates the SSC component of the stellar mass function covariance matrix
        
        Returns
        -------
        smf_ssc : array
            with shape (csmf_mass_bins, csmf_mass_bins, csmf_tomo_bins, csmf_tomo_bins)
        """
        ell, sum_m_a_lm = \
            self.calc_a_lm('mm', 'mm', survey_params_dict)
        if ell is not None:
            ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
            x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
            flat_idx = 0
            for i_chi in range(len(self.los_integration_chi)):
                for i_ell in range(len(ell)):
                    x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                    x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                    flat_idx +=1
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
        survey_variance_mmmm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust']**2/self.deg2torad2**2)
        self.survey_variance_mmmm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_mmmm, k=1, s=0, ext=0)
        result = np.zeros((len(self.log10csmf_mass_bins), len(self.log10csmf_mass_bins), self.n_tomo_csmf, self.n_tomo_csmf))
        for i_tomo in range(self.n_tomo_csmf):
            for j_tomo in range(i_tomo, self.n_tomo_csmf):
                for i_mass in range(len(self.log10csmf_mass_bins)):
                    for j_mass in range(i_mass, len(self.log10csmf_mass_bins)):
                        los_integration_chi_update = self.los_integration_chi[np.where(self.spline_zcsmf_total(self.los_integration_chi) > 0)[0]]
                        integrand = self.spline_zcsmf[i_tomo](los_integration_chi_update)*self.spline_zcsmf[j_tomo](los_integration_chi_update) \
                            *los_integration_chi_update**2*self.survey_variance_mmmm_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)**2 *self.phi_tilde_spline[j_mass](los_integration_chi_update)
                        result[i_mass, j_mass, i_tomo, j_tomo] = survey_params_dict['survey_area_clust']**2/self.deg2torad2**2*self.f_tomo[i_tomo]*self.f_tomo[j_tomo]/self.Vmax[i_mass, i_tomo]/self.Vmax[j_mass, j_tomo]*simpson(integrand, x = los_integration_chi_update)
                        result[i_mass, j_mass, j_tomo, i_tomo] = result[i_mass, j_mass, i_tomo, j_tomo]
                        result[j_mass, i_mass, j_tomo, i_tomo] = result[i_mass, j_mass, i_tomo, j_tomo]
                        result[j_mass, i_mass, i_tomo, j_tomo] = result[i_mass, j_mass, i_tomo, j_tomo]
        return result

    
    def covELL_csmf_cross_LSS_sva(self,
                                  covELLspacesettings):
        r"""
        Calculates the sample variance term for the stellar mass function cross LSS
        covariance

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.

        Returns
        -------
        covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm : list of arrays
            with shapes (number of ell bins, number of smf bins, sample dims, n_tomo_smf, n_tomo_clust/lens, n_tomo_clust/lens) 
        """
        if self.gg:
            covELL_smf_cross_gg = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
            
            for i_mass in range(len(self.log10csmf_mass_bins)):
                spline = RegularGridInterpolator((np.log(self.mass_func.k), self.los_chi), np.log(self.csmf_count_matter_bispectrum[:, :, i_mass]),bounds_error= False, fill_value = None)
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(i_tomo, self.n_tomo_clust):
                            chi_low = max(
                                    self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                            chi_high = min(
                                self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                            if chi_low >= chi_high:
                                continue
                            if chi_low < self.los_integration_chi[0]:
                                chi_low = self.los_integration_chi[0]
                            if chi_high > self.los_integration_chi[-1]:
                                chi_high = self.los_integration_chi[-1]
                            los_integration_chi_update = self.__get_updated_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            los_integration_chi_update = los_integration_chi_update[np.where(self.spline_zcsmf_total(los_integration_chi_update) > 0)[0]]
                            weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_zclust[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)/los_integration_chi_update**2
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                            covELL_smf_cross_gg[: ,i_mass, :, i_smf_tomo, i_tomo, j_tomo] = self.f_tomo[i_smf_tomo]/self.Vmax[i_mass, i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)[:, None]
        else:
            covELL_smf_cross_gg = 0
        
        if self.gm:
            covELL_smf_cross_gm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
            
            for i_mass in range(len(self.log10csmf_mass_bins)):
                spline = RegularGridInterpolator((np.log(self.mass_func.k), self.los_chi), np.log(self.csmf_count_matter_bispectrum[:, :, i_mass]),bounds_error= False, fill_value = None)
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_lens):
                            chi_low = self.chi_min_clust[i_tomo]
                            chi_high = self.chi_max_clust[i_tomo]
                            if chi_low < self.los_integration_chi[0]:
                                chi_low = self.los_integration_chi[0]
                            if chi_high > self.los_integration_chi[-1]:
                                chi_high = self.los_integration_chi[-1]
                            los_integration_chi_update = self.__get_updated_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            los_integration_chi_update = los_integration_chi_update[np.where(self.spline_zcsmf_total(los_integration_chi_update) > 0)[0]]
                            weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)/los_integration_chi_update**2
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                            covELL_smf_cross_gm[: ,i_mass, :, i_smf_tomo, i_tomo, j_tomo] = self.f_tomo[i_smf_tomo]/self.Vmax[i_mass, i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)[:, None]
        else:
            covELL_smf_cross_gm = 0
        if self.mm:
            covELL_smf_cross_mm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
            for i_mass in range(len(self.log10csmf_mass_bins)):
                spline = RegularGridInterpolator((np.log(self.mass_func.k), self.los_chi), np.log(self.csmf_count_matter_bispectrum[:, :, i_mass]),bounds_error= False, fill_value = None)
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_lens):
                            los_integration_chi_update = self.los_integration_chi[np.where(self.spline_zcsmf_total(self.los_integration_chi) > 0)[0]]
                            weight = self.spline_lensweight[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)/los_integration_chi_update**2
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                            covELL_smf_cross_mm[: ,i_mass, 0, i_smf_tomo, i_tomo, j_tomo] = self.f_tomo[i_smf_tomo]/self.Vmax[i_mass, i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
        else:
            covELL_smf_cross_mm = 0
        return covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm
    

    def covELL_csmf_cross_LSS_ssc(self,
                                  covELLspacesettings,
                                  survey_params_dict):
        r"""
        Calculates the super sample variance term for the stellar mass function cross LSS
        covariance

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.

        Returns
        -------
        covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm : list of arrays
            with shapes (number of ell bins, number of smf bins, sample dims, n_tomo_smf, n_tomo_clust/lens, n_tomo_clust/lens) 
        """
            
        if self.gg:
            covELL_smf_cross_gg = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
            
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'gg', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust']**2/self.deg2torad2**2)
            survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)
        
            for i_mass in range(len(self.log10csmf_mass_bins)):
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(i_tomo, self.n_tomo_clust):
                            chi_low = max(
                                    self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                            chi_high = min(
                                self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                            if chi_low >= chi_high:
                                continue
                            if chi_low < self.los_integration_chi[0]:
                                chi_low = self.los_integration_chi[0]
                            if chi_high > self.los_integration_chi[-1]:
                                chi_high = self.los_integration_chi[-1]
                            los_integration_chi_update = self.__get_updated_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            los_integration_chi_update = los_integration_chi_update[np.where(self.spline_zcsmf_total(los_integration_chi_update) > 0)[0]]
                            weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_zclust[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)
                            weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)/los_integration_chi_update**2
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            for i_sample in range(self.sample_dim):
                                spline_eval = self.spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                covELL_smf_cross_gg[: ,i_mass, i_sample, i_smf_tomo, i_tomo, j_tomo] = survey_params_dict['survey_area_clust']/self.deg2torad2/self.Vmax[i_mass, i_smf_tomo]*self.f_tomo[i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
        else:
            covELL_smf_cross_gg = 0

        if self.gm:
            covELL_smf_cross_gm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
            
            ell, sum_m_a_lm = \
                self.calc_a_lm('gm', 'gm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_ggl']**2/self.deg2torad2**2)
            survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)

            for i_mass in range(len(self.log10csmf_mass_bins)):
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_lens):
                            chi_low = self.chi_min_clust[i_tomo]
                            chi_high =self.chi_max_clust[i_tomo]
                            if chi_low >= chi_high:
                                continue
                            if chi_low < self.los_integration_chi[0]:
                                chi_low = self.los_integration_chi[0]
                            if chi_high > self.los_integration_chi[-1]:
                                chi_high = self.los_integration_chi[-1]
                            los_integration_chi_update = self.__get_updated_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            los_integration_chi_update = los_integration_chi_update[np.where(self.spline_zcsmf_total(los_integration_chi_update) > 0)[0]]
                            weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)/los_integration_chi_update**2
                            weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            for i_sample in range(self.sample_dim):
                                spline_eval = self.spline_responsePgm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                covELL_smf_cross_gm[: ,i_mass, i_sample, i_smf_tomo, i_tomo, j_tomo] = survey_params_dict['survey_area_clust']/self.deg2torad2/self.Vmax[i_mass, i_smf_tomo]*self.f_tomo[i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
        else:
            covELL_smf_cross_gm = 0

        if self.mm:
            covELL_smf_cross_mm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
            
            ell, sum_m_a_lm = \
                self.calc_a_lm('mm', 'mm', survey_params_dict)
            if ell is not None:
                ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                x_values = np.zeros((2,len(ell)*len(self.los_integration_chi)))
                flat_idx = 0
                for i_chi in range(len(self.los_integration_chi)):
                    for i_ell in range(len(ell)):
                        x_values[0,flat_idx] = self.los_integration_chi[i_chi]
                        x_values[1,flat_idx] = (ell[i_ell] + .5)/self.los_integration_chi[i_chi]
                        flat_idx +=1
                power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_lens']**2/self.deg2torad2**2)
            survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)

            for i_mass in range(len(self.log10csmf_mass_bins)):
                for i_smf_tomo in range(self.n_tomo_csmf):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            los_integration_chi_update = self.__get_updated_los_integration_chi(
                                self.chimin, self.chimax, covELLspacesettings)
                            los_integration_chi_update = los_integration_chi_update[np.where(self.spline_zcsmf_total(los_integration_chi_update) > 0)[0]]
                            weight = self.spline_lensweight[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self.spline_zcsmf[i_smf_tomo](los_integration_chi_update)/self.spline_zcsmf_total(los_integration_chi_update)/los_integration_chi_update**2
                            weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)
                            x_values = np.zeros((2,len(self.ellrange)*len(los_integration_chi_update)))
                            flat_idx = 0
                            for i_chi in range(len(los_integration_chi_update)):
                                for i_ell in range(len(self.ellrange)):
                                    ki = np.log((self.ellrange[i_ell] + 0.5)/los_integration_chi_update[i_chi])
                                    x_values[1,flat_idx] = los_integration_chi_update[i_chi]
                                    x_values[0,flat_idx] = ki
                                    flat_idx +=1
                            spline_eval = np.exp(self.spline_responsePmm((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                            covELL_smf_cross_mm[: ,i_mass, :, i_smf_tomo, i_tomo, j_tomo] = survey_params_dict['survey_area_clust']/self.deg2torad2/self.Vmax[i_mass, i_smf_tomo]*self.f_tomo[i_smf_tomo]*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)[:, None]
        else:
            covELL_smf_cross_mm = 0        
        return covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm
        
        

    def covELL_non_gaussian_non_Limber(self,
                                       covELLspacesettings,
                                       output_dict,
                                       bias_dict,
                                       hod_dict,
                                       prec,
                                       tri_tab,
                                       nongaussELLgggg):
        r"""
        Calculates the non-Gaussian part of the covariance using the
        full expression. This will slow down the code significantly. (still work in progress)

        Parameters
        ----------
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
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
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'


        Returns
        -------
        nongauss : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['nongauss'] or covELLspacesettings['nglimber']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating non-Gaussian covariance in ell space using the full projection")
        splines_gggg = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gggm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_ggmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]

        zet = self.zet_min
        # We want the trispectrum at least at two zet
        if (self.zet_min + covELLspacesettings['tri_delta_z']) >= self.zet_max:
            covELLspacesettings['tri_delta_z'] = self.zet_max - self.zet_min
        trispec_at_z, chi_list, idx_z = [], [], 0
        while zet < self.zet_max:
            zet = self.zet_min + \
                covELLspacesettings['tri_delta_z']*idx_z
            if zet > self.zet_max:
                zet = self.zet_max
            if (idx_z == 0):
                chi_list.append(self.los_integration_chi[0])
            else:
                chi_list.append(self.cosmology.comoving_distance(
                    zet).value * self.cosmology.h)
            idx_z += 1
            self.update_mass_func(
                zet, bias_dict, hod_dict, prec)
            trispec_at_z.append(self.trispectra(
                output_dict, bias_dict, hod_dict, prec['hm'], tri_tab))

        gggg_z = np.zeros(len(chi_list))
        gggm_z = np.zeros(len(chi_list))
        ggmm_z = np.zeros(len(chi_list))
        gmgm_z = np.zeros(len(chi_list))
        mmgm_z = np.zeros(len(chi_list))
        mmmm_z = np.zeros(len(chi_list))
        k_spline = 2
        if len(chi_list) < 4:
            k_spline = 1
        print('Producing splines for non-Gaussian computation')
        for i_k in range(len(self.mass_func.k)):
            for j_k in range(len(self.mass_func.k)):
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        for i_chi in range(len(chi_list)):
                            if self.gg:
                                gggg_z[i_chi] = trispec_at_z[i_chi][0][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gg and self.gm and self.cross_terms:
                                gggm_z[i_chi] = trispec_at_z[i_chi][1][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gg and self.mm and self.cross_terms:
                                ggmm_z[i_chi] = trispec_at_z[i_chi][2][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gm:
                                gmgm_z[i_chi] = trispec_at_z[i_chi][3][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gm and self.mm and self.cross_terms:
                                mmgm_z[i_chi] = trispec_at_z[i_chi][4][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.mm:
                                mmmm_z[i_chi] = trispec_at_z[i_chi][5][i_k,
                                                                       j_k, i_sample, j_sample]
                        splines_gggg[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gggg_z, k=k_spline, s=0, ext=1))
                        splines_gggm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gggm_z, k=k_spline, s=0, ext=1))
                        splines_ggmm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, ggmm_z, k=k_spline, s=0, ext=1))
                        splines_gmgm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gmgm_z, k=k_spline, s=0, ext=1))
                        splines_mmgm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, mmgm_z, k=k_spline, s=0, ext=1))
                        splines_mmmm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, mmmm_z, k=k_spline, s=0, ext=1))

        ell_max_ng_nonlimber = 20
        n_ell_ng_nonlimber = 2
        ell_non_tri_non_limber = np.unique(np.geomspace(2,ell_max_ng_nonlimber, n_ell_ng_nonlimber).astype(int)).astype(float)
        nonlimbernongauss = np.zeros((len(ell_non_tri_non_limber),len(ell_non_tri_non_limber),self.sample_dim,self.sample_dim,self.n_tomo_clust,self.n_tomo_clust,self.n_tomo_clust,self.n_tomo_clust))
        
        integral_over_k1 = np.zeros((len(self.ellrange),len(self.mass_func.k),len(self.los_integration_chi), len(self.los_integration_chi)))

        lev = levin.Levin(3, 8, 32, 1e-6, self.integration_intervals)
        t0 = time.time()
        print('Carrying out first integral over two Bessel functions')
        
        for i_ell in range(len(self.ellrange)):
            for i_chi_1 in range(len(self.los_integration_chi)):
                for i_chi_x in range(len(self.los_integration_chi)):
                    for i_k1 in range(len(self.mass_func.k)):
                        integrand = np.zeros_like(self.mass_func.k)
                        for i_k3 in range(len(self.mass_func.k)):
                            integrand[i_k3] = np.sqrt(splines_gggg[i_k1][i_k3][0](self.los_integration_chi[i_chi_1]))*self.mass_func.k[i_k3]**2
                        lev.init_integral(
                            self.mass_func.k, integrand[:, None], True, True)
                        integral_over_k1[i_ell, i_chi_1, i_chi_x, i_k1] = float(lev.double_bessel(self.los_integration_chi[i_chi_1], self.los_integration_chi[i_chi_x], int(ell_non_tri_non_limber[i_ell]), int(ell_non_tri_non_limber[i_ell]), self.mass_func.k[0], self.mass_func.k[-1])[0])

        print('Carrying out second integral over two Bessel functions')
        for i_ell in range(len(self.ellrange)):
            for j_ell in range(i_ell, len(self.ellrange)):
                for i_chi_1 in range(len(self.los_integration_chi)):
                    for i_chi_3 in range(len(self.los_integration_chi)):
                        for i_chi_x in range(len(self.los_integration_chi)):
                            integrand = integral_over_k1[i_ell, i_chi_1, i_chi_x,:]
                            lev.init_integral(
                            self.mass_func.k, integrand[:, None], True, True)
                        

        
        return nongaussELLgggg
 

        

