import warnings
import itertools
import numpy as np
from scipy.integrate import quad, IntegrationWarning, simpson
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator

import multiprocessing as mp


try:
    from onecov.cov_halo_model import HaloModel
    from onecov.cov_output import Output
except:
    from cov_halo_model import HaloModel
    from cov_output import Output





class PolySpectra(HaloModel):
    """
    This class calculates the power spectra, power spectra responses,
    and trispectra that are required for the covariance in k-space.
    Spectra for matter and tracer are evaluated at a single redshift. 
    Inherits the functionality of the HaloModel class.

    Attributes
    ----------
    zet : float
        Redshift at which the covariance should be evaluated.
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
    hod_dict : dictionary
        Specifies all the information about the halo occupation 
        distribution used. This defines the shot noise level of the 
        covariance and includes the mass bin definition of the different 
        galaxy populations. To be passed from the read_input method of 
        the Input class.
    survey_params_dict : dictionary
        Specifies all the information unique to a specific survey. To be 
        passed from the read_input method of the Input class.
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

    Private Variables :
    -------------------
    see HaloModel class
    mm : bool
        flag that only performs necessary calculation for the matter-
        matter observable. To be passed from the read_input method of 
        the Input class (part of [observables]).
    gm : bool
        same as above but for the matter-tracer observable
    gg : bool
        same as above but for the tracer-tracer observable
    cross_terms : bool
        Same as above but for the cross-terms between different 
        observables.
    cov_dict : dictionary
        See above in Parameters : cov_dict
    Pmm : array
        with unit ?
        with shape (powspec_prec['log10k_bins'], sample bins)
        matter-matter power spectrum either from a look-up table,
        calculated through the halo model, or None if not needed
    Pgm : array
        with unit ?
        same as above but for the matter-tracer power spectrum
    Pgg : array
        with unit ?
        same as above but for the tracer-tracer power spectrum
    Plin_spline : UnivariateSpline
        spline object for log linear-power-spectrum as a function of
        log(k), given by the halo model
    krange_tri : array
        with unit ?
        with shape (trispec_prec['log10k_bins'])
        linear k-range on which the trispectra will be evaluated
    logks_tri : array
        with unit ?
        with shape (trispec_prec['log10k_bins'])
        logarithmic k-range on which the trispectra will be evaluated
    tri_idxlist : list
        a list containing the indices of a matrix with shape
        (trispec_prec['log10k_bins'], trispec_prec['log10k_bins'],
        sample_dim, sample_dim), where we only loop over independent
        matrix elements (the matrix is symmetric in k-space and 
        sample-space)
    trispec_matter_klim : float
        with unit ?
        precision parameter for the minimum wavenumber for which the
        trispectrum is evaluated precisely
    trispec_matter_mulim : float
        precision parameter mu = cos(phi) for the minimum angle mu 
        between two k-vectors for which the trispectrum is evaluated 
        precisely
    num_cores : int
        specifies number of cores for trispectra .calculation, if no
        input is given or '[misc]: num_cores = 0' all available cores 
        are used


    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_polyspectra import PolySpectra
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    zet = 0
    polys = PolySpectra(zet, covterms, observables, cosmo, bias, hod, 
        survey_params, prec, read_in_tables)

    """

    def __init__(self,
                 zet,
                 cov_dict,
                 obs_dict,
                 cosmo_dict,
                 bias_dict,
                 hod_dict,
                 survey_params_dict,
                 prec,
                 read_in_tables):
        HaloModel.__init__(self,
                           zet,
                           cosmo_dict,
                           bias_dict,
                           hod_dict,
                           survey_params_dict,
                           prec,
                           read_in_tables)
        self.mm = obs_dict['observables']['cosmic_shear']
        self.gm = obs_dict['observables']['ggl']
        self.gg = obs_dict['observables']['clustering']
        self.unbiased_clustering = obs_dict['observables']['unbiased_clustering']
        if read_in_tables['zet_dep_bias'] is not None and not self.unbiased_clustering and (self.gg or self.gm):
            self.unbiased_clustering = True
            self.redshift_dep_bias = True
        else:
            self.redshift_dep_bias = False
       # if self.gg == False and self.gm == False:
       #     self.unbiased_clustering = False
        self.int_2h = None
        if self.unbiased_clustering:
            self.mm = True
        self.cross_terms = obs_dict['observables']['cross_terms']
        self.int_list = None
        self.cov_dict = cov_dict
        self.Pmm = self.P_mm(bias_dict, hod_dict, prec)
        self.Pgm = self.P_gm(bias_dict, hod_dict, prec['hm'])
        self.Pgg = self.P_gg(bias_dict, hod_dict, prec['hm'])
        self.Plin_spline = UnivariateSpline(np.log(self.mass_func.k),
                                            np.log(self.mass_func.power),
                                            k=2, s=0, ext=0)
        self.krange_tri = np.logspace(prec['trispec']['log10k_min'],
                                      prec['trispec']['log10k_max'],
                                      prec['trispec']['log10k_bins'])
        self.logks_tri = np.linspace(prec['trispec']['log10k_min'],
                                     prec['trispec']['log10k_max'],
                                     prec['trispec']['log10k_bins'])
        self.tri_idxlist = self.__get_idxlist_for_trispec()
        self.trispec_matter_klim = prec['trispec']['matter_klim']
        self.trispec_matter_mulim = prec['trispec']['matter_mulim']
        self.trispec_small_k_damping = prec['trispec']['small_k_damping']
        self.tri_lowlim = prec['trispec']['lower_calc_limit']
        self.num_cores = prec['misc']['num_cores'] \
            if prec['misc']['num_cores'] > 0 else mp.cpu_count()
        if self.unbiased_clustering:
            self.lensing = obs_dict['observables']['cosmic_shear']
            self.mm = obs_dict['observables']['cosmic_shear']

    def __get_idxlist_for_trispec(self):
        """
        Returns a list which contains the indices of a matrix with shape
        (trispec_prec['log10k_bins'], trispec_prec['log10k_bins'],
        sample_dim, sample_dim), where we only loop over independent
        matrix elements. The matrix is symmetric in k-space and 
        sample-space.

        Returns
        -------
        idxlist : list
            with length 4 indices per trispec_prec['log10k_bins'] \
            *(trispec_prec['log10k_bins']+1)/2 * sample*(sample_dim+1)/2
            entries

        """
        idxlist = []
        for idxi in range(len(self.krange_tri)):
            for idxj in range(idxi, len(self.krange_tri)):
                for nbin in range(self.sample_dim):
                    for mbin in range(nbin, self.sample_dim):
                        idxlist.append((idxi, idxj, nbin, mbin))

        return idxlist

    def update_mass_func(self,
                         zet,
                         bias_dict,
                         hod_dict,
                         prec):
        """
        Updates the redshift of the halo mass function and recalculates 
        required, subsequent quantities which depend on redshift and are 
        called multiple times to avoid overhead. Should be called if the
        halo mass function object already exists and only the only 
        updated parameter is the redshift.

        Parameters
        ----------
        zet : float
            Redshift at which the covariance should be evaluated.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        """
        self.zet = zet
        self.mass_func.update(z=zet)
        self.ngal = self.nbar(hod_dict)
        self.ncen = self.nbar_cen(hod_dict)
        self.nsat = self.nbar_sat(hod_dict)
        self.norm_bias = -1
        self.effective_bias = \
            self.calc_effective_bias(
                bias_dict, hod_dict, prec['hm'])
        self.Pmm = self.P_mm(bias_dict, hod_dict, prec)
        self.Pgm = self.P_gm(bias_dict, hod_dict, prec['hm'])
        self.Pgg = self.P_gg(bias_dict, hod_dict, prec['hm'])
        self.Plin_spline = UnivariateSpline(np.log(self.mass_func.k),
                                            np.log(self.mass_func.power),
                                            k=2, s=0, ext=0)
        if bias_dict['has_csmf']:
            self.set_spline_galaxy_stellar_mf(hod_dict)
            self.set_spline_galaxy_stellar_mf_bias(hod_dict, bias_dict, prec['hm'])
        return True

    def __P_xy_1h(self,
                  bias_dict,
                  hod_dict,
                  hm_prec,
                  type_x,
                  type_y):
        """
        Calculates the 1-halo contribution of the xy power spectra (with
        x,y either matter 'm', central 'cen', or satellite 'sat').

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        type_x : string
            - "m" for matter
            - "cen" for centrals
            - "sat" for satellites
        type_y : string
            same as type_x

        Returns
        -------
        integral : array
            with unit ?
            with shape (log10k bins, sample bins, sample bins)

        Reference
        ---------
        Dvornik et al. (2018), their Sect 3.1, where the normalization
        in Eq. (24) and (25) is different, i.e. in their notation n_g.

        """

        if type_x == 'm' and type_y == 'm':
            # update mass range of the halo mass function
            M_min_save = hm_prec["log10M_min"]
            step_save = self.mass_func.dlog10m
            Mmin = 2.0
            hm_prec["log10M_min"] = Mmin
            step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
            self.mass_func.update(Mmin=Mmin, dlog10m=step)
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.hod.hod_update(bias_dict, hm_prec)

            hurlyX = self.hurly_x(bias_dict, hod_dict, type_x)
            hurlyY = self.hurly_x(bias_dict, hod_dict, type_y)
            integral = simpson(
                hurlyX*hurlyY*self.mass_func.dndm, x = self.mass_func.m)

            # resets mass range of the halo mass function
            hm_prec["log10M_min"] = M_min_save
            self.mass_func.update(Mmin=M_min_save, dlog10m=step_save)
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.hod.hod_update(bias_dict, hm_prec)

        else:
            hurlyX = self.hurly_x(bias_dict, hod_dict, type_x)
            hurlyY = self.hurly_x(bias_dict, hod_dict, type_y)
            integral = simpson(hurlyX[:,:,None,:]*hurlyY[:,None,:,:]*self.mass_func.dndm,
                               x = self.mass_func.m)

        return integral

    def __P_xy_2h(self,
                  bias_dict,
                  hod_dict,
                  hm_prec,
                  type_x,
                  type_y):
        """
        Calculates the 2-halo contribution of the xy power spectra (with
        x,y either matter 'm', central 'cen', or satellite 'sat').

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        type_x : string
            - "m" for matter
            - "cen" for centrals
            - "sat" for satellites
        type_y : string
            same as type_x

        Returns
        -------
        integralXY : array
            with unit ?
            with shape (log10k bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Sect 3.1, where the normalization
        in Eq. (24) and (25) is different, i.e. in their notation n_g.

        """
        hurlyX = self.hurly_x(bias_dict, hod_dict, type_x)
        hurlyY = self.hurly_x(bias_dict, hod_dict, type_y)
        if type_x == 'sat':
            hurlyX /= self.uk(bias_dict['Mc_relation_cen'])[:, None]
        if type_x == 'm':
            hurlyX /= self.uk(bias_dict['Mc_relation_cen'])[:, None]
        if type_y == 'sat':
            hurlyY /= self.uk(bias_dict['Mc_relation_cen'])[:, None]
        if type_y == 'm':
            hurlyY /= self.uk(bias_dict['Mc_relation_cen'])[:, None]

        bias = self.bias(bias_dict, hm_prec) * bias_dict['bias_2h']
        bias_fac = 1
        if type_x == 'm':
            bias_fac /= bias_dict['bias_2h']
        if type_y == 'm':
            bias_fac /= bias_dict['bias_2h']


        integralX = bias_fac*simpson(self.mass_func.dndm * bias * hurlyX,
                                      x = self.mass_func.m)
        integralY = bias_fac*simpson(self.mass_func.dndm * bias * hurlyY,
                                      x = self.mass_func.m)

        return integralX[:,:,None]*integralY[:,None,:]*self.mass_func.power[:,None,None]

    def P_mm(self,
             bias_dict,
             hod_dict,
             prec):
        """
        Either calculations the matter-matter power spectrum or 
        interpolates the power spectrum from a look-up table for the 
        k-range set in powspec_prec and redshift zet. Returns None, if 
        Pmm is not needed for the Gaussian covariance.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        Pmm : array
            with unit (h/Mpc)^3 [so far, needs to be changed]
            with shape (log10k bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Sect 3.1.

        """
        if (self.mm or self.gm) and self.cov_dict['gauss']:
            if self.Pxy_tab['mm'] is None:
                if prec['hm']['transfer_model'] != 'CAMB':
                    Pmm = self.__P_xy_1h(
                        bias_dict, hod_dict, prec['hm'], 'm', 'm') \
                        * self.small_k_damping(
                            prec['hm']['small_k_damping'],
                            self.mass_func.k)[:, None] \
                        + self.mass_func.power[:, None]
                else:
                    return None

            else:
                Pmm = np.zeros((len(self.mass_func.k), self.sample_dim))
                for mbin in range(self.sample_dim):
                    mm_2dspline = RectBivariateSpline(
                        np.log(self.Pxy_tab['k']),
                        self.Pxy_tab['z'],
                        np.log(self.Pxy_tab['mm'][:, mbin, :]),
                        kx=1, ky=2, s=0)
                    Pmm[:, mbin] = np.squeeze(np.exp(mm_2dspline(
                        np.log(self.mass_func.k), self.mass_func.z)))

            return Pmm
        else:
            return None

    def P_gm(self,
             bias_dict,
             hod_dict,
             hm_prec):
        """
        Either calculations the matter-tracer power spectrum or 
        interpolates the power spectrum from a look-up table for the 
        k-range set in powspec_prec and redshift zet. Returns None, if 
        Pgm is not needed for the Gaussian or super-sample covariance.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        Pgm : array
            with unit ?
            with shape (log10k bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Sect 3.1., Eq. (17)

        """
        if self.gm or (self.mm and self.gg) and (self.cov_dict['gauss'] or self.cov_dict['ssc']):
            if self.Pxy_tab['gm'] is None:
                if self.unbiased_clustering and hm_prec['transfer_model'] != 'CAMB':
                    Pgm = self.mass_func.nonlinear_power[:, None] \
                        * np.ones(self.sample_dim)
                else:
                    if self.unbiased_clustering:
                        Pgm = self.__P_xy_1h(
                            bias_dict, hod_dict, hm_prec, 'm', 'm') \
                            * self.small_k_damping(
                                hm_prec['small_k_damping'],
                                self.mass_func.k)[:, None] \
                            + self.mass_func.power[:, None]
                    else:
                        Pgm = \
                            np.diagonal(self.__P_xy_1h(bias_dict, hod_dict, hm_prec, 'm', 'sat'), axis1 = 1, axis2 = 2) \
                            + np.diagonal(self.__P_xy_1h(bias_dict, hod_dict, hm_prec, 'm', 'cen'), axis1 = 1, axis2 = 2) \
                            * self.small_k_damping(
                                hm_prec['small_k_damping'],
                                self.mass_func.k)[:, None] \
                            + self.mass_func.power[:, None] \
                            * self.effective_bias[None, :] * bias_dict['bias_2h']
            else:
                Pgm = np.zeros((len(self.mass_func.k), self.sample_dim))
                for mbin in range(self.sample_dim):
                    gm_2dspline = RectBivariateSpline(
                        np.log(self.Pxy_tab['k']),
                        self.Pxy_tab['z'],
                        np.log(self.Pxy_tab['gm'][:, mbin, :]),
                        kx=1, ky=2, s=0)
                    Pgm[:, mbin] = np.squeeze(np.exp(gm_2dspline(
                        np.log(self.mass_func.k), self.mass_func.z)[:]))

            return Pgm
        else:
            return None

    def P_gg(self,
             bias_dict,
             hod_dict,
             hm_prec):
        """
        Either calculations the tracer-tracer power spectrum or 
        interpolates the power spectrum from a look-up table for the 
        k-range set in powspec_prec and redshift zet. Returns None, if 
        Pgg is not needed for the Gaussian or super-sample covariance.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        Pgg : array
            with unit ?
            with shape (log10k bins, sample bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Sect 3.1., Eq. (16)

        """
        if (self.gg or self.gm) and \
                (self.cov_dict['gauss'] or self.cov_dict['ssc']):

            if self.Pxy_tab['gg'] is None:
                if self.unbiased_clustering and self.sample_dim < 2 and hm_prec['transfer_model'] != 'CAMB':
                    Pgg = self.mass_func.nonlinear_power[:, None, None]
                else:
                    if self.unbiased_clustering:
                        Pgg = (self.__P_xy_1h(
                            bias_dict, hod_dict, hm_prec, 'm', 'm') \
                            * self.small_k_damping(
                                hm_prec['small_k_damping'],
                                self.mass_func.k)[:, None] \
                            + self.mass_func.power[:, None])[:,:,None]
                    else:
                        Pgg = (2 *
                            self.__P_xy_1h(bias_dict, hod_dict,
                                            hm_prec, 'sat', 'cen')
                            + self.__P_xy_1h(bias_dict, hod_dict,
                                                hm_prec, 'sat', 'sat')
                            #+ self.__P_xy_1h(bias_dict, hod_dict,
                            #                hm_prec, 'cen', 'cen')
                            ) \
                            * self.small_k_damping(
                                hm_prec['small_k_damping'],
                                self.mass_func.k)[:, None, None] \
                            + self.mass_func.power[:, None, None] \
                            * (bias_dict['bias_2h'])**2.0* (self.effective_bias[None,:,None])*( self.effective_bias[None,None,:])

            else:
                Pgg = np.zeros((len(self.mass_func.k), self.sample_dim))
                for mbin in range(self.sample_dim):
                    gg_2dspline = RectBivariateSpline(
                        np.log(self.Pxy_tab['k']),
                        self.Pxy_tab['z'],
                        np.log(self.Pxy_tab['gg'][:, mbin, :]),
                        kx=1, ky=2, s=0)
                    Pgg[:, mbin] = np.squeeze(np.exp(gg_2dspline(
                        np.log(self.mass_func.k), self.mass_func.z)))

            return Pgg
        else:
            return None

    def powspec_responses(self,
                          bias_dict,
                          hod_dict,
                          hm_prec):
        """
        Calculates the power spectra responses 
        del Pxy (k) / del delta_b. Returns None, if the a response term
        is not needed for the covariance.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        response_P_gg, response_P_gm, response_P_mm : list of arrays
            with unit ?
            with shape (log10k bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Appendix A, Eq. (A12) - (A14)

        """
        lnk = np.log(self.mass_func.k)

        if self.gm or self.mm or self.unbiased_clustering:
            integral_m = \
                self.halo_model_integral_I_alpha_x(
                    bias_dict, hod_dict, hm_prec, 1, 'm')
        if self.gg or self.gm:
            integral_g = \
                self.halo_model_integral_I_alpha_x(
                    bias_dict, hod_dict, hm_prec, 1, 'g')
            bias_g = self.calc_effective_bias(bias_dict, hod_dict, hm_prec)
        if self.mm or self.unbiased_clustering:
            integral_mm = \
                self.halo_model_integral_I_alpha_xy(
                    bias_dict, hod_dict, hm_prec, 1, 'm', 'm')
            integral_mm = np.einsum('iijj->ij', integral_mm)
        if self.gm and not self.unbiased_clustering:
            integral_gm = \
                self.halo_model_integral_I_alpha_xy(
                    bias_dict, hod_dict, hm_prec, 1, 'g', 'm')
            integral_gm = np.einsum('iijj->ij', integral_gm)
        if self.gg and not self.unbiased_clustering:
            integral_gg = \
                self.halo_model_integral_I_alpha_xy(
                    bias_dict, hod_dict, hm_prec, 1, 'g', 'g')
            integral_gg = np.diagonal(np.diagonal(integral_gg, axis1 = 0, axis2 = 1), axis1 = 0, axis2 = 1)
        k3lnPkspl = UnivariateSpline(np.log(self.mass_func.k),np.log(self.mass_func.k**3*self.mass_func.power),k = 3,s = 5*min(self.mass_func.k[1:]- self.mass_func.k[:-1]),ext=0)
        dk3lnPkspl_dlnk = k3lnPkspl.derivative()
        deriv_Plin = dk3lnPkspl_dlnk(np.log(self.mass_func.k))
        

        if self.mm:
            response_P_mm = \
                (68/21 - deriv_Plin[:, None]/3) \
                * integral_m**2 * self.mass_func.power[:, None] \
                + integral_mm
        else:
            response_P_mm = None
        if self.unbiased_clustering:
            if self.gm:
                response_P_gm = \
                    (68/21 - deriv_Plin[:, None]/3) \
                    * integral_m**2 * self.mass_func.power[:, None] \
                    + integral_mm
                if not self.redshift_dep_bias:
                    response_P_gm -= self.Pgm
            else:
                response_P_gm = None
            if self.gg:
                response_P_gg = \
                    (68/21 - deriv_Plin[:, None]/3) \
                    * integral_m**2 * self.mass_func.power[:, None] \
                    + integral_mm 
                if not self.redshift_dep_bias:
                    response_P_gg -= 2*np.diagonal(self.Pgg, axis1 = -2, axis2 =-1)
            else:
                response_P_gg = None
        else:
            if self.gm:
                response_P_gm = \
                    (68/21 - deriv_Plin[:, None]/3) \
                    * integral_g * integral_m * self.mass_func.power[:, None] \
                    + integral_gm - bias_g[None, :] * self.Pgm
            else:
                response_P_gm = None

            if self.gg:
                response_P_gg = \
                    (68/21 - deriv_Plin[:, None]/3) \
                    * integral_g**2 * self.mass_func.power[:, None] \
                    + integral_gg - 2 * bias_g[None, :] * (np.diagonal(self.Pgg,axis1=-2,axis2=-1) - (np.diagonal(self.Pgg,axis1=-2,axis2=-1)[-1,:])[None,:])
            else:
                response_P_gg = None
        return response_P_gg, response_P_gm, response_P_mm

    def __check_for_tabulated_trispectra(self, tri_tab):
        """
        Checks whether pre-tabulated trispectra are available. If
        necessary, interpolates the trispectra in redshift, 
        extrapolation is not supported.

        Parameters
        ----------
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of 
            matter 'm' and tracer 'g', optional) for different
            wavenumbers and redshifts. To be passed from the read_input 
            method of the FileInput class.

        Returns
        -------
        tri_gggg, tri_gggm, tri_ggmm, \
        tri_gmgm, tri_mmgm, tri_mmmm : list of bools

        and

        trispec_gggg, trispec_gggm, trispec_ggmm, \
        trispec_gmgm, trispec_mmgm, trispec_mmmm : list of arrays
            with unit ?
            with shape (powspec_prec['log10k_bins'], 
                        powspec_prec['log10k_bins'],
                        sample bins,
                        sample bins)

        """

        k_dim = len(self.mass_func.k)

        tri_gggg, tri_gggm, tri_ggmm, tri_gmgm, tri_mmgm, tri_mmmm = \
            True, True, True, True, True, True

        if tri_tab is None:
            tri_tab = {'z': None}
        if tri_tab['z'] is not None:
            if self.mass_func.z >= min(tri_tab['z']) and \
               self.mass_func.z <= max(tri_tab['z']):
                if abs(np.log(self.mass_func.k[0]) - tri_tab['log10k'][0]) < 1e-3 or \
                   abs(np.log(self.mass_func.k[-1]) - tri_tab['log10k'][-1]) < 1e-3 or \
                   k_dim == len(tri_tab['log10k']):
                    if self.gg and tri_tab['gggg'].any() != 0:
                        tri_gggg = False
                    if self.gg and self.gm and self.cross_terms and \
                       tri_tab['gggm'].any() != 0:
                        tri_gggm = False
                    if self.gg and self.mm and self.cross_terms and \
                       tri_tab['ggmm'].any() != 0:
                        tri_ggmm = False
                    if self.gm and tri_tab['gmgm'].any() != 0:
                        tri_gmgm = False
                    if self.mm and self.gm and self.cross_terms and \
                       tri_tab['mmgm'].any() != 0:
                        tri_mmgm = False
                    if self.mm and tri_tab['mmmm'].any() != 0:
                        tri_mmmm = False

        if not tri_gggg:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_gggg = tri_tab['gggg'][:, :, :, :, idx]
            else:
                trispec_gggg = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    gggg_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['gggg'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    gggg = np.squeeze(np.exp(gggg_spline(self.mass_func.z)))
                    trispec_gggg[idxi][idxj][nbin][mbin] = gggg
                    trispec_gggg[idxi][idxj][mbin][nbin] = gggg
                    trispec_gggg[idxj][idxi][nbin][mbin] = gggg
                    trispec_gggg[idxj][idxi][mbin][nbin] = gggg
        else:
            trispec_gggg = 0
            if not self.gg:
                tri_gggg = False

        if not tri_gggm:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_gggm = tri_tab['gggm'][:, :, :, :, idx]
            else:
                trispec_gggm = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    gggm_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['gggm'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    gggm = np.squeeze(np.exp(gggm_spline(self.mass_func.z)))
                    trispec_gggm[idxi][idxj][nbin][mbin] = gggm
                    trispec_gggm[idxi][idxj][mbin][nbin] = gggm
                    trispec_gggm[idxj][idxi][nbin][mbin] = gggm
                    trispec_gggm[idxj][idxi][mbin][nbin] = gggm
        else:
            trispec_gggm = 0
            if not (self.gg and self.gm and self.cross_terms):
                tri_gggm = False

        if not tri_ggmm:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_ggmm = tri_tab['ggmm'][:, :, :, :, idx]
            else:
                trispec_ggmm = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    ggmm_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['ggmm'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    ggmm = np.squeeze(np.exp(ggmm_spline(self.mass_func.z)))
                    trispec_ggmm[idxi][idxj][nbin][mbin] = ggmm
                    trispec_ggmm[idxi][idxj][mbin][nbin] = ggmm
                    trispec_ggmm[idxj][idxi][nbin][mbin] = ggmm
                    trispec_ggmm[idxj][idxi][mbin][nbin] = ggmm
        else:
            trispec_ggmm = 0
            if not (self.gg and self.mm and self.cross_terms):
                tri_ggmm = False

        if not tri_gmgm:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_gmgm = tri_tab['gmgm'][:, :, :, :, idx]
            else:
                trispec_gmgm = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    gmgm_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['gmgm'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    gmgm = np.squeeze(np.exp(gmgm_spline(self.mass_func.z)))
                    trispec_gmgm[idxi][idxj][nbin][mbin] = gmgm
                    trispec_gmgm[idxi][idxj][mbin][nbin] = gmgm
                    trispec_gmgm[idxj][idxi][nbin][mbin] = gmgm
                    trispec_gmgm[idxj][idxi][mbin][nbin] = gmgm
        else:
            trispec_gmgm = 0
            if not self.gm:
                tri_gmgm = False

        if not tri_mmgm:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_mmgm = tri_tab['mmgm'][:, :, :, :, idx]
            else:
                trispec_mmgm = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    mmgm_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['mmgm'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    mmgm = np.squeeze(np.exp(mmgm_spline(self.mass_func.z)))
                    trispec_mmgm[idxi][idxj][nbin][mbin] = mmgm
                    trispec_mmgm[idxi][idxj][mbin][nbin] = mmgm
                    trispec_mmgm[idxj][idxi][nbin][mbin] = mmgm
                    trispec_mmgm[idxj][idxi][mbin][nbin] = mmgm
        else:
            trispec_mmgm = 0
            if not (self.mm and self.gm and self.cross_terms):
                tri_mmgm = False

        if not tri_mmmm:
            if self.mass_func.z in tri_tab['z']:
                idx = np.where(tri_tab['z'] == self.mass_func.z)[0][0]
                trispec_mmmm = tri_tab['mmmm'][:, :, :, :, idx]
            else:
                trispec_mmmm = np.zeros((k_dim, k_dim,
                                         self.sample_dim, self.sample_dim))
                for idxi, idxj, nbin, mbin in self.tri_idxlist:
                    mmmm_spline = UnivariateSpline(
                        tri_tab['z'],
                        np.log(tri_tab['mmmm'][idxi, idxj, nbin, mbin, :]),
                        k=2, s=0)
                    mmmm = np.squeeze(np.exp(mmmm_spline(self.mass_func.z)))
                    trispec_mmmm[idxi][idxj][nbin][mbin] = mmmm
                    trispec_mmmm[idxi][idxj][mbin][nbin] = mmmm
                    trispec_mmmm[idxj][idxi][nbin][mbin] = mmmm
                    trispec_mmmm[idxj][idxi][mbin][nbin] = mmmm
        else:
            trispec_mmmm = 0
            if not self.mm:
                tri_mmmm = False

        return tri_gggg, tri_gggm, tri_ggmm, \
            tri_gmgm, tri_mmgm, tri_mmmm, \
            trispec_gggg, trispec_gggm, trispec_ggmm, \
            trispec_gmgm, trispec_mmgm, trispec_mmmm

    def trispectra(self,
                   output_dict,
                   bias_dict,
                   hod_dict,
                   hm_prec,
                   tri_tab):
        """
        Calculates the trispectra for all combinations of matter ('m')
        and tracer ('g'). Extrapolates and interpolates the trispectra 
        from the k-range specified in trispec_prec to the k-range in 
        powspec_prec. Returns None, if a trispectrum term is not needed 
        for the covariance.

        Parameters
        ----------
        output_dict : dictionary
            Specifies whether a file for the trispectra should be 
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from 
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of 
            matter 'm' and tracer 'g', optional) for different
            wavenumbers and redshifts. To be passed from the read_input 
            method of the FileInput class.

        Returns
        -------
        trispec_gggg, trispec_gggm, trispec_ggmm, \
        trispec_gmgm, trispec_mmgm, trispec_mmmm : list of arrays
            with unit ?
            with shape (log10k bins, log10k bins, 
                        sample bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Appendix A, Eq. (A21), (A22)
        Pielorz et al. (2010) for the formalism of the 234-halo term
            following and fitting perturbation theory

        """
        if self.unbiased_clustering:
            self.mm = True
            gg_save = self.gg
            self.gg = False
        tri_gggg, tri_gggm, tri_ggmm, \
            tri_gmgm, tri_mmgm, tri_mmmm, \
            trispec_gggg, trispec_gggm, trispec_ggmm, \
            trispec_gmgm, trispec_mmgm, trispec_mmmm = \
            self.__check_for_tabulated_trispectra(tri_tab)
        trispec1h_gggg, trispec1h_gggm, trispec1h_ggmm, \
            trispec1h_gmgm, trispec1h_mmgm, trispec1h_mmmm = \
            self.__trispectra_1h(bias_dict, hod_dict, hm_prec,
                                 tri_gggg, tri_gggm, tri_ggmm,
                                 tri_gmgm, tri_mmgm, tri_mmmm)
        trispectra234h = \
            self.__trispectra_234h(bias_dict, hod_dict, hm_prec)
        self.trispectra234h = trispectra234h
        self.trispec1h_gggg = trispec1h_gggg
        self.trispec1h_mmmm = trispec1h_mmmm
        if output_dict['trispec']:
            out = Output(output_dict)
            trispecs = [trispec1h_gggg + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :]**2,
                        trispec1h_gggm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :],
                        trispec1h_ggmm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2,
                        trispec1h_gmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]*self.effective_bias[None,None,None, :],
                        trispec1h_mmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,None,:],
                        trispec1h_mmmm + trispectra234h[:, :, None, None]]
            tri_bool = [tri_gggg, tri_gggm, tri_ggmm,
                        tri_gmgm, tri_mmgm, tri_mmmm]
            out.write_trispectra(self.mass_func.z,
                                 self.logks_tri,
                                 self.sample_dim,
                                 trispecs,
                                 tri_bool)

        k_dim = len(self.mass_func.k)

        if self.gg and tri_gggg:
            if len(np.where(trispec1h_gggg + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :]**2 <0)[0]) == 0:
                logtrispec_gggg_ktri = \
                    np.log10(trispec1h_gggg + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :]**2)
            else:
                logtrispec_gggg_ktri = trispec1h_gggg + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :]**2
            trispec_gggg = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
        else:
            logtrispec_gggg_ktri = None

        if self.gg and self.gm and self.cross_terms and tri_gggm:
            if len(np.where(trispec1h_gggm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :] <0)[0]) == 0:
                logtrispec_gggm_ktri = \
                    np.log10(trispec1h_gggm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :])
            else:
                logtrispec_gggm_ktri = \
                    (trispec1h_gggm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2*self.effective_bias[None,None,None, :])
            trispec_gggm = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
        else:
            logtrispec_gggm_ktri = None

        if self.gg and self.mm and self.cross_terms and tri_ggmm:
            if len(np.where(trispec1h_ggmm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2 < 0)[0]) == 0:
                logtrispec_ggmm_ktri = \
                    np.log10(trispec1h_ggmm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2)
            else:
                logtrispec_ggmm_ktri = trispec1h_ggmm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]**2
            trispec_ggmm = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
        else:
            logtrispec_ggmm_ktri = None

        if self.gm and tri_gmgm:
            if len(np.where(trispec1h_gmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]*self.effective_bias[None,None,None, :] < 0)[0]) == 0:
                logtrispec_gmgm_ktri = \
                    np.log10(trispec1h_gmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]*self.effective_bias[None,None,None, :])
            else:
                logtrispec_gmgm_ktri = trispec1h_gmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,:,None]*self.effective_bias[None,None,None, :]
            trispec_gmgm = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
        else:
            logtrispec_gmgm_ktri = None

        if self.mm and self.gm and self.cross_terms and tri_mmgm:
            if len(np.where(trispec1h_mmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,None,:] < 0)[0]) == 0:
                logtrispec_mmgm_ktri = \
                    np.log10(trispec1h_mmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,None,:])
            else:
                logtrispec_mmgm_ktri = trispec1h_mmgm + trispectra234h[:, :, None, None]*self.effective_bias[None,None,None,:]
            trispec_mmgm = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
        else:
            logtrispec_mmgm_ktri = None
        if self.mm and tri_mmmm:
            if len(np.where(trispec1h_mmmm + trispectra234h[:, :, None, None] <0)[0]) == 0:
                logtrispec_mmmm_ktri = \
                    np.log10(trispec1h_mmmm + trispectra234h[:, :, None, None])
            else:
                logtrispec_mmmm_ktri = trispec1h_mmmm + trispectra234h[:, :, None, None]
            trispec_mmmm = np.zeros((k_dim, k_dim,
                                     self.sample_dim, self.sample_dim))
            
        else:
            logtrispec_mmmm_ktri = None


        
        '''
        # prepare extrapolation
        idx_min = np.min(np.where(self.mass_func.k > self.krange_tri[0]))
        idx_max = np.max(np.where(self.mass_func.k < self.krange_tri[-1]))
        
        extr_idx = np.arange(idx_min)
        extr_idx = np.append(extr_idx,
                             np.arange(idx_max, len(self.mass_func.k)))
        extr_idx = extr_idx.astype(int)
        trilist = zip([trispec_gggg, trispec_gggm, trispec_ggmm,
                       trispec_gmgm, trispec_mmgm, trispec_mmmm],
                      [logtrispec_gggg_ktri, logtrispec_gggm_ktri,
                       logtrispec_ggmm_ktri, logtrispec_gmgm_ktri,
                       logtrispec_mmgm_ktri, logtrispec_mmmm_ktri])
        obslist = [self.gg and tri_gggg,
                   self.gg and self.gm and self.cross_terms and tri_gggm,
                   self.gg and self.mm and self.cross_terms and tri_ggmm,
                   self.gm and tri_gmgm,
                   self.mm and self.gm and self.cross_terms and tri_mmgm,
                   self.mm and tri_mmmm]
        iterlist = list(itertools.compress(trilist, obslist))

        # BivariateSpline does no sensible extrapolation, so we do our
        # own
        for trispec, logtrispec_ktri in iterlist:
            logtrispec = np.zeros((len(self.mass_func.k),
                                   len(self.mass_func.k),
                                   self.sample_dim,
                                   self.sample_dim))

            for nbin in range(self.sample_dim):
                for mbin in range(nbin, self.sample_dim):
                    # interpolation
                    spline_trispec = RectBivariateSpline(
                        self.logks_tri, self.logks_tri,
                        logtrispec_ktri[:, :, nbin, mbin], kx=2, ky=2, s=0)
                    logtrispec[idx_min:idx_max, idx_min:idx_max, nbin, mbin] = \
                        spline_trispec(
                            np.log10(self.mass_func.k[idx_min:idx_max]),
                            np.log10(self.mass_func.k[idx_min:idx_max]))
                    logtrispec[:, :, mbin, nbin] = \
                        np.copy(logtrispec[:, :, nbin, mbin])

                    # extrapolation in one k-direction
                    for idxi in range(idx_min, idx_max):
                        spline_trispec = UnivariateSpline(
                            np.log10(self.mass_func.k[idx_min:idx_max]),
                            logtrispec[idxi, idx_min:idx_max, nbin, mbin],
                            k=1, s=0, ext=0)
                        trispec[idxi, :, nbin, mbin] = \
                            10**spline_trispec(np.log10(self.mass_func.k))
                        trispec[:, idxi, nbin, mbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])
                        trispec[idxi, :, mbin, nbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])
                        trispec[:, idxi, mbin, nbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])

                    # extrapolation in the second k-direction
                    for idxi in extr_idx:
                        spline_trispec = UnivariateSpline(
                            np.log10(self.mass_func.k[idx_min:idx_max-20]),
                            np.log10(
                                trispec[idxi, idx_min:idx_max-20, nbin, mbin]),
                            k=1, s=0, ext=0)
                        trispec[idxi, :, nbin, mbin] = 10**spline_trispec(
                            np.log10(self.mass_func.k))
                        trispec[:, idxi, nbin, mbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])
                        trispec[idxi, :, mbin, nbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])
                        trispec[:, idxi, mbin, nbin] = \
                            np.copy(trispec[idxi, :, nbin, mbin])
        
        '''
        
        x_values = np.zeros((2,len(self.mass_func.k)*len(self.mass_func.k)))
        flat_idx = 0
        for i_k in range(len(self.mass_func.k)):
            for j_k in range(len(self.mass_func.k)):
                x_values[0,flat_idx] = np.log10(self.mass_func.k[i_k])
                x_values[1,flat_idx] = np.log10(self.mass_func.k[j_k])
                flat_idx +=1
        np.seterr(over='ignore')
        for nbin in range(self.sample_dim):
            for mbin in range(nbin, self.sample_dim):
                if self.gg and tri_gggg:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_gggg_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_gggg_ktri)):
                        trispec_gggg[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else:
                        trispec_gggg[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                if self.gg and self.gm and self.cross_terms:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_gggm_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_gggm_ktri)):
                        trispec_gggm[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else: 
                        trispec_gggm[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                if self.gg and self.mm and self.cross_terms:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_ggmm_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_ggmm_ktri)):
                        trispec_ggmm[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else:
                        trispec_ggmm[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                if self.gm:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_gmgm_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_gmgm_ktri)):
                        trispec_gmgm[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else:
                        trispec_gmgm[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                if self.gm and self.mm and self.cross_terms:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_mmgm_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_mmgm_ktri)):
                        trispec_mmgm[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else:
                        trispec_mmgm[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                if self.mm:
                    tri_interpolator = RegularGridInterpolator((self.logks_tri, self.logks_tri),
                                                    logtrispec_mmmm_ktri,bounds_error= False, fill_value = None)
                    if np.isinf(10**np.max(logtrispec_mmmm_ktri)):
                        trispec_mmmm[:,:,nbin,mbin] = tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
                    else:
                        trispec_mmmm[:,:,nbin,mbin] = 10**tri_interpolator((x_values[0,:],x_values[1,:])).reshape((len(self.mass_func.k),len(self.mass_func.k)))
        np.seterr(over='warn')
        if self.gg:
            trispec_gggg[np.where(trispec_gggg < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.gg and self.gm and self.cross_terms:
            trispec_gggm[np.where(trispec_gggm < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.gg and self.mm and self.cross_terms:
            trispec_ggmm[np.where(trispec_ggmm < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.gm:
            trispec_gmgm[np.where(trispec_gmgm < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.gm and self.mm and self.cross_terms:
            trispec_mmgm[np.where(trispec_mmgm < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.mm:
            trispec_mmmm[np.where(trispec_mmmm < self.tri_lowlim)] = \
                self.tri_lowlim
        if self.unbiased_clustering:
            if not self.lensing:
                self.mm = False
            self.gg = gg_save
            return trispec_mmmm, trispec_mmmm, trispec_mmmm, \
                trispec_mmmm, trispec_mmmm, trispec_mmmm
        else:
            return trispec_gggg, trispec_gggm, trispec_ggmm, \
                trispec_gmgm, trispec_mmgm, trispec_mmmm

    def __poisson(self,
                  lam,
                  fac=0.9):
        """
        Computing poisson factors necessary for halomodel integrals and HoD functions
        for the trispectrum.

        Parameters
        ----------
        lam : int
            ...
        fac : float
            ...

        Returns
        -------
        res : float

        """
        res = 1
        for _ in range(2, lam):
            res *= (lam - 1) * fac - lam + 2
        return res

    def __trispectra_1h(self,
                        bias_dict,
                        hod_dict,
                        hm_prec,
                        tri_gggg,
                        tri_gggm,
                        tri_ggmm,
                        tri_gmgm,
                        tri_mmgm,
                        tri_mmmm):
        """
        Calculates the 1-halo term for the trispectra for all 
        combinations of matter ('m') and tracer ('g'). Uses the
        multiprocessing library to speed up the calculation with all
        available cores. Returns None, if a trispectrum term is not 
        needed for the covariance or if it is available as a look-up
        table.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        tri_xxxx : bool
            Is False if this particular trispectrum is already available
            as a look-up table. 'x' is either 'm' for matter or 'g' for 
            tracer.

        Returns
        -------
        trispec1h_gggg, trispec1h_gggm, trispec1h_ggmm, \
        trispec1h_gmgm, trispec1h_mmgm, trispec1h_mmmm : list of arrays
            with unit ?
            with shape (tri log10k bins, tri log10k bins,
                        sample bins, sample bins)

        References
        ----------
        Dvornik et al. (2018), their Appendix A, Eq. (A22)

        """

        k_dim = len(self.krange_tri)
        M_dim = len(self.mass_func.m)

        trispec1h_gggg = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        trispec1h_gggm = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        trispec1h_ggmm = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        trispec1h_gmgm = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        trispec1h_mmgm = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        trispec1h_mmmm = np.zeros((k_dim, k_dim,
                                   self.sample_dim, self.sample_dim))
        damp_1h = np.sqrt(self.small_k_damping(
            self.trispec_small_k_damping,
            self.krange_tri,
            1e-2)[:, None, None, None]*self.small_k_damping(
            self.trispec_small_k_damping,
            self.krange_tri,
            1e-2)[None, :, None, None])

        '''if (self.gg or self.gm) and \
           (tri_gggg or tri_gggm or tri_ggmm or tri_gmgm or tri_mmgm):
            hurly_c_spline = \
                self.hurly_x_spline_logk(bias_dict, hod_dict, 'cen')
            hurly_s_spline = \
                self.hurly_x_spline_logk(bias_dict, hod_dict, 'sat')
            hurly_c = np.zeros((k_dim, self.sample_dim, M_dim))
            hurly_s = np.zeros((k_dim, self.sample_dim, M_dim))
        if (self.mm or self.gm) and \
           (tri_gggm or tri_ggmm or tri_gmgm or tri_mmgm):
            hurly_m_spline = \
                self.hurly_x_spline_logk(bias_dict, hod_dict, 'm')
            hurly_m = np.zeros((k_dim, self.sample_dim, M_dim))
            for nbin in range(self.sample_dim):
                for idxM in range(M_dim):
                    if self.gg or self.gm:
                        hurly_c[:, nbin, idxM] = \
                            hurly_c_spline[nbin][idxM](self.logks_tri)
                        hurly_s[:, nbin, idxM] = \
                            hurly_s_spline[nbin][idxM](self.logks_tri)
                    if self.mm or self.gm:
                        hurly_m[:, nbin, idxM] = \
                            hurly_m_spline[nbin][idxM](self.logks_tri)'''

        hurly_c_spline = \
            self.hurly_x_spline_logk(bias_dict, hod_dict, 'cen')
        hurly_s_spline = \
            self.hurly_x_spline_logk(bias_dict, hod_dict, 'sat')
        hurly_c = np.zeros((k_dim, self.sample_dim, M_dim))
        hurly_s = np.zeros((k_dim, self.sample_dim, M_dim))
        hurly_m_spline = \
            self.hurly_x_spline_logk(bias_dict, hod_dict, 'm')
        hurly_m = np.zeros((k_dim, self.sample_dim, M_dim))
        for nbin in range(self.sample_dim):
            for idxM in range(M_dim):
                hurly_c[:, nbin, idxM] = \
                    hurly_c_spline[nbin][idxM](self.logks_tri)
                hurly_s[:, nbin, idxM] = \
                    hurly_s_spline[nbin][idxM](self.logks_tri)
                hurly_m[:, nbin, idxM] = \
                    hurly_m_spline[nbin][idxM](self.logks_tri)


        if self.gg and tri_gggg:
            integrand = np.sqrt((4 * hurly_c[:, None, :, None, :] 
                        * hurly_s[:, None, :, None, :]**3 + hurly_s[:, None, :, None, :]**4)
                    * (4 * hurly_c[None, :, None, :, :] * hurly_s[None, :, None, :, :]**3
                        + hurly_s[None, :, None, :, :]**4)) \
                    * self.mass_func.dndm[None, None, None, None, :] * self.__poisson(4)
            trispec1h_gggg = simpson(integrand, x = self.mass_func.m)
        if self.gg and self.gm and self.cross_terms and tri_gggm:
            integrand = np.sqrt(
                    (3 * hurly_c[:, None, :, None, :] * hurly_s[:, None, :, None, :]**2
                     + hurly_s[:, None, :, None, :]**3)
                    * hurly_m[:, None, :, None, :]
                    * (3 * hurly_c[None, :, None, :, :] * hurly_s[None, :, None, :, :]**2
                        + hurly_s[None, :, None, :, :]**3)
                    * hurly_m[None, :, None, :, :]) \
                    * self.mass_func.dndm[None, None, None, None, :] * self.__poisson(3)
            trispec1h_gggm = simpson(integrand, x = self.mass_func.m)
        
        if self.gg and self.mm and self.cross_terms and tri_ggmm:
            integrand = np.sqrt(
                (4 * hurly_c[:, None, :, None, :] * hurly_s[:, None, :, None, :]**3
                    + hurly_s[None, :, None, :, :]**4)
                * hurly_m[None, :, None, :, :]**4) \
                * self.mass_func.dndm[None, None, None, None, :] 
            trispec1h_ggmm = simpson(integrand, x = self.mass_func.m)
 
        if self.gm and tri_gmgm:
            integrand = np.sqrt(
                (2 * hurly_c[None, :, None, :, :] * hurly_s[None, :, None, :, :]
                    + hurly_s[None, :, None, :, :]**2)
                * hurly_m[None, :, None, :, :]**2
                * (2 * hurly_c[None, :, None, :, :] * hurly_s[None, :, None, :, :]
                    + hurly_s[None, :, None, :, :]**2)
                * hurly_m[None, :, None, :, :]**2) \
                * self.mass_func.dndm[None, None, None, None, :] 
            trispec1h_gmgm = simpson(integrand, x = self.mass_func.m)
       
        if self.mm and self.gm and self.cross_terms and tri_mmgm:
            integrand = np.sqrt(
                hurly_m[None, :, None, :, :]**4
                * (2 * hurly_c[None, :, None, :, :] * hurly_s[None, :, None, :, :]
                    + hurly_s[None, :, None, :, :]**2)
                * hurly_m[None, :, None, :, :]**2) \
                * self.mass_func.dndm[None, None, None, None, :] 
            trispec1h_mmgm = simpson(integrand, x = self.mass_func.m)

        if self.mm and tri_mmmm:
            # update mass range of the halo mass function
            M_min_save = hm_prec["log10M_min"]
            step_save = self.mass_func.dlog10m
            Mmin = 2.0
            hm_prec["log10M_min"] = Mmin
            step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.mass_func.update(Mmin=Mmin, dlog10m=step)
            self.hod.hod_update(bias_dict, hm_prec)
            M_dim = len(self.mass_func.m)

            hurly_m_spline = self.hurly_x_spline_logk(bias_dict, hod_dict, 'm')
            hurly_m = np.zeros((k_dim, self.sample_dim, M_dim))
            for nbin in range(self.sample_dim):
                for idxM in range(M_dim):
                    hurly_m[:, nbin, idxM] = \
                        hurly_m_spline[nbin][idxM](self.logks_tri)
            
            
            integrand = hurly_m[:, None, 0, :]**2 * hurly_m[None, :, 0, :]**2 * self.mass_func.dndm[None, None, :]
            trispec1h_mmmm = simpson(integrand, x = self.mass_func.m)[:, :, None, None]*np.ones((self.sample_dim,self.sample_dim))[None, None, :, :]
            # resets mass range of the halo mass function
            hm_prec["log10M_min"] = M_min_save
            self.mass_func.update(Mmin=M_min_save, dlog10m=step_save)
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.hod.hod_update(bias_dict, hm_prec)
        return trispec1h_gggg * damp_1h, trispec1h_gggm * damp_1h, \
            trispec1h_ggmm * damp_1h, trispec1h_gmgm * damp_1h, \
            trispec1h_mmgm * damp_1h, trispec1h_mmmm * damp_1h

    def __calc_int_for_trispec_2h(self,
                                  phi,
                                  ki,
                                  kj):
        """
        Calculates the 2-halo term for the trispectra. Is only needed by
        the integration routine in the trispectra_234h method.

        Parameters
        ----------
        phi : float
            angle between the ki and kj vector
        ki : float
            wavenumber
        kj : float
            wavenumber

        Returns:
        --------
        pp + pm : float

        """
        mu = np.cos(phi)

        km, pm = 0, 0
        if not ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1-mu < self.trispec_matter_mulim)):
            km = np.sqrt(ki*ki + kj*kj - 2*ki*kj*mu)
            pm = np.exp(self.Plin_spline(np.log(km)))

        kp, pp = 0, 0
        if not ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1+mu < self.trispec_matter_mulim)):
            kp = np.sqrt(ki*ki + kj*kj + 2*ki*kj*mu)
            pp = np.exp(self.Plin_spline(np.log(kp)))
        return pp + pm

    def __calc_int_for_trispec_3h(self,
                                  phi,
                                  ki,
                                  kj):
        """
        Calculates the 3-halo term for the trispectra. Is only needed by
        the integration routine in the trispectra_234h method.

        Parameters
        ----------
        See documentation of the __calc_int_for_trispec_2h - method.

        Returns:
        --------
        res : float

        """
        mu = np.cos(phi)
        res = self.__bispec_pt(mu, ki, kj) \
            + self.__bispec_pt(-mu, ki, kj)

        return res

    def __bispec_pt(self,
                    mu,
                    ki,
                    kj):
        """
        Auxiliary function to calculate the 3-halo term for the 
        trispectra. Is only needed by the __calc_int_for_trispec_3h
        method.

        Parameters
        ----------
        mu : float
            cosine of angle between the ki and kj vector
        ki : float
            wavenumber
        kj : float
            wavenumber

        Returns:
        --------
        res : float

        """

        pi = np.exp(self.Plin_spline(np.log(ki)))
        pj = np.exp(self.Plin_spline(np.log(kj)))

        term1 = 2 * self.__pt_kernel_f2(mu, ki, kj) * pi*pj

        if ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1+mu < self.trispec_matter_mulim)):
            return term1

        kp = np.sqrt(ki*ki + kj*kj + 2*ki*kj*mu)
        pp = np.exp(self.Plin_spline(np.log(kp)))
        mu_ip = ki/kp + mu * kj/kp
        mu_jp = kj/kp + mu * ki/kp
        term2 = 2 * self.__pt_kernel_f2(-mu_ip, ki, kp) * pi*pp
        term3 = 2 * self.__pt_kernel_f2(-mu_jp, kj, kp) * pj*pp

        return term1 + term2 + term3

    def __pt_kernel_f2(self,
                       mu,
                       ki,
                       kj):
        """
        Auxiliary function to calculate the 3 and 4-halo term for the 
        trispectra. Is only needed by the __bispec_pt and
        __calc_int_for_trispec_4h method.

        Parameters
        ----------
        See documentation of the __bispec_pt - method.

        Returns:
        --------
        float

        """
        return 5/7 + 2/7 * mu*mu + .5*mu * (ki/kj + kj/ki)

    def calc_int_for_trispec_4h(self,
                                  phi,
                                  ki,
                                  kj):
        return self.__calc_int_for_trispec_4h(phi,
                                  ki,
                                  kj)
    
    def __calc_int_for_trispec_4h(self,
                                  phi,
                                  ki,
                                  kj):
        """
        Calculates the 4-halo term for the trispectra. Is only needed by
        the integration routine in the trispectra_234h method.

        Parameters
        ----------
        See documentation of the __calc_int_for_trispec_2h - method.

        Returns:
        --------
        term1 + term2 + term3 + term4 : float

        """
        mu = np.cos(phi)

        pm = 0
        F2_im, F2_jm = 0, 0
        km = np.sqrt(ki*ki + kj*kj - 2*ki*kj*mu)
        if km == 0:
            F2_im = 13./28
            F2_jm = 13./28
        else:
            mu_im = kj/km * mu - ki/km
            mu_jm = kj/km - mu * ki/km
            pm = np.exp(self.Plin_spline(np.log(km)))
            F2_im = self.__pt_kernel_f2(mu_im, ki, km)
            F2_jm = self.__pt_kernel_f2(-mu_jm, kj, km)

        pp = 0
        F2_ip, F2_jp = 0, 0
        kp = np.sqrt(ki*ki + kj*kj + 2*ki*kj*mu)
        if kp == 0:
            F2_ip = 13./28
            F2_jp = 13./28
        else:
            mu_ip = kj/kp * mu + ki/kp
            mu_jp = ki/kp * mu + kj/kp
            pp = np.exp(self.Plin_spline(np.log(kp)))
            F2_ip = self.__pt_kernel_f2(-mu_ip, ki, kp)
            F2_jp = self.__pt_kernel_f2(-mu_jp, kj, kp)

        pi = np.exp(self.Plin_spline(np.log(ki)))
        pj = np.exp(self.Plin_spline(np.log(kj)))

        F3_ij = self.__pt_kernel_f3(mu, ki, kj)
        F3_ji = self.__pt_kernel_f3(mu, kj, ki)

        term1 = 4 * pi*pi * (F2_ip*F2_ip*pp + F2_im*F2_im*pm)
        term2 = 4 * pj*pj * (F2_jp*F2_jp*pp + F2_jm*F2_jm*pm)
        term3 = 8 * pi*pj * (F2_ip*F2_jp*pp + F2_im*F2_jm*pm)
        term4 = 12 * pi*pj * (F3_ij*pi + F3_ji*pj)

        return term1 + term2 + term3 + term4

    def __pt_kernel_f3(self,
                       mu,
                       ki,
                       kj):
        """
        Auxiliary function to calculate the 4-halo term for the 
        trispectra. Is only needed by the __calc_int_for_trispec_4h 
        method.

        Parameters
        ----------
        See documentation of the __bispec_pt - method.

        Returns:
        --------
        res : float

        """
        '''
        if ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1-mu < self.trispec_matter_mulim)):
            return 0.0
        km = 0
        mu_m, alpha_m, beta_m = 0, 0, 0
        if not ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1-mu < self.trispec_matter_mulim)):
            km = np.sqrt(ki*ki + kj*kj - 2*ki*kj*mu)
            mu_m = kj/km * mu - ki/km
            alpha_m = 1 + ki/km * mu_m
            beta_m = mu_m/2 * (ki/km + km/ki + 2*mu_m)

        kp = 0
        mu_p, alpha_p, beta_p = 0, 0, 0
        if not ((np.fabs(ki-kj) < self.trispec_matter_klim)
                and (1+mu < self.trispec_matter_mulim)):
            kp = np.sqrt(ki*ki + kj*kj + 2*ki*kj*mu)
            mu_p = kj/kp * mu + ki/kp
            alpha_p = 1 - ki/kp * mu_p
            beta_p = -mu_p/2 * (ki/kp + kp/ki - 2*mu_p)

        F2_plus = self.__pt_kernel_f2(mu, ki, kj)
        F2_minus = self.__pt_kernel_f2(-mu, ki, kj)
        G2_plus = self.__pt_kernel_g2(mu, ki, kj)
        G2_minus = self.__pt_kernel_g2(-mu, ki, kj)
        
        res = 7/54 * F2_minus * (1 + km/ki * mu_m) \
            + 7/54 * F2_plus * (1 - kp/ki * mu_p) \
            + 4/54 * (G2_minus * beta_m + G2_plus * beta_p) \
            + 7/54 * (G2_minus * alpha_m + G2_plus * alpha_p)
        '''
        #if not ((np.fabs(ki-kj) < self.trispec_matter_klim)
        #        and (1-mu < self.trispec_matter_mulim)):
        kj_minus_ki_squared = ki**2 + kj**2 - 2*ki*kj*mu
        kj_plus_ki_squared = ki**2 + kj**2 + 2*ki*kj*mu
        if kj_minus_ki_squared == 0:
            beta_m = 1
            alpha_m = 1
        else:
            beta_m = kj**2.0*(ki*kj*mu-ki**2)/2.0/ki**2/kj_minus_ki_squared
            alpha_m = (kj**2-ki*kj*mu)/kj_minus_ki_squared
        if kj_plus_ki_squared == 0:
            beta_p = 1
            alpha_p = 1
        else:
            beta_p = kj**2*(-ki**2 -ki*kj*mu)/2.0/ki**2/kj_plus_ki_squared
            alpha_p = (kj**2+ki*kj*mu)/kj_plus_ki_squared
        F2_plus = self.__pt_kernel_f2(mu, ki, kj)
        F2_minus = self.__pt_kernel_f2(-mu, ki, kj)
        G2_plus = self.__pt_kernel_g2(mu, ki, kj)
        G2_minus = self.__pt_kernel_g2(-mu, ki, kj)
        res = 7/54 * F2_minus * kj/ki*mu \
            - 7/54 * F2_plus * kj/ki*mu \
            + 4/54 * (G2_minus * beta_m + G2_plus * beta_p) \
            + 7/54 * (G2_minus * alpha_m + G2_plus * alpha_p)

        return res
        #else:
        #    return 0.0

    def __pt_kernel_g2(self,
                       mu,
                       ki,
                       kj):
        """
        Auxiliary function to calculate the 4-halo term for the 
        trispectra. Is only needed by the __calc_int_for_trispec_4h 
        method.

        Parameters
        ----------
        See documentation of the __bispec_pt - method.

        Returns:
        --------
        float

        """
        return 3/7 + 4/7 * mu*mu + .5*mu * (ki/kj + kj/ki)

    def __trispectra_234h(self,
                          bias_dict,
                          hod_dict,
                          hm_prec):
        """
        Calculates the 234-halo terms for the trispectra. Uses the
        multiprocessing library to speed up the calculation with all
        available cores.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        trispec_2h + trispec_3h + trispec_4h : array
            with unit ?
            with shape (tri log10k bins,tri log10k bins)

        References
        ----------
        Pielorz et al. (2010) for the formalism of the 234-halo term
            following and fitting perturbation theory

        """

        k_dim = len(self.krange_tri)

        trispec_2h = np.zeros((k_dim, k_dim))
        trispec_3h = np.zeros((k_dim, k_dim))
        trispec_4h = np.zeros((k_dim, k_dim))

        halomod_integral_m_spline = \
            self.halo_model_integral_I_alpha_x_spline_loglog(
                bias_dict, hod_dict, hm_prec, 1, 'm')
        halomod_integral_mm_spline = \
            self.halo_model_integral_I_alpha_xy_spline_loglog(
                bias_dict, hod_dict, hm_prec, 1, 'm', 'm')
        halomod_integral_mmm_spline = \
            self.halo_model_integral_I_alpha_mmm_spline_loglog(
                bias_dict, hod_dict, hm_prec, 1)
        
        integral_m = 10**halomod_integral_m_spline[0](self.logks_tri)
        integral_mm = \
            10**halomod_integral_mm_spline[0][0](self.logks_tri,
                                                 self.logks_tri)
        integral_mmm = \
            10**halomod_integral_mmm_spline(self.logks_tri, self.logks_tri)

        phis = np.linspace(0, np.pi/2, 300)
        kidxlist = \
            list(itertools.combinations_with_replacement(np.arange(k_dim), 2))

        if self.int_2h is None:
            self.int_4h = np.zeros((len(self.krange_tri),len(self.krange_tri)))
            self.int_3h = np.zeros((len(self.krange_tri),len(self.krange_tri)))
            self.int_2h = np.zeros((len(self.krange_tri),len(self.krange_tri)))
            for i in range(len(self.krange_tri)):
                for j in range(i, len(self.krange_tri)):
                    ki = self.krange_tri[i]
                    kj = self.krange_tri[j]
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        try:
                            self.int_2h[i,j] = 2/np.pi \
                                * quad(self.__calc_int_for_trispec_2h, 0, .5*np.pi,
                                        args=(ki, kj), limit=50, epsrel=1e-05)[0]
                        except IntegrationWarning:
                            integrand = np.zeros_like(phis)
                            for idx, phi in enumerate(phis):
                                integrand[idx] = \
                                    self.__calc_int_for_trispec_2h(phi, ki, kj)
                            self.int_2h[i,j] = 2/np.pi * simpson(integrand, x = phis)
                        try:
                            self.int_3h[i,j] = 2/np.pi \
                                * quad(self.__calc_int_for_trispec_3h, 0, .5*np.pi,
                                        args=(ki, kj), limit=50, epsrel=1e-05)[0]
                        except IntegrationWarning:
                            integrand = np.zeros_like(phis)
                            for idx, phi in enumerate(phis):
                                integrand[idx] = \
                                    self.__calc_int_for_trispec_3h(phi, ki, kj)
                            self.int_3h[i,j] = 2/np.pi * simpson(integrand, x = phis)

                        try:
                            self.int_4h[i,j] = 1/np.pi \
                                * quad(self.__calc_int_for_trispec_4h, 0, np.pi,
                                        args=(ki, kj), limit=100, epsrel=1e-05)[0]    
                        except IntegrationWarning:
                            integrand = np.zeros_like(phis)
                            for idx, phi in enumerate(phis):
                                integrand[idx] = \
                                    self.__calc_int_for_trispec_4h(phi, ki, kj)
                            self.int_4h[i,j] = 2/np.pi * simpson(integrand, x = phis)  
                        self.int_4h[j,i] = self.int_4h[i,j]
                        self.int_3h[j,i] = self.int_3h[i,j]
                        self.int_2h[j,i] = self.int_2h[i,j]
                                 
            Dp = self.mass_func.growth_factor
            self.int_2h /= Dp**2
            self.int_3h /= Dp**4
            self.int_4h /= Dp**6
            
            
        
        
        
        Dp = self.mass_func.growth_factor
        self.int_2h *= Dp**2
        self.int_3h *= Dp**4
        self.int_4h *= Dp**6
        Pspline_eval = np.exp(self.Plin_spline(np.log(self.krange_tri)))
        trispec_2h = 2 * integral_mmm * integral_m[:, None] \
                * Pspline_eval[:, None]  \
                + 2 * integral_mmm * integral_m[None,:] \
                *  Pspline_eval[None, : ] \
                + integral_mm**2 * self.int_2h
        trispec_3h = \
                2 * integral_mm * integral_m[:, None] \
                * integral_m[None, :] * self.int_3h
        trispec_4h = \
                integral_m[:, None]**2 * integral_m[None, :]**2 * self.int_4h
        self.int_2h /= Dp**2
        self.int_3h /= Dp**4
        self.int_4h /= Dp**6
        
        for idxi, idxj in kidxlist:
            trispec_2h[idxj][idxi] = trispec_2h[idxi][idxj]
            trispec_3h[idxj][idxi] = trispec_3h[idxi][idxj]
            trispec_4h[idxj][idxi] = trispec_4h[idxi][idxj]

        if np.isnan(trispec_2h).any():
            print("TrispectraWarning: One or several elements in the 2-halo " +
                  "trispectrum term evaluated to nan. This might bias the " +
                  "result.")
            trispec_2h = np.nan_to_num(trispec_2h,
                                       nan=np.nanmean(trispec_2h))
        if np.isnan(trispec_3h).any():
            print("TrispectraWarning: One or several elements in the 3-halo " +
                  "trispectrum term evaluated to nan. This might bias the " +
                  "result.")
            trispec_3h = np.nan_to_num(trispec_3h,
                                       nan=np.nanmean(trispec_2h))
        if np.isnan(trispec_4h).any():
            print("TrispectraWarning: One or several elements in the 4-halo " +
                  "trispectrum term evaluated to nan. This might bias the " +
                  "result.")
            trispec_4h = np.nan_to_num(trispec_4h,
                                       nan=np.nanmean(trispec_4h))
        return trispec_2h + trispec_3h + trispec_4h

    def get_trispec_files(self,
                          output_dict,
                          bias_dict,
                          hod_dict,
                          prec,
                          zmax,
                          deltaz=0.5,
                          zmin=0):
        """
        Calculates and outputs the trispectra for all desired 
        combinations of matter ('m') and tracer ('g') for all redshifts 
        between zmin and zmax with deltaz steps in the k-range specified 
        in trispec_prec.

        Parameters
        ----------
        output_dict : dictionary
            Specifies whether a file for the trispectra should be 
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from 
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        zmax : float
            Maximum redshift for which the trispectrum should be 
            calculated.
        deltaz : float
            default : 0.5
            Step size for which trispectra are calculated. zmin is 
            increased until the redshift exceeds zmax.
        zmin : float
            default : 0
            Minimum redshift for which the trispectrum should be 
            calculated.

        """
        if output_dict['trispec'] == False:
            print("InputWarning: This function will output trispectra files " +
                  "as './trispectra_>z<.ascii', although the config file " +
                  "indicates [output settings]: 'save_trispectra = False'.")
            output_dict['trispec'] = './trispectra.ascii'

        zets = np.arange(zmin, zmax+deltaz, deltaz)
        for zet in zets:
            print('\rz=', round(zet, 3), '   ',
                  np.round(100*np.argwhere(zets == zet)[0][0]/len(zets), 1),
                  '%',
                  end=' ')
            self.update_mass_func(zet, bias_dict, hod_dict, prec)
            self.trispectra(output_dict, bias_dict, hod_dict, prec['hm'], None)

        print('\rtrispectra output files for z=[', round(zets[0], 3), ',',
              round(zets[-1], 3), ']   100%')
        output_dict['trispec'] = False

        return True
