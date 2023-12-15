import numpy as np
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy.special import sici, erf
from astropy import units as u   
import camb
from camb import model, initialpower


import hmf
try:
    from onecov.cov_setup import Setup
    from onecov.cov_hod import HOD
except:
    from cov_setup import Setup
    from cov_hod import HOD


class HaloModel(Setup):
    """
    This class calculates the necessary ingredients of the halo model 
    which are later used to calculate the covariance. Important 
    quantities are the mass function, the galaxy bias and the halo model 
    integrals. All quantities are calculated at a single redshift. 
    Inherits the functionality of the Setup class and contains instances 
    of the hmf and hod class.

    Atrributes
    ----------
    zet : float
        Redshift at which the covariance should be evaluated.
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

    Private Variables
    -----------------
    see Setup class
    mass_func : class
        This class specifies the mass function. It originates from the 
        hmf module.
    hod : class 
        This class specifies the halo occupation distribution. It 
        originates from the cov_hod module.
    ngal : array
        with unit 1 / (Mpc/h)^3
        with shape (sample_bins)
        average number density of tracers given an HOD
    ncen : array
        with unit 1 / (Mpc/h)^3
        with shape (sample_bins)
        average number density of central galaxies given an HOD
    nsat : array
        with unit 1 / (Mpc/h)^3
        with shape (sample_bins)
        average number density of satellite galaxies given an HOD
    norm_bias : float
        normalization constant for the bias model
    effective_bias : array
        with shape (sample_bins)
        effective bias for tracers assuming an HOD

    Example
    -------
    from cov_input import Input, FileInput
    from cov_halo_model import HaloModel
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    zet = 0
    hm = HaloModel(zet, cosmo, bias, hod, survey_params, prec,
                   read_in_tables)

    """

    def __init__(self,
                 zet,
                 cosmo_dict,
                 bias_dict,
                 hod_dict,
                 survey_params_dict,
                 prec,
                 read_in_tables):
        Setup.__init__(self,
                       cosmo_dict,
                       bias_dict,
                       survey_params_dict,
                       prec,
                       read_in_tables)
        
        self.mass_func = \
            self.calc_mass_func(zet, cosmo_dict, prec['hm'], prec['powspec'])
        self.hod = HOD(bias_dict, prec['hm'])
        self.ngal = self.nbar(hod_dict)
        self.ncen = self.nbar_cen(hod_dict)
        self.nsat = self.nbar_sat(hod_dict)
        self.norm_bias = -1
        self.effective_bias = self.calc_effective_bias(
            bias_dict, hod_dict, prec['hm'])
        self.__set_spline_galaxy_stellar_mf(hod_dict)
        self.__set_spline_galaxy_stellar_mf_bias(hod_dict, bias_dict, prec['hm'])
        

    def calc_mass_func(self,
                       zet,
                       cosmo_dict,
                       hm_prec,
                       powspec_prec):
        """
        Calculates the mass function from the hmf module for a given 
        cosmology at the redshift 'zet'.

        Parameters 
        ----------
        zet : float
            Redshift at which the covariance should be evaluated.
        cosmo_dict : dictionary
            Specifies all cosmological parameters. To be passed from the 
            read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        powspec_prec : dictionary
            Contains precision information about the power spectra, this 
            includes k-range and spacing. To be passed from the 
            read_input method of the Input class.

        Returns
        -------
            mass_func : class

        """


        self.camb_pars = camb.CAMBparams()
        self.camb_pars.set_cosmology(H0=100*self.cosmology.h, 
                            ombh2=self.cosmology.h**2*self.cosmology.Ob0,
                            omch2=self.cosmology.h**2*(self.cosmology.Om0-self.cosmology.Ob0),
                            omk = 0.0,
                            mnu= cosmo_dict['m_nu'])
        self.camb_pars.set_dark_energy(w=self.cosmology.w0, wa=self.cosmology.wa, dark_energy_model='fluid') 
        self.camb_pars.InitPower.set_params(ns=cosmo_dict['ns'],
                                    As = 1.8e-9)
        self.camb_pars.set_matter_power(kmax=1000, redshifts = [0])
        self.camb_pars.NonLinear = model.NonLinear_none

        mstep = (hm_prec['log10M_max'] - hm_prec['log10M_min']) \
            / (hm_prec['M_bins'] - .5)
        kstep = (np.log(10**powspec_prec['log10k_max'])
                 - np.log(10**powspec_prec['log10k_min'])) \
            / powspec_prec['log10k_bins']

        transfmodel = {
                       'sigma_8': cosmo_dict['sigma8'],
                       'n': cosmo_dict['ns'],
                       'lnk_min': np.log(10**powspec_prec['log10k_min']),
                       'lnk_max': np.log(10**powspec_prec['log10k_max']),
                       'dlnk': kstep,
                       'takahashi': False}
        self.transfmodel = transfmodel
        
        prefix = 'hmf.mass_function.fitting_functions.'
        try:
            hm_prec_model = eval(prefix+hm_prec['hmf_model'])
        except AttributeError:
            raise Exception("ConfigError: hmf does not support the fitting " +
                            "function " + hm_prec['hmf_model'] + ". Available functions " +
                            "can be found at https://hmf.readthedocs.io/en/latest/" +
                            "_autosummary/hmf.mass_function.fitting_functions.html .")
        

        mass_func = hmf.MassFunction(
            Mmin=hm_prec['log10M_min'],
            Mmax=hm_prec['log10M_max'],
            dlog10m=mstep,
            hmf_model=hm_prec_model,
            mdef_model=hm_prec['mdef_model'],
            mdef_params=hm_prec['mdef_params'],
            disable_mass_conversion=hm_prec['disable_mass_conversion'],
            delta_c=hm_prec['delta_c'],
            cosmo_model=self.cosmology,
            z=zet,
            transfer_model=hm_prec['transfer_model'],
            transfer_params = {'extrapolate_with_eh':False,'camb_params':self.camb_pars},
            **transfmodel)

        self.camb_pars_new = self.camb_pars.copy()

        hm_prec['M_bins'] = len(mass_func.m)

        return mass_func

    def nbar(self,
             hod_dict):
        """
        Calculates the average number density of tracers given an HoD
        description, i.e., the integral of the mass function over the 
        occupation number of satellite and central galaxies.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.

        Returns
        -------
        nbar : array
            with unit 1 / (Mpc/h)^3
            with shape (sample_bins)

        """
        return np.trapz(self.mass_func.dndm
                        * self.hod.occ_num_and_prob(
                            hod_dict,
                            self.mor_tab,
                            self.occprob_tab,
                            self.occnum_tab
                        )[0],
                        self.hod.Mrange[None, :])

    def nbar_cen(self,
                 hod_dict):
        """
        Calculates the average number density of central galaxies given 
        an HoD description, i.e., the integral of the mass function over 
        the occupation number of central galaxies.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.

        Returns
        -------
        nbar_cen : array
            with unit 1 / (Mpc/h)^3
            with shape (sample_bins)
        """
        return self.nbar(hod_dict) - self.nbar_sat(hod_dict)

    def nbar_sat(self,
                 hod_dict):
        """
        Calculates the average number density of satellite galaxies 
        given an HoD description, i.e., the integral of the mass 
        function over the occupation number of satellite galaxies.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.

        Returns
        -------
        nbar_sat : array
            with unit 1 / (Mpc/h)^3
            with shape (sample_bins)
        """
        return np.trapz(self.mass_func.dndm
                        * self.hod.occ_num_and_prob_per_pop(
                            hod_dict,
                            'sat',
                            self.mor_tab,
                            self.occprob_tab,
                            self.occnum_tab
                        )[0],
                        self.hod.Mrange[None, :])

    def __bias_tinker10_fittfunc(self,
                                 nu):
        """
        Evaluates the fitting function for the bias at the peak height, 
        nu.

        Parameters
        ----------
        nu : float
            The height of the peak in units of the variance.

        Returns
        -------
        tinker : float
            Fitting function evaluated at nu.

        References
        ----------
        Tinker et al. (2010)

        """
        y = np.log10(self.mass_func.mdef_params['overdensity'])
        A = 1 + 0.24*y*np.exp(-(4/y)**4)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4/y)**4)
        c = 2.4

        tinker = 1 \
            - A * nu**a / (nu**a + self.mass_func.delta_c**a) \
            + B * nu**b \
            + C * nu**c

        return tinker

    def bias(self,
             bias_dict,
             hm_prec):
        """
        Implements the bias model as a function of mass specified in the 
        bias dictionary with the following options:
            (i) 'Tinker10' empirical model (see 
                __bias_tinker10_fittfunc)
            (ii) shame we didn't do more yet 
        All bias models are normalized, i.e., the integral over the peak 
        height is required to be unity.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        tinker : array
            with shape (M_bins)

        """

        if bias_dict['model'] == 'Tinker10':
            step_save = self.mass_func.dlog10m
            nu_save = self.mass_func.nu**0.5

            norm_Mmin = 2.
            norm_Mbins = 500.

            Mmin_new = norm_Mmin
            step_new = (self.mass_func.Mmax - Mmin_new) / norm_Mbins
            self.mass_func.update(Mmin=Mmin_new, dlog10m=step_new)
            nu_new = self.mass_func.nu**0.5
            if self.norm_bias == -1:
                self.norm_bias = \
                    np.trapz(self.mass_func.fsigma
                             / nu_new
                             * self.__bias_tinker10_fittfunc(nu_new),
                             nu_new)
            spline_tinker = \
                UnivariateSpline(nu_new, self.__bias_tinker10_fittfunc(nu_new),
                                 k=2, s=0, ext=0)
            tinker = spline_tinker(nu_save)
            self.mass_func.update(Mmin=hm_prec['log10M_min'],
                                  dlog10m=step_save)

            return tinker / self.norm_bias
        else:
            raise Exception("For now the only valid bias model is " +
                            "'Tinker10', sorry for the inconvenience, work in progress.")
            return False

    def calc_effective_bias(self,
                            bias_dict,
                            hod_dict,
                            hm_prec):
        """
        Calculates the effective bias of tracers.

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
        bias : array
            with shape (sample_bins)
 
        """

        if self.effbias_tab['bias'] is None:
            occ_num = self.hod.occ_num_and_prob(
                hod_dict,
                self.mor_tab,
                self.occprob_tab,
                self.occnum_tab
            )[0]
            integral = np.trapz(self.mass_func.dndm
                                * occ_num
                                * self.bias(bias_dict, hm_prec),
                                self.mass_func.m)

            bias = integral / self.ngal
        else:
            bias = np.zeros(self.sample_dim)
            for mbin in range(self.sample_dim):
                bias_2dspline = UnivariateSpline(
                    self.effbias_tab['z'],
                    self.effbias_tab['bias'][mbin, :], k=2, s=0)
                bias[mbin] = bias_2dspline(self.mass_func.z)
        return bias

    def uk(self,
           Mc_relation,
           type = 'cen'):
        """
        Calculates the normalized Fourier transform of the NFW density
        profile. Requires a mass-concentration relation and the
        definition of 'overdensity' (default : 200, can be adjusted in
        [halomodel evaluation]: 'mdef_params'.

        Parameters
        ----------
        Mc_relation : string
            mass-concentration relation

        Returns
        -------
        u_k : array
            with shape(log10k_bins, M_bins)

        """

        overdensity = self.mass_func.mdef_params['overdensity']
        con = self.__concentration(Mc_relation)
        if type == 'sat':
            con *= 0.6289028827810339
        if type == 'halo':
            con *= 0.9841
        deltac = overdensity * con**3 / (3 * (np.log(1+con) - con/(1+con)))

        rvir = self.__virial_radius()
        rs = rvir / con
        # rsk[idx] gives all rs's for each k
        rsk = np.outer(self.mass_func.k, rs)
        bsin, bcos = sici(rsk)
        asin, acos = sici((1+con) * rsk)

        u_k = 4*np.pi * self.rho_bg * deltac * rs**3 / self.mass_func.m \
            * (np.sin(rsk) * (asin-bsin) - np.sin(con*rsk) / ((1+con)*rsk)
               + np.cos(rsk) * (acos - bcos))

        return u_k

    def __virial_radius(self):
        """
        Calculates the virial radius given a halo overdensity 
        definition.

        Returns
        -------
        virial_radius : array
            with shape(M_bins)

        """

        denom = 4 * np.pi * self.rho_bg * \
            self.mass_func.mdef_params['overdensity']

        return (3 * self.mass_func.m / denom)**(1/3)

    def __concentration(self,
                        Mc_relation):
        """
        Calculates a mass-concentration relation for a given model with
        the following options
            (i) 'duffy08' : Duffy et al. 2008
            (II) 'flat' : concentration is one
            (ii) all this laziness...

        Parameters
        ----------
        Mc_relation : string
            mass-concentration relation

        Returns
        -------
        con : array
            with shape(M_bins)

        """
        if Mc_relation == 'duffy08':
            con = 10.14 / (self.mass_func.m/2e12)**0.081 \
                / (1 + self.mass_func.z)**1.01
        elif Mc_relation == 'flat':
            con = 1
        else:
            raise Exception("ConfigError: The mass-concentration relation " +
                            Mc_relation + " is not implemented (yet). Available options " +
                            "are: 'duffy08', 'flat'.")
        return con

    def small_k_damping(self,
                        mode,
                        krange,
                        scale=.1):
        """
        Function to exponentially damp the power of the 1-halo term on 
        large scales / small wavenumbers.

        Parameters
        ----------
        mode : string
            Can be either 'damped' or 'none' for damping and no damping,
            respectively.

        Returns
        -------
        erf(k) : array
            with shape (len(krange))
            for 'damped'
        1(k) : array
            with shape (len(krange))
            for 'none'

        """
        if mode == 'damped':
            return erf(krange / scale)
        elif mode == 'none':
            return np.ones_like(krange)
        else:
            raise Exception("KeyError: The mode in [halomodel (or) trispec " +
                            "evaluation]: 'small_k_damping_for1h' must be either " +
                            "'damped' or 'none'.")

    def hurly_x(self,
                bias_dict,
                hod_dict,
                type_x):
        """
        Calculates quantities for the halo model.

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
        type_x : string
            - 'm' for matter (Eq. 23 in reference paper)
            - 'cen' for centrals (Eq. 24 in reference paper)
            - 'sat' for satellites (Eq. 25 in reference paper)

        Returns
        -------
        hurly_x : array
            with shape (log10k_bins, sample_bins, M_bins)

        References
        ----------
        Dvornik et al. (2018), their Sect 3.1.
            Note: The normalization is different, we use n_gal for all
                  cases.

        """

        # if type == 'sat'
        uk = self.uk(bias_dict['Mc_relation_sat'], 'sat')
        norm = self.ngal
        pop = self.hod.occ_num_and_prob_per_pop(
            hod_dict,
            'sat',
            self.mor_tab,
            self.occprob_tab,
            self.occnum_tab
        )[0]
        if (type_x == 'cen'):
            uk = np.ones_like(uk)
            #norm = self.ncen
            pop = self.hod.occ_num_and_prob_per_pop(
                hod_dict,
                'cen',
                self.mor_tab,
                self.occprob_tab,
                self.occnum_tab
            )[0]
        if (type_x == 'm'):
            uk = self.uk(bias_dict['Mc_relation_cen'],'halo')
            norm = np.ones_like(norm) * self.rho_bg
            pop = self.mass_func.m[None, :]
        return (uk[:, None, :]*pop[None, :, :]) / norm.T[None, :, None]

    def hurly_x_spline_logk(self,
                            bias_dict,
                            hod_dict,
                            type_x):
        """
        Calculates a spline of log10(k) and the halo model quantity 
        hurly_x which is needed to calculate the trispectrum.

        Parameters
        ----------
        See documentation of the hurly_x - method.

        Returns
        -------
        hurly_x : nested list of UnivariateSplines
            with shape (sample_bins, M_bins)

        """
        hurlyX = self.hurly_x(bias_dict, hod_dict, type_x)

        hurly_shape = hurlyX.shape
        hurlyX_spline = [[] for _ in range(hurly_shape[1])]
        for nbin in range(hurly_shape[1]):
            for idxM in range(hurly_shape[2]):
                hurlyX_spline[nbin].append(UnivariateSpline(
                    np.log10(self.mass_func.k),
                    hurlyX[:, nbin, idxM], s=0, ext=0))

        return hurlyX_spline

    def halo_model_integral_I_alpha_x(self,
                                      bias_dict,
                                      hod_dict,
                                      hm_prec,
                                      alpha,
                                      type_x):
        """
        Calculates quantities for the halo model.

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
        alpha : int
            Auxillary label
        type_x : string
            - "m" for matter
            - "g" for tracers

        Returns
        -------
        halo_model_integral_I_alpha_x : array
            with shape (log10k_bins, sample_bins)

        References
        ---------
        Dvornik et al. (2018), their Appendix A, Eq. (A15)

        """
        if alpha == 0:
            integral_x = 0
        else:
            integral_x = 1.0
            if type_x == 'g':
                hurlyX = \
                    self.hurly_x(bias_dict, hod_dict, 'cen') \
                    + self.hurly_x(bias_dict, hod_dict, 'sat')
                bias = self.bias(bias_dict, hm_prec) * bias_dict['bias_2h']
                integral_x = np.trapz(self.mass_func.dndm
                                      * bias
                                      * hurlyX,
                                      self.mass_func.m)
            if type_x == 'm':
                M_min_save = hm_prec["log10M_min"]
                step_save = self.mass_func.dlog10m
                Mmin = 2.0
                hm_prec["log10M_min"] = Mmin
                step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.mass_func.update(Mmin=Mmin, dlog10m=step)
                self.hod.hod_update(bias_dict, hm_prec)

                hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
                bias = self.bias(bias_dict, hm_prec)
                integral_x = np.trapz(
                    self.mass_func.dndm * hurlyX * bias, self.mass_func.m)

                hm_prec["log10M_min"] = M_min_save
                self.mass_func.update(Mmin=M_min_save, dlog10m=step_save)
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.hod.hod_update(bias_dict, hm_prec)

        return integral_x

    def halo_model_integral_I_alpha_x_spline_loglog(self,
                                                    bias_dict,
                                                    hod_dict,
                                                    hm_prec,
                                                    alpha,
                                                    type_x):
        """
        Calculates a spline of log10(k) and the log10 halo model 
        quantity I_alpha_x which is needed to calculate the trispectrum.

        Parameters
        ----------
        See documentation of the halo_model_integral_I_alpha_x - method.

        Returns
        -------
        integralX_spline : list of UnivariateSplines
            with shape (sample_bins)

        """

        integralX = np.log10(self.halo_model_integral_I_alpha_x(bias_dict,
                                                                hod_dict,
                                                                hm_prec,
                                                                alpha,
                                                                type_x))
        integralX_shape = integralX.shape

        integralX_spline = []
        for nbin in range(integralX_shape[1]):
            integralX_spline.append(
                UnivariateSpline(np.log10(self.mass_func.k),
                                 integralX[:, nbin],
                                 k=1, s=0, ext=0))

        return integralX_spline

    def halo_model_integral_I_alpha_xy(self,
                                       bias_dict,
                                       hod_dict,
                                       hm_prec,
                                       alpha,
                                       type_x,
                                       type_y):
        """
        Calculates quantities for the halo model.

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
        alpha : int
            Auxillary label
        type_x : string
            - "m" for matter
            - "g" for tracers
        type_y : string
            - "m" for matter
            - "g" for tracers

        Returns
        -------
        integral_xy : array
            with shape (log10k_bins, log10k_bins, 
                        sample_bins, sample_bins)

        References
        ----------
        Dvornik et al. (2018), their Appendix A, Eq. (A15)

        """

        if alpha == 0:
            integral_xy = 0
        else:
            integral_xy = 1
            hurlyX = 1
            hurlyY = 1
            correct = 0
            bias = 1

            if type_x == 'g':
                hurlyX = \
                    self.hurly_x(bias_dict, hod_dict, 'cen') \
                    + self.hurly_x(bias_dict, hod_dict, 'sat')
                bias = self.bias(bias_dict, hm_prec) * bias_dict['bias_2h']
            elif type_x == 'm':
                hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')

            if (type_y == 'g'):
                hurlyY = \
                    self.hurly_x(bias_dict, hod_dict, 'cen') \
                    + self.hurly_x(bias_dict, hod_dict, 'sat')
                bias = self.bias(bias_dict, hm_prec) * bias_dict['bias_2h']
            elif type_y == 'm':
                hurlyY = self.hurly_x(bias_dict, hod_dict, 'm')
                bias = self.bias(bias_dict, hm_prec)

            if type_x == 'g' and type_y == 'g':
                correct = \
                    self.hurly_x(
                        bias_dict, hod_dict, 'cen')[:, None, :,  None, :] \
                    * self.hurly_x(
                        bias_dict, hod_dict, 'cen')[None, :, None, :, :]
                bias = self.bias(bias_dict, hm_prec)

            if type_x == 'm' and type_y == 'm':
                M_min_save = hm_prec["log10M_min"]
                step_save = self.mass_func.dlog10m
                Mmin = 2.0
                hm_prec["log10M_min"] = Mmin
                step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.mass_func.update(Mmin=Mmin, dlog10m=step)
                self.hod.hod_update(bias_dict, hm_prec)

                hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
                bias = self.bias(bias_dict, hm_prec)
                integral_xy = np.trapz(self.mass_func.dndm
                                       * hurlyX[:, None, :,  None, :]
                                       * hurlyX[None, :, None, :, :]
                                       * bias[None, None, None, None, :],
                                       self.mass_func.m)

                hm_prec["log10M_min"] = M_min_save
                self.mass_func.update(Mmin=M_min_save, dlog10m=step_save)
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.hod.hod_update(bias_dict, hm_prec)
            else:
                integral_xy = np.trapz(self.mass_func.dndm
                                       * (hurlyX[:, None, :,  None, :]
                                          * hurlyY[None, :, None, :, :]
                                          - correct)
                                       * bias[None, None, None, None, :],
                                       self.mass_func.m)

        return integral_xy

    def halo_model_integral_I_alpha_xy_spline_loglog(self,
                                                     bias_dict,
                                                     hod_dict,
                                                     hm_prec,
                                                     alpha,
                                                     type_x,
                                                     type_y):
        """
        Calculates a spline of log10(k) and the log10 halo model 
        quantity I_alpha_xy which is needed to calculate the 
        trispectrum.

        Parameters
        ----------
        See documentation of the halo_model_integral_I_alpha_xy - 
        method.

        Returns
        -------
        integralXY_spline : nested list of UnivariateSplines
            with shape (sample_bins, sample_bins)

        """

        integralXY = np.log10(self.halo_model_integral_I_alpha_xy(bias_dict,
                                                                  hod_dict,
                                                                  hm_prec,
                                                                  alpha,
                                                                  type_x,
                                                                  type_y))

        integralXY_shape = integralXY.shape
        integralXY_spline = []
        for nbin in range(integralXY_shape[2]):
            integralXY_spline.append([])
            for mbin in range(integralXY_shape[3]):
                integralXY_spline[nbin].append(RectBivariateSpline(
                    np.log10(self.mass_func.k),
                    np.log10(self.mass_func.k),
                    integralXY[:, :, nbin, mbin],
                    kx=1, ky=1, s=0))

        return integralXY_spline

    def halo_model_integral_I_alpha_mmm(self,
                                        bias_dict,
                                        hod_dict,
                                        hm_prec,
                                        alpha):
        """
        Calculates quantities for the halo model which is needed to 
        calculate the trispectrum.

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
        alpha : int
            Auxillary label

        Returns
        -------
        integral_mmm : array
            with shape (log10k_bins, log10k_bins, sample_bins)

        Reference :
        -----------
        Takada ? checkme
        Dvornik et al. (2018), their Appendix A, Eq. (A15)

        """

        if alpha == 0:
            integral_mmm = 0
        else:
            M_min_save = hm_prec["log10M_min"]
            step_save = self.mass_func.dlog10m
            Mmin = 2.0
            hm_prec["log10M_min"] = Mmin
            step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.mass_func.update(Mmin=Mmin, dlog10m=step)
            self.hod.hod_update(bias_dict, hm_prec)

            hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
            bias = self.bias(bias_dict, hm_prec)
            integral_mmm = np.trapz(self.mass_func.dndm[None, None, None, :]
                                    * hurlyX[:, None, :, :]
                                    * hurlyX[None, :, :, :]**2.0
                                    * bias[None, None, None, :],
                                    self.mass_func.m)

            hm_prec["log10M_min"] = M_min_save
            self.mass_func.update(Mmin=M_min_save, dlog10m=step_save)
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.hod.hod_update(bias_dict, hm_prec)

        return integral_mmm

    def halo_model_integral_I_alpha_mmm_spline_loglog(self,
                                                      bias_dict,
                                                      hod_dict,
                                                      hm_prec,
                                                      alpha):
        """
        Calculates a 2d spline of log10(k), log10(k) and the log10 halo 
        model quantity I_alpha_mmm which is needed to calculate the
        trispectrum.

        Parameters
        ----------
        See documentation of the halo_model_integral_I_alpha_mmm - 
        method.

        Returns
        -------
        integralmmm_spline : RectBivariateSpline

        """

        integralmmm = np.log10(self.halo_model_integral_I_alpha_mmm(bias_dict,
                                                                    hod_dict,
                                                                    hm_prec,
                                                                    alpha))

        return RectBivariateSpline(
            np.log10(self.mass_func.k),
            np.log10(self.mass_func.k),
            integralmmm[:, :, 0],
            kx=1, ky=1, s=0)

    def galaxy_stellar_mf(self,
                          hod_dict,
                          type):
        """
        Calculates the stellar mass function for centrals and satellites
        
        Parameters
        ----------
        type : string
            can be 'sat' or 'cen' for satellites and centrals respectively
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        
        Returns
        -------
        smf : array 
            the galaxy stellar mass with shape (sample bins, 300)

        """
        if type == 'sat':
            csmf_sat = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'sat',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return np.trapz(self.mass_func.dndm[None, None, :]*csmf_sat, self.mass_func.m, axis = -1)
        if type == 'cen':
            csmf_sat = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'cen',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return np.trapz(self.mass_func.dndm[None, None, :]*csmf_sat, self.mass_func.m, axis = -1)
    
    def galaxy_stellar_mf_bias(self,
                               hod_dict,
                               bias_dict,
                               hm_prec,
                               type):
        """
        Calculates the stellar mass function for centrals and satellites times
        the halo bias (phitilde)
        
        Parameters
        ----------
        type : string
            can be 'sat' or 'cen' for satellites and centrals respectively
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        
        Returns
        -------
        smf : array 
            the galaxy stellar mass with shape (sample bins, 300)

        """
        if type == 'sat':
            csmf_sat = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'sat',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return np.trapz((self.bias(bias_dict,hm_prec)*self.mass_func.dndm)[None, None, :]*csmf_sat, self.mass_func.m, axis = -1)
        if type == 'cen':
            csmf_sat = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'cen',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return np.trapz((self.bias(bias_dict,hm_prec)*self.mass_func.dndm)[None, None, :]*csmf_sat, self.mass_func.m, axis = -1)

    def __set_spline_galaxy_stellar_mf(self,
                                       hod_dict):
        """
        Sets up the splines for the galaxy stellar mass function as private variables
        of the halo_model class for both satellites and centrals

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        """
        aux_M = np.zeros((len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]) + len(self.hod.Mbins[0,:]))
        aux_smf_s = self.galaxy_stellar_mf(hod_dict, 'sat')
        aux_smf_c = self.galaxy_stellar_mf(hod_dict, 'cen')
        aux_s = np.zeros_like(aux_M)
        aux_c = np.zeros_like(aux_M)
        for i_bins in range(len(self.hod.Mbins[:,0]) - 1):
            aux_M[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = self.hod.Mbins[i_bins,:-1]
            aux_c[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = aux_smf_c[i_bins,:-1]
            aux_s[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = aux_smf_s[i_bins,:-1]
        aux_M[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = self.hod.Mbins[len(self.hod.Mbins[:,0]) - 1,:]
        aux_c[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = aux_smf_c[len(self.hod.Mbins[:,0]) - 1,:]
        aux_s[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = aux_smf_s[len(self.hod.Mbins[:,0]) - 1,:]
        self.stellar_mass = aux_M
        self.galaxy_smf_c = UnivariateSpline(np.array(aux_M), np.array(aux_c) , k=2, s=0, ext=0)
        self.galaxy_smf_s = UnivariateSpline(np.array(aux_M), np.array(aux_s) , k=2, s=0, ext=0)

    def __set_spline_galaxy_stellar_mf_bias(self,
                                            hod_dict,bias_dict, hm_prec):
        """
        Sets up the splines for the galaxy stellar mass function as private variables
        of the halo_model class for both satellites and centrals

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        """
        aux_M = np.zeros((len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]) + len(self.hod.Mbins[0,:]))
        aux_smf_s = self.galaxy_stellar_mf_bias(hod_dict,bias_dict, hm_prec, 'sat')
        aux_smf_c = self.galaxy_stellar_mf_bias(hod_dict,bias_dict, hm_prec, 'cen')
        aux_s = np.zeros_like(aux_M)
        aux_c = np.zeros_like(aux_M)
        for i_bins in range(len(self.hod.Mbins[:,0]) - 1):
            aux_M[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = self.hod.Mbins[i_bins,:-1]
            aux_c[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = aux_smf_c[i_bins,:-1]
            aux_s[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = aux_smf_s[i_bins,:-1]
        aux_M[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = self.hod.Mbins[len(self.hod.Mbins[:,0]) - 1,:]
        aux_c[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = aux_smf_c[len(self.hod.Mbins[:,0]) - 1,:]
        aux_s[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = aux_smf_s[len(self.hod.Mbins[:,0]) - 1,:]
        self.galaxy_smf_bias_c = UnivariateSpline(np.array(aux_M), np.array(aux_c) , k=2, s=0, ext=0)
        self.galaxy_smf_bias_s = UnivariateSpline(np.array(aux_M), np.array(aux_s) , k=2, s=0, ext=0)        

    def conditional_galaxy_stellar_mf(self,
                                      hod_dict,
                                      type):
        """
        Calculates the conditional stellar mass function for centrals and satellites
        
        Parameters
        ----------
        type : string
            can be 'sat' or 'cen' for satellites and centrals respectively
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        
        Returns
        -------
        csmf_type : array 
            the galaxy stellar mass with shape (300, M_bins)

        """
        if type == 'sat':
            return self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                     'sat',
                                                      self.mor_tab,
                                                      self.occprob_tab,
                                                      self.occnum_tab)[1][0,:,:]
        if type == 'cen':
            return self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                     'cen',
                                                      self.mor_tab,
                                                      self.occprob_tab,
                                                      self.occnum_tab)[1][0,:,:]
        
    def count_matter_bispectrum(self,
                                bias_dict,
                                hm_prec):
        """
        Calculates the three dimensional count-matter density cross-bispectrum
        for a collapsed triange at the wavenumbers specified in log10kbins
        and redshift specified in the class. This is used for later computation
        of the cross-correlation between the conditional stellar mass function and
        2pt statistics

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        Returns
        -------
        nBcmm_mu : array
            with shape (log10k bins, 
                        sample bins, sample bins)

        References
        ----------
        Takada and Bridle 2007,  New Journal of Physics, 9, 446

        """
        halo_profile = self.uk(bias_dict['Mc_relation_cen'])
        halo_bias = self.bias(bias_dict,hm_prec)
        csmf = self.conditional_galaxy_stellar_mf('cen')
        term1 = self.mass_func.dndm[None, None,:]*csmf[None, :, :]*(self.mass_func.m**2)[None, None,:]/self.rho_bg**2*(halo_profile**2)[:, None, :]
        term2 = self.mass_func.dndm[None, None,:]*csmf[None, :, :]*(self.mass_func.m)[None, None,:]/self.rho_bg*(halo_profile)[:, None, :]*halo_bias[None,None,:]
        term3 = self.mass_func.dndm[None,:]*(self.mass_func.m)[None,:]/self.rho_bg*(halo_profile)[:, :]*halo_bias[None,:]

        I1 = np.trapz(term1, self.mass_func.m, axis=-1)
        I2 = np.trapz(term2, self.mass_func.m, axis=-1) * np.trapz(term3, self.mass_func.m, axis=-1)[:, None]

        return I1 + 2.0 * self.mass_func.power[:, None] * I2
    
    def get_count_matter_bispectrum(self,
                                    bias_dict,
                                    hm_prec,
                                    log10csmf_mass_bins):
        """
        Sets the splines for the matter count matter bispectrum

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.
        obs_dict : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
        """

        aux_M = np.zeros((len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]) + len(self.hod.Mbins[0,:]))
        aux_bispec_count_mm = self.count_matter_bispectrum(bias_dict, hm_prec)
        aux_bispec = np.zeros((len(aux_M)),self.mass_func.k)
        for i_bins in range(len(self.hod.Mbins[:,0]) - 1):
            aux_M[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1])] = self.hod.Mbins[i_bins,:-1]
            aux_bispec[i_bins*len(self.hod.Mbins[0,:-1]) : (i_bins+1)*len(self.hod.Mbins[0,:-1]), :] = aux_bispec_count_mm[:, i_bins,:-1]
        aux_M[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = self.hod.Mbins[len(self.hod.Mbins[:,0]) - 1,:]
        aux_bispec[(len(self.hod.Mbins[:,0]) - 1)*len(self.hod.Mbins[0,:-1]):] = aux_bispec_count_mm[:, len(self.hod.Mbins[:,0]) - 1,:]
        count_matter_bispec = np.zeros((len(self.mass_func.k), len(log10csmf_mass_bins)))
        for i_k in range(len(self.mass_func.k)):
            count_matter_bispec[i_k, :] = np.exp(np.intep(log10csmf_mass_bins,np.log10(aux_M), np.log(aux_bispec)))
        return count_matter_bispec