import copy

import camb
import hmf
import numpy as np
from camb import model
from hmf.density_field import transfer_models as hmf_transfer_models
from hmf._internals._framework import Component
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d, make_interp_spline
from scipy.special import erf, sici

try:
    from onecov.cov_hod import HOD
    from onecov.cov_setup import Setup
except:
    from cov_hod import HOD
    from cov_setup import Setup

_camb_transfer_cache = {}

# maximum wavenumber in h/Mpc up to which CAMB calculates the linear transfer
# function for the halo model, hmf extrapolates ln T linearly in ln k beyond
# (extrapolate_with_eh = False). Changes the non-Gaussian and super-sample
# covariance by ~0.3% (within CAMB's own sampling noise at high k), the total
# covariance by ~2e-4 in correlation units.
CAMB_KMAX_LINEAR = 100.


def camb_transfer_functions(params):
    """
    camb.get_transfer_functions, but CAMB is only run once per distinct set
    of parameters within a run (the transfer functions are only read).
    """
    key = repr(params)
    if key not in _camb_transfer_cache:
        _camb_transfer_cache[key] = camb.get_transfer_functions(params)
    return _camb_transfer_cache[key]


class _CAMBWithCachedTransfers:
    """
    Stands in for the camb module in hmf's transfer models, so that hmf
    does not rerun CAMB for transfer functions it has already calculated.
    """

    def __getattr__(self, name):
        return getattr(camb, name)

    @staticmethod
    def get_transfer_functions(params):
        return camb_transfer_functions(params)


hmf_transfer_models.camb = _CAMBWithCachedTransfers()

_simpson_weights_cache = {}


def simpson_weights(x):
    """
    Weights w such that simpson(y, x=x, axis=-1) = y @ w for any y (Simpson's
    rule is linear in y). Cached per grid.
    """
    key = x.tobytes()
    if key not in _simpson_weights_cache:
        _simpson_weights_cache[key] = simpson(np.eye(len(x)), x=x, axis=-1)
    return _simpson_weights_cache[key]



def _copy_mass_function(mass_func):
    """
    Deep copy of an hmf MassFunction that shares its components (transfer
    and growth model, filter, ...), the CAMB parameters and the CAMB results
    with the original. hmf replaces a component instead of changing it when
    its parameters change, so sharing is safe; copying is not: the CAMB
    transfer model only keeps part of its parameters when copied, and CAMB
    results cannot be copied at all.
    """
    memo = {}
    for value in mass_func.__dict__.values():
        if isinstance(value, Component):
            memo[id(value)] = value
    camb_params = (mass_func.transfer_params or {}).get('camb_params')
    if camb_params is not None:
        memo[id(camb_params)] = camb_params
    dispatch = copy._deepcopy_dispatch
    previous = dispatch.get(camb.CAMBdata)
    dispatch[camb.CAMBdata] = lambda obj, memo: obj
    try:
        return copy.deepcopy(mass_func, memo)
    finally:
        if previous is None:
            del dispatch[camb.CAMBdata]
        else:
            dispatch[camb.CAMBdata] = previous

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
        self.hod = HOD(bias_dict, prec['hm'], self.mass_func.m)
        self.ngal = self.nbar(hod_dict)
        self.ncen = self.nbar_cen(hod_dict)
        self.nsat = self.nbar_sat(hod_dict)
        self.effective_bias = self.calc_effective_bias(
            bias_dict, hod_dict, prec['hm'])
        if self.hod.mass_bins_disagree:
            saved_bins = self.__widen_mass_bins(bias_dict)
            self.hod.hod_update(bias_dict, prec['hm'], self.mass_func.m)
            self.set_spline_galaxy_stellar_mf(hod_dict)
            self.set_spline_galaxy_stellar_mf_bias(hod_dict, bias_dict, prec['hm'])
            self.__restore_mass_bins(bias_dict, saved_bins)
            self.hod.hod_update(bias_dict, prec['hm'], self.mass_func.m)
        else:
            self.set_spline_galaxy_stellar_mf(hod_dict)
            self.set_spline_galaxy_stellar_mf_bias(hod_dict, bias_dict, prec['hm'])
        self.zet= zet

    @staticmethod
    def __widen_mass_bins(bias_dict):
        """
        Extends the stellar mass bins of the galaxy samples in bias_dict to
        the stellar mass range of the SMF (for the SMF splines and the
        count bispectra) and returns copies of the original bins for
        __restore_mass_bins.
        """
        keys = ('logmass_bins', 'logmass_bins_upper', 'logmass_bins_lower')
        saved_bins = {key: None if bias_dict[key] is None else np.copy(bias_dict[key]) for key in keys}
        csmf_min, csmf_max = HOD.csmf_mass_range(bias_dict)
        if bias_dict['logmass_bins_upper'] is not None and bias_dict['logmass_bins_lower'] is not None:
            upper, lower = np.copy(bias_dict['logmass_bins_upper']), np.copy(bias_dict['logmass_bins_lower'])
            upper[np.argmax(upper)] = max(np.max(upper), csmf_max)
            lower[np.argmin(lower)] = min(np.min(lower), csmf_min)
            bias_dict['logmass_bins_upper'], bias_dict['logmass_bins_lower'] = upper, lower
        else:
            bins = np.copy(bias_dict['logmass_bins'])
            bins[0], bins[-1] = csmf_min, csmf_max
            bias_dict['logmass_bins'] = bins
        return saved_bins

    @staticmethod
    def __restore_mass_bins(bias_dict, saved_bins):
        """
        Restores the stellar mass bins of the galaxy samples in bias_dict
        after __widen_mass_bins.
        """
        for key, value in saved_bins.items():
            bias_dict[key] = value

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
        self.camb_pars.set_matter_power(kmax=CAMB_KMAX_LINEAR, redshifts = [0])
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
        return simpson(self.mass_func.dndm
                        * self.hod.occ_num_and_prob(
                            hod_dict,
                            self.mor_tab,
                            self.occprob_tab,
                            self.occnum_tab
                        )[0],
                        x = self.hod.Mrange[None, :])

    def nbar_overlap(self,
                     hod_dict):
        """
        Calculates the average number density of the galaxies which belong
        to two galaxy samples at the same time, i.e. with stellar masses in
        the overlap of their stellar mass bins (log10mass_bins_lower/upper),
        n_overlap = int dM n(M) int_overlap dM* Phi(M*|M).

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. To be passed from the read_input method of
            the Input class.

        Returns
        -------
        nbar_overlap : array
            with unit 1 / (Mpc/h)^3
            with shape (sample_bins, sample_bins), nbar on the diagonal and
            zero for disjoint samples
        """
        nbar = np.diag(np.array(self.ngal, dtype=float))
        occ_prob = self.hod.occ_num_and_prob(hod_dict, self.mor_tab, self.occprob_tab, self.occnum_tab)[1]
        if occ_prob is None:
            return nbar
        grid = self.hod.Mbins
        for i_sample in range(len(grid)):
            # occupation integrated over stellar mass of sample i_sample
            cumulative = None
            for j_sample in range(len(grid)):
                low = max(grid[i_sample, 0], grid[j_sample, 0])
                high = min(grid[i_sample, -1], grid[j_sample, -1])
                if j_sample == i_sample or high <= low:
                    continue
                if cumulative is None:
                    cumulative = cumulative_trapezoid(occ_prob[i_sample], grid[i_sample], axis=0, initial=0)
                occ_num = np.zeros(len(self.mass_func.m))
                for edge, sign in ((high, 1.0), (low, -1.0)):
                    idx = np.clip(np.searchsorted(grid[i_sample], edge) - 1, 0, len(grid[i_sample]) - 2)
                    frac = (edge - grid[i_sample, idx])/(grid[i_sample, idx + 1] - grid[i_sample, idx])
                    occ_num += sign*((1 - frac)*cumulative[idx] + frac*cumulative[idx + 1])
                nbar[i_sample, j_sample] = simpson(self.mass_func.dndm*occ_num, x=self.hod.Mrange)
        return nbar

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
        return simpson(self.mass_func.dndm
                        * self.hod.occ_num_and_prob_per_pop(
                            hod_dict,
                            'sat',
                            self.mor_tab,
                            self.occprob_tab,
                            self.occnum_tab
                        )[0],
                        x = self.hod.Mrange[None, :])

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
        The bias is not renormalised: the integral of n(M) b(M) M/rho_bg over
        the mass range is in general not unity. This is corrected for in the
        large-scale limit of the matter integrals by assigning the missing mass
        to the lowest mass (see halo_model_integral_I_alpha_x), as in CCL.

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
            # Tinker et al. (2010) at the peak height of the current masses
            return self.__bias_tinker10_fittfunc(self.mass_func.nu**0.5)
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
            integral = simpson(self.mass_func.dndm
                                * occ_num
                                * self.bias(bias_dict, hm_prec),
                                 x = self.mass_func.m)

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
           bias_dict,
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
        # cached for the current redshift (hurly_x needs it many times per
        # redshift), the cached arrays are read-only
        if type in ('cen', 'sat'):
            key = (type, bias_dict['Mc_relation_'+type],
                   bias_dict['norm_Mc_relation_'+type], overdensity,
                   self.rho_bg, self.mass_func.k.tobytes(),
                   self.mass_func.m.tobytes())
            cache = self.__dict__.setdefault('_uk_cache', {'z': None})
            if cache['z'] != self.mass_func.z or len(cache) > 8:
                cache.clear()
                cache['z'] = self.mass_func.z
            if key in cache:
                return cache[key]
        if type == 'sat':
            con = self.__concentration(bias_dict['Mc_relation_sat'])
            con *= bias_dict['norm_Mc_relation_sat']
        if type == 'cen':
            con = self.__concentration(bias_dict['Mc_relation_cen'])
            con *= bias_dict['norm_Mc_relation_cen']
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
        u_k.setflags(write=False)
        cache[key] = u_k

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

    def _mass_weights(self):
        """
        Simpson weights for integrals over the halo mass of the current
        mass function, simpson(y, x=m, axis=-1) = y @ weights.
        """
        return simpson_weights(self.mass_func.m)

    @staticmethod
    def _outer_mass_integral(x, y, weights):
        """
        sum_M weights[M] x[k1, s1, M] y[k2, s2, M] for all k1, k2, s1, s2,
        returned with shape (k1, k2, s1, s2).
        """
        n_k1, n_s1, n_m = x.shape
        n_k2, n_s2, _ = y.shape
        res = (x*weights).reshape(-1, n_m) @ y.reshape(-1, n_m).T
        return res.reshape(n_k1, n_s1, n_k2, n_s2).transpose(0, 2, 1, 3)

    def _use_extended_mass_range(self, Mmin, step):
        """
        Replaces the mass function by one on the extended mass range starting
        at 10**Mmin (with spacing step) as needed for the matter integrals,
        until _use_main_mass_range is called. A separate mass function object,
        synchronised in redshift and k range, is kept for this instead of
        changing the mass range of the main one back and forth, which makes
        hmf recompute sigma(M) and everything derived from it every time.
        """
        main = self.mass_func
        extended = getattr(self, '_mass_func_extended', None)
        if extended is None:
            extended = _copy_mass_function(main)
            self._mass_func_extended = extended
        extended.update(z=main.z, lnk_min=main.lnk_min, lnk_max=main.lnk_max,
                        dlnk=main.dlnk, Mmin=Mmin, dlog10m=step)
        self.__dict__.setdefault('_mass_func_stack', []).append(main)
        self.mass_func = extended

    def _use_main_mass_range(self):
        """Restores the mass function replaced by _use_extended_mass_range."""
        self.mass_func = self._mass_func_stack.pop()

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

    @staticmethod
    def satellite_moment_factor(hod_dict,
                                order):
        """
        Ratio of the factorial moment of the satellite occupation to the
        Poisson case, <N_s (N_s - 1) ... (N_s - n + 1)|M> / <N_s|M>^n,
        for n = order galaxies in the same halo. The second moment is set
        by the Poisson parameter P, <N_s (N_s - 1)|M> = P <N_s|M>^2 (Eq. 21
        of Dvornik et al. 2023), the higher ones follow from the count
        distribution with this second moment,
        prod_{j=1}^{n-1} max(0, 1 + j (P - 1)), which is exact for a
        Poisson (P = 1), negative binomial (P > 1) or binomial (P < 1)
        distribution; the factors are set to zero where they would turn
        negative.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used, here 'poisson_sat' (default 1).
        order : int
            number of satellite galaxies in the same halo

        Returns
        -------
        factor : float
        """
        poisson = hod_dict.get('poisson_sat', 1.0)
        factor = 1.0
        for j in range(1, order):
            factor *= max(0.0, 1.0 + j*(poisson - 1.0))
        return factor

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
        uk = self.uk(bias_dict, 'sat')
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
            uk = self.uk(bias_dict,'cen')
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
        hurly_x : BSpline
            in log10(k), evaluates to shape (..., sample_bins, M_bins)

        """
        hurlyX = self.hurly_x(bias_dict, hod_dict, type_x)

        # interpolating cubic spline (not-a-knot, as UnivariateSpline with
        # s=0) for all samples and masses at once
        return make_interp_spline(np.log10(self.mass_func.k), hurlyX, k=3, axis=0)

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
                norm = simpson(self.mass_func.dndm * bias *
                              self.mass_func.m, x = self.mass_func.m) / self.rho_bg
                Abmin = (1-norm)*self.rho_bg/self.mass_func.m[0]
                integral_x = hurlyX @ (self.mass_func.dndm * bias
                                       * self._mass_weights()) + Abmin*hurlyX[:, :, 0]
            if type_x == 'm':
                M_min_save = hm_prec["log10M_min"]
                Mmin = 2.0
                hm_prec["log10M_min"] = Mmin
                step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
                hm_prec['M_bins'] = len(self.mass_func.m)
                self._use_extended_mass_range(Mmin, step)
                self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)

                hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
                bias = self.bias(bias_dict, hm_prec) * bias_dict['bias_2h']
                norm = simpson(self.mass_func.dndm * bias *
                              self.mass_func.m, x = self.mass_func.m) / self.rho_bg
                Abmin = (1-norm)*self.rho_bg/self.mass_func.m[0]
                integral_x = hurlyX @ (self.mass_func.dndm * bias
                                       * self._mass_weights()) + Abmin*hurlyX[:, :, 0]

                hm_prec["log10M_min"] = M_min_save
                self._use_main_mass_range()
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)

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
        log10k = np.log10(self.mass_func.k)
        integralX_spline = []
        for nbin in range(integralX_shape[1]):
            integralX_spline.append(
                UnivariateSpline(log10k,
                                 integralX[:, nbin],
                                 k=1, s=0, ext=0))

        return integralX_spline

    def halo_model_integral_I_alpha_xy(self,
                                       bias_dict,
                                       hod_dict,
                                       hm_prec,
                                       alpha,
                                       type_x,
                                       type_y,
                                       diagonal=False):
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
        diagonal : bool
            If True, only the diagonal (k_1 = k_2, same sample bins) is
            calculated, which avoids the (log10k_bins^2 * sample_bins^2 *
            M_bins) integrand.

        Returns
        -------
        integral_xy : array
            with shape (log10k_bins, log10k_bins, 
                        sample_bins, sample_bins)
            or (log10k_bins, sample_bins) if diagonal

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

            is_mm = (type_x == 'm' and type_y == 'm')
            if type_x == 'g' or type_y == 'g':
                hurly_cen = self.hurly_x(bias_dict, hod_dict, 'cen')
                hurly_sat = self.hurly_x(bias_dict, hod_dict, 'sat')
            if (type_x == 'm' or type_y == 'm') and not is_mm:
                hurly_m = self.hurly_x(bias_dict, hod_dict, 'm')
            if not is_mm:
                bias_val = self.bias(bias_dict, hm_prec)

            if type_x == 'g':
                hurlyX = hurly_cen + hurly_sat
                bias = bias_val * bias_dict['bias_2h']
            elif type_x == 'm' and not is_mm:
                hurlyX = hurly_m

            if type_y == 'g':
                hurlyY = hurly_cen + hurly_sat
                bias = bias_val * bias_dict['bias_2h']
            elif type_y == 'm' and not is_mm:
                hurlyY = hurly_m
                bias = bias_val

            if type_x == 'g' and type_y == 'g':
                # galaxy pairs in the same halo, 2 c s + P s^2
                poisson_correction = 1.0 - self.satellite_moment_factor(hod_dict, 2)
                if diagonal:
                    correct = hurly_cen * hurly_cen + poisson_correction * hurly_sat * hurly_sat
                bias = bias_val

            if type_x == 'm' and type_y == 'm':
                M_min_save = hm_prec["log10M_min"]
                Mmin = 2.0
                hm_prec["log10M_min"] = Mmin
                step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
                hm_prec['M_bins'] = len(self.mass_func.m)
                self._use_extended_mass_range(Mmin, step)
                self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)

                hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
                bias = self.bias(bias_dict, hm_prec)
                weights = self.mass_func.dndm * bias * self._mass_weights()
                if diagonal:
                    integral_xy = (hurlyX * hurlyX) @ weights
                else:
                    integral_xy = self._outer_mass_integral(hurlyX, hurlyX, weights)
                hm_prec["log10M_min"] = M_min_save
                self._use_main_mass_range()
                hm_prec['M_bins'] = len(self.mass_func.m)
                self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
            elif diagonal:
                integral_xy = (hurlyX * hurlyY - correct) @ (
                    self.mass_func.dndm * bias * self._mass_weights())
            else:
                weights = self.mass_func.dndm * bias * self._mass_weights()
                integral_xy = self._outer_mass_integral(hurlyX, hurlyY, weights)
                if type_x == 'g' and type_y == 'g':
                    integral_xy -= self._outer_mass_integral(hurly_cen, hurly_cen, weights)
                    integral_xy -= poisson_correction*self._outer_mass_integral(hurly_sat, hurly_sat, weights)

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
        log10k = np.log10(self.mass_func.k)
        integralXY_spline = []
        for nbin in range(integralXY_shape[2]):
            integralXY_spline.append([])
            for mbin in range(integralXY_shape[3]):
                integralXY_spline[nbin].append(RectBivariateSpline(
                    log10k,
                    log10k,
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
            Mmin = 2.0
            hm_prec["log10M_min"] = Mmin
            step = (self.mass_func.Mmax - Mmin) / hm_prec["M_bins"]
            hm_prec['M_bins'] = len(self.mass_func.m)
            self._use_extended_mass_range(Mmin, step)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)

            hurlyX = self.hurly_x(bias_dict, hod_dict, 'm')
            bias = self.bias(bias_dict, hm_prec)
            weights = self.mass_func.dndm * bias * self._mass_weights()
            integral_mmm = np.stack([(hurlyX[:, i_s, :]*weights) @ (hurlyX[:, i_s, :]**2.0).T
                                     for i_s in range(hurlyX.shape[1])], axis=-1)

            hm_prec["log10M_min"] = M_min_save
            self._use_main_mass_range()
            hm_prec['M_bins'] = len(self.mass_func.m)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
            

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

        log10k = np.log10(self.mass_func.k)
        return RectBivariateSpline(
            log10k,
            log10k,
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
            return simpson(self.mass_func.dndm[None, None, :]*csmf_sat, x = self.mass_func.m, axis = -1)
        if type == 'cen':
            csmf_sat = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'cen',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return simpson(self.mass_func.dndm[None, None, :]*csmf_sat, x = self.mass_func.m, axis = -1)
    
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
            csmf = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'sat',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return simpson((self.bias(bias_dict,hm_prec)*self.mass_func.dndm)[None, None, :]*csmf, x = self.mass_func.m, axis = -1)
        if type == 'cen':
            csmf = self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                         'cen',
                                                         self.mor_tab,
                                                         self.occprob_tab,
                                                         self.occnum_tab)[1]
            return simpson((self.bias(bias_dict,hm_prec)*self.mass_func.dndm)[None, None, :]*csmf, x = self.mass_func.m, axis = -1)

    def set_spline_galaxy_stellar_mf(self,
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
        indices = np.unique(aux_M,return_index = True)[1]
        
        self.galaxy_smf_c = UnivariateSpline(aux_M[indices], np.array(aux_c)[indices] , k=2, s=0, ext=0)
        self.galaxy_smf_s = UnivariateSpline(aux_M[indices], np.array(aux_s)[indices] , k=2, s=0, ext=0)

    def set_spline_galaxy_stellar_mf_bias(self,
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
        
        indices = np.unique(aux_M,return_index = True)[1]
        self.galaxy_smf_bias_c = UnivariateSpline(np.array(aux_M)[indices], np.array(aux_c)[indices], k=2, s=0, ext=0)
        self.galaxy_smf_bias_s = UnivariateSpline(np.array(aux_M)[indices], np.array(aux_s)[indices], k=2, s=0, ext=0)        

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
                                                      self.occnum_tab)[1]
        if type == 'cen':
            return self.hod.occ_num_and_prob_per_pop(hod_dict,
                                                     'cen',
                                                      self.mor_tab,
                                                      self.occprob_tab,
                                                      self.occnum_tab)[1]
        
    def count_matter_bispectrum(self,
                                bias_dict,
                                hod_dict,
                                hm_prec):
        """
        Calculates the three dimensional count-matter density cross-bispectrum
        for a collapsed triange at the wavenumbers specified in log10kbins
        and redshift specified in the class. This is used for later computation
        of the cross-correlation between the conditional stellar mass function and
        2pt statistics.

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
        nBcmm_mu : array
            with shape (log10k bins, sample bins)

        References
        ----------
        Takada and Bridle 2007,  New Journal of Physics, 9, 446

        """
        halo_profile = self.uk(bias_dict)
        halo_bias = self.bias(bias_dict,hm_prec)
        csmf = self.conditional_galaxy_stellar_mf(hod_dict,'cen') + self.conditional_galaxy_stellar_mf(hod_dict,'sat')
        m = self.mass_func.m
        # the mass integrals (Simpson's rule) as matrix products over the
        # halo mass, csmf_weighted has shape (sample bins * stellar masses, M)
        weights = simpson_weights(m)
        csmf_weighted = csmf.reshape(-1, len(m))*(self.mass_func.dndm*weights)
        I1 = ((halo_profile**2)*(m**2/self.rho_bg**2)) @ csmf_weighted.T
        I2 = (halo_profile*(m/self.rho_bg*halo_bias)) @ csmf_weighted.T
        # matter I_1^1 with the missing mass assigned to the lowest mass, as in
        # halo_model_integral_I_alpha_x
        mass_weights = self.mass_func.dndm*m/self.rho_bg*halo_bias*weights
        I3 = halo_profile @ mass_weights + (1 - np.sum(mass_weights))*halo_profile[:, 0]
        return (I1 + 2.0 * self.mass_func.power[:, None] * I2 * I3[:, None]).reshape((len(halo_profile),) + csmf.shape[:-1])
    
    def get_count_matter_bispectrum(self,
                                    bias_dict,
                                    hod_dict,
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
        if self.hod.mass_bins_disagree:
            saved_bins = self.__widen_mass_bins(bias_dict)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
            aux_M = self.hod.Mbins.reshape(len(self.hod.Mbins[0,:])*len(self.hod.Mbins[:,0]))
            indices = np.unique(aux_M,return_index = True)[1]
            aux_bispec_count_mm = self.count_matter_bispectrum(bias_dict, hod_dict, hm_prec)
            aux_bispec_count_mm = aux_bispec_count_mm.reshape((len(aux_bispec_count_mm[:,0,0]), len(self.hod.Mbins[0,:])*len(self.hod.Mbins[:,0])))
            log10_aux_M = np.log10(aux_M[indices])
            count_matter_bispec = np.zeros((len(self.mass_func.k), len(log10csmf_mass_bins)))
            for i_k in range(len(self.mass_func.k)):
                count_matter_bispec[i_k, :] = np.exp(np.interp(log10csmf_mass_bins, log10_aux_M, np.log(aux_bispec_count_mm[i_k,indices])))
            self.__restore_mass_bins(bias_dict, saved_bins)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
        else:
            aux_M = self.hod.Mbins.reshape(len(self.hod.Mbins[0,:])*len(self.hod.Mbins[:,0]))
            indices = np.unique(aux_M,return_index = True)[1]
            aux_bispec_count_mm = self.count_matter_bispectrum(bias_dict, hod_dict, hm_prec)
            aux_bispec_count_mm = aux_bispec_count_mm.reshape((len(aux_bispec_count_mm[:,0,0]), len(self.hod.Mbins[0,:])*len(self.hod.Mbins[:,0])))
            log10_aux_M = np.log10(aux_M[indices])
            count_matter_bispec = np.zeros((len(self.mass_func.k), len(log10csmf_mass_bins)))
            for i_k in range(len(self.mass_func.k)):
                count_matter_bispec[i_k, :] = np.exp(np.interp(log10csmf_mass_bins, log10_aux_M, np.log(aux_bispec_count_mm[i_k,indices])))
        return count_matter_bispec

    def __csmf_at_stellar_masses(self,
                                 bias_dict,
                                 hod_dict,
                                 hm_prec,
                                 log10csmf_mass_bins):
        """
        Conditional stellar mass function of centrals and satellites,
        Phi_c(M*|M) and Phi_s(M*|M), at the stellar masses of the SMF bins.
        If the SMF bins extend beyond the stellar mass bins of the galaxy
        samples, the stellar mass grid of the HOD is extended for this
        evaluation only.

        Returns
        -------
        csmf_cen, csmf_sat : arrays
            with shape (SMF bins, M_bins)
        """
        if self.hod.mass_bins_disagree:
            saved_bins = self.__widen_mass_bins(bias_dict)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
        log10_aux_M = np.log10(self.hod.Mbins.reshape(-1))
        indices = np.unique(log10_aux_M, return_index=True)[1]
        csmf = []
        for pop in ('cen', 'sat'):
            aux_csmf = self.conditional_galaxy_stellar_mf(hod_dict, pop).reshape(-1, len(self.mass_func.m))[indices]
            csmf.append(interp1d(log10_aux_M[indices], aux_csmf, axis=0, bounds_error=False,
                                 fill_value=0.0)(log10csmf_mass_bins))
        if self.hod.mass_bins_disagree:
            self.__restore_mass_bins(bias_dict, saved_bins)
            self.hod.hod_update(bias_dict, hm_prec, self.mass_func.m)
        return csmf[0], csmf[1]

    def get_count_galaxy_bispectra(self,
                                   bias_dict,
                                   hod_dict,
                                   hm_prec,
                                   log10csmf_mass_bins):
        """
        Calculates the count-galaxy-galaxy and count-galaxy-matter
        bispectra for a collapsed triangle (the counts at k = 0), i.e. the
        three-point functions of the galaxies in the stellar mass bins of the
        SMF with two galaxies of each galaxy sample, or with one galaxy and
        the matter, in analogy to the count-matter bispectrum. Only
        triplets of distinct galaxies contribute: a halo hosts at most one
        central and the moments of the satellite occupation follow from the
        Poisson parameter (see satellite_moment_factor). The 1-halo terms
        read, with Phi_c/s(M*|M) the conditional stellar mass function of
        the counts, H_c/s/m(k, M) as in hurly_x and P the Poisson parameter,
            nB_cgg^1h = int dM n(M) [Phi_c P H_s^2 + 2 Phi_s P H_c H_s
                                     + Phi_s P(2P - 1) H_s^2],
            nB_cgm^1h = int dM n(M) [Phi_c H_s + Phi_s H_c + P Phi_s H_s] H_m,
        and the 2-halo terms (count and one tracer in the same halo)
            nB_cgg^2h = 2 P_lin(k) A_cg(k) I_g(k),
            nB_cgm^2h = P_lin(k) [A_cg(k) I_m(k) + A_cm(k) I_g(k)],
        with A_cg = int dM n b [Phi_c H_s + Phi_s H_c + P Phi_s H_s],
        A_cm = int dM n b (Phi_c + Phi_s) H_m and I_x = int dM n b H_x.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be
            passed from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. To be passed from the read_input method of
            the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel. To be
            passed from the read_input method of the Input class.
        log10csmf_mass_bins : array
            stellar masses of the SMF bins

        Returns
        -------
        nBcgg, nBcgm : arrays
            with shape (log10k bins, SMF bins, sample bins)

        References
        ----------
        Takada and Bridle 2007,  New Journal of Physics, 9, 446
        """
        csmf_cen, csmf_sat = self.__csmf_at_stellar_masses(bias_dict, hod_dict, hm_prec, log10csmf_mass_bins)
        hurly_c = self.hurly_x(bias_dict, hod_dict, 'cen')
        hurly_s = self.hurly_x(bias_dict, hod_dict, 'sat')
        hurly_m = self.hurly_x(bias_dict, hod_dict, 'm')[:, 0, :]
        poisson_2 = self.satellite_moment_factor(hod_dict, 2)
        poisson_3 = self.satellite_moment_factor(hod_dict, 3)
        m = self.mass_func.m
        halo_bias = self.bias(bias_dict, hm_prec)
        weights = self.mass_func.dndm*simpson_weights(m)
        def mass_integral(csmf, hurly, mass_weights):
            # int dM w(M) Phi(M*_mu|M) H(k, s, M) as a matrix product, with
            # shape (k, SMF bins, samples)
            n_k, n_sample, n_m = hurly.shape
            return (hurly.reshape(n_k*n_sample, n_m) @ (csmf*mass_weights).T).reshape(n_k, n_sample, -1).transpose(0, 2, 1)
        # count and one galaxy in the same halo: Phi_c H_s + Phi_s (H_c + P H_s)
        pair_sat = hurly_c + poisson_2*hurly_s
        bispec_cgg_1h = mass_integral(csmf_cen, poisson_2*hurly_s**2, weights) \
            + mass_integral(csmf_sat, (2*poisson_2*hurly_c + poisson_3*hurly_s)*hurly_s, weights)
        bispec_cgm_1h = mass_integral(csmf_cen, hurly_s*hurly_m[:, None, :], weights) \
            + mass_integral(csmf_sat, pair_sat*hurly_m[:, None, :], weights)
        a_cg = mass_integral(csmf_cen, hurly_s, weights*halo_bias) + mass_integral(csmf_sat, pair_sat, weights*halo_bias)
        a_cm = ((csmf_cen + csmf_sat)[None, :, :]*hurly_m[:, None, :]) @ (weights*halo_bias)
        integral_g = (hurly_c + hurly_s) @ (weights*halo_bias)
        # matter I_1^1 with the missing mass assigned to the lowest mass, as in
        # count_matter_bispectrum
        mass_weights = weights*m/self.rho_bg*halo_bias
        halo_profile = self.uk(bias_dict)
        integral_m = halo_profile @ mass_weights + (1 - np.sum(mass_weights))*halo_profile[:, 0]
        power = self.mass_func.power
        bispec_cgg = bispec_cgg_1h + 2*power[:, None, None]*a_cg*integral_g[:, None, :]
        bispec_cgm = bispec_cgm_1h + power[:, None, None]*(a_cg*integral_m[:, None, None]
                                                           + a_cm[:, :, None]*integral_g[:, None, :])
        return bispec_cgg, bispec_cgm
