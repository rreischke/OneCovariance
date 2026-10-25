import hashlib
from os import getpid, makedirs, path, replace

import astropy.cosmology
import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema



def _hash_inputs(digest, value):
    """Feeds numbers, strings, arrays and (nested) lists/tuples/dicts into a hash."""
    if isinstance(value, dict):
        for key in sorted(value):
            _hash_inputs(digest, key)
            _hash_inputs(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b'(%d' % len(value))
        for item in value:
            _hash_inputs(digest, item)
        digest.update(b')')
    elif isinstance(value, np.ndarray) or isinstance(value, np.generic):
        value = np.ascontiguousarray(value)
        digest.update(f'{value.dtype.str}{value.shape}'.encode())
        digest.update(value.tobytes())
    else:
        digest.update(repr(value).encode())


_levin_hash = []


def _levin_build_hash():
    """SHA-256 of the compiled levin extension (computed once per run)."""
    if not _levin_hash:
        try:
            import levin
            with open(levin.__file__, 'rb') as binary:
                _levin_hash.append(hashlib.sha256(binary.read()).hexdigest())
        except (ImportError, OSError, AttributeError):
            _levin_hash.append('unknown')
    return _levin_hash[0]


def cached_computation(cache_dir, name, inputs, compute):
    """
    Returns compute(), an array or a dictionary of arrays. If cache_dir is
    not None, the result is stored there under a hash of name and inputs
    (which must determine the result completely) and read back by later
    runs with the same inputs. Failing to read or write the cache only
    causes the result to be computed.
    """
    if cache_dir is None:
        return compute()
    digest = hashlib.sha256(name.encode())
    digest.update(_levin_build_hash().encode())
    _hash_inputs(digest, inputs)
    fname = path.join(cache_dir, name + '_' + digest.hexdigest()[:24] + '.npz')
    if path.exists(fname):
        try:
            with np.load(fname) as cached:
                if 'result' in cached.files:
                    return cached['result']
                return {key[5:]: cached[key] for key in cached.files}
        except Exception as error:
            print("CacheWarning: could not read " + fname + " (" + str(error) + "), recalculating.")
    result = compute()
    try:
        makedirs(cache_dir, exist_ok=True)
        tmp = fname + '.' + str(getpid()) + '.tmp.npz'
        if isinstance(result, dict):
            np.savez(tmp, **{'dict_' + key: np.asarray(val) for key, val in result.items()})
        else:
            np.savez(tmp, result=np.asarray(result))
        replace(tmp, fname)
    except OSError as error:
        print("CacheWarning: could not write " + fname + " (" + str(error) + ").")
    return result

class Setup:
    """
    This class provides various methods that set the cosmological model
    via astropy, and normalizes the tabulated redshift distributions. It
    also contains consistency checks between input in the configuration
    file and the look-up tables, the redshift range, number of
    tomographic bins, the k-range, the mass range and the number of
    galaxy samples (e.g., split in stellar mass bins). All consistency
    checks are automatically performed when the Setup class is called.

    Atrributes
    ----------
    cosmo_dict : dictionary
        Specifies all cosmological parameters. To be passed from the 
        read_input method of the Input class.
    bias_dict : dictionary
        Specifies all the information about the bias model. To be passed
        from the read_input method of the Input class.
    survey_params_dict : dictionary
        Specifies all the information unique to a specific survey.
        Relevant values are the effective number density of galaxies for 
        all tomographic bins. To be passed from the read_input method of 
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
    cosmology : class
        This class specifies the cosmological model and basic relations 
        such as distances. It originates from the 
        astropy.cosmology.w0waCDM module.
    rho_bg : float
        with unit M_sun / pc**3
        matter density parameter times critical density at redshift 0
    sample_dim : int
        number of samples of, e.g., stellar mass the tracers are divided 
        in, specified by [bias]: 'log10mass_bins' or the look-up tables
        specifying the HOD
    zet_clust : dictionary
        private variable for read_in_tables['zclust'] but zero-padded 
        and 'value_loc_in_clustbin'-adjusted (see above)
    zet_lens : dictionary
        private variable for read_in_tables['zlens'] but zero-padded 
        and 'value_loc_in_lensbin'-adjusted (see above)
    zet_min : float
        minimum redshift considered as given by zet_clust
    zet_max : float
        maximum redshift considered as given by zet_lens
    n_tomo_clust : int
        number of tomographic bins relevant for clustering and/or 
        galaxy-galaxy lensing analysis
    n_tomo_lens : int
        number of tomographic bins relevant for cosmic shear and/or 
        galaxy-galaxy lensing analysis
    Pxy_tab : dictionary
        See above in Parameters Pxy_tab
    Cxy_tab : dictionary
        See above in Parameters Cxy_tab
    effbias_tab : dictionary
        See above in Parameters effbias_tab
    mor_tab : dictionary
        See above in Parameters mor_tab
    occprob_tab : dictionary
        See above in Parameters occprob_tab
    occnum_tab : dictionary
        See above in Parameters occnum_tab

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_setup import Setup
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    setting = Setup(cosmo, bias, survey_params, prec, read_in_tables)

    """

    def __init__(self,
                 cosmo_dict,
                 bias_dict,
                 survey_params_dict,
                 prec,
                 read_in_tables):

        self.cosmology, self.rho_bg = self.__set_cosmology(cosmo_dict)
        if (bias_dict['logmass_bins_upper'] is not None and bias_dict['logmass_bins_lower'] is not None):
            self.sample_dim = len(bias_dict['logmass_bins_upper'])
        else:
            self.sample_dim = len(bias_dict['logmass_bins']) - 1
        self.bias_dict = bias_dict

        self.Pxy_tab = read_in_tables['Pxy']
        self.Cxy_tab = read_in_tables['Cxy']
        self.effbias_tab = read_in_tables['effbias']
        self.mor_tab = read_in_tables['mor']
        self.occprob_tab = read_in_tables['occprob']
        self.occnum_tab = read_in_tables['occnum']

        self.zet_clust = read_in_tables['zclust']
        self.zet_lens = read_in_tables['zlens']
        self.zet_csmf = read_in_tables['zcsmf']
        self.zet_min, self.zet_max, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_csmf = \
            self.__consistency_checks_for_z_support_in_tabs(read_in_tables)
        self.tomos_6x2pt_clust = \
            self.__consistency_checks_for_tomographic_dims(survey_params_dict)
        self.__consistency_checks_for_k_support_in_tabs(prec['powspec'])
        self.__consistency_checks_for_M_support_in_tabs(bias_dict, prec['hm'])

        self.zet_clust, self.zet_lens = \
            self.__check_z_distribution_value_loc_in_bin()

    def __set_cosmology(self,
                        cosmo_dict):
        """
        Fixes the cosmology with a class from the astropy library for 
        all future calculations. Uses a general 'w0waCDM' model.

        Parameters
        ----------
        cosmo_dict : dictionary
            Specifies all cosmological parameters. To be passed from the 
            read_input method of the Input class.

        Return
        ---------
        astropy.cosmology.w0waCDM : class

        """
        cosmology = astropy.cosmology.w0waCDM(
            H0=cosmo_dict['h']*100.0,
            Om0=cosmo_dict['omega_m'],
            Ob0=cosmo_dict['omega_b'],
            Ode0=cosmo_dict['omega_de'],
            w0=cosmo_dict['w0'],
            wa=cosmo_dict['wa'],
            Neff=cosmo_dict['neff'],
            m_nu=[cosmo_dict['m_nu'],0,0]*u.eV,
            Tcmb0=cosmo_dict['Tcmb0'])

        rho_bg = (cosmology.Om(0)*(cosmology.critical_density(0).to(u.M_sun/u.Mpc**3)/cosmology.h**2)).value
        return cosmology, rho_bg

    def __consistency_checks_for_z_support_in_tabs(self, read_in_tables):
        """
        Performs consistency checks for the tabulated power spectra and
        effective bias against the tabulated redshift distribution(s). 
        Checks are done for redshift 0 to the maximum redshift. Gives 
        warnings or errors if they are incompatible. 

        """
        # the redshift range covers the redshift distributions of all tracers
        n_tomo_clust = len(self.zet_clust['nz']) if self.zet_clust['z'] is not None else 0
        n_tomo_lens = len(self.zet_lens['photoz']) if self.zet_lens['z'] is not None else 0
        n_tomo_csmf = len(self.zet_csmf['pz']) if self.zet_csmf['z'] is not None else 0
        zets = [tab['z'] for tab in (self.zet_clust, self.zet_lens, self.zet_csmf) if tab['z'] is not None]
        zet_min = min(zet[0] for zet in zets) if zets else None
        zet_max = max(zet[-1] for zet in zets) if zets else None

        # check for power spectra look-up tables
        if self.Pxy_tab['z'] is not None:
            if self.Pxy_tab['z'][0] > 1e-4:
                raise Exception("SetupError: The minimum redshift support " +
                                "of the tabulated power spectra is " +
                                str(round(self.Pxy_tab['z'][0], 4)) + " " +
                                "while structure of this code requires the " +
                                "first entry to be for z=0. Must be " +
                                "adjusted to go on.")
            if zet_max - self.Pxy_tab['z'][-1] > 1e-3:
                raise Exception("SetupError: The maximum redshift support " +
                                "of the tabulated power spectra is "
                                + str(round(self.Pxy_tab['z'][-1], 4)) + " " +
                                "while the redshift distribution requires " +
                                str(round(zet_max, 4)) + ". Must be " +
                                "adjusted to go on.")

        # check for effective bias look-up tables
        if self.effbias_tab['z'] is not None:
            if self.effbias_tab['z'][0] > 1e-3:
                raise Exception("SetupError: The minimum redshift support " +
                                "of the tabulated effective bias is " +
                                str(round(self.effbias_tab['z'][0], 3)) + " " +
                                "while structure of this code requires the " +
                                "first entry to be for z=0. Must be " +
                                "adjusted to go on.")
            if zet_max - self.effbias_tab['z'][-1] > 1e-3:
                raise Exception("SetupError: The maximum redshift support " +
                                "of the tabulated effective bias is " +
                                str(round(self.effbias_tab['z'][-1], 3)) +
                                " while the redshift distribution requires " +
                                str(round(zet_max, 4)) + ". Must be " +
                                "adjusted to go on.")
        if read_in_tables['arb_radial']['do_arbitrary_radial_weights']:
            if read_in_tables['arb_radial']['number_radial_weights_gg'] is not None:
                if read_in_tables['arb_radial']['add_to_galaxy']:
                    n_tomo_clust += read_in_tables['arb_radial']['number_radial_weights_gg']
                else:
                    n_tomo_clust = read_in_tables['arb_radial']['number_radial_weights_gg']
            if read_in_tables['arb_radial']['number_radial_weights_mm'] is not None:
                if read_in_tables['arb_radial']['add_to_matter']:
                    n_tomo_lens += read_in_tables['arb_radial']['number_radial_weights_mm']
                else:
                    n_tomo_lens = read_in_tables['arb_radial']['number_radial_weights_mm']
            try:
                zet_min_arb = min(read_in_tables['arb_radial']['z_gg'][0], read_in_tables['arb_radial']['z_mm'][0])
                zet_max_arb = max(read_in_tables['arb_radial']['z_gg'][-1], read_in_tables['arb_radial']['z_mm'][-1])
            except TypeError:
                if read_in_tables['arb_radial']['z_gg'] is not None:
                    zet_min_arb = read_in_tables['arb_radial']['z_gg'][0]
                    zet_max_arb = read_in_tables['arb_radial']['z_gg'][-1]
                if read_in_tables['arb_radial']['z_mm'] is not None:
                    zet_min_arb = read_in_tables['arb_radial']['z_mm'][0]
                    zet_max_arb = read_in_tables['arb_radial']['z_mm'][-1]
            zet_min = min(zet_min, zet_min_arb)
            zet_max = max(zet_max, zet_max_arb)
        return zet_min, zet_max, n_tomo_clust, n_tomo_lens, n_tomo_csmf

    def __consistency_checks_for_tomographic_dims(self,
                                                  survey_params_dict):
        """
        Performs consistency checks for the number of tomographic bins 
        for parameters specific to the survey. Also, checks against the 
        number of sample bins. Enables smooth processing of 6x2pt 
        analysis. Gives warnings or errors if they are incompatible. 

        Parameters
        ------------
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies 
            for all tomographic bins. To be passed from the read_input 
            method of the Input class.

        """
        if survey_params_dict['n_eff_clust'] is not None and \
           self.n_tomo_clust > 0:
            if len(survey_params_dict['n_eff_clust']) == 1:
                if len(survey_params_dict['n_eff_clust']) != self.n_tomo_clust:
                    raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_clust) + " but 'n_eff_clust' has " +
                                str(len(survey_params_dict['n_eff_clust'])) + " " +
                                "entries. Must be adjusted to go on.")
                else:
                    ...
                #survey_params_dict['n_eff_clust'] = \
                #    np.ones((self.sample_dim, self.n_tomo_clust)) \
                #    * survey_params_dict['n_eff_clust'][0]
            elif len(survey_params_dict['n_eff_clust']) == \
                    self.sample_dim*self.n_tomo_clust:
                ...
                #survey_params_dict['n_eff_clust'] = \
                #    survey_params_dict['n_eff_clust'].reshape(
                #        (self.n_tomo_clust, self.sample_dim)).swapaxes(0, 1)
            elif len(survey_params_dict['n_eff_clust']) != self.n_tomo_clust:
                if len(survey_params_dict['n_eff_clust']) == self.sample_dim:
                    raise Exception("SetupError: The number of tomographic " +
                                    "bins is " + str(self.n_tomo_clust) + " with " +
                                    str(self.sample_dim) + " galaxy samples but " +
                                    "'n_eff_clust' has " +
                                    str(len(survey_params_dict['n_eff_clust'])) + " " +
                                    "entries. The number of values given coincides " +
                                    "with the number of galaxy samples. If you want " +
                                    "these values for all tomographic bins, you " +
                                    "have to repeat them for all tomographic bins, " +
                                    "e.g. for 2 samples and 3 tomographic bins: " +
                                    "n_eff_clust = 0.3, 0.4, 0.3, 0.4, 0.3, 0.4. " +
                                    "In any case, must be adjusted to go on.")
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_clust) + " but 'n_eff_clust' has " +
                                str(len(survey_params_dict['n_eff_clust'])) + " " +
                                "entries. Must be adjusted to go on.")

        if survey_params_dict['shot_noise_clust'] is not None and \
           self.n_tomo_clust > 0:
            if len(survey_params_dict['shot_noise_clust']) == 1:
                survey_params_dict['shot_noise_clust'] = \
                    np.ones(self.n_tomo_clust) \
                    * survey_params_dict['shot_noise_clust'][0]
            elif len(survey_params_dict['shot_noise_clust']) == \
                    self.sample_dim*self.n_tomo_clust:
                survey_params_dict['shot_noise_clust'] = \
                    survey_params_dict['shot_noise_clust'].reshape(
                        (self.n_tomo_clust, self.sample_dim))
            elif len(survey_params_dict['shot_noise_clust']) != \
                    self.n_tomo_clust:
                if len(survey_params_dict['shot_noise_clust']) == \
                   self.sample_dim:
                    raise Exception("SetupError: The number of tomographic " +
                                    "bins is " + str(self.n_tomo_clust) + " with " +
                                    str(self.sample_dim) + " galaxy samples but " +
                                    "'shot_noise_clust' has " +
                                    str(len(survey_params_dict['shot_noise_clust'])) +
                                    " entries. The number of values given coincides " +
                                    "with the number of galaxy samples. If you want " +
                                    "these values for all tomographic bins, you " +
                                    "have to repeat them for all tomographic bins, " +
                                    "e.g. for 2 samples and 3 tomographic bins: " +
                                    "shot_noise_clust = 0.3, 0.4, 0.3, 0.4, 0.3, " +
                                    "0.4. In any case, must be adjusted to go on.")
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_clust) + " but " +
                                "'shot_noise_clust' has " +
                                str(len(survey_params_dict['shot_noise_clust'])) + " " +
                                "entries. Must be adjusted to go on.")

        if survey_params_dict['n_eff_ggl'] is not None and self.n_tomo_clust > 0:
            if len(survey_params_dict['n_eff_ggl']) == 1:
                survey_params_dict['n_eff_ggl'] = \
                    np.ones((self.sample_dim, self.n_tomo_clust)) \
                    * survey_params_dict['n_eff_ggl'][0]
            elif len(survey_params_dict['n_eff_ggl']) == \
                    self.sample_dim*self.n_tomo_clust:
                survey_params_dict['n_eff_ggl'] = \
                    survey_params_dict['n_eff_ggl'].reshape(
                        (self.n_tomo_clust, self.sample_dim)).swapaxes(0, 1)
            elif len(survey_params_dict['n_eff_ggl']) != self.n_tomo_clust:
                if len(survey_params_dict['n_eff_ggl']) == self.sample_dim:
                    raise Exception("SetupError: The number of tomographic " +
                                    "bins is " + str(self.n_tomo_clust) + " with " +
                                    str(self.sample_dim) + " galaxy samples but " +
                                    "'n_eff_ggl' has " +
                                    str(len(survey_params_dict['n_eff_ggl'])) + " " +
                                    "entries. The number of values given coincides " +
                                    "with the number of galaxy samples. If you want " +
                                    "these values for all tomographic bins, you " +
                                    "have to repeat them for all tomographic bins, " +
                                    "e.g. for 2 samples and 3 tomographic bins: " +
                                    "n_eff_ggl = 0.3, 0.4, 0.3, 0.4, 0.3, 0.4. " +
                                    "In any case, must be adjusted to go on.")
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_clust) + " but 'n_eff_ggl' " +
                                str(len(survey_params_dict['n_eff_ggl'])) + " " +
                                "has entries. Must be adjusted to go on.")

        if survey_params_dict['n_eff_lens'] is not None and \
           self.n_tomo_lens > 0:
            if len(survey_params_dict['n_eff_lens']) == 1:
                survey_params_dict['n_eff_lens'] = \
                    np.ones(self.n_tomo_lens) \
                    * survey_params_dict['n_eff_lens'][0]
            if len(survey_params_dict['n_eff_lens']) != self.n_tomo_lens:
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_lens) + " but 'n_eff_lens' " +
                                "has " + str(len(survey_params_dict['n_eff_lens'])) +
                                " entries. Must be adjusted to go on.")

        if survey_params_dict['ellipticity_dispersion'] is not None and \
           self.n_tomo_lens > 0:
            if len(survey_params_dict['ellipticity_dispersion']) == 1:
                survey_params_dict['ellipticity_dispersion'] = \
                    np.ones(self.n_tomo_lens) \
                    * survey_params_dict['ellipticity_dispersion'][0]
            if len(survey_params_dict['n_eff_lens']) != self.n_tomo_lens:
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_lens) + " but ellipticity_dispersion " +
                                "has " +
                                str(len(survey_params_dict['ellipticity_dispersion'])) +
                                " entries. Must be adjusted to go on.")

        if survey_params_dict['shot_noise_gamma'] is not None and \
           self.n_tomo_lens > 0:
            if len(survey_params_dict['shot_noise_gamma']) == 1:
                survey_params_dict['shot_noise_gamma'] = \
                    np.ones(self.n_tomo_lens) \
                    * survey_params_dict['shot_noise_gamma'][0]
            if len(survey_params_dict['n_eff_lens']) != self.n_tomo_lens:
                raise Exception("SetupError: The number of tomographic bins " +
                                "is " + str(self.n_tomo_lens) + " but " +
                                "shot_noise_gamma has " +
                                str(len(survey_params_dict['shot_noise_gamma'])) +
                                " entries. Must be adjusted to go on.")

        if survey_params_dict['tomos_6x2pt_clust'] is None and \
           self.zet_clust['tomos_6x2pt'] is None:
            tomos_6x2pt_clust = [None]
        elif survey_params_dict['tomos_6x2pt_clust'] is not None:
            tomos_6x2pt_clust = survey_params_dict['tomos_6x2pt_clust']
            if self.zet_clust['tomos_6x2pt'] is not None and \
               any(tomos_6x2pt_clust != self.zet_clust['tomos_6x2pt']):
                raise Exception("not the same")
        elif self.zet_clust['tomos_6x2pt'] is not None:
            tomos_6x2pt_clust = self.zet_clust['tomos_6x2pt']

        if tomos_6x2pt_clust[0] is not None:
            if sum(tomos_6x2pt_clust) != self.n_tomo_clust:
                raise Exception('tomo clust 6x2pt mismatch')
            if len(tomos_6x2pt_clust) != 2:
                raise Exception("too ambitious 6x2pt")

            if survey_params_dict['survey_area_clust'] is None:
                ...
            elif len(survey_params_dict['survey_area_clust']) == 1:
                survey_params_dict['survey_area_clust'] = \
                    np.array([survey_params_dict['survey_area_clust'][0]]*3)
                print("The survey areas for the 3 clustering measurements " +
                      "within the 6x2pt analysis are implicitely set to be " +
                      "all the same: " +
                      str(survey_params_dict['survey_area_clust'][0]) + "sqdeg.")
            elif len(survey_params_dict['survey_area_clust']) != 3:
                raise Exception("shouldn't happen, there must be 3 clust " +
                                "areas")
            if survey_params_dict['survey_area_ggl'] is None:
                ...
            elif len(survey_params_dict['survey_area_ggl']) == 1:
                survey_params_dict['survey_area_ggl'] = \
                    np.array([survey_params_dict['survey_area_ggl'][0]]*2)
                print("The survey areas for the 2 galaxy-galaxy lensing " +
                      "measurements within the 6x2pt analysis are implicitely " +
                      "set to be all the same: " +
                      str(survey_params_dict['survey_area_ggl'][0]) + "sqdeg.")
            elif len(survey_params_dict['survey_area_ggl']) != 2:
                raise Exception("shouldn't happen, there must be 2 ggl areas")

            if survey_params_dict['survey_area_lens'] is None:
                ...
            elif len(survey_params_dict['survey_area_lens']) != 1:
                raise Exception("shouldnt happen, no specz shapes")

        return tomos_6x2pt_clust[0]

    def __consistency_checks_for_k_support_in_tabs(self,
                                                   powspec_prec):
        """
        Performs consistency checks for the tabulated power spectra and
        the configuration in [powspec evaluation]. Gives warnings if
        they are incompatible.

        Parameters
        ------------
        powspec_prec : dictionary
            Contains precision information about the power spectra, 
            this includes k-range and spacing. To be passed from the 
            read_input method of the Input class.

        """
        k_min = 10**(powspec_prec['log10k_min'])
        k_max = 10**(powspec_prec['log10k_max'])
        if self.Pxy_tab['k'] is not None:
            if k_min < self.Pxy_tab['k'][0]:
                print("SetupWarning: The minimum wavenumber support of the " +
                      "tabulated power spectra is log10k = " +
                      str(round(np.log10(self.Pxy_tab['k'][0]), 4)) + " while " +
                      "the requested range for the power spectra is " +
                      str(round(np.log10(k_min), 4)) + ". The minimum " +
                      "wavenumber will be truncated to the tabulated input.")
                powspec_prec['log10k_min'] = np.log10(self.Pxy_tab['k'][0])
            if k_max > self.Pxy_tab['k'][-1]:
                print("SetupWarning: The maximum wavenumber support of the " +
                      "tabulated power spectra is log10k = " +
                      str(round(np.log10(self.Pxy_tab['k'][-1]), 4)) + " " +
                      "while the requested range for the power spectra is " +
                      str(round(np.log10(k_max), 4)) + ". The maximum " +
                      "wavenumber will be truncated to the tabulated input.")
                powspec_prec['log10k_max'] = np.log10(self.Pxy_tab['k'][-1])

        return True

    def __consistency_checks_for_M_support_in_tabs(self,
                                                   bias_dict,
                                                   hm_prec):
        """
        Performs consistency checks for the halo occupation distribution
        and the configuration in [bias] and [halomodel evaluation]. For 
        the mass range, the priority is occnum_tab > occprob_tab > 
        mor_tab, only the highest priority is checked since only the
        HOD class will ignore the rest. Gives warnings if they are 
        incompatible.

        Parameters
        ------------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be 
            passed from the read_input method of the Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also, 
            see hmf documentation by Steven Murray), this includes mass 
            range and spacing for the mass integrations in the halo 
            model. To be passed from the read_input method of the Input 
            class.

        """
        if self.occnum_tab['M'] is not None:
            checkaux = \
                abs(hm_prec['log10M_min'] - np.log10(self.occnum_tab['M'][0]))
            if checkaux > 1e-3:
                print("SetupWarning: The minimum log10(mass) support of the " +
                      "tabulated occupation number is log10(M) = " +
                      str(round(np.log10(self.occnum_tab['M'][0]), 2)) + " " +
                      "while the requested range for the mass is " +
                      str(round(hm_prec['log10M_min'], 2)) + ". The minimum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_min'] = np.log10(self.occnum_tab['M'][0])
            checkaux = \
                abs(hm_prec['log10M_max'] - np.log10(self.occnum_tab['M'][-1]))
            if checkaux > 1e-3:
                print("SetupWarning: The maximum log10(mass) support of the " +
                      "tabulated occupation number is log10(M) = " +
                      str(round(np.log10(self.occnum_tab['M'][-1]), 2)) + " " +
                      "while the requested range for the mass is " +
                      str(round(hm_prec['log10M_max'], 2)) + ". The maximum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_max'] = np.log10(self.occnum_tab['M'][-1])
            if hm_prec['M_bins'] != len(self.occnum_tab['M']):
                print("SetupWarning: The tabulated occupation number has " +
                      str(len(self.occnum_tab['M'])) + " mass bins, while " +
                      "the number of requested mass bins is " +
                      str(hm_prec['M_bins']) + ". The number of mass bins " +
                      "will be set to the tabulated input.")
                hm_prec['M_bins'] = len(self.occnum_tab['M'])

        elif self.occprob_tab['M'] is not None:
            checkaux = \
                abs(hm_prec['log10M_min'] - np.log10(self.occprob_tab['M'][0]))
            if checkaux > 1e-3:
                print("SetupWarning: The minimum log10(mass) support of the " +
                      "tabulated occupation probability is log10(M) = " +
                      str(round(np.log10(self.occprob_tab['M'][0]), 2)) + " " +
                      "while the requested range for the mass is " +
                      str(round(hm_prec['log10M_min'], 2)) + ". The minimum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_min'] = np.log10(self.occprob_tab['M'][0])
            checkaux = \
                abs(hm_prec['log10M_max']
                    - np.log10(self.occprob_tab['M'][-1]))
            if checkaux > 1e-3:
                print("SetupWarning: The maximum log10(mass) support of the " +
                      "tabulated occupation probability is log10(M) = " +
                      str(round(np.log10(self.occprob_tab['M'][-1]), 2)) +
                      " while the requested range for the mass is " +
                      str(round(hm_prec['log10M_max'], 2)) + ". The maximum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_max'] = np.log10(self.occprob_tab['M'][-1])
            if hm_prec['M_bins'] != len(self.occprob_tab['M']):
                print("SetupWarning: The tabulated occupation probability " +
                      "has " + str(len(self.occprob_tab['M'])) + " mass " +
                      "bins, while the number of requested mass bins is " +
                      str(hm_prec['M_bins']) + ". The number of mass bins " +
                      "will be set to the tabulated input.")
                hm_prec['M_bins'] = len(self.occprob_tab['M'])

        elif self.mor_tab['M'] is not None:
            checkaux = \
                abs(hm_prec['log10M_min'] - np.log10(self.mor_tab['M'][0]))
            if checkaux > 1e-3:
                print("SetupWarning: The minimum log10(mass) support of the " +
                      "tabulated mass-observable relation is log10(M) = " +
                      str(round(np.log10(self.mor_tab['M'][0]), 2)) + " " +
                      "while the requested range for the mass is " +
                      str(round(hm_prec['log10M_min'], 2)) + ". The minimum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_min'] = np.log10(self.mor_tab['M'][0])
            checkaux = \
                abs(hm_prec['log10M_max'] - np.log10(self.mor_tab['M'][-1]))
            if checkaux > 1e-3:
                print("SetupWarning: The maximum log10(mass) support of the " +
                      "tabulated mass-observable relation is log10(M) = " +
                      str(round(np.log10(self.mor_tab['M'][-1]), 2)) + " " +
                      "while the requested range for the mass is " +
                      str(round(hm_prec['log10M_max'], 2)) + ". The maximum " +
                      "mass will be set to the tabulated input.")
                hm_prec['log10M_max'] = np.log10(self.mor_tab['M'][-1])
            if hm_prec['M_bins'] != len(self.mor_tab['M']):
                print("SetupWarning: The tabulated mass-observable relation " +
                      "has " + str(len(self.mor_tab['M'])) + " mass bins, " +
                      "while the number of requested mass bins is " +
                      str(hm_prec['M_bins']) + ". The number of mass bins " +
                      "will be set to the tabulated input.")
                hm_prec['M_bins'] = len(self.mor_tab['M'])

        if self.occprob_tab['M'] is not None and self.sample_dim > 1:
            for sample in range(self.sample_dim):
                checkaux = \
                    abs(bias_dict['logmass_bins'][sample]
                        - np.log10(self.occprob_tab['Mbins'][sample, 0]))
                if checkaux > 1e-3:
                    print("SetupWarning: The minimum log10(mass) per sample " +
                          "bin " + str(sample) + " support of the tabulated " +
                          "occupation probability is log10(M) = " +
                          str(round(np.log10(
                              self.occprob_tab['Mbins'][sample, 0]), 2)) +
                          " while the requested range for the mass is " +
                          str(round(bias_dict['logmass_bins'][sample], 2)) +
                          ". The maximum mass will be set to the tabulated " +
                          "input.")
                    bias_dict['logmass_bins'][sample] = \
                        np.log10(self.occprob_tab['Mbins'][sample, 0])
                checkaux = \
                    abs(bias_dict['logmass_bins'][sample+1]
                        - np.log10(self.occprob_tab['Mbins'][sample, -1]))
                if checkaux > 1e-3:
                    print("SetupWarning: The maximum log10(mass) per sample " +
                          "bin " + str(sample) + " support of the tabulated " +
                          "occupation probability is log10(M) = " +
                          str(round(np.log10(
                              self.occprob_tab['Mbins'][sample, -1]), 2)) +
                          " while the requested range for the mass is " +
                          str(round(bias_dict['logmass_bins'][sample+1], 2)) +
                          ". The maximum mass will be set to the tabulated " +
                          "input.")
                    bias_dict['logmass_bins'][sample+1] = \
                        np.log10(self.occprob_tab['Mbins'][sample, -1])

        return True

    def __check_z_distribution_value_loc_in_bin(self):
        """
        Ensuring that the redshift distribution contains values which 
        are situated in the middle of any redshift bin.
        """
        if self.zet_clust['value_loc_in_bin'] == 'cen':
            self.zet_clust['z'] = self.zet_clust['z']
        if self.zet_lens['value_loc_in_bin'] == 'cen':
            self.zet_lens['z'] = self.zet_lens['z']
        if self.n_tomo_clust != 0:
            if self.zet_clust['value_loc_in_bin'] != 'left' and self.zet_clust['z'][0] == 0:
                if self.zet_clust['value_loc_in_bin'] != 'cen':
                    raise Exception("If the lowest redshift value in the clustering redshift distribution is zero, 'value_loc_in_clustbin' must be set to 'left' in the config file.")
        if self.n_tomo_lens != 0:
            if self.zet_lens['value_loc_in_bin'] != 'left' and self.zet_lens['z'][0] == 0:
                if self.zet_lens['value_loc_in_bin'] != 'cen':
                    raise Exception("If the lowest redshift value in the lensing redshift distribution is zero, 'value_loc_in_lensbin' must be set to 'left' in the config file.")

        if self.n_tomo_clust == 0:
            ...
        elif self.zet_clust['value_loc_in_bin'] == 'left':
            self.zet_clust['z'] = np.append(
                .5*(self.zet_clust['z'][1:] + self.zet_clust['z'][:-1]),
                1.5*self.zet_clust['z'][-1] - .5*self.zet_clust['z'][-2])
        elif self.zet_clust['value_loc_in_bin'] == 'right':
            self.zet_clust['z'] = np.append(
                1.5*self.zet_clust['z'][:-1] - .5*self.zet_clust['z'][1:],
                .5*(self.zet_clust['z'][-1] + self.zet_clust['z'][-2]))
        else:
            ...
        # if self.n_tomo_clust > 0:
        #     if self.zet_clust['z'][0] > 1e-4:
        #         self.zet_clust['z'] = np.insert(self.zet_clust['z'], 0, 0)
        #         self.zet_clust['nz'] = np.insert(self.zet_clust['nz'], 0, 0)
        #         if self.n_tomo_lens > 0:
        #             self.zet_lens['photoz'] = \
        #                 np.insert(self.zet_lens['photoz'], 0, 0)
        #     if self.zet_clust['nz'][-1] > 0:
        #         self.zet_max += self.zet_clust['z'][-1] - self.zet_clust['z'][-2]
        #         self.zet_clust['z'] = \
        #             np.append(self.zet_clust['z'], self.zet_max)
        #         self.zet_clust['nz'] = np.append(self.zet_clust['nz'], 0)
        #         if self.n_tomo_lens > 0:
        #             self.zet_lens['photoz'] = \
        #                 np.append(self.zet_lens['photoz'], 0)
        if self.n_tomo_lens == 0:
            ...
        elif self.zet_lens['value_loc_in_bin'] == 'left':
            self.zet_lens['z'] = np.append(
                .5*(self.zet_lens['z'][1:] + self.zet_lens['z'][:-1]),
                1.5*self.zet_lens['z'][-1] - .5*self.zet_lens['z'][-2])
        elif self.zet_lens['value_loc_in_bin'] == 'right':
            self.zet_lens['z'] = np.append(
                1.5*self.zet_lens['z'][:-1] - .5*self.zet_lens['z'][1:],
                .5*(self.zet_lens['z'][-1] + self.zet_lens['z'][-2]))
        else:
            ...
        # if self.n_tomo_lens > 0:
        #     if self.zet_lens['z'][0] > 1e-4:
        #         self.zet_lens['z'] = np.insert(self.zet_lens['z'], 0, 0)
        #         self.zet_lens['photoz'] = \
        #             np.insert(self.zet_lens['photoz'], 0, 0)
        #     if self.zet_lens['photoz'][-1] > 0:
        #         self.zet_max += self.zet_lens['z'][-1] - self.zet_lens['z'][-2]
        #         self.zet_lens['z'] = \
        #             np.append(self.zet_lens['z'], self.zet_max)
        #         self.zet_lens['photoz'] = np.append(self.zet_lens['photoz'], 0)

        return self.zet_clust, self.zet_lens

    def consistency_checks_for_Cell_calculation(self,
                                                obs_dict,
                                                cosmo_dict,
                                                powspec_prec,
                                                ellrange,
                                                los_chi,
                                                do_not_update_ell):

        if abs(cosmo_dict['omega_m'] + cosmo_dict['omega_de'] - 1) > 1e-5:
            print("SetupWarning: The cosmology is not flat and the Cell " +
                  "calculation (currently) assumes a flat cosmology. This " +
                  "might cause a bias.")

        update_massfunc, update_ellrange = False, False
        calc_theta = False
        if not obs_dict['arbitrary_summary']['do_arbitrary_summary']:
            if obs_dict['observables']['cosmic_shear'] and obs_dict['observables']['est_shear'] == 'xi_pm':
                calc_theta = True
            if obs_dict['observables']['ggl'] and obs_dict['observables']['est_ggl'] == 'gamma_t':
                calc_theta = True
            if obs_dict['observables']['clustering'] and obs_dict['observables']['est_clust'] == 'w':
                calc_theta = True
        calc_cosebi = False
        if not obs_dict['arbitrary_summary']['do_arbitrary_summary']:
            if obs_dict['observables']['cosmic_shear'] and obs_dict['observables']['est_shear'] == 'cosebi':
                calc_cosebi = True
            if obs_dict['observables']['ggl'] and obs_dict['observables']['est_ggl'] == 'cosebi':
                calc_cosebi = True
            if obs_dict['observables']['clustering'] and obs_dict['observables']['est_clust'] == 'cosebi':
                calc_cosebi = True
        calc_bandpower = False
        if not obs_dict['arbitrary_summary']['do_arbitrary_summary']:
            if obs_dict['observables']['cosmic_shear'] and obs_dict['observables']['est_shear'] == 'bandpowers':
                calc_bandpower = True
            if obs_dict['observables']['ggl'] and obs_dict['observables']['est_ggl'] == 'bandpowers':
                calc_bandpower = True
            if obs_dict['observables']['clustering'] and obs_dict['observables']['est_clust'] == 'bandpowers':
                calc_bandpower = True
        calc_ell = False
        if not obs_dict['arbitrary_summary']['do_arbitrary_summary']:
            for observables in obs_dict['observables'].values():
                if np.any(observables == 'C_ell'):
                    calc_ell = True
            calc_ell = True if calc_theta else False
            calc_ell = True if calc_cosebi else False
        else:
            calc_ell = True
        
            
        ellmin, ellmax = ellrange[0], ellrange[-1]
        ell_bins = len(ellrange)
        if calc_cosebi or calc_bandpower:
            if ellrange[0] > 2 and not do_not_update_ell:
                print("SetupWarning: The COSEBI covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the minimum ell mode "
                      "to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] < 2 and not do_not_update_ell:
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] == 0 and not do_not_update_ell:
                print("SetupWarning: Setting to minimum ell mode to 0 is " +
                      "quite brave. For numerical safety reasons, we adjust " +
                      "it to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True

            if ellrange[-1] < 1e4 and not obs_dict['ELLspace']['pixelised_cell'] and not do_not_update_ell:
                print("SetupWarning: The COSEBI covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the maximum ell mode "
                      "to '[covELLspace settings]: ell_max = 1e4'.")
                ellmax = 1e4
                update_ellrange = True
            if len(ellrange) < 100 and not do_not_update_ell:
                print("SetupWarning: The COSEBI covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the number of ell modes "
                      "to '[covELLspace settings]: ell_bins = 100'.")
                ell_bins = 100
                update_ellrange = True
        if calc_theta:
            if ellrange[0] > 2 and not do_not_update_ell:
                print("SetupWarning: The xi_pm covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the minimum ell mode "
                      "to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] < 2 and not do_not_update_ell:
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] == 0 and not do_not_update_ell:
                print("SetupWarning: Setting to minimum ell mode to 0 is " +
                      "quite brave. For numerical safety reasons, we adjust " +
                      "it to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True

            if ellrange[-1] < 1e4 and not obs_dict['ELLspace']['pixelised_cell'] and not do_not_update_ell:
                print("SetupWarning: The xi_pm covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the maximum ell mode "
                      "to '[covELLspace settings]: ell_max = 1e4'.")
                ellmax = 1e4
                update_ellrange = True
            if len(ellrange) < 100 and not do_not_update_ell:
                print("SetupWarning: The xi_pm covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the number of ell modes "
                      "to '[covELLspace settings]: ell_bins = 100'.")
                ell_bins = 100
                update_ellrange = True
        if obs_dict['arbitrary_summary']['do_arbitrary_summary']:
            if ellrange[0] > 2 and not do_not_update_ell:
                print("SetupWarning: The arbitrary summary covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the minimum ell mode "
                      "to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] < 2 and not do_not_update_ell:
                ellmin = 2
                update_ellrange = True
            elif ellrange[0] == 0 and not do_not_update_ell:
                print("SetupWarning: Setting to minimum ell mode to 0 is " +
                      "quite brave. For numerical safety reasons, we adjust " +
                      "it to '[covELLspace settings]: ell_min = 2'.")
                ellmin = 2
                update_ellrange = True

            if ellrange[-1] < 1e4 and not obs_dict['ELLspace']['pixelised_cell'] and not do_not_update_ell:
                print("SetupWarning: The arbitrary summary covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the maximum ell mode "
                      "to '[covELLspace settings]: ell_max = 1e4'.")
                ellmax = 1e4
                update_ellrange = True
            if len(ellrange) < 100 and not do_not_update_ell:
                print("SetupWarning: The arbitrary summary covariance is currently " +
                      "calculated via the projected powerspectra (C_ells), " +
                      "the projection integral runs formally from 0 to " +
                      "infinity. We, therefore, adjust the number of ell modes "
                      "to '[covELLspace settings]: ell_bins = 100'.")
                ell_bins = 100
                update_ellrange = True

        kmin, kmax = None, None
        if calc_ell or calc_theta or calc_cosebi:
            kmin = np.log10(ellmin / los_chi[-1])
            kmax = np.log10(ellmax / los_chi[0])
            if kmax > 1e4:
                print("SetupWarning: The required kmax is " +
                      str(round(kmax, 2)) + "Mpc which is quite demanding " +
                      "regarding our knowledge of an accurate power " +
                      "spectrum at these scales. Calculations will still be " +
                      "done, but caution is advised.")

            if powspec_prec['log10k_min'] > kmin:
                print("According to the input minimum ell mode and redshift " +
                      "distribution, the minimum logarithmic wavenumber " +
                      "[powspec evaluation]: 'log10k_min' is set to " +
                      str(np.round(kmin, 2)) + ".")
                update_massfunc = True
            elif powspec_prec['log10k_min'] < kmin-1:
                print("According to the input minimum ell mode and redshift " +
                      "distribution, the minimum logarithmic wavenumber " +
                      "[powspec evaluation]: 'log10k_min' set in the config " +
                      "file is at least an order of magnitude lower than " +
                      "required. Consider updating your settings to " +
                      "'log10kmin = " + str(np.round(kmin*1.01, 2)) + ".")
                kmin = powspec_prec['log10k_min']
            else:
                kmin = powspec_prec['log10k_min']

            if powspec_prec['log10k_max'] < kmax:
                print("According to the input maximum ell mode and redshift " +
                      "distribution, the maximum logarithmic wavenumber " +
                      "[powspec evaluation]: 'log10k_max' is set to " +
                      str(np.round(kmax, 2)) + ".")
                update_massfunc = True
            elif powspec_prec['log10k_max'] > kmax+1:
                print(powspec_prec['log10k_max'], kmax)
                print("According to the input maximum ell mode and redshift " +
                      "distribution, the maximum logarithmic wavenumber " +
                      "[powspec evaluation]: 'log10k_max' set in the config " +
                      "file is at least an order of magnitude larger than " +
                      "required. Consider updating your settings to " +
                      "'log10kmax = " + str(np.round(kmax*1.01, 2)) + ".")
                kmax = powspec_prec['log10k_max']
            else:
                kmax = powspec_prec['log10k_max']

            if update_massfunc:
                powspec_prec['log10k_min'] = kmin
                powspec_prec['log10k_max'] = kmax
                self.__consistency_checks_for_k_support_in_tabs(powspec_prec)

        return update_massfunc, update_ellrange, kmin, kmax, ellmin, ellmax, ell_bins

    def get_unique_elements(self,
                            mode,
                            covELL):
        """
        Returns the unique elements of a data vector
        """

        ldim, sdim = covELL.shape[0], covELL.shape[2]
        uniqu_elem = np.ones(covELL.shape)
        uniqu_elem = uniqu_elem \
            * np.triu(np.ones(ldim))[:, :, None, None, None, None, None, None]
        uniqu_elem = uniqu_elem \
            * np.triu(np.ones(sdim))[None, None, :, :, None, None, None, None]
        if mode == 'diag':
            t1dim, t2dim = covELL.shape[4], covELL.shape[5]
            triu4d = np.triu(np.ones((t1dim*t2dim, t1dim*t2dim))
                             ).reshape((t1dim, t2dim, t1dim, t2dim))
            uniqu_elem = uniqu_elem * \
                triu4d[None, None, None, None, :, :, :, :]

        return uniqu_elem

    def symmetrise_matrix(self,
                          mode,
                          covELL):
        ldim, sdim = covELL.shape[0], covELL.shape[2]
        tril = np.tril(np.ones(ldim), -1)[:, :, None, None, None, None, None, None] \
            * covELL.transpose(1, 0, 2, 3, 4, 5, 6, 7)
        covELL_sym = covELL + tril
        tril = np.tril(np.ones(sdim), -1)[None, None, :, :, None, None, None, None] \
            * covELL.transpose(0, 1, 3, 2, 4, 5, 6, 7)
        covELL_sym = covELL_sym + tril
        if mode == 'diag':
            t1dim, t2dim = covELL.shape[4], covELL.shape[5]
            tril4d = np.tril(np.ones((t1dim*t2dim, t1dim*t2dim)), -
                             1).reshape((t1dim, t2dim, t1dim, t2dim))
            tril = tril4d * covELL.transpose(0, 1, 2, 3, 6, 7, 4, 5)
            covELL_sym = covELL_sym + tril

        return covELL_sym

    def __rebin_npair(self, npairs, theta_npair, theta_ul_bins, theta_bins, analytic, name):
        """
        Sums the tabulated pair counts npairs (angular bins at theta_npair,
        the nominal centres of logarithmic bins) into the angular bins with
        edges theta_ul_bins. Bins that the table does not fully cover and
        combinations without a file (NaN) get the analytic pair counts (same
        shape as the result); covered bins without any tabulated pairs are
        interpolated (in log theta) from the covered bins with pairs.
        """
        n_theta = len(theta_ul_bins) - 1
        pair_rebinned = np.zeros((n_theta,) + npairs.shape[1:])
        bin_index = np.searchsorted(theta_ul_bins, theta_npair, side='left') - 1
        inside = (bin_index >= 0) & (bin_index < n_theta)
        np.add.at(pair_rebinned, bin_index[inside], npairs[inside])
        # range covered by the table (half a bin beyond the outermost centres, with half a bin of tolerance for rounding)
        dlog_lo = np.log(theta_npair[1]/theta_npair[0]) if len(theta_npair) > 1 else 0.0
        dlog_hi = np.log(theta_npair[-1]/theta_npair[-2]) if len(theta_npair) > 1 else 0.0
        covered = (np.log(theta_ul_bins[:-1]) >= np.log(theta_npair[0]) - dlog_lo) & \
                  (np.log(theta_ul_bins[1:]) <= np.log(theta_npair[-1]) + dlog_hi)
        log_theta = np.log(theta_bins)
        for index in np.ndindex(npairs.shape[1:]):
            column = pair_rebinned[(slice(None),) + index]
            if np.all(np.isnan(npairs[(slice(None),) + index])):
                column[:] = analytic[(slice(None),) + index]
                continue
            good = covered & (column > 0)
            holes = covered & ~(column > 0)
            if np.any(holes):
                if np.sum(good) < 2:
                    raise Exception("SetupError: The paircount file you have passed does not support the desired angular range "
                                    "and a rebinning is not possible. There is at least one theta bin without any pairs. Please "
                                    "check the files or the theta range")
                column[holes] = np.interp(log_theta[holes], log_theta[good], column[good])
            column[~covered] = analytic[(~covered,) + index]
        if not np.all(covered):
            print("Warning: The pair counts for " + name + " do not cover the angular bin(s) " +
                  ", ".join("[" + str(round(lo, 4)) + ", " + str(round(hi, 4)) + "]"
                            for lo, hi, c in zip(theta_ul_bins[:-1], theta_ul_bins[1:], covered) if not c) +
                  " (arcmin) fully, using the analytic number of pairs from the effective number density there.")
        return pair_rebinned

    @staticmethod
    def __dnpair_from_table(npair, theta_npair, theta_ul_bins, analytic, name):
        """
        The number of pairs per arcmin, dN_pair/dtheta, at theta_ul_bins
        from the tabulated pair counts npair (angular bins at theta_npair,
        then tomographic bin i, j and sample). Outside of the tabulated
        range and for combinations without a file (NaN) the analytic
        dN_pair/dtheta (same shape as the result) is used.
        """
        dnpair = (npair[1:] + npair[:-1])/2./(theta_npair[1:] - theta_npair[:-1])[:, None, None, None]
        theta_mid = (theta_npair[1:] - theta_npair[:-1])/2 + theta_npair[:-1]
        if theta_ul_bins[0] < theta_npair[0] or theta_ul_bins[-1] > theta_npair[-1]:
            print("Warning: The provided file for the pair counts for the " + name + " has a smaller angular range than required. "
                  "In particular we require theta between: " + str(theta_ul_bins[0]) + " and " + str(theta_ul_bins[-1]) + ", but "
                  "the pair counts file only provides it from " + str(theta_npair[0]) + " to " + str(theta_npair[-1]) +
                  ". Will use the analytic formula over the extended range. This might cause unwanted behaviour. Please provide an npair file "
                  "extending over the requested angular range.")
        outside = (theta_ul_bins < theta_npair[0]) | (theta_ul_bins > theta_npair[-1])
        result = np.copy(analytic)
        for index in np.ndindex(dnpair.shape[1:]):
            column = dnpair[(slice(None),) + index]
            if np.all(np.isnan(column)):
                continue
            spline = interp1d(theta_mid, column, fill_value="extrapolate")
            result[(~outside,) + index] = spline(theta_ul_bins[~outside])
        return result

    def get_dnpair(self,
                   obsbool,
                   theta_ul_bins,
                   survey_params_dict,
                   npair_tab):
        """
        The number of pairs per arcmin, dN_pair/dtheta, at theta_ul_bins for
        clustering, galaxy-galaxy lensing and cosmic shear with shape
        (theta, sample, tomographic bin i, j), from the pair count files if
        given and from the effective number densities otherwise.
        """
        gg, gm, mm = obsbool
        ring = 2.0*np.pi*theta_ul_bins[:, None, None, None]*60*60
        dnpair_gg, dnpair_gm, dnpair_mm = None, None, None
        theta_gg, theta_gm, theta_mm = None, None, None
        if gg:
            analytic = ring*survey_params_dict['survey_area_clust'][0] \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_clust'][None, None, :, :]
            if npair_tab['npair_gg'] is not None:
                dnpair_gg = self.__dnpair_from_table(npair_tab['npair_gg'], npair_tab['theta_gg'], theta_ul_bins,
                                                     analytic, 'shot noise')
            else:
                print("Approximating the clustering real space shot noise " +
                      "contribution with the effective number density.")
                dnpair_gg = analytic
            dnpair_gg = dnpair_gg.transpose(0, 3, 1, 2)
            theta_gg = theta_ul_bins
        if gm:
            analytic = ring*survey_params_dict['survey_area_ggl'][0] \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
            if npair_tab['npair_gm'] is not None:
                dnpair_gm = self.__dnpair_from_table(npair_tab['npair_gm'], npair_tab['theta_gm'], theta_ul_bins,
                                                     analytic, 'GGL noise')
            else:
                print("Approximating the ggl real space shot noise " +
                      "contribution with the effective number density.")
                dnpair_gm = analytic
            dnpair_gm = dnpair_gm.transpose(0, 3, 1, 2)
            theta_gm = theta_ul_bins
        if mm:
            analytic = ring*survey_params_dict['survey_area_lens'][0] \
                * survey_params_dict['n_eff_lens'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
            if npair_tab['npair_mm'] is not None:
                dnpair_mm = self.__dnpair_from_table(npair_tab['npair_mm'], npair_tab['theta_mm'], theta_ul_bins,
                                                     analytic, 'shape noise')
            else:
                print("Approximating the lensing real space shape noise " +
                      "contribution with the effective number density.")
                dnpair_mm = analytic
            dnpair_mm = dnpair_mm.transpose(0, 3, 1, 2)
            theta_mm = theta_ul_bins
        return dnpair_gg, dnpair_gm, dnpair_mm, theta_gg, theta_gm, theta_mm

    def get_npair(self,
                  obsbool,
                  theta_ul_bins,
                  theta_bins,
                  survey_params_dict,
                  npair_tab):

        gg, gm, mm = obsbool
        theta_lin = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])
        theta_log = 10**(.5 * (np.log10(theta_ul_bins[1:])
                               + np.log10(theta_ul_bins[:-1])))

        area = np.pi*(theta_ul_bins[1:]**2 - theta_ul_bins[:-1]**2)[:, None, None, None]*60*60
        if gg:
            analytic_gg = area*survey_params_dict['survey_area_clust'][0] \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_clust'][None, None, :, :]
        if gm:
            analytic_gm = area*survey_params_dict['survey_area_ggl'][0] \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
        if mm:
            analytic_mm = area*survey_params_dict['survey_area_lens'][0] \
                * survey_params_dict['n_eff_lens'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
        if gg and npair_tab['npair_gg'] is not None:
            if len(npair_tab['theta_gg']) == len(theta_ul_bins)-1 and \
               (np.allclose(npair_tab['theta_gg'], theta_lin) or
                    np.allclose(npair_tab['theta_gg'], theta_log)):
                # combinations without a file: analytic number of pairs
                npair_gg = np.where(np.isnan(npair_tab['npair_gg']), analytic_gg, npair_tab['npair_gg'])
            else:
                print("Warning: The number of pairs for clustering is not " +
                      "given in the same angular bins / range as is queried " +
                      "in the config. The rebinning routine just sums up " +
                      "N_pairs until they match the requested binning. This " +
                      "may not be correct, e.g., in the presence of weights. ")
                npair_gg = self.__rebin_npair(npair_tab['npair_gg'],
                                              npair_tab['theta_gg'],
                                              theta_ul_bins,
                                              theta_bins,
                                              analytic_gg, 'clustering')
            npair_gg = npair_gg.transpose(0, 3, 1, 2)
        elif gg:
            print("Approximating the clustering real space shot noise " +
                  "contribution with the effective number density.")
            npair_gg = np.pi \
                * (theta_ul_bins[1:, None, None, None]**2
                   - theta_ul_bins[:-1, None, None, None]**2) \
                * survey_params_dict['survey_area_clust'][0] * 60*60 \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_clust'][None, None, :, :]
            npair_gg = npair_gg.transpose(0, 3, 1, 2)
        else:
            npair_gg = None

        if gm and npair_tab['npair_gm'] is not None:
            if len(npair_tab['theta_gm']) == len(theta_ul_bins)-1 and \
               (np.allclose(npair_tab['theta_gm'], theta_lin) or
                    np.allclose(npair_tab['theta_gm'], theta_log)):
                # combinations without a file: analytic number of pairs
                npair_gm = np.where(np.isnan(npair_tab['npair_gm']), analytic_gm, npair_tab['npair_gm'])
            else:
                print("Warning: The number of pairs for galaxy-galaxy " +
                      "lensing is not given in the same angular bins / " +
                      "range as is queried in the config. The rebinning " +
                      "routine just sums up N_pairs until they match the " +
                      "requested binning. This may not be correct, e.g., in " +
                      "the presence of weights. ")
                npair_gm = self.__rebin_npair(npair_tab['npair_gm'],
                                              npair_tab['theta_gm'],
                                              theta_ul_bins,
                                              theta_bins,
                                              analytic_gm, 'galaxy-galaxy lensing')
            npair_gm = npair_gm.transpose(0, 3, 1, 2)
        elif gm:
            print("Approximating the galaxy-galaxy lensing real space shot " +
                  "noise contribution with the effective number density of " +
                  "lens galaxies.")
            npair_gm = np.pi \
                * (theta_ul_bins[1:, None, None, None]**2
                   - theta_ul_bins[:-1, None, None, None]**2) \
                * survey_params_dict['survey_area_ggl'][0] * 60*60 \
                * survey_params_dict['n_eff_clust'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
            npair_gm = npair_gm.transpose(0, 3, 1, 2)
        else:
            npair_gm = None

        if mm and npair_tab['npair_mm'] is not None:
            if len(npair_tab['theta_mm']) == len(theta_ul_bins)-1 and \
               (np.allclose(npair_tab['theta_mm'], theta_lin) or
                    np.allclose(npair_tab['theta_mm'], theta_log)):
                # combinations without a file: analytic number of pairs
                npair_mm = np.where(np.isnan(npair_tab['npair_mm']), analytic_mm, npair_tab['npair_mm'])
            else:
                print("Warning: The number of pairs for cosmic shear is not " +
                      "given in the same angular bins / range as is queried " +
                      "in the config. The rebinning routine just sums up " +
                      "N_pairs until they match the requested binning. This " +
                      "may not be correct, e.g., in the presence of weights. ")
                npair_mm = self.__rebin_npair(npair_tab['npair_mm'],
                                              npair_tab['theta_mm'],
                                              theta_ul_bins,
                                              theta_bins,
                                              analytic_mm, 'cosmic shear')
            npair_mm = npair_mm.transpose(0, 3, 1, 2)
        elif mm:
            print("Approximating the cosmic shear real space shot noise " +
                  "contribution with the effective number density of source " +
                  "galaxies.")
            npair_mm = np.pi \
                * (theta_ul_bins[1:, None, None, None]**2
                   - theta_ul_bins[:-1, None, None,  None]**2) \
                * survey_params_dict['survey_area_lens'][0] * 60*60 \
                * survey_params_dict['n_eff_lens'][None, :, None, :] \
                * survey_params_dict['n_eff_lens'][None, None, :, :]
            npair_mm = npair_mm.transpose(0, 3, 1, 2)
        else:
            npair_mm = None

        return npair_gg, npair_gm, npair_mm

    @staticmethod
    def _read_healpix_mask(mfile):
        """
        Reads a HEALPix mask with healpy, which handles full-sky and
        partial-sky (explicitly indexed) maps and NESTED ordering (the map
        is returned in RING ordering as needed by anafast). Unobserved
        pixels (UNSEEN, nan) are set to zero.
        """
        import healpy
        data = np.asarray(healpy.read_map(mfile, field=0, dtype=None), dtype=np.float64)
        data[~np.isfinite(data) | (data == healpy.UNSEEN)] = 0.0
        return data

    def calc_survey_area(self,
                         survey_params_dict):
        """
        Reads in the survey area from a mask file with healpy for which 
        the covariance should be calculated.

        Parameters
        ------------
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            This function needs the name of the mask files and the 
            read_in bool. To be passed from the read_input method of the 
            Input class. Updates the dictionary survey_params_dict with 
            the calculated survey areas.

        """
        if survey_params_dict['read_mask_clust_clust'] or \
           survey_params_dict['read_mask_ggl_ggl'] or \
           survey_params_dict['read_mask_lens_lens']:
            try:
                import healpy
            except ImportError:
                raise Exception("ImportError: To process the the mask file, " +
                                "healpy>=1.15.0 must be installed.")

        for est in ['clust', 'ggl', 'lens']:
            if not survey_params_dict['read_mask_'+est+'_'+est]:
                continue
            if survey_params_dict['survey_area_'+est] is not None:
                continue
            survey_params_dict['survey_area_'+est] = []
            save_mfile = []
            for mfile in survey_params_dict['mask_file_'+est+'_'+est]:
                if mfile in save_mfile:
                    survey_params_dict['survey_area_'+est].append(
                        survey_params_dict['survey_area_'+est]
                        [save_mfile.index(mfile)])
                    save_mfile.append(mfile)
                    continue
                save_mfile.append(mfile)

                print("Reading in the mask file " + mfile + " to get " +
                      "the survey area.")
                data = self._read_healpix_mask(mfile)
                Nside = healpy.npix2nside(len(data))
                survey_params_dict['survey_area_'+est].append(
                    np.sum(data)
                    * healpy.nside2pixarea(Nside, degrees=True))
        if survey_params_dict['survey_area_lens'] is None and \
           not survey_params_dict['read_mask_lens_lens']:
            if survey_params_dict['mask_file_lens_lens'] == \
               survey_params_dict['mask_file_clust_clust']:
                if survey_params_dict['survey_area_clust'] is not None:
                    survey_params_dict['survey_area_lens'] = \
                        survey_params_dict['survey_area_clust']

        if survey_params_dict['survey_area_ggl'] is None and \
           not survey_params_dict['read_mask_ggl_ggl']:
            if survey_params_dict['mask_file_ggl_ggl'] == \
               survey_params_dict['mask_file_clust_clust']:
                if survey_params_dict['survey_area_clust'] is not None:
                    survey_params_dict['survey_area_ggl'] = \
                        survey_params_dict['survey_area_clust']
            elif survey_params_dict['mask_file_ggl_ggl'] == \
                    survey_params_dict['mask_file_lens_lens']:
                if survey_params_dict['survey_area_lens'] is not None:
                    survey_params_dict['survey_area_ggl'] = \
                        survey_params_dict['survey_area_lens']

        if self.tomos_6x2pt_clust is not None:
            if len(survey_params_dict['survey_area_clust']) == 1:
                survey_params_dict['survey_area_clust'] = \
                    [survey_params_dict['survey_area_clust'][0]]*3
            elif len(survey_params_dict['survey_area_clust']) == 3:
                ...
            else:
                raise Exception("shouldn't happen clust area")

            if survey_params_dict['survey_area_ggl'] is None:
                ...
            elif len(survey_params_dict['survey_area_ggl']) == 1:
                survey_params_dict['survey_area_ggl'] = \
                    [survey_params_dict['survey_area_clust'][0]]*2
            elif len(survey_params_dict['survey_area_ggl']) == 2:
                ...
            else:
                raise Exception("shouldn't happen ggl area")

            if survey_params_dict['survey_area_lens'] is not None and \
               len(survey_params_dict['survey_area_lens']) != 1:
                raise Exception("shouldn't happen lens area")

        # probes that are not calculated have no survey area (None)
        for est in ['clust', 'ggl', 'lens']:
            if survey_params_dict['survey_area_'+est] is not None:
                survey_params_dict['survey_area_'+est] = \
                    np.array(survey_params_dict['survey_area_'+est])

        return True

    def calc_a_lm(self, est1, est2, survey_params_dict):
        """
        Calculates the angular modes that fit into the survey area from 
        a mask file with healpy or reads in those modes from an 
        alm_file. Checks whether the survey area(s) are available and 
        calculates them if not.

        Parameters
        ------------
        est1 : string
            specifies the estimator for which the a_lm's are needed,
            'gg' for clustering, or 'mm' cosmic shear
        est2 : string
            specifies the estimator for which the a_lm's are needed,
            'gm' for galaxy-galaxy lensing, or 'mm' cosmic shear
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            This function needs the name of the mask files and alm files 
            and the read_in bools. To be passed from the read_input 
            method of the Input class.

        Return
        ---------
        ell : list of arrays
            angular modes that fit into the survey area
        sum_m_a_lm : list of arrays
            with the same shape as ell
            the sum over all m modes per angular mode ell (l) 

        """

        import healpy
        from astropy.io import ascii
        if not hasattr(self, '_mask_cl_cache'):
            self._mask_cl_cache = {}
        if (est1 == 'mm' and est2 == 'gg') or est1 == 'gm':
            est1, est2 = est2, est1
        if est1 == 'gg':
            est1 = 'clust'
        elif est1 == 'gm':
            est1 = 'ggl'
        elif est1 == 'mm':
            est1 = 'lens'
        else:
            raise Exception("KeyError: The name of the first estimator must " +
                            "be either 'gg' for clustering, 'gm' for galaxy-galaxy " +
                            "lensing or 'mm' for cosmic shear.")
        if est2 == 'gg':
            est2 = 'clust'
        elif est2 == 'gm':
            est2 = 'ggl'
        elif est2 == 'mm':
            est2 = 'lens'
        else:
            raise Exception("KeyError: The name of the second estimator " +
                            "must be either 'gg' for clustering, 'gm' for galaxy-galaxy " +
                            "lensing or 'mm' for cosmic shear.")
        if survey_params_dict['survey_area_'+est1] is None or \
           survey_params_dict['survey_area_'+est2] is None:
            self.calc_survey_area(survey_params_dict)

        ell, sum_m_a_lm = [], []
        failure = False
        if survey_params_dict['read_alm_'+est1+'_'+est2]:
            if survey_params_dict['alm_file_'+est1+'_'+est2] is not None:
                for afile in survey_params_dict['alm_file_'+est1+'_'+est2]:
                    try:
                        print("Reading in a_lm file " + afile + ".")
                        data = ascii.read(afile)
                        ell.append(data[data.colnames[0]])
                        sum_m_a_lm.append(data[data.colnames[1]])
                    except:
                        print("WARNING: a_lm file " + afile + " not found. Will use circular mask for response")
                        ell, sum_m_a_lm = None, None
                        failure = True
            else:
                print("WARNING: Will use circular mask for response")
                ell, sum_m_a_lm = None, None
                failure = True
        elif survey_params_dict['read_mask_'+est1+'_'+est2]:
            for mfile in survey_params_dict['mask_file_'+est1+'_'+est1]:
                try:
                    print("Reading in the mask file " + mfile + " to get the " +
                        "survey area modes.")
                    data = self._read_healpix_mask(mfile)
                    Nside = healpy.npix2nside(len(data))
                    ellmax = 3 * Nside - 1
                    '''if est1 == est2:
                        # default range suggested by healpy
                        aux_ell = np.arange(ellmax)
                        ell.append(aux_ell)
                        C_ell = healpy.anafast(data, use_weights=True)
                        sum_m_a_lm.append((2 * aux_ell + 1) * C_ell[:ellmax])
                    else:'''
                    for mfile2 in \
                            survey_params_dict['mask_file_'+est1+'_'+est2]:
                        print("Reading in the second mask file " + mfile +
                            " to get the cross survey area modes.")
                        data2 = self._read_healpix_mask(mfile2)
                        Nside2 = healpy.npix2nside(len(data2))
                        ellmax2 = 3 * Nside2 - 1
                        ellmax = ellmax if ellmax < ellmax2 else ellmax2

                        aux_ell = np.arange(ellmax)
                        ell.append(aux_ell)
                        # the survey modes only depend on the masks, calculate them once per run
                        key = ('mask', mfile, mfile2)
                        if key not in self._mask_cl_cache:
                            self._mask_cl_cache[key] = healpy.anafast(data, data2, use_weights=True)
                        C_ell = self._mask_cl_cache[key]
                        sum_m_a_lm.append((2 * aux_ell + 1) * C_ell[:ellmax])
                    failure = False
                except Exception as error:
                    print("WARNING: could not process the mask file " + mfile + " (" +
                          type(error).__name__ + ": " + str(error) + "). Will use circular mask for response")
                    ell, sum_m_a_lm = None, None
                    failure = True
        else:
            ell, sum_m_a_lm = None, None

        if ell is not None and survey_params_dict['save_alms'] and not failure:
            if est1 == est2:
                filename = survey_params_dict['save_alms'] + \
                    '_' + est1 + '.ascii'
            else:
                filename = survey_params_dict['save_alms'] + '_' + est1 + '_' \
                    + est2 + '.ascii'
            for ellx, sum_m_a_lmx in zip(ell, sum_m_a_lm):
                print("Writing '" + filename + "'.")
                ascii.write([ellx, sum_m_a_lmx],
                            filename,
                            names=['#ell', 'sum_m_a_lm'],
                            overwrite=True)
        if ell is None:
            ell, sum_m_a_lm = [], []
            NSIDE = 1024
            NPIX = healpy.nside2npix(NSIDE)
            for i in range(len(survey_params_dict['survey_area_'+est1])):
                for j in range(len(survey_params_dict['survey_area_'+est2])):
                    areas = (survey_params_dict['survey_area_'+est1][i], survey_params_dict['survey_area_'+est2][j])
                    alms = []
                    for survey_area in areas:
                        key = ('circle_alm', survey_area, NSIDE)
                        if key not in self._mask_cl_cache:
                            survey_area_in_rad = survey_area*np.pi**2/(180**2)
                            radius_mask = np.arccos(1 - survey_area_in_rad / (2 * np.pi))
                            center_coord = healpy.ang2vec(0, 0)
                            ipix_disc = healpy.query_disc(nside=NSIDE, vec=center_coord, radius=radius_mask)
                            m = np.zeros(NPIX)
                            m[ipix_disc] = 1
                            self._mask_cl_cache[key] = healpy.map2alm(m, use_weights=True)
                        alms.append(self._mask_cl_cache[key])
                    ellmax = 3 * NSIDE - 1
                    aux_ell = np.arange(ellmax)
                    ell.append(aux_ell)
                    key = ('circle', min(areas), max(areas), NSIDE)
                    if key not in self._mask_cl_cache:
                        if areas[0] == areas[1]:
                            self._mask_cl_cache[key] = healpy.alm2cl(alms[0])
                        else:
                            self._mask_cl_cache[key] = healpy.alm2cl(alms[0], alms[1])
                    C_ell = self._mask_cl_cache[key]
                    sum_m_a_lm.append((2 * aux_ell + 1) * C_ell[:ellmax])
                    if areas[0] == areas[1]:
                        print("Assuming a circular mask with size",areas[0],"deg2 for",est1,"and",est2,".")
                    else:
                        print("Assuming concentric circular masks with sizes",areas[0],"and",areas[1],"deg2 for",est1,"and",est2,".")
        return ell, sum_m_a_lm


# setting = Setup(cosmo, bias, survey_params, hm_prec, powspec_prec,
#     read_in_tables)

class GaussLegendre:
    """
    This class implements Gauss legendre integration using the Legendre 
    polynomials to approximate an integral over an arbitrary function
    by a polynomial up to some order.

    Parameters
    ------------
    support : 1D array
        the support of the integrand, i.e. the integration variable
    integrand : 1D array
        the integrand, must be of the same length as support
    log_support: bool
        should the integration variable be interpolated logarithmically
    log_integrand: bool
        should the integrand be interpolated logarithmically
    order : int
        the order of the Gauss-Legendre quadrature
    rel_acc : float
        relative accuracy to be reached. This is always estimated by the
        relative difference between the integral evaluated with
        polynomial order n and n + 2
    """

    def __init__(self,
                 support,
                 integrand,
                 log_support,
                 log_integrand,
                 order,
                 rel_acc):
        if len(support) != len(integrand):
            raise Exception("The support and the integrand need to have " +
                            "the same shape. Currently: " + str(len(support)) +
                            " and " + str(len(integrand)) + ".")
        self.support = support
        self.logx = True if log_support else False
        self.x = np.log(support) if self.logx else support

        self.integrand = integrand
        self.logy = True if log_integrand else False
        y = np.log(integrand) if self.logy else integrand
        self.integrand_interpolation = interp1d(
            self.x, y, bounds_error=True)  # fill_value="extrapolate")
        self.maxima = argrelextrema(integrand, np.greater)

        self.has_max = True if len(self.maxima[0][:] > 0) else False
        self.minima = argrelextrema(integrand, np.less)
        self.has_min = True if len(self.minima[0][:] > 0) else False

        self.n = order
        self.n_input = order
        self.rel_acc = rel_acc

    def update_integrand(self,
                         new_integrand):
        """
        The extrema for one kernel function do not changed when 
        multiplied with a smooth function (e.g., ell*Cell). Therefore, 
        we can update the integrand to safe computational time with this
        function.
        """
        y = np.log(new_integrand) if self.logy else new_integrand
        self.integrand_interpolation = interp1d(self.x, y, bounds_error=True)

    def __quadrature(self, lo, hi, roots, weights):
        """
        Calculates the Gauss-Legendre quadrature in the interval lo and 
        hi. The roots and weights must be provided as tabulated input.

        Parameters
        ------------
        lo : float
            lower integration limit
        hi : float
            upper integrationl limit
        roots : 1D array
            roots of the polynomials
        weights : 1D array
            weights of the quadrature

        Return
        ---------
        the integral between lo and hi using roots and weights (float)

        """

        integral = 0
        for i in range(self.n):
            aux_x = (hi-lo)/2 * roots[i] + (lo+hi)/2
            if self.logx:
                aux_x = np.log(aux_x)
            aux_y = self.integrand_interpolation(aux_x)
            if self.logy:
                aux_y = np.exp(aux_y)
            integral += weights[i] * aux_y
        return integral * (hi-lo)/2

    def __integrate_fixed_order(self):
        """
        Calculates the integral of an oscillatory integrand using the 
        extrema of the integrand and integrating piecewise between them 
        using the quadrature rule.

        Returns
        ------------
        the integral over the support

        """

        x, w = np.polynomial.legendre.leggauss(self.n)
        if self.has_max and self.has_min:
            N = len(self.maxima[0][:]) if len(self.maxima[0][:]) > len(
                self.minima[0][:]) else len(self.minima[0][:])
            if self.maxima[0][0] > self.minima[0][0]:
                integral = self.__quadrature(
                    self.support[0], self.support[self.minima[0][0]], x, w)
                for i in range(0, N-1):
                    integral += self.__quadrature(self.support[self.minima[0][i]],
                                                  self.support[self.maxima[0][i]], x, w)
                    integral += self.__quadrature(self.support[self.maxima[0][i]],
                                                  self.support[self.minima[0][i+1]], x, w)
                if len(self.maxima[0][:]) == len(self.minima[0][:]):
                    integral += self.__quadrature(self.support[self.minima[0][i+1]],
                                                  self.support[self.maxima[0][i+1]], x, w)

            else:
                integral = self.__quadrature(
                    self.support[0], self.support[self.maxima[0][0]], x, w)
                for i in range(0, N-1):
                    integral += self.__quadrature(self.support[self.maxima[0][i]],
                                                  self.support[self.minima[0][i]], x, w)
                    integral += self.__quadrature(self.support[self.minima[0][i]],
                                                  self.support[self.maxima[0][i+1]], x, w)
                if len(self.maxima[0][:]) == len(self.minima[0][:]):
                    integral += self.__quadrature(self.support[self.maxima[0][i+1]],
                                                  self.support[self.minima[0][i+1]], x, w)
            if self.maxima[0][-1] > self.minima[0][-1]:
                integral += self.__quadrature(self.support[self.maxima[0][-1]],
                                              self.support[-1], x, w)
            else:
                integral += self.__quadrature(self.support[self.minima[0][-1]],
                                              self.support[-1], x, w)
        else:
            integral = self.__quadrature(
                self.support[0], self.support[-1], x, w)

        return integral

    def integrate(self):
        """
        Calculates the integral of an oscillatory integrand using the 
        extrema of the integrand and integrating piecewise between them 
        using the quadrature rule. The integral is calculated in the 
        supported interval by increasing the polynomial order until a
        fixed relative accuracy is reached.


        Returns
        ------------
        the integral over the support

        """

        result_n = self.__integrate_fixed_order()
        self.n += 2
        result_np2 = self.__integrate_fixed_order()
        rel_error = np.abs(result_n-result_np2)/result_n
        while rel_error > self.rel_acc:
            result_n = self.__integrate_fixed_order()
            self.n += 2
            result_np2 = self.__integrate_fixed_order()
            rel_error = np.abs(result_n-result_np2)/result_n
            if self.n > 150:
                print("IntegrationWarning: The Gauss-Legendre integrator has " +
                    "reached its supported accuracy limit (%.2e" % rel_error +
                    ") which is still above the desired accuracy " +
                    "(%.2e" % self.rel_acc + ").")
        self.n = self.n_input
        return result_np2
