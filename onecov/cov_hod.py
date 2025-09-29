import numpy as np
from scipy.integrate import simpson


class HOD():
    """
    This class provides different models for the halo occupation 
    distribution. All quantities are calculated at as a function halo
    mass.

    Attributes
    ----------
    bias_dict : dictionary
        Specifies all the information about the bias model. To be passed
        from the read_input method of the Input class.
    hm_prec : dictionary
        Contains precision information about the HaloModel (also, see 
        hmf documentation by Steven Murray), this includes mass range 
        and spacing for the mass integrations in the halo model.

    Private Variables
    -----------------
    Mrange : array
        with unit M_sun/h
        with shape (M_bins)
        logarithmically spaced masses in the range 
        [hm_prec['log10M_min'], hm_prec['log10M_max']]
    Mbins : array
        with unit M_sun/h
        with shape (sample_dims, self.N_stellar_mass)
        logarithmically spaced masses in the range 
        [bias_dict['logmass_bins'][:-1], bias_dict['logmass_bins'][1:]],
        returns Mrange if no samples are specified

    Example
    -------
    from cov_input import Input, FileInput
    from cov_setup import Setup
    from cov_hod import HOD
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    # autoperforms consistency check of look-up tables with 
    # configuration
    setting = Setup(cosmo, bias, survey_params, prec, read_in_tables)
    my_hod = HOD(bias, prec['hm'])

    Instructions how to add another relation :
    ------------------------------------------
    1) add your own relation with preferably hard-coded variables
    2) change the function occ_num_and_prob_per_pop
      2.1) add if pop == 'cen': and else if pop == 'sat': to the
           function if your relations differ for these populations
      2.2) your mass-observable relation can be called at L302 
           Mobs = ..., make sure you provide an array with shape 
           (M_bins) as does, e.g., the function double_powerlaw
      2.3) your scattering relation can be called at L303 (and L290) 
           occ_prob = ..., make sure you provide an array with shape 
           (sample bins, self.N_stellar_mass, M_bins) as does, e.g., the function 
           lognormal

    """

    def __init__(self, 
                 bias_dict, 
                 hm_prec):
        self.mass_bins_disagree = False
        if bias_dict['has_csmf']:
            if bias_dict['logmass_bins_upper'] is not None and bias_dict['logmass_bins_upper'] is not None:
                if bias_dict['logmass_bins_lower'][0] > bias_dict['csmf_log10M_bins'][0] or bias_dict['logmass_bins_upper'][-1] < bias_dict['csmf_log10M_bins'][-1]:
                    self.mass_bins_disagree = True
            else:
                if bias_dict['logmass_bins'][0] > bias_dict['csmf_log10M_bins'][0] or bias_dict['logmass_bins'][-1] < bias_dict['csmf_log10M_bins'][-1]:
                    self.mass_bins_disagree = True
        self.N_stellar_mass = 300
        self.Mrange = np.logspace(
            hm_prec['log10M_min'], hm_prec['log10M_max'], hm_prec['M_bins'])
        self.Mbins = self.mass_bins(bias_dict, hm_prec)
        
    def mass_bins(self, 
                  bias_dict,
                  hm_prec):
        """
        If individual mass-ranges for samples of galaxies are given,
        compute their logaarithmically spaced mass ranges per sample

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
        bins : array
            Returns the mass bins define in hm_prec.

        """
        if bias_dict['logmass_bins'][0] == bias_dict['logmass_bins'][1] and (bias_dict['logmass_bins_upper'] is None and bias_dict['logmass_bins_lower'] is None):
            return self.Mrange.reshape((1,hm_prec['M_bins']))
        else:
            try:
                if (bias_dict['logmass_bins_upper'] is not None and bias_dict['logmass_bins_lower'] is not None):
                    bins =  np.logspace(
                        bias_dict['logmass_bins_lower'], 
                        bias_dict['logmass_bins_upper'], 
                        self.N_stellar_mass).T
                else:
                    bins =  np.logspace(
                        bias_dict['logmass_bins'][:-1], 
                        bias_dict['logmass_bins'][1:], 
                        self.N_stellar_mass).T
            # for Python 3.6 or earlier
            except ValueError:
                bins = np.array([])
                if (bias_dict['logmass_bins_upper'] is not None and bias_dict['logmass_bins_lower'] is not None):
                    for mbin in range(len(bias_dict['logmass_bins'])-1):
                        bins = np.concatenate((bins, np.logspace(
                                    bias_dict['logmass_bins_lower'][mbin], 
                                    bias_dict['logmass_bins_upper'][mbin], 
                                    self.N_stellar_mass)))
                else:
                    for mbin in range(len(bias_dict['logmass_bins'])-1):
                        bins = np.concatenate((bins, np.logspace(
                                    bias_dict['logmass_bins'][mbin], 
                                    bias_dict['logmass_bins'][mbin+1], 
                                    self.N_stellar_mass)))
                bins = bins.reshape(len(bias_dict['logmass_bins'])-1,self.N_stellar_mass)

        return bins

    def hod_update(self, 
                   bias_dict, 
                   hm_prec):
        """
        Updates the private variables Mrange and Mbins to a new mass 
        range

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

        """

        self.Mrange = np.logspace(
            hm_prec['log10M_min'], hm_prec['log10M_max'], hm_prec['M_bins'])
        self.Mbins = self.mass_bins(bias_dict, hm_prec)

        return True

    def double_powerlaw(self, 
                        hod_dict, 
                        pop):
        """
        Calculates the mass-observable relation for a specified 
        range of halo masses.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        pop : string
            Is either 'cen' for central galaxies or 'sat' for satellite 
            galaxies.

        Returns
        -------
        Mobs : array
            with unit M_sun/h
            with shape (M_bins)
        """
        mrange = self.Mrange / 10**hod_dict['dpow_logM1_'+pop]
        Mobs = \
              np.log10(hod_dict['dpow_norm_'+pop]) \
            + hod_dict['dpow_logM0_'+pop] \
            + hod_dict['dpow_a_'+pop] * np.log10(mrange) \
            - (hod_dict['dpow_a_'+pop] \
            - hod_dict['dpow_b_'+pop]) * np.log10(1 + mrange)
        return Mobs

    def lognormal(self, 
                  hod_dict, 
                  Mobs,
                  pop):
        """
        One model to calculate the occupation probability per halo mass 
        for a specified population of galaxies. Follows Eq. (15) of 
        van Uitert et al. (2016)

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        Mobs : array
            Results from the mass-observable relation. To be calculated
            first.
        pop : string
            - "cen" for centrals
            - "sat" for satellites

        Returns
        -------
        logn : array
            with shape (sample_bins, self.N_stellar_mass, M_bins)
        """



        logMbinsMobs = np.log10(self.Mbins[:, :, None]/Mobs[None, None, :])
        return 1.0/np.sqrt(2.*np.pi)/np.log(10)/hod_dict['logn_sigma_c_'+pop]/Mobs[None, None, :]*np.exp(-logMbinsMobs**2./2/hod_dict['logn_sigma_c_'+pop]**2.)

    def modschechter(self, 
                     hod_dict, 
                     Mobs,
                     pop):
        """
        One model to calculate the occupation probability per halo mass 
        for a specified population of galaxies. Follows Eq. (17) of 
        van Uitert et al. (2016)

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        Mobs : array
            Results from the mass-observable relation. To be calculated
            first.
        pop : string
            - "cen" for centrals
            - "sat" for satellites

        Returns 
        -------
        schech : array
            with shape (sample_bins, self.N_stellar_mass, M_bins)
        """
        MbinsMobs = self.Mbins[:, :, None]/Mobs[None, None, :]
        phi_s = 10**(-hod_dict['modsch_b_'+pop][0] + hod_dict['modsch_b_'+pop][1]   *np.log10(self.Mrange/10**hod_dict['modsch_logMref_'+pop]))
        return (phi_s/Mobs)[None, None, :]*(MbinsMobs**hod_dict['modsch_alpha_s_'+pop]*np.exp(-MbinsMobs**2.0))[:,:,:]

    

    # Expected number of 
    def occ_num_and_prob_per_pop(self, 
                                 hod_dict, 
                                 pop, 
                                 mor_tab, 
                                 occprob_tab, 
                                 occnum_tab):
        """
        Calculates the expected number of objects of a given type of a 
        given mass, < N_cen/sat | M >.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        pop : string
            - "cen" for centrals
            - "sat" for satellites
        mor_tab : dictionary
            default : None
            Look-up table for the mass-observable relation (optional). 
            To be passed from the read_input method of the FileInput 
            class.
        occprob_tab : dictionary
            default : None
            Look-up table for the occupation probability as a function 
            of halo mass per galaxy sample (optional). To be passed from 
            the read_input method of the FileInput class.
        occnum_tab : dictionary
            default : None
            Look-up table for the occupation number as a function of 
            halo mass per galaxy sample (optional). To be passed from 
            the read_input method of the FileInput class.

        Returns
        -------
        occ_num : array
            with shape (sample_bins, M_bins)
        occ_prob : array
            with shape (sample_bins, self.N_stellar_mass, M_bins) [Note1]
            
        [Note1] If the occupation number is given as a look-up table,
        then occ_prob will be NoneType
        """
        
        if pop != 'cen' and pop != 'sat':
            raise Exception(
                "InputError: The galaxy population for the mass-observable "
                + "relation must be either centrals ('cen') or satellites "
                + "('sat').")
            
        if occnum_tab['M'] is not None:
            occ_num = occnum_tab[pop]
            occ_prob = None
        
        elif occprob_tab['M'] is not None:
            occ_prob = occprob_tab[pop]
            if np.isnan(occ_prob).any():
                print("HODWarning: One or several elements in the occupation "
                    + "probability evaluated to nan. They are replaced with "
                    + "zeros. This might bias the result.")
                occ_prob = np.nan_to_num(occ_prob)
            occ_num = simpson(occ_prob, x = self.Mbins[:, :, None], axis=1)
        
        elif mor_tab['M'] is not None:
            Mobs = mor_tab[pop]
            occ_prob = \
                eval(hod_dict['model_scatter_'+pop])(hod_dict, Mobs, pop)
            if np.isnan(occ_prob).any():
                print("HODWarning: One or several elements in the occupation "
                    + "probability evaluated to nan. They are replaced with "
                    + "zeros. This might bias the result.")
                occ_prob = np.nan_to_num(occ_prob)
            occ_num = simpson(occ_prob, x = self.Mbins[:, :, None], axis=1)

        else:
            Mobs = 10**eval(hod_dict['model_mor_'+pop])(hod_dict, pop)
            occ_prob = \
                eval(hod_dict['model_scatter_'+pop])(hod_dict, Mobs, pop)
            if np.isnan(occ_prob).any():
                print("HODWarning: One or several elements in the occupation "
                    + "probability evaluated to nan. They are replaced with "
                    + "zeros. This might bias the result.")
                occ_prob = np.nan_to_num(occ_prob)
            occ_num = simpson(occ_prob, x = self.Mbins[:, :, None], axis=1)
        return occ_num, occ_prob

    def occ_num_and_prob(self, 
                         hod_dict, 
                         mor_tab, 
                         occprob_tab, 
                         occnum_tab):
        """
        Calculates the expected number of galaxies of a given mass, 
        < N_gal | M >.

        Parameters
        ----------
        hod_dict : dictionary
            Specifies all the information about the halo occupation 
            distribution used. This defines the shot noise level of the 
            covariance and includes the mass bin definition of the 
            different galaxy populations. To be passed from the 
            read_input method of the Input class.
        mor_tab : dictionary
            default : None
            Look-up table for the mass-observable relation (optional). 
            To be passed from the read_input method of the FileInput 
            class.
        occprob_tab : dictionary
            default : None
            Look-up table for the occupation probability as a function 
            of halo mass per galaxy sample (optional). To be passed from 
            the read_input method of the FileInput class.
        occnum_tab : dictionary
            default : None
            Look-up table for the occupation number as a function of 
            halo mass per galaxy sample (optional). To be passed from 
            the read_input method of the FileInput class.

        Returns
        -------
        occ_num : array
            with shape (sample_bins, M_bins)
        occ_prob : array
            with shape (sample_bins, self.N_stellar_mass, M_bins)
        """

        occ_num_cen, occ_prob_cen = \
            self.occ_num_and_prob_per_pop(hod_dict, 'cen', 
                mor_tab, occprob_tab, occnum_tab)
        occ_num_sat, occ_prob_sat = \
            self.occ_num_and_prob_per_pop(hod_dict, 'sat', 
                mor_tab, occprob_tab, occnum_tab)
        if occ_prob_cen is None or occ_prob_sat is None:
            return occ_num_cen + occ_num_sat, None

        return occ_num_cen + occ_num_sat, occ_prob_cen + occ_prob_sat
    