import numpy as np

from .cov_polyspectra import PolySpectra

class CovKSpace(PolySpectra):
    """
    This class calculates the k-space covariance for power spectra 
    estimators (matter-matter, tracer-tracer and tracer-matter) at a 
    single redshift. Inherits the functionality of the PolySpectra
    class.

    Parameters :
    ------------
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
    see PolySpectra class

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_k_space import CovKSpace
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    zet = 0
    covk = CovKSpace(zet, covterms, observables, cosmo, bias, hod,
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
        PolySpectra.__init__(self,
                             zet,
                             cov_dict,
                             obs_dict,
                             cosmo_dict,
                             bias_dict,
                             hod_dict,
                             survey_params_dict,
                             prec,
                             read_in_tables)
    
    def calc_covK(self,
                  output_dict,
                  bias_dict, 
                  hod_dict,
                  survey_params_dict,  
                  prec, 
                  read_in_tables,
                  tomo_clust_idx=0, 
                  tomo_lens_idx=0):
        """
        Calculates the full covariance between two observables in 
        k-space as specified in the config file.

        Parameters :
        ------------
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
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies 
            for all tomographic bins as well as the ellipticity 
            dispersion for galaxy shapes. To be passed from the 
            read_input method of the Input class.
        prec : dictionary
            with the following keys (To be passed from the read_input  
            method of the Input class.)
            'hm' : dictionary
                Contains precision information about the HaloModel 
                (also, see hmf documentation by Steven Murray), this 
                includes mass range and spacing for the mass halo 
                integrations in the model.
        read_in_tables : dictionary with the following keys (To be 
            passed  from the read_input method of the FileInput class.)
            'tri' : dictionary
                Look-up table for the trispectra (for all combinations  
                of matter 'm' and tracer 'g', optional) for different
                wavenumbers and redshifts.
        tomo_clust_idx : int
            default : 0
            If n_eff_clust is not None, tomo_clust_idx specifies which 
            tomographic bin should be used for the calculation.
        tomo_lens_idx : int
            default : 0
            If n_eff_lens is not None, tomo_lens_idx specifies which
            tomographic bin should be used for the calculation.

        Returns :
        ---------
        gauss, nongauss, ssc : list of arrays
            each with 6 entries for the observables 
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape
                each with shape (log10k bins, log10k bins, 
                                 sample bins, sample bins)
        """

        print("Calculating covariance in k_space.")

        if not self.cov_dict['split_gauss']:
            gaussgggg, gaussgggm, gaussggmm, \
            gaussgmgm, gaussmmgm, gaussmmmm, \
            gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn = \
                self.covK_gaussian(survey_params_dict,
                                tomo_clust_idx, 
                                tomo_lens_idx)
            gauss = [gaussgggg + gaussgggg_sn, gaussgggm,
                    gaussggmm, gaussgmgm + gaussgmgm_sn,
                    gaussmmgm, gaussmmmm + gaussmmmm_sn]
        else:
            gauss = self.covK_gaussian(survey_params_dict,
                                       tomo_clust_idx, 
                                       tomo_lens_idx)

        nongauss = self.covK_non_gaussian(output_dict,
                                          bias_dict, 
                                          hod_dict,
                                          prec['hm'], 
                                          read_in_tables['tri'])
                                    
        ssc = self.covK_ssc(bias_dict, 
                            hod_dict, 
                            prec['hm'])
                    
        return list(gauss), list(nongauss), list(ssc)

    def covK_gaussian(self,
                      survey_params_dict,
                      tomo_clust_idx=0, 
                      tomo_lens_idx=0):
        """
        Calculates the Gaussian (disconnected) covariance in k-space 
        between two observables. If the shot_noise is not explicitely
        given (i.e., survey_params['shot_noise_clust'] for clustering 
        and survey_params['shot_noise_gamma'] for lensing) the Gaussian
        covariance will not contain any contributions of noise.

        Parameters :
        ------------
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the 'shot_noise_clust' which is
            basically 1 / n_3dgal (from simulations or approximations 
            probably) and 'shot_noise_gamma' which is ~ ellipticity
            dispersion over n_3dgal. To be passed from the read_input
            method of the Input class.
        tomo_clust_idx : int
            default : 0
            If n_eff_clust is not None, tomo_clust_idx specifies which 
            tomographic bin should be used for the calculation.
        tomo_lens_idx : int
            default : 0
            If n_eff_lens is not None, tomo_lens_idx specifies which
            tomographic bin should be used for the calculation.

        Returns :
        ---------
        gaussgggg, gaussgggm, gaussggmm, \
        gaussgmgm, gaussmmgm, gaussmmmm, \
        gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn : list of arrays
            with shape (log10k bins, log10k bins, 
                        sample bins, sample bins)

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the pure matter-matter ('mmmm'),
        tracer-tracer ('gggg'), and matter-tracer ('gmgm') terms, one 
        needs to add gaussxxyy + gaussxxyy_sn. They are kept separate 
        for a later numerical integration.

        """
        
        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0

        print("Calculating gaussian covariance in k_space.")

        gaussgggg_sva, gaussgggg_mix, gaussgggg_sn, \
        gaussgggm_sva, gaussgggm_mix, gaussgggm_sn, \
        gaussggmm_sva, gaussggmm_mix, gaussggmm_sn, \
        gaussgmgm_sva, gaussgmgm_mix, gaussgmgm_sn, \
        gaussmmgm_sva, gaussmmgm_mix, gaussmmgm_sn, \
        gaussmmmm_sva, gaussmmmm_mix, gaussmmmm_sn = \
            self.__covK_split_gaussian(survey_params_dict,
                                       tomo_clust_idx, 
                                       tomo_lens_idx)

        if not self.cov_dict['split_gauss']:
            gaussgggg = gaussgggg_sva + gaussgggg_mix
            gaussgggm = gaussgggm_sva + gaussgggm_mix
            gaussggmm = gaussggmm_sva + gaussggmm_mix
            gaussgmgm = gaussgmgm_sva + gaussgmgm_mix
            gaussmmgm = gaussmmgm_sva + gaussmmgm_mix
            gaussmmmm = gaussmmmm_sva + gaussmmmm_mix
            return gaussgggg, gaussgggm, gaussggmm, \
                   gaussgmgm, gaussmmgm, gaussmmmm, \
                   gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn
        else:
            return gaussgggg_sva, gaussgggg_mix, gaussgggg_sn, \
                   gaussgggm_sva, gaussgggm_mix, gaussgggm_sn, \
                   gaussggmm_sva, gaussggmm_mix, gaussggmm_sn, \
                   gaussgmgm_sva, gaussgmgm_mix, gaussgmgm_sn, \
                   gaussmmgm_sva, gaussmmgm_mix, gaussmmgm_sn, \
                   gaussmmmm_sva, gaussmmmm_mix, gaussmmmm_sn
    
    def __covK_split_gaussian(self,
                              survey_params_dict,
                              tomo_clust_idx=0, 
                              tomo_lens_idx=0):
        """
        Calculates the Gaussian (disconnected) covariance in k-space 
        between two observables. Explicitly, splits the full Gaussian 
        covariances into a term depending on the power spectra, the 
        shot-noise and a mix term. If the shot_noise is not explicitely
        given (i.e., survey_params['shot_noise_clust'] for clustering 
        and survey_params['shot_noise_gamma'] for lensing) the Gaussian
        covariance will not contain any contributions of noise.

        Parameters :
        ------------
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the 'shot_noise_clust' which is
            basically 1 / n_3dgal (from simulations or approximations 
            probably) and 'shot_noise_gamma' which is ~ ellipticity
            dispersion over n_3dgal. To be passed from the read_input
            method of the Input class.
        tomo_clust_idx : int
            default : 0
            If n_eff_clust is not None, tomo_clust_idx specifies which 
            tomographic bin should be used for the calculation.
        tomo_lens_idx : int
            default : 0
            If n_eff_lens is not None, tomo_lens_idx specifies which
            tomographic bin should be used for the calculation.

        Returns :
        ---------
        gaussgggg_sva, gaussgggg_mix, gaussgggg_sn, \
        gaussgggm_sva, gaussgggm_mix, gaussgggm_sn, \
        gaussggmm_sva, gaussggmm_mix, gaussggmm_sn, \
        gaussgmgm_sva, gaussgmgm_mix, gaussgmgm_sn, \
        gaussmmgm_sva, gaussmmgm_mix, gaussmmgm_sn, \
        gaussmmmm_sva, gaussmmmm_mix, gaussmmmm_sn : list of arrays
            with shape (log10k bins, log10k bins,
                        sample bins, sample bins)

        Note :
        ------
        To get the full covariance contribution, one needs to add 
        gausswxyz_sva + gausswxyz_mix + gausswxyz_sn. Some terms are
        zero by definition (e.g., gaussgggm_sn).

        """
        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        if survey_params_dict['shot_noise_clust'] is None:
            shot_noise_g = np.zeros((self.sample_dim, self.sample_dim))
        elif survey_params_dict['shot_noise_clust'][0] == 0:
            shot_noise_g = 1 / self.ngal * np.eye(self.sample_dim)
        else:
            shot_noise_g = \
                survey_params_dict['shot_noise_clust'][tomo_clust_idx] \
                * np.eye(self.sample_dim)

        if survey_params_dict['shot_noise_gamma'] is None:
            shot_noise_gamma = np.zeros((self.sample_dim, self.sample_dim))
        else:
            shot_noise_gamma = \
                survey_params_dict['shot_noise_gamma'][tomo_lens_idx] \
                * np.eye(self.sample_dim)

        reshape_mat = \
            np.eye(len(self.mass_func.k))[:,:,None,None]
        
        if self.gg:
            gaussgggg_sva = 2 * self.Pgg[:,:,None] * self.Pgg[:,None,:]
            gaussgggg_mix = 4 * self.Pgg[:,:,None] * shot_noise_g[None,:,:]
            gaussgggg_sn = 2 * shot_noise_g**2

            gaussgggg_sva = gaussgggg_sva * reshape_mat
            gaussgggg_mix = gaussgggg_mix * reshape_mat
            gaussgggg_sn = gaussgggg_sn[None,None,:,:] * reshape_mat
        else:
            gaussgggg_sva, gaussgggg_mix, gaussgggg_sn = 0, 0, 0
            
        if self.gg and self.gm and self.cross_terms:
            gaussgggm_sva = self.Pgg[:,:,None] * self.Pgm[:,:,None] \
                          + self.Pgg[:,None,:] * self.Pgm[:,None,:]
            gaussgggm_mix = 2 * self.Pgm[:,:,None] * shot_noise_g[None,:,:]
            gaussgggm_sn = 0

            gaussgggm_sva = gaussgggm_sva * reshape_mat
            gaussgggm_mix = gaussgggm_mix * reshape_mat
        else:
            gaussgggm_sva, gaussgggm_mix, gaussgggm_sn = 0, 0, 0
                
        if self.gg and self.mm and self.cross_terms:
            gaussggmm_sva = 2* self.Pgm[:,:,None] * self.Pgm[:,None,:]
            gaussggmm_mix = 0
            gaussggmm_sn = 0

            gaussggmm_sva = gaussggmm_sva * reshape_mat
        else:
            gaussggmm_sva, gaussggmm_mix, gaussggmm_sn = 0, 0, 0
          
        if self.gm:
            gaussgmgm_sva = self.Pgm[:,:,None] * self.Pgm[:,None,:] \
                                + self.Pgg[:,:,None] * self.Pmm[:,None,:]
            gaussgmgm_mix = self.Pgg[:,:,None] * shot_noise_gamma[None,:,:] \
                          + self.Pmm[:,None,:] * shot_noise_g[None,:,:]
            gaussgmgm_sn = shot_noise_g * shot_noise_gamma

            gaussgmgm_sva = gaussgmgm_sva * reshape_mat
            gaussgmgm_mix = gaussgmgm_mix * reshape_mat
            gaussgmgm_sn = gaussgmgm_sn[None,None,:,:] * reshape_mat
        else:
            gaussgmgm_sva, gaussgmgm_mix, gaussgmgm_sn = 0, 0, 0

        if self.mm and self.gm and self.cross_terms:
            gaussmmgm_sva = self.Pmm[:,:,None] * self.Pgm[:,:,None] \
                          + self.Pmm[:,None,:] * self.Pgm[:,None,:]
            gaussmmgm_mix = 2 * self.Pmm[:,:,None] * shot_noise_gamma[None,:,:]
            gaussmmgm_sn = 0

            gaussmmgm_sva = gaussmmgm_sva * reshape_mat
            gaussmmgm_mix = gaussmmgm_mix * reshape_mat
        else:
            gaussmmgm_sva, gaussmmgm_mix, gaussmmgm_sn = 0, 0, 0
  
        if self.mm:
            gaussmmmm_sva = 2* self.Pmm[:,:,None] * self.Pmm[:,None,:]
            gaussmmmm_mix = 4 * self.Pmm[:,:,None] * shot_noise_gamma[None,:,:]
            gaussmmmm_sn = 2 * shot_noise_gamma**2

            gaussmmmm_sva = gaussmmmm_sva * reshape_mat
            gaussmmmm_mix = gaussmmmm_mix * reshape_mat
            gaussmmmm_sn = gaussmmmm_sn[None,None,:,:] * reshape_mat
        else:
            gaussmmmm_sva, gaussmmmm_mix, gaussmmmm_sn = 0, 0, 0

        return gaussgggg_sva, gaussgggg_mix, gaussgggg_sn, \
               gaussgggm_sva, gaussgggm_mix, gaussgggm_sn, \
               gaussggmm_sva, gaussggmm_mix, gaussggmm_sn, \
               gaussgmgm_sva, gaussgmgm_mix, gaussgmgm_sn, \
               gaussmmgm_sva, gaussmmgm_mix, gaussmmgm_sn, \
               gaussmmmm_sva, gaussmmmm_mix, gaussmmmm_sn
    
    def covK_non_gaussian(self, 
                          output_dict,
                          bias_dict, 
                          hod_dict,
                          hm_prec,
                          tri_tab):
        """
        Calculates the non-Gaussian (connected) covariance in k-space 
        between two observables.

        Parameters :
        ------------
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

        Returns :
        ---------
        trigggg, trigggm, triggmm, \
        trigmgm, trimmgm, trimmmm : list of arrays
            with shape (log10k bins, log10k bins,
                        sample bins, sample bins)

        """
        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0

        print("Calculating nongaussian covariance in k_space.")

        return self.trispectra(output_dict,
                               bias_dict, 
                               hod_dict,
                               hm_prec, 
                               tri_tab)
    
    def covK_ssc(self, 
                 bias_dict, 
                 hod_dict, 
                 hm_prec):
        """
        Calculates the super-sample covariance in k-space between two 
        observables.

        Parameters :
        ------------
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

        Returns :
        ---------
        sscgggg, sscgggm, sscggmm, \
        sscgmgm, sscmmgm, sscmmmm : list of arrays
            with shape (log10k bins, log10k bins,
                        sample bins, sample bins)

        """
        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0

        print("Calculating super-sample covariance in k_space.")
        
        resp_Pgg, resp_Pgm ,resp_Pmm = \
            self.powspec_responses(bias_dict, hod_dict, hm_prec)
        
        if self.gg:
            sscgggg = resp_Pgg[:,None,:,None] * resp_Pgg[None,:,None,:]
        else:
            sscgggg = 0
            
        if self.gg and self.gm and self.cross_terms:
            sscgggm = resp_Pgg[:,None,:,None] * resp_Pgm[None,:,None,:]
        else:
            sscgggm = 0
            
        if self.gg and self.mm and self.cross_terms:
            sscggmm = resp_Pgg[:,None,:,None] * resp_Pmm[None,:,None,:]
        else:
            sscggmm = 0
            
        if self.gm:
            sscgmgm = resp_Pgm[:,None,:,None] * resp_Pgm[None,:,None,:]
        else:
            sscgmgm = 0
            
        if self.mm and self.gm and self.cross_terms:
            sscmmgm = resp_Pmm[:,None,:,None] * resp_Pgm[None,:,None,:]
        else:
            sscmmgm = 0
            
        if self.mm:
            sscmmmm = resp_Pmm[:,None,:,None] * resp_Pmm[None,:,None,:]
        else:
            sscmmmm = 0

        return sscgggg, sscgggm, sscggmm, sscgmgm, sscmmgm, sscmmmm
                
# covk = CovKSpace(0, covterms, observables, cosmo, bias, hod,
#         survey_params, hm_prec, powspec_prec, trispec_prec, 
#         read_in_tables)