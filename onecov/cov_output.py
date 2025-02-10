import enum
import numpy as np
import os


from astropy.units.cgs import K
# from astropy.io import fits

#plot stuff
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('font', family='sans-serif')
#
rcParams['figure.figsize'] = (8., 6.)
rcParams['axes.linewidth'] = 2
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 30
rcParams['font.size'] = 16
rcParams['lines.linewidth'] = 3
rcParams['lines.markersize'] = 10
rcParams['lines.markeredgewidth'] = 2
rcParams['xtick.major.size'] = 8
rcParams['xtick.minor.size'] = 4
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.labelsize'] = 16
rcParams['ytick.major.size'] = 8
rcParams['ytick.minor.size'] = 4
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.labelsize'] = 16

class Output(): 
    """
    Class writing the output of the OneCovariance code. Methods of this class collect
    all the necessary blocks of the covariance matrix.

    Parameters:
    -----------
    output_dict : dictionary
    
    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_ell_space import CovELLSpace
    from cov_output import Output

    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input()
    covell = CovELLSpace(covterms, observables, output, cosmo, bias,
                         hod, survey_params, prec, read_in_tables)
    gauss, nongauss, ssc = \
        covell.calc_covELL(observables, output, bias, hod, 
                           survey_params, prec, read_in_tables)

    out = Output(output)
    out.write_cov(covterms, observables, covell.n_tomo_clust, 
                  covell.n_tomo_lens, covell.ellrange, gauss, nongauss,
                  ssc)
    """

    def __init__(self, output_dict, projected_clust = None, projected_lens = None):
        self.filename = output_dict['file']
        self.__check_filetype()
        self.style = output_dict['style']
        
        self.plot = output_dict['make_plot']
        self.trispecfile = output_dict['trispec']
        self.Cellfile = output_dict['Cell']
        self.tex = output_dict['use_tex']
        self.save_as_binary = output_dict['save_as_binary']
        self.list_style_spatial_first = output_dict['list_style_spatial_first']
        self.projected_lens = projected_lens
        self.projected_clust = projected_clust
    
    def __check_filetype(self):
        for idx,fn in enumerate(self.filename):
            if fn == '':
                continue

            dotloc = fn[::-1].find('.')

            if dotloc == -1 or dotloc == 0:
                filetype = 'dat'
            else:
                dotloc = len(fn) - dotloc
                filetype = fn[dotloc:]

            if filetype == 'fits':
                print("ConfigWarning: Fits output is not implemented yet, sorry " +
                    ":(. The file extension will be changed to 'dat'.")
                self.filename[idx] = fn[:dotloc] + 'dat'

        return True

    def __add_string_to_filename(self, 
                                 addin, 
                                 fname):
        dotloc = fname[::-1].find('.')
        if dotloc == -1 or dotloc == 0:
            dotloc = 1
        dotloc = len(fname) - dotloc - 1

        if type(addin) == str:
            fname = fname[:dotloc] + '_' + addin + fname[dotloc:]
        else:
            fname = fname[:dotloc] + '_' + str(round(addin,4)) + fname[dotloc:]

        return fname

    def write_arbitrary_cov(self,
                            cov_dict,
                            obs_dict,
                            n_tomo_clust,
                            n_tomo_lens,
                            read_in_tables,
                            gauss,
                            nongauss,
                            ssc):
        """
        Writes the covariance matrix to a file depending on the specifications in the config.ini.

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
        n_tomo_clust : int
            Number of clustering (lens) bins
        n_tomo_lens : int
            Number of lensing (source) bins
        proj_quant : array
            The projected quantity associated with the covariance, e.g. theta, or ell
        gauss : list of arrays
            Gaussian covariance split into the different components
        nongauss : list of arrays
            Non-Gaussian covariance
        ssc : list of arrays
            Supersample covariance
        """
        self.has_gauss, self.has_nongauss, self.has_ssc = cov_dict['gauss'], cov_dict['nongauss'], cov_dict['ssc']
        self.has_csmf = obs_dict['observables']['csmf']
        gauss, nongauss, ssc = self.__none_to_zero(gauss, nongauss, ssc)

        obslist, obsbool, obslength = self.__get_obslist(obs_dict)
        gg, gm, mm = obsbool[0], obsbool[3], obsbool[5]
        xipp, xipm, ximm = None, None, None
        mult = 1
        self.conditional_stellar_mass_function_cov = []
        if self.has_csmf:
            self.conditional_stellar_mass_function_cov = gauss[-5:]
            gauss = gauss[:-5]
        if len(gauss) == obslength:
            ...
        elif len(gauss) == obslength+3:
            gauss = [gauss[0]+gauss[6], gauss[1],
                     gauss[2], gauss[3]+gauss[7],
                     gauss[4], gauss[5]+gauss[8]]
        elif len(gauss) == 10:
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == 14:
            gauss = [gauss[0]+gauss[10], gauss[1], gauss[2], gauss[3],
                     gauss[4]+gauss[11], gauss[5], gauss[6],
                     gauss[7]+gauss[12], gauss[8], 
                     gauss[9]+gauss[13]]
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == obslength*3:
            mult = 3
        elif len(gauss) == 30:
            mult = 3
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == 22 and (obs_dict['ELLspace']['n_spec'] is not None or obs_dict['ELLspace']['n_spec'] != 0):
            ...
        elif len(gauss) == 66 and (obs_dict['ELLspace']['n_spec'] is not None or obs_dict['ELLspace']['n_spec'] != 0):
            mult = 3
        else:
            raise Exception("OutputError: The gaussian covariance needs at " +
                "least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(nongauss) != obslength and (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0):
            raise Exception("OutputError: The nongaussian covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(ssc) != obslength and (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0):
            raise Exception("OutputError: The super-sample covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
                
        sampledim = self.__get_sampledim(gauss, nongauss, ssc)
        gaussidx = 0
        for idx in range(obslength):
            gausstest = 0
            if n_tomo_lens is None and n_tomo_clust is None:
                shape = [len(sampledim)]*2 + [sampledim]*2
            else:
                tomodim = \
                    self.__get_tomodim(gauss[gaussidx], nongauss[idx], ssc[idx])
                if tomodim[0] == -1:
                    for _ in range(mult):
                        gausstest += gauss[gaussidx]
                        gaussidx += 1
                    if type(gausstest+nongauss[idx]+ssc[idx]) == int and \
                       obsbool[idx]:
                        obsbool[idx] = False
                        gg = False if obslist[idx] == 'gggg' else gg
                        gm = False if obslist[idx] == 'gmgm' else gm
                        mm = False if obslist[idx] == 'mmmm' else mm
                        print("OutputWarning: According to the config file " +
                              "the covariance for " + obslist[idx] + " ('m' " +
                              "might be 'kappa' in your case) is supposed " +
                              "to be calculated but neither the Gaussian, " +
                              "non-Gaussian or super-sample covariance has " +
                              "any values. This term will be manually set " +
                              "to False.")
                    continue
            for _ in range(mult):
                if isinstance(gauss[gaussidx], np.ndarray):    
                    gauss[gaussidx] = self.__check_for_empty_input(gauss[gaussidx], gauss[gaussidx].shape)
                    gaussidx += 1
            if isinstance(nongauss[idx], np.ndarray):
                nongauss[idx] = self.__check_for_empty_input(nongauss[idx], nongauss[idx].shape)
            if isinstance(ssc[idx], np.ndarray):
                ssc[idx] = self.__check_for_empty_input(ssc[idx], ssc[idx].shape)

        
        if ('terminal' in self.style or 'list' in self.style):
            fct_args = [obslist, obsbool]
            if self.list_style_spatial_first:
                self.__write_cov_list_arbitrary_cosmosis_style(cov_dict, obs_dict, n_tomo_clust, 
                                    n_tomo_lens, sampledim, read_in_tables, 
                                    gauss, nongauss, ssc, fct_args)
            else:
                self.__write_cov_list_arbitrary(cov_dict, obs_dict, n_tomo_clust, 
                                    n_tomo_lens, sampledim, read_in_tables, 
                                    gauss, nongauss, ssc, fct_args)
        if 'matrix' in self.style or self.plot:
            fct_args = [obslist, obsbool, obslength, mult, 
                        gg, gm, mm, xipp, xipm, ximm]
            self.__write_cov_matrix_arbitrary(obs_dict, cov_dict, n_tomo_clust,
                                    n_tomo_lens, sampledim, read_in_tables, 
                                    gauss, nongauss, ssc, fct_args)
        return True

    def write_cov(self,
                  cov_dict,
                  obs_dict,
                  n_tomo_clust,
                  n_tomo_lens,
                  proj_quant,
                  gauss,
                  nongauss,
                  ssc):
        """
        Writes the covariance matrix to a file depending on the specifications in the config.ini.

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
        n_tomo_clust : int
            Number of clustering (lens) bins
        n_tomo_lens : int
            Number of lensing (source) bins
        proj_quant : array
            The projected quantity associated with the covariance, e.g. theta, or ell
        gauss : list of arrays
            Gaussian covariance split into the different components
        nongauss : list of arrays
            Non-Gaussian covariance
        ssc : list of arrays
            Supersample covariance
            
        """
        self.has_gauss, self.has_nongauss, self.has_ssc = cov_dict['gauss'], cov_dict['nongauss'], cov_dict['ssc']
        self.has_csmf = obs_dict['observables']['csmf']
        self.is_cell = obs_dict['observables']['is_cell']
        gauss, nongauss, ssc = self.__none_to_zero(gauss, nongauss, ssc)

        obslist, obsbool, obslength = self.__get_obslist(obs_dict)
        gg, gm, mm = obsbool[0], obsbool[3], obsbool[5]
        xipp, xipm, ximm = None, None, None
        mult = 1
        self.conditional_stellar_mass_function_cov = []
        if self.has_csmf:
            if self.is_cell:
                self.conditional_stellar_mass_function_cov = gauss[-4:]
                gauss = gauss[:-4]
            else:
                self.conditional_stellar_mass_function_cov = gauss[-5:]
                gauss = gauss[:-5]
        if len(gauss) == obslength:
            ...
        elif len(gauss) == obslength+3:
            gauss = [gauss[0]+gauss[6], gauss[1],
                     gauss[2], gauss[3]+gauss[7],
                     gauss[4], gauss[5]+gauss[8]]
        elif len(gauss) == 10:
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == 14:
            gauss = [gauss[0]+gauss[10], gauss[1], gauss[2], gauss[3],
                     gauss[4]+gauss[11], gauss[5], gauss[6],
                     gauss[7]+gauss[12], gauss[8], 
                     gauss[9]+gauss[13]]
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == obslength*3:
            mult = 3
        elif len(gauss) == 30:
            mult = 3
            obslist, obsbool, obslength = self.__get_obslist(obs_dict, True)
            xipp, xipm, ximm = obsbool[7], obsbool[8], obsbool[9]
        elif len(gauss) == 22 and (obs_dict['ELLspace']['n_spec'] is not None or obs_dict['ELLspace']['n_spec'] != 0):
            ...
        elif len(gauss) == 66 and (obs_dict['ELLspace']['n_spec'] is not None or obs_dict['ELLspace']['n_spec'] != 0):
            mult = 3
        else:
            raise Exception("OutputError: The gaussian covariance needs at " +
                "least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(nongauss) != obslength and (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0):
            raise Exception("OutputError: The nongaussian covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(ssc) != obslength and (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0):
            raise Exception("OutputError: The super-sample covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
                
        sampledim = self.__get_sampledim(gauss, nongauss, ssc)

        gaussidx = 0
        for idx in range(obslength):
            gausstest = 0
            if n_tomo_lens is None and n_tomo_clust is None:
                shape = [len(proj_quant)]*2 + [sampledim]*2
            else:
                tomodim = \
                    self.__get_tomodim(gauss[gaussidx], nongauss[idx], ssc[idx])
                if tomodim[0] == -1:
                    for _ in range(mult):
                        gausstest += gauss[gaussidx]
                        gaussidx += 1
                    if type(gausstest+nongauss[idx]+ssc[idx]) == int and \
                       obsbool[idx]:
                        obsbool[idx] = False
                        gg = False if obslist[idx] == 'gggg' else gg
                        gm = False if obslist[idx] == 'gmgm' else gm
                        mm = False if obslist[idx] == 'mmmm' else mm
                        print("OutputWarning: According to the config file " +
                              "the covariance for " + obslist[idx] + " ('m' " +
                              "might be 'kappa' in your case) is supposed " +
                              "to be calculated but neither the Gaussian, " +
                              "non-Gaussian or super-sample covariance has " +
                              "any values. This term will be manually set " +
                              "to False.")
                    continue
            for _ in range(mult):
                if isinstance(gauss[gaussidx], np.ndarray):    
                    gauss[gaussidx] = self.__check_for_empty_input(gauss[gaussidx], gauss[gaussidx].shape)
                    gaussidx += 1
            if isinstance(nongauss[idx], np.ndarray):
                nongauss[idx] = self.__check_for_empty_input(nongauss[idx], nongauss[idx].shape)
            if isinstance(ssc[idx], np.ndarray):
                ssc[idx] = self.__check_for_empty_input(ssc[idx], ssc[idx].shape)

        if ('terminal' in self.style or 'list' in self.style):
            fct_args = [obslist, obsbool]
            if self.list_style_spatial_first:
                self.__write_cov_list_cosmosis_style(cov_dict, obs_dict, n_tomo_clust, 
                                    n_tomo_lens, sampledim, proj_quant, 
                                    gauss, nongauss, ssc, fct_args)
            else:
                self.__write_cov_list(cov_dict, obs_dict, n_tomo_clust, 
                                    n_tomo_lens, sampledim, proj_quant, 
                                    gauss, nongauss, ssc, fct_args)
        if 'matrix' in self.style or self.plot:
            fct_args = [obslist, obsbool, obslength, mult, 
                        gg, gm, mm, xipp, xipm, ximm]
            self.__write_cov_matrix_new(obs_dict, cov_dict, n_tomo_clust,
                                    n_tomo_lens, sampledim, proj_quant, 
                                    gauss, nongauss, ssc, fct_args)

    def plot_corrcoeff_matrix(self,
                              obs_dict,
                              covmatrix, 
                              cov_diag,
                              proj_quant,
                              n_tomo_clust,
                              n_tomo_lens,
                              sampledim,
                              filename = None,
                              fct_args = None):
        """
        Plots the Pearson correlation coefficient of the covariance matrix
        to a file depending on the specifications in the config.ini.

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
        covmatrix : 2d array
            The full covariance matrix with all contributions 
        cov_diag : 2d array
            The diagonal block part of the covariance matrix
        n_tomo_clust : int
            Number of clustering (lens) bins
        n_tomo_lens : int
            Number of lensing (source) bins
        sampledim : int
            Number of sample bins
        filename : str
            Filename of the plot
        """
        obslist, obsbool, obslength, mult, gg, gm, mm, xipp, xipm, ximm = \
            fct_args
        ratio = len(covmatrix) / 140
        if self.tex:
            plt.rc('text', usetex=True)
            #plt.rc('image', interpolation='none')

        else:
            plt.rc('text', usetex=False)
            #plt.rc('image', interpolation='none')

        fig, ax = plt.subplots(1, 1, figsize=(12,12))

        corr_covmatrix = self.__correlation_matrix(covmatrix)
        
        limit = max(-min(corr_covmatrix.flatten()), max(corr_covmatrix.flatten()))
        cbar = ax.imshow(corr_covmatrix, cmap = 'seismic', 
                         extent = [0, len(corr_covmatrix), 0, len(corr_covmatrix)],
                         vmin=-limit, vmax=limit, interpolation='nearest')
        fig.colorbar(cbar, location='bottom', shrink=.775, aspect=30, pad=0.055).ax.tick_params(axis='x', direction='in')
        ax.text(len(covmatrix)/2, -6*ratio, 'Correlation coefficients', fontsize=16, ha='center', va='center')

        
        labels_position = []
        labels_position_y = []
        labels_text = []
        position = 0
        old_position = 0
        if gg:
            if np.any(self.projected_clust):
                proj_quant = self.projected_clust
            sub_position_tomo = 0
            for sub_tomo in range(int(n_tomo_clust*(n_tomo_clust + 1)/2)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += len(proj_quant)*sampledim
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                for sub_sample in range(sampledim):
                    sub_position_sample += len(proj_quant)
                    ax.axhline(y=len(covmatrix)-sub_position_sample, color='black', linewidth=.15, ls = ":")
                    ax.axvline(x=sub_position_sample, color='black', linewidth=.15, ls = ":")
            position += len(proj_quant)*sampledim*n_tomo_clust*(n_tomo_clust + 1)/2
            old_position = position
            labels_position.append(position/2)
            labels_position_y.append(len(covmatrix) - position/2)
            if obs_dict['observables']['est_clust'] == 'k_space':
                labels_text.append(r'$P_\mathrm{gg}(k)$')
            if obs_dict['observables']['est_clust'] == 'C_ell':
                labels_text.append(r'$C_\mathrm{gg}(\ell)$')
            if obs_dict['observables']['est_clust'] == 'w':
                labels_text.append(r'$w(\theta)$')
            if obs_dict['observables']['est_clust'] == 'cosebi':
                labels_text.append(r'$\Psi^\mathrm{gg}_n$')
            if obs_dict['observables']['est_clust'] == 'bandpowers':
                labels_text.append(r'$\mathcal{C}_\mathrm{gg}(L)$')
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        if gm:
            if np.any(self.projected_clust):
                proj_quant = self.projected_clust
            sub_position_tomo = old_position
            for sub_tomo in range(int(n_tomo_clust*n_tomo_lens)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += len(proj_quant)*sampledim
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                for sub_sample in range(sampledim):
                    sub_position_sample += len(proj_quant)
                    ax.axhline(y=len(covmatrix)-sub_position_sample, color='black', linewidth=.15, ls = ":")
                    ax.axvline(x=sub_position_sample, color='black', linewidth=.15, ls = ":")
            position += len(proj_quant)*sampledim*n_tomo_clust*n_tomo_lens
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            
            old_position = position
            if obs_dict['observables']['est_ggl'] == 'k_space':
                labels_text.append(r'$P_\mathrm{gm}(k)$')
            if obs_dict['observables']['est_ggl'] == 'C_ell':
                labels_text.append(r'$C_\mathrm{gm}(\ell)$')
            if obs_dict['observables']['est_ggl'] == 'gamma_t':
                labels_text.append(r'$\gamma_\mathrm{t}(\theta)$')
            if obs_dict['observables']['est_ggl'] == 'cosebi':
                labels_text.append(r'$\Psi^\mathrm{gm}_n$')
            if obs_dict['observables']['est_ggl'] == 'bandpowers':
                labels_text.append(r'$\mathcal{C}_\mathrm{gm}(L)$')
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        if mm:
            if np.any(self.projected_lens):
                proj_quant = self.projected_lens
            sub_position_tomo = old_position
            for sub_tomo in range(int(n_tomo_lens*(n_tomo_lens + 1)/2)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += len(proj_quant)
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += len(proj_quant)*n_tomo_lens*(n_tomo_lens + 1)/2
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            
            old_position = position
            if obs_dict['observables']['est_shear'] == 'k_space':
                labels_text.append(r'$P_\mathrm{mm}(k)$')
            if obs_dict['observables']['est_shear'] == 'C_ell':
                labels_text.append(r'$C_\mathrm{mm}(\ell)$')
            if obs_dict['observables']['est_shear'] == 'xi_pm':
                labels_text.append(r'$\xi_+(\theta)$')
                labels_text.append(r'$\xi_-(\theta)$')
            if obs_dict['observables']['est_shear'] == 'cosebi':
                labels_text.append(r'$E_n$')
                labels_text.append(r'$B_n$')
            if obs_dict['observables']['est_shear'] == 'bandpowers':
                labels_text.append(r'$\mathcal{C}_\mathrm{E}(L)$')
                labels_text.append(r'$\mathcal{C}_\mathrm{B}(L)$')
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
            sub_position_tomo = old_position
            if obs_dict['observables']['est_shear'] != 'C_ell':
                for sub_tomo in range(int(n_tomo_lens*(n_tomo_lens + 1)/2)):
                    sub_position_sample = sub_position_tomo
                    sub_position_tomo += len(proj_quant)
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                position += len(proj_quant)*n_tomo_lens*(n_tomo_lens + 1)/2
                labels_position.append(old_position + (position- old_position)/2)
                labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
                old_position = position
                ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
                ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        ax.xaxis.tick_top()

        plt.yticks(labels_position_y, labels_text)
        plt.xticks(labels_position, labels_text)

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)

        print("Plotting correlation matrix")
   

    def plot_corrcoeff_matrix_arbitrary(self,
                                        obs_dict,
                                        covmatrix, 
                                        cov_diag,
                                        summary,
                                        n_tomo_clust,
                                        n_tomo_lens,
                                        sampledim,
                                        filename = None,
                                        fct_args = None):
        """
        Plots the Pearson correlation coefficient of the covariance matrix
        to a file depending on the specifications in the config.ini.

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
        covmatrix : 2d array
            The full covariance matrix with all contributions 
        cov_diag : 2d array
            The diagonal block part of the covariance matrix
        n_tomo_clust : int
            Number of clustering (lens) bins
        n_tomo_lens : int
            Number of lensing (source) bins
        sampledim : int
            Number of sample bins
        filename : str
            Filename of the plot
        """
        obslist, obsbool, obslength, mult, gg, gm, mm, xipp, xipm, ximm = \
            fct_args
        ratio = len(covmatrix) / 140
        if self.tex:
            plt.rc('text', usetex=True)
            #plt.rc('image', interpolation='none')

        else:
            plt.rc('text', usetex=False)
            #plt.rc('image', interpolation='none')

        fig, ax = plt.subplots(1, 1, figsize=(12,12))

        corr_covmatrix = self.__correlation_matrix(covmatrix)
        
        limit = max(-min(corr_covmatrix.flatten()), max(corr_covmatrix.flatten()))
        cbar = ax.imshow(corr_covmatrix, cmap = 'seismic', 
                         extent = [0, len(corr_covmatrix), 0, len(corr_covmatrix)],
                         vmin=-limit, vmax=limit, interpolation='nearest')
        fig.colorbar(cbar, location='bottom', shrink=.775, aspect=30, pad=0.055).ax.tick_params(axis='x', direction='in')
        ax.text(len(covmatrix)/2, -6*ratio, 'Correlation coefficients', fontsize=16, ha='center', va='center')

        
        labels_position = []
        labels_position_y = []
        labels_text = []
        position = 0
        old_position = 0
        if gg:
            gg_summary_length = []
            gg_summary_length.append(int(summary['arb_number_first_summary_gg']))
            if summary['number_summary_gg'] > 1:
                gg_summary_length.append(int(len(summary['WL_gg']) - summary['arb_number_first_summary_gg']))
            sub_position_tomo = 0
            for i in range(summary['number_summary_gg']):
                for sub_tomo in range(int(n_tomo_clust*(n_tomo_clust + 1)/2)):
                    sub_position_sample = sub_position_tomo
                    sub_position_tomo += gg_summary_length[i]*sampledim
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    for sub_sample in range(sampledim):
                        sub_position_sample += gg_summary_length[i]
                        ax.axhline(y=len(covmatrix)-sub_position_sample, color='black', linewidth=.15, ls = ":")
                        ax.axvline(x=sub_position_sample, color='black', linewidth=.15, ls = ":")
                position += gg_summary_length[i]*sampledim*n_tomo_clust*(n_tomo_clust + 1)/2
                labels_position.append(position - gg_summary_length[i]*sampledim*n_tomo_clust*(n_tomo_clust + 1)/2/2)
                old_position = position
                labels_position_y.append(len(covmatrix) - (position - gg_summary_length[i]*sampledim*n_tomo_clust*(n_tomo_clust + 1)/2/2))
                labels_text.append(r'$\mathcal{O}_{\mathrm{gg},p_'+str(i+1)+'}(L)$')
                ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
                ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        if gm:
            gm_summary_length = []
            gm_summary_length.append(int(summary['arb_number_first_summary_gm']))
            if summary['number_summary_gm'] > 1:
                gm_summary_length.append(int(len(summary['WL_gm']) - summary['arb_number_first_summary_gm']))
            sub_position_tomo = old_position
            for i in range(summary['number_summary_gm']):    
                for sub_tomo in range(int(n_tomo_clust*n_tomo_lens)):
                    sub_position_sample = sub_position_tomo
                    sub_position_tomo += gm_summary_length[i]*sampledim
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    for sub_sample in range(sampledim):
                        sub_position_sample += gm_summary_length[i]
                        ax.axhline(y=len(covmatrix)-sub_position_sample, color='black', linewidth=.15, ls = ":")
                        ax.axvline(x=sub_position_sample, color='black', linewidth=.15, ls = ":")
                position += gm_summary_length[i]*sampledim*n_tomo_clust*n_tomo_lens
                labels_position.append(position - gm_summary_length[i]*sampledim*n_tomo_clust*n_tomo_lens/2)
                labels_position_y.append(len(covmatrix) - (position - gm_summary_length[i]*sampledim*n_tomo_clust*n_tomo_lens/2))
                old_position = position
                labels_text.append(r'$\mathcal{O}_{\mathrm{gm},p_'+str(i+1)+'}(L)$')
                ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
                ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        if mm:
            mm_summary_length = []
            mm_summary_length.append(int(summary['arb_number_first_summary_mm']))
            if summary['number_summary_mm'] > 1:
                mm_summary_length.append(int(len(summary['WL_mmE']) - summary['arb_number_first_summary_mm']))
            sub_position_tomo = old_position
            for i in range(summary['number_summary_mm']):
                for sub_tomo in range(int(n_tomo_lens*(n_tomo_lens + 1)/2)):
                    sub_position_sample = sub_position_tomo
                    sub_position_tomo += mm_summary_length[i]
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                position += mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2
                labels_position.append(position -  mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2/2)
                labels_position_y.append(len(covmatrix) - (position -  mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2/2))
                old_position = position
                labels_text.append(r'$\mathcal{O}_{\mathrm{mmE},p_'+str(i+1)+'}(L)$')
                ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
                ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
                sub_position_tomo = old_position
    
            for i in range(summary['number_summary_mm']):
                for sub_tomo in range(int(n_tomo_lens*(n_tomo_lens + 1)/2)):
                    sub_position_sample = sub_position_tomo
                    sub_position_tomo += mm_summary_length[i]
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                position += mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2
                labels_text.append(r'$\mathcal{O}_{\mathrm{mmB},p_'+str(i+1)+'}(L)$')
                labels_position.append(position - mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2/2)
                labels_position_y.append(len(covmatrix) - (position - mm_summary_length[i]*n_tomo_lens*(n_tomo_lens + 1)/2/2))
                old_position = position
                ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
                ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        ax.xaxis.tick_top()

        plt.yticks(labels_position_y, labels_text)
        plt.xticks(labels_position, labels_text, rotation=90)

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)

        print("Plotting correlation matrix")

    def __write_cov_list(self,
                         cov_dict,
                         obs_dict,
                         n_tomo_clust,
                         n_tomo_lens,
                         sampledim,
                         proj_quant,
                         gauss,
                         nongauss,
                         ssc,
                         fct_args):
        obslist, obsbool = fct_args
        proj_quant_str = 'x1\tx2\t'

        for observables in obs_dict['observables'].values():
            if np.any(observables == 'C_ell') or np.any(observables == 'bandpowers'):
                proj_quant_str = 'ell1\tell2\t'       
        if (obs_dict['observables']['est_shear'] == 'xi_pm' or \
              obs_dict['observables']['est_ggl'] == 'gamma_t' or \
              obs_dict['observables']['est_clust'] == 'w'):
            proj_quant_str = 'theta1\ttheta2\t'
        for observables in obs_dict['observables'].values():
            if np.any(observables == 'k_space'):   
                proj_quant_str = 'log10k1\t\tlog10k2'
        for observables in obs_dict['observables'].values():
            if np.any(observables == 'cosebi'):  
                proj_quant_str = 'n1\t\tn2'
        if not cov_dict['split_gauss']:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e'
        else:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'

        olist = []
        splitidx = 0
        write_header = True

        if obs_dict['ELLspace']['n_spec'] is not None and obs_dict['ELLspace']['n_spec'] != 0:
            obs_copy = ['gggg_ssss',  'gggg_sssp',  'gggg_sspp',  \
                'gggg_spsp',  'gggg_ppsp',  'gggg_pppp',  \
                'gggm_sssm',  'gggm_sspm',  'gggm_spsm',  \
                'gggm_sppm',  'gggm_ppsm',  'gggm_pppm',  \
                'ggmm_ssmm',  'ggmm_spmm',  'ggmm_ppmm',  \
                'gmgm_smsm',  'gmgm_smpm',  'gmgm_pmsm',  \
                'gmgm_pmpm',  'mmgm_mmsm',  'mmgm_mmpm',  \
                'mmmm_mmmm']
            obs_type = ['gggg', 'gggg', 'gggg', 'gggg', 'gggg', 'gggg',
                        'gggm', 'gggm', 'gggm', 'gggm', 'gggm', 'gggm',
                        'ggmm', 'ggmm', 'ggmm',
                        'gmgm', 'gmgm', 'gmgm', 'gmgm', 'gmgm',
                        'mmgm', 'mmgm',
                        'mmmm']
            if not cov_dict['split_gauss']:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                 tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                    write_header = False
                for i_probe in range(22):
                    if not isinstance(gauss[i_probe], np.ndarray):
                        continue
                    if not isinstance(nongauss[i_probe], np.ndarray):
                        nongauss[i_probe] = np.zeros_like(gauss[i_probe])
                    if not isinstance(ssc[i_probe], np.ndarray):
                        ssc[i_probe] = np.zeros_like(gauss[i_probe])
                    r1 = gauss[i_probe].shape[0]
                    r2 = gauss[i_probe].shape[1]
                    tomo1 = gauss[i_probe].shape[4]
                    tomo2 = gauss[i_probe].shape[5]
                    tomo3 = gauss[i_probe].shape[6]
                    tomo4 = gauss[i_probe].shape[7]
                    sampledim1 = 1
                    sampledim2 = 1
                    for i_r1 in range(r1):
                        for i_r2 in range(r2):
                            p1 = proj_quant[i_r1]
                            p2 = proj_quant[i_r2]                    
                            if obs_type[i_probe] == 'gggg' or obs_type[i_probe] == 'mmmm' or obs_type[i_probe] == 'ggmm':
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(t3, tomo4):
                                                if obs_type[i_probe] == 'gggg':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmmm':
                                                    sampledim1 = 1
                                                    sampledim2 = 1
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                        p2 = self.projected_lens[i_r2]
                                                if obs_type[i_probe] == 'ggmm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = 1
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                    if np.any(self.projected_lens):
                                                        p2 = self.projected_lens[i_r2]        
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                            if obs_type[i_probe] == 'gggm' or obs_type[i_probe] == 'mmgm':
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                if obs_type[i_probe] == 'gggm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmgm':
                                                    sampledim1 = 1
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                    if np.any(self.projected_clust):
                                                        p2 = self.projected_clust[i_r2]    
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                            if obs_type[i_probe] == 'gmgm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                                for t1 in range(tomo1):
                                    for t2 in range(tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        if np.any(self.projected_clust):
                                                            p1 = self.projected_clust[i_r1]
                                                            p2 = self.projected_clust[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                if 'terminal' in self.style:
                    print("Writing result to terminal. (Brace yourself...).'")
                    for ostr in olist:
                        print(ostr)
                elif 'list' in self.style:
                    fn = self.filename[self.style.index('list')]
                    with open(fn, 'w') as file:
                        print("Writing '" + fn + "'.")
                        for ostr in olist:
                            file.write("%s\n" % ostr)
            else:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                '\tcovg sn\t\tcovng\t\tcovssc')
                    write_header = False
                for i_probe in range(22):
                    if not isinstance(gauss[3*i_probe], np.ndarray):
                        continue
                    if not isinstance(gauss[3*i_probe + 1], np.ndarray):
                        gauss[3*i_probe + 1] = np.zeros_like(gauss[3*i_probe])
                    if not isinstance(gauss[3*i_probe + 2], np.ndarray):
                        gauss[3*i_probe + 2] = np.zeros_like(gauss[3*i_probe])
                    if not isinstance(nongauss[i_probe], np.ndarray):
                        nongauss[i_probe] = np.zeros_like(gauss[3*i_probe])
                    if not isinstance(ssc[i_probe], np.ndarray):
                        ssc[i_probe] = np.zeros_like(gauss[3*i_probe])
                    r1 = gauss[3*i_probe].shape[0]
                    r2 = gauss[3*i_probe].shape[1]
                    tomo1 = gauss[3*i_probe].shape[4]
                    tomo2 = gauss[3*i_probe].shape[5]
                    tomo3 = gauss[3*i_probe].shape[6]
                    tomo4 = gauss[3*i_probe].shape[7]
                    for i_r1 in range(r1):
                        for i_r2 in range(r2):
                            p1 = proj_quant[i_r1]
                            p2 = proj_quant[i_r2]                    
                            if obs_type[i_probe] == 'gggg' or obs_type[i_probe] == 'mmmm' or obs_type[i_probe] == 'ggmm':
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(t3, tomo4):
                                                if obs_type[i_probe] == 'gggg':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmmm':
                                                    sampledim1 = 1
                                                    sampledim2 = 1
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                        p2 = self.projected_lens[i_r2]
                                                if obs_type[i_probe] == 'ggmm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = 1
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                    if np.any(self.projected_lens):
                                                        p2 = self.projected_lens[i_r2]    
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                            if obs_type[i_probe] == 'gggm' or obs_type[i_probe] == 'mmgm':
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                if obs_type[i_probe] == 'gggm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmgm':
                                                    sampledim1 = 1
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                    if np.any(self.projected_clust):
                                                        p2 = self.projected_clust[i_r2]
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                            if obs_type[i_probe] == 'gmgm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                                if np.any(self.projected_clust):
                                    p1 = self.projected_clust[i_r1]
                                    p2 = self.projected_clust[i_r2]
                                for t1 in range(tomo1):
                                    for t2 in range(tomo2):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                if not self.save_as_binary:
                    if 'terminal' in self.style:
                            print("Writing result to terminal. (Brace yourself...).'")
                            for ostr in olist:
                                print(ostr)
                    elif 'list' in self.style:
                        fn = self.filename[self.style.index('list')]
                        with open(fn, 'w') as file:
                            print("Writing '" + fn + "'.")
                            for ostr in olist:
                                file.write("%s\n" % ostr)
                
        else:
            olist = []
            splitidx = 0
            write_header = True
            for oidx, obs in enumerate(obslist):
                obs_copy = np.copy(obs)
                if obs == 'xipxip' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_mmCE_mm'
                if obs == 'xipxim' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_mmCB_mm'
                if obs == 'ximxim' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CB_mmCB_mm'
                if obs == 'gggg' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['clustering'] == True:
                    obs_copy = 'CE_ggCE_gg'
                if obs == 'gmgm' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_gmCE_gm'
                if obs == 'gggm' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_ggCE_gm'
                if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_ggCE_mm'
                if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_ggCB_mm'
                if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_gmCE_mm'
                if obs == 'gmxim' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_gmCB_mm'
                

                if obs == 'xipxip' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'EmmEmm'
                if obs == 'xipxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'EmmBmm'
                if obs == 'ximxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'BmmBmm'
                if obs == 'gggg' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['clustering'] == True:
                    obs_copy = 'PsiggPsigg'
                if obs == 'gmgm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsigmPsigm'
                if obs == 'gggm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsiggPsigm'
                if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsiggEmm'
                if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsiggBmm'
                if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsigmEmm'
                if obs == 'gmxim' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsigmBmm'
                
                
                if not obsbool[oidx]:
                    splitidx += 3
                    continue
                sampledim1 = sampledim
                sampledim2 = sampledim
                if not cov_dict['split_gauss']:
                    if write_header:
                        olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                    tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                        write_header = False
                    if not isinstance(gauss[oidx], np.ndarray):
                        continue
                    if not isinstance(nongauss[oidx], np.ndarray):
                        nongauss[oidx] = np.zeros_like(gauss[oidx])
                    if not isinstance(ssc[oidx], np.ndarray):
                        ssc[oidx] = np.zeros_like(gauss[oidx])
                    r1 = gauss[oidx].shape[0]
                    r2 = gauss[oidx].shape[1]
                    for i_r1 in range(r1):
                        for i_r2 in range(r2):                    
                            if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                tomo1 = gauss[oidx].shape[4]
                                if obs == 'gggg':
                                    sampledim1 = sampledim
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        ri = self.projected_clust[i_r1]
                                        rj = self.projected_clust[i_r2]
                                    else:
                                        ri = proj_quant[i_r1]
                                        rj = proj_quant[i_r2]
                                if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                    sampledim1 = 1
                                    sampledim2 = 1
                                    if np.any(self.projected_lens):
                                        ri = self.projected_lens[i_r1]
                                        rj = self.projected_lens[i_r2]
                                    else:
                                        ri = proj_quant[i_r1]
                                        rj = proj_quant[i_r2]
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo1):
                                            for t4 in range(t3, tomo1):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[oidx][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[oidx][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs == 'gmgm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                                tomo1 = gauss[oidx].shape[4]
                                tomo2 = gauss[oidx].shape[5]
                                if np.any(self.projected_clust):
                                    ri = self.projected_clust[i_r1]
                                    rj = self.projected_clust[i_r2]
                                else:
                                    ri = proj_quant[i_r1]
                                    rj = proj_quant[i_r2]
                                for t1 in range(tomo1):
                                    for t2 in range(tomo2):
                                        for t3 in range(tomo1):
                                            for t4 in range(tomo2):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[oidx][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[oidx][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                                tomo1 = gauss[oidx].shape[4]
                                tomo3 = gauss[oidx].shape[6]
                                tomo4 = gauss[oidx].shape[7]
                                if obs == 'gggm':
                                    sampledim1 = sampledim
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        ri = self.projected_clust[i_r1]
                                        rj = self.projected_clust[i_r2]
                                    else:
                                        ri = proj_quant[i_r1]
                                        rj = proj_quant[i_r2]
                                if obs in ['mmgm', 'gmxip', 'gmxim']:
                                    sampledim1 = 1
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        rj = self.projected_clust[i_r2]
                                    else:
                                        rj = proj_quant[i_r2]
                                    if np.any(self.projected_lens):
                                        ri = self.projected_lens[i_r1]
                                    else:
                                        ri = proj_quant[i_r1]
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[oidx][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[oidx][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs in ['ggmm', 'ggxip', 'ggxim']:
                                tomo1 = gauss[oidx].shape[4]
                                tomo2 = gauss[oidx].shape[6]
                                sampledim1 = sampledim
                                sampledim2 = 1
                                if np.any(self.projected_lens):
                                    rj = self.projected_lens[i_r2]
                                else:
                                    rj = proj_quant[i_r2]
                                if np.any(self.projected_clust):
                                    ri = self.projected_clust[i_r1]
                                else:
                                    ri = proj_quant[i_r1]
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo2):
                                            for t4 in range(t3, tomo2):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[oidx][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[oidx][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                else:
                    if write_header:
                        olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                    tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                    '\tcovg sn\t\tcovng\t\tcovssc')
                        write_header = False
                    if not isinstance(gauss[3*oidx], np.ndarray):
                        continue
                    if not isinstance(gauss[3*oidx + 1], np.ndarray):
                        gauss[3*oidx + 1] = np.zeros_like(gauss[3*oidx])
                    if not isinstance(gauss[3*oidx + 2], np.ndarray):
                        gauss[3*oidx + 2] = np.zeros_like(gauss[3*oidx])
                    if not isinstance(nongauss[oidx], np.ndarray):
                        nongauss[oidx] = np.zeros_like(gauss[3*oidx])
                    if not isinstance(ssc[oidx], np.ndarray):
                        ssc[oidx] = np.zeros_like(gauss[3*oidx])
                    r1 = gauss[3*oidx].shape[0]
                    r2 = gauss[3*oidx].shape[1]
                    for i_r1 in range(r1):
                        for i_r2 in range(r2):
                            if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                tomo1 = gauss[splitidx].shape[4]
                                if obs == 'gggg':
                                    sampledim1 = sampledim
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        rj = self.projected_clust[i_r2]
                                        ri = self.projected_clust[i_r1]
                                    else:
                                        ri = proj_quant[i_r1]
                                        rj = proj_quant[i_r2]
                                if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                    sampledim1 = 1
                                    sampledim2 = 1
                                    if np.any(self.projected_lens):
                                        rj = self.projected_lens[i_r2]
                                        ri = self.projected_lens[i_r1]
                                    else:
                                        ri = proj_quant[i_r1]
                                        rj = proj_quant[i_r2]
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo1):
                                            for t4 in range(t3, tomo1):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[splitidx][idxs] \
                                                            + gauss[splitidx+1][idxs] \
                                                            + gauss[splitidx+2][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[splitidx][idxs],
                                                            gauss[splitidx+1][idxs],
                                                            gauss[splitidx+2][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs == 'gmgm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                                if np.any(self.projected_clust):
                                    rj = self.projected_clust[i_r2]
                                    ri = self.projected_clust[i_r1]
                                tomo1 = gauss[splitidx].shape[4]
                                tomo2 = gauss[splitidx].shape[5]
                                for t1 in range(tomo1):
                                    for t2 in range(tomo2):
                                        for t3 in range(tomo1):
                                            for t4 in range(tomo2):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[splitidx][idxs] \
                                                            + gauss[splitidx+1][idxs] \
                                                            + gauss[splitidx+2][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[splitidx][idxs],
                                                            gauss[splitidx+1][idxs],
                                                            gauss[splitidx+2][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                                tomo1 = gauss[splitidx].shape[4]
                                tomo3 = gauss[splitidx].shape[6]
                                tomo4 = gauss[splitidx].shape[7]
                                if obs == 'gggm':
                                    sampledim1 = sampledim
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        rj = self.projected_clust[i_r2]
                                        ri = self.projected_clust[i_r1]
                                    
                                if obs in ['mmgm', 'gmxip', 'gmxim']:
                                    sampledim1 = 1
                                    sampledim2 = sampledim
                                    if np.any(self.projected_clust):
                                        rj = self.projected_clust[i_r2]
                                    else:
                                        rj = proj_quant[i_r2]
                                    if np.any(self.projected_lens):
                                        ri = self.projected_lens[i_r1]
                                    else:
                                        ri = proj_quant[i_r1]                                    
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo3):
                                            for t4 in range(tomo4):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[splitidx][idxs] \
                                                            + gauss[splitidx+1][idxs] \
                                                            + gauss[splitidx+2][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[splitidx][idxs],
                                                            gauss[splitidx+1][idxs],
                                                            gauss[splitidx+2][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                            elif obs in ['ggmm', 'ggxip', 'ggxim']:
                                tomo1 = gauss[splitidx].shape[4]
                                tomo2 = gauss[splitidx].shape[6]
                                sampledim1 = sampledim
                                sampledim2 = 1
                                if np.any(self.projected_lens):
                                    rj = self.projected_lens[i_r2]
                                else:
                                    rj = proj_quant[i_r2]
                                if np.any(self.projected_clust):
                                    ri = self.projected_clust[i_r1]
                                else:
                                    ri = proj_quant[i_r1]
                                for t1 in range(tomo1):
                                    for t2 in range(t1, tomo1):
                                        for t3 in range(tomo2):
                                            for t4 in range(t3, tomo2):
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t1, t3, t4)
                                                        cov = gauss[splitidx][idxs] \
                                                            + gauss[splitidx+1][idxs] \
                                                            + gauss[splitidx+2][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[splitidx][idxs],
                                                            gauss[splitidx+1][idxs],
                                                            gauss[splitidx+2][idxs],
                                                            nongauss[oidx][idxs],
                                                            ssc[oidx][idxs])
                                                        olist.append(ostr)
                    splitidx += 3
            
            index = 0
            if self.has_csmf:
                if self.is_cell:
                    obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmm"]
                else:
                    obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmmE", "csmfmmB"]
                for obs in self.conditional_stellar_mass_function_cov:
                    csmf_auto = False
                    if index == 0:
                        csmf_auto = True
                    if not isinstance(obs, np.ndarray):
                        continue
                    if csmf_auto:
                        for i_r1 in range(len(obs[:, 0,0,0])):
                            for i_r2 in range(len(obs[0, :, 0, 0])):
                                for t1 in range(len(obs[0,0,:,0])):
                                    for t2 in range(len(obs[0,0,0,:])):
                                        ri = i_r1
                                        rj = i_r2              
                                        cov = obs[i_r1, i_r2, t1, t2]
                                        if not cov_dict['split_gauss']:
                                            ostr = ostr_format \
                                                % (obs_copy[index],  ri, rj, 
                                                1, 1, t1+1, t2+1, t1+1, t2+1, 
                                                cov, 
                                                0,
                                                0,
                                                0)
                                        else:
                                            ostr = ostr_format \
                                                % (obs_copy[index], ri, rj,
                                                1, 1, t1+1, t2+1, t1+1, t2+1, 
                                                cov, 
                                                0,
                                                0,
                                                0,
                                                0,
                                                0)
                                        olist.append(ostr)
                    else:
                        for i_r1 in range(len(obs[:, 0,0,0,0,0])):
                            for i_r2 in range(len(obs[0, :, 0, 0,0,0])):                
                                for s1 in range(len(obs[0,0,:,0,0,0])):
                                    for t1 in range(len(obs[0,0,0,:,0,0])):
                                        for t2 in range(len(obs[0,0,0,0,:,0])):
                                            for t3 in range(len(obs[0,0,0,0,0,:])):
                                                ri = i_r1
                                                rj = i_r2              
                                                cov = obs[i_r1, i_r2, s1, t1, t2, t3]
                                                if not cov_dict['split_gauss']:
                                                    ostr = ostr_format \
                                                        % (obs_copy[index],  ri, rj, 
                                                        s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                        cov, 
                                                        0,
                                                        0,
                                                        0)
                                                else:
                                                    ostr = ostr_format \
                                                        % (obs_copy[index], ri, rj,
                                                        s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                        cov, 
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0)
                                                olist.append(ostr)
                    index += 1
            if not self.save_as_binary:
                if 'terminal' in self.style:
                    print("Writing result to terminal. (Brace yourself...).'")
                    for ostr in olist:
                        print(ostr)
                elif 'list' in self.style:
                    fn = self.filename[self.style.index('list')]
                    with open(fn, 'w') as file:
                        print("Writing '" + fn + "'.")
                        for ostr in olist:
                            file.write("%s\n" % ostr)
        return True
    
    def __write_cov_list_cosmosis_style(self,
                                        cov_dict,
                                        obs_dict,
                                        n_tomo_clust,
                                        n_tomo_lens,
                                        sampledim,
                                        proj_quant,
                                        gauss,
                                        nongauss,
                                        ssc,
                                        fct_args):
        obslist, obsbool = fct_args
        proj_quant_str = 'x1\tx2\t'

        for observables in obs_dict['observables'].values():
            if np.any(observables == 'C_ell') or np.any(observables == 'bandpowers'):
                proj_quant_str = 'ell1\tell2\t'       
        if (obs_dict['observables']['est_shear'] == 'xi_pm' or \
              obs_dict['observables']['est_ggl'] == 'gamma_t' or \
              obs_dict['observables']['est_clust'] == 'w'):
            proj_quant_str = 'theta1\ttheta2\t'
        for observables in obs_dict['observables'].values():
            if np.any(observables == 'k_space'):   
                proj_quant_str = 'log10k1\t\tlog10k2'
        for observables in obs_dict['observables'].values():
            if np.any(observables == 'cosebi'):  
                proj_quant_str = 'n1\t\tn2'
        if not cov_dict['split_gauss']:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e'
        else:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%.2e\t%.2e\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'

        idxlist = self.__get_idxlist(proj_quant, sampledim)
        olist = []
        splitidx = 0
        write_header = True

        if obs_dict['ELLspace']['n_spec'] is not None and obs_dict['ELLspace']['n_spec'] != 0:
            obs_copy = ['gggg_ssss',  'gggg_sssp',  'gggg_sspp',  \
                'gggg_spsp',  'gggg_ppsp',  'gggg_pppp',  \
                'gggm_sssm',  'gggm_sspm',  'gggm_spsm',  \
                'gggm_sppm',  'gggm_ppsm',  'gggm_pppm',  \
                'ggmm_ssmm',  'ggmm_spmm',  'ggmm_ppmm',  \
                'gmgm_smsm',  'gmgm_smpm',  'gmgm_pmsm',  \
                'gmgm_pmpm',  'mmgm_mmsm',  'mmgm_mmpm',  \
                'mmmm_mmmm']
            obs_type = ['gggg', 'gggg', 'gggg', 'gggg', 'gggg', 'gggg',
                        'gggm', 'gggm', 'gggm', 'gggm', 'gggm', 'gggm',
                        'ggmm', 'ggmm', 'ggmm',
                        'gmgm', 'gmgm', 'gmgm', 'gmgm', 'gmgm',
                        'mmgm', 'mmgm',
                        'mmmm']
            if not cov_dict['split_gauss']:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                 tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                    write_header = False
                for i_probe in range(22):
                    if not isinstance(gauss[i_probe], np.ndarray):
                        continue
                    r1 = gauss[i_probe].shape[0]
                    r2 = gauss[i_probe].shape[1]
                    tomo1 = gauss[i_probe].shape[4]
                    tomo2 = gauss[i_probe].shape[5]
                    tomo3 = gauss[i_probe].shape[6]
                    tomo4 = gauss[i_probe].shape[7]
                    sampledim1 = 1
                    sampledim2 = 1
                    if obs_type[i_probe] == 'gggg' or obs_type[i_probe] == 'mmmm' or obs_type[i_probe] == 'ggmm':
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(t3, tomo4):
                                        for i_r1 in range(r1):
                                            for i_r2 in range(r2):
                                                p1 = proj_quant[i_r1]
                                                p2 = proj_quant[i_r2] 
                                                if obs_type[i_probe] == 'gggg':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmmm':
                                                    sampledim1 = 1
                                                    sampledim2 = 1
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                        p2 = self.projected_lens[i_r2]
                                                if obs_type[i_probe] == 'ggmm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = 1
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                    if obs_type[i_probe] == 'gggm' or obs_type[i_probe] == 'mmgm':
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        for i_r1 in range(r1):
                                            for i_r2 in range(r2):
                                                p1 = proj_quant[i_r1]
                                                p2 = proj_quant[i_r2] 
                                                if obs_type[i_probe] == 'gggm':
                                                    sampledim1 = sampledim
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                        p2 = self.projected_clust[i_r2]
                                                if obs_type[i_probe] == 'mmgm':
                                                    sampledim1 = 1
                                                    sampledim2 = sampledim
                                                    if np.any(self.projected_lens):
                                                        p1 = self.projected_lens[i_r1]
                                                    if np.any(self.projected_clust):
                                                        p1 = self.projected_clust[i_r1]
                                                for i_s1 in range(sampledim1):
                                                    for i_s2 in range(sampledim2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                    if obs_type[i_probe] == 'gmgm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                        for t1 in range(tomo1):
                            for t2 in range(tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        p1 = proj_quant[i_r1]
                                                        p2 = proj_quant[i_r2] 
                                                        if np.any(self.projected_clust):
                                                            p1 = self.projected_clust[i_r1]
                                                            p2 = self.projected_clust[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        cov = gauss[i_probe][idxs] \
                                                            + nongauss[i_probe][idxs] \
                                                            + ssc[i_probe][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss[i_probe][idxs],
                                                            nongauss[i_probe][idxs],
                                                            ssc[i_probe][idxs])
                                                        olist.append(ostr)
                if not self.save_as_binary:
                    if 'terminal' in self.style:
                        print("Writing result to terminal. (Brace yourself...).'")
                        for ostr in olist:
                            print(ostr)
                    elif 'list' in self.style:
                        fn = self.filename[self.style.index('list')]
                        with open(fn, 'w') as file:
                            print("Writing '" + fn + "'.")
                            for ostr in olist:
                                file.write("%s\n" % ostr)
            else:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                '\tcovg sn\t\tcovng\t\tcovssc')
                    write_header = False
                for i_probe in range(22):
                    if not isinstance(gauss[3*i_probe], np.ndarray):
                        continue
                    r1 = gauss[3*i_probe].shape[0]
                    r2 = gauss[3*i_probe].shape[1]
                    tomo1 = gauss[3*i_probe].shape[4]
                    tomo2 = gauss[3*i_probe].shape[5]
                    tomo3 = gauss[3*i_probe].shape[6]
                    tomo4 = gauss[3*i_probe].shape[7]
                    if obs_type[i_probe] == 'gggg' or obs_type[i_probe] == 'mmmm' or obs_type[i_probe] == 'ggmm':
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(t3, tomo4):
                                        if obs_type[i_probe] == 'gggg':
                                            sampledim1 = sampledim
                                            sampledim2 = sampledim
                                        if obs_type[i_probe] == 'mmmm':
                                            sampledim1 = 1
                                            sampledim2 = 1
                                        if obs_type[i_probe] == 'ggmm':
                                            sampledim1 = sampledim
                                            sampledim2 = 1                  
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        p1 = proj_quant[i_r1]
                                                        p2 = proj_quant[i_r2]
                                                        if obs_type[i_probe] == 'gggg':
                                                            if np.any(self.projected_clust):
                                                                p1 = self.projected_clust[i_r1]
                                                                p2 = self.projected_clust[i_r2]
                                                        if obs_type[i_probe] == 'mmmm':
                                                            if np.any(self.projected_lens):
                                                                p1 = self.projected_lens[i_r1]
                                                                p2 = self.projected_lens[i_r2]
                                                        if obs_type[i_probe] == 'ggmm':
                                                            if np.any(self.projected_clust):
                                                                p1 = self.projected_clust[i_r1]
                                                            if np.any(self.projected_lens):
                                                                p1 = self.projected_lens[i_r1]         
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    if obs_type[i_probe] == 'gggm' or obs_type[i_probe] == 'mmgm':
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        if obs_type[i_probe] == 'gggm':
                                            sampledim1 = sampledim
                                            sampledim2 = sampledim  
                                        if obs_type[i_probe] == 'mmgm':
                                            sampledim1 = 1
                                            sampledim2 = sampledim   
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        p1 = proj_quant[i_r1]
                                                        p2 = proj_quant[i_r2]
                                                        if obs_type[i_probe] == 'gggm':
                                                            if np.any(self.projected_clust):
                                                                p1 = self.projected_clust[i_r1]
                                                                p2 = self.projected_clust[i_r2]
                                                        if obs_type[i_probe] == 'mmgm':
                                                            if np.any(self.projected_lens):
                                                                p1 = self.projected_lens[i_r1]
                                                            if np.any(self.projected_clust):
                                                                p2 = self.projected_clust[i_r2]                                                       
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    if obs_type[i_probe] == 'gmgm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                        for t1 in range(tomo1):
                            for t2 in range(tomo2):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        p1 = proj_quant[i_r1]
                                                        p2 = proj_quant[i_r2]
                                                        if np.any(self.projected_clust):
                                                            p1 = self.projected_clust[i_r1]
                                                            p2 = self.projected_clust[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + nongauss_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy[i_probe], p1, p2,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                if not self.save_as_binary:
                    if 'terminal' in self.style:
                            print("Writing result to terminal. (Brace yourself...).'")
                            for ostr in olist:
                                print(ostr)
                    elif 'list' in self.style:
                        fn = self.filename[self.style.index('list')]
                        with open(fn, 'w') as file:
                            print("Writing '" + fn + "'.")
                            for ostr in olist:
                                file.write("%s\n" % ostr)
                
        else:
            
            idxlist = self.__get_idxlist(proj_quant, sampledim)
            olist = []
            splitidx = 0
            write_header = True        
            for oidx, obs in enumerate(obslist):
                i_probe = oidx
                obs_copy = obs
                if obs == 'xipxip' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_mmCE_mm'
                if obs == 'xipxim' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_mmCB_mm'
                if obs == 'ximxim' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CB_mmCB_mm'
                if obs == 'gggg' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['clustering'] == True:
                    obs_copy = 'CE_ggCE_gg'
                if obs == 'gmgm' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_gmCE_gm'
                if obs == 'gggm' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_ggCE_gm'
                if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_ggCE_mm'
                if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_ggCB_mm'
                if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'CE_gmCE_mm'
                if obs == 'gmxim' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'CE_gmCB_mm'
                

                if obs == 'xipxip' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'EmmEmm'
                if obs == 'xipxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'EmmBmm'
                if obs == 'ximxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'BmmBmm'
                if obs == 'gggg' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['clustering'] == True:
                    obs_copy = 'PsiggPsigg'
                if obs == 'gmgm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsigmPsigm'
                if obs == 'gggm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsiggPsigm'
                if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsiggEmm'
                if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsiggBmm'
                if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                    obs_copy = 'PsigmEmm'
                if obs == 'gmxim' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                    obs_copy = 'PsigmBmm'

                if not obsbool[oidx]:
                    splitidx += 3
                    continue
                sampledim1 = sampledim
                sampledim2 = sampledim
                if not cov_dict['split_gauss']:
                    if not isinstance(gauss[oidx], np.ndarray):
                        continue
                    r1 = gauss[oidx].shape[0]
                    r2 = gauss[oidx].shape[1]
                    
                    if write_header:
                        olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                    tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                        write_header = False
                    if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                        tomo1 = gauss[oidx].shape[4]
                        if obs == 'gggg':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                        if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            sampledim1 = 1
                            sampledim2 = 1
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo1):
                                    for t4 in range(t3, tomo1):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        if obs == 'gggg':
                                                            if np.any(self.projected_clust):
                                                                ri = self.projected_clust[i_r1]
                                                                rj = self.projected_clust[i_r2]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]
                                                        if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                                            if np.any(self.projected_lens):
                                                                ri = self.projected_lens[i_r1]
                                                                rj = self.projected_lens[i_r2]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[i_probe], int):
                                                            gauss_aux = 0.0
                                                        else:
                                                            gauss_aux = gauss[i_probe][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]    
                                                        cov = gauss_aux + nongauss_aux + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_aux,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs == 'gmgm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                        tomo1 = gauss[oidx].shape[4]
                        tomo2 = gauss[oidx].shape[5]
                        for t1 in range(tomo1):
                            for t2 in range(tomo2):
                                for t3 in range(tomo1):
                                    for t4 in range(tomo2):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        ri = proj_quant[i_r1]
                                                        rj = proj_quant[i_r2]
                                                        if np.any(self.projected_clust):
                                                            ri = self.projected_clust[i_r1]
                                                            rj = self.projected_clust[i_r2]
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                            rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[i_probe], int):
                                                            gauss_aux = 0.0
                                                        else:
                                                            gauss_aux = gauss[i_probe][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]    
                                                        cov = gauss_aux + nongauss_aux + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_aux,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                        tomo1 = gauss[oidx].shape[4]
                        tomo3 = gauss[oidx].shape[6]
                        tomo4 = gauss[oidx].shape[7]
                        if obs == 'gggm':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                        if obs in ['mmgm', 'gmxip', 'gmxim']:
                            sampledim1 = 1
                            sampledim2 = sampledim
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        if obs == 'gggm':
                                                            if np.any(self.projected_clust):
                                                                ri = self.projected_clust[i_r1]
                                                                rj = self.projected_clust[i_r2]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]
                                                        if obs in ['mmgm', 'gmxip', 'gmxim']:
                                                            if np.any(self.projected_lens):
                                                                ri = self.projected_lens[i_r1]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                            if np.any(self.projected_clust):
                                                                rj = self.projected_clust[i_r2]
                                                            else:
                                                                rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[i_probe], int):
                                                            gauss_aux = 0.0
                                                        else:
                                                            gauss_aux = gauss[i_probe][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]    
                                                        cov = gauss_aux + nongauss_aux + ssc_aux 
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_aux,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs in ['ggmm', 'ggxip', 'ggxim']:
                        tomo1 = gauss[oidx].shape[4]
                        tomo2 = gauss[oidx].shape[6]
                        sampledim1 = sampledim
                        sampledim2 = 1
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo2):
                                    for t4 in range(t3, tomo2):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        ri = proj_quant[i_r1]
                                                        rj = proj_quant[i_r2]
                                                        if np.any(self.projected_lens):
                                                            rj = self.projected_lens[i_r2]
                                                        else:
                                                            rj = proj_quant[i_r2]
                                                        if np.any(self.projected_clust):
                                                            ri = self.projected_clust[i_r1]
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if isinstance(gauss[i_probe], int):
                                                            gauss_aux = 0.0
                                                        else:
                                                            gauss_aux = gauss[i_probe][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]    
                                                        cov = gauss_aux + nongauss_aux + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy,  ri, rj, 
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_aux,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                else:
                    if not isinstance(gauss[3*oidx], np.ndarray):
                        continue
                    r1 = gauss[3*oidx].shape[0]
                    r2 = gauss[3*oidx].shape[1]
                    
                    if write_header:
                        olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                    tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                    '\tcovg sn\t\tcovng\t\tcovssc')
                        write_header = False
                    if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                        tomo1 = gauss[splitidx].shape[4]
                        if obs == 'gggg':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                        if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            sampledim1 = 1
                            sampledim2 = 1
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo1):
                                    for t4 in range(t3, tomo1):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if obs == 'gggg':
                                                            if np.any(self.projected_clust):
                                                                ri = self.projected_clust[i_r1]
                                                                rj = self.projected_clust[i_r2]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]
                                                        if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                                            if np.any(self.projected_lens):
                                                                rj = self.projected_lens[i_r2]
                                                                ri = self.projected_lens[i_r1]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]
                                                        
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy, ri, rj,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs == 'gmgm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                        tomo1 = gauss[splitidx].shape[4]
                        tomo2 = gauss[splitidx].shape[5]
                        for t1 in range(tomo1):
                            for t2 in range(tomo2):
                                for t3 in range(tomo1):
                                    for t4 in range(tomo2):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        if np.any(self.projected_clust):
                                                            ri = self.projected_clust[i_r1]
                                                            rj = self.projected_clust[i_r2]             
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                            rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy, ri, rj,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                        tomo1 = gauss[splitidx].shape[4]
                        tomo3 = gauss[splitidx].shape[6]
                        tomo4 = gauss[splitidx].shape[7]
                        if obs == 'gggm':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                        if obs in ['mmgm', 'gmxip', 'gmxim']:
                            sampledim1 = 1
                            sampledim2 = sampledim
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo3):
                                    for t4 in range(tomo4):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        if obs == 'gggm':
                                                            if np.any(self.projected_clust):
                                                                ri = self.projected_clust[i_r1]
                                                                rj = self.projected_clust[i_r2]           
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                                rj = proj_quant[i_r2]  
                                                        if obs in ['mmgm', 'gmxip', 'gmxim']:
                                                            if np.any(self.projected_lens):
                                                                ri = self.projected_lens[i_r1]
                                                            else:
                                                                ri = proj_quant[i_r1]
                                                            if np.any(self.projected_clust): 
                                                                rj = self.projected_clust[i_r2] 
                                                            else:
                                                                rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy, ri, rj,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    elif obs in ['ggmm', 'ggxip', 'ggxim']:
                        tomo1 = gauss[splitidx].shape[4]
                        tomo2 = gauss[splitidx].shape[6]
                        sampledim1 = sampledim
                        sampledim2 = 1
                        for t1 in range(tomo1):
                            for t2 in range(t1, tomo1):
                                for t3 in range(tomo2):
                                    for t4 in range(t3, tomo2):
                                        for i_s1 in range(sampledim1):
                                            for i_s2 in range(sampledim2):
                                                for i_r1 in range(r1):
                                                    for i_r2 in range(r2):
                                                        if np.any(self.projected_lens):
                                                            rj = self.projected_lens[i_r2]
                                                        else:
                                                            rj = proj_quant[i_r2]
                                                        if np.any(self.projected_clust): 
                                                            ri = self.projected_clust[i_r1] 
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        gauss_sva = gauss[3*i_probe]
                                                        if isinstance(gauss[3*i_probe], int):
                                                            gauss_sva = 0.0
                                                        else:
                                                            gauss_sva = gauss[3*i_probe][idxs]
                                                        if isinstance(gauss[3*i_probe+1], int):
                                                            gauss_mix = 0.0
                                                        else:
                                                            gauss_mix = gauss[3*i_probe+1][idxs]
                                                        if isinstance(gauss[3*i_probe+2], int):
                                                            gauss_sn = 0.0
                                                        else:
                                                            gauss_sn = gauss[3*i_probe+2][idxs]
                                                        if isinstance(nongauss[i_probe], int):
                                                            nongauss_aux = 0.0
                                                        else:
                                                            nongauss_aux = nongauss[i_probe][idxs]
                                                        if isinstance(ssc[i_probe], int):
                                                            ssc_aux = 0.0
                                                        else:
                                                            ssc_aux = ssc[i_probe][idxs]
                                                        cov = gauss_sva \
                                                            + gauss_mix \
                                                            + gauss_sn \
                                                            + nongauss_aux \
                                                            + ssc_aux
                                                        ostr = ostr_format \
                                                            % (obs_copy, ri, rj,
                                                            i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                            cov, 
                                                            gauss_sva,
                                                            gauss_mix,
                                                            gauss_sn,
                                                            nongauss_aux,
                                                            ssc_aux)
                                                        olist.append(ostr)
                    splitidx += 3
            
            index = 0
            if self.has_csmf:
                if self.is_cell:
                    obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmm"]
                else:
                    obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmmE", "csmfmmB"]
                for obs in self.conditional_stellar_mass_function_cov:
                    csmf_auto = False
                    if index == 0:
                        csmf_auto = True
                    if not isinstance(obs, np.ndarray):
                        continue
                    if csmf_auto:
                        for t1 in range(len(obs[0,0,:,0])):
                            for t2 in range(len(obs[0,0,0,:])):
                                for i_r1 in range(len(obs[:, 0,0,0])):
                                    for i_r2 in range(len(obs[0, :, 0, 0])):
                                        ri = i_r1
                                        rj = i_r2              
                                        cov = obs[i_r1, i_r2, t1, t2]
                                        if not cov_dict['split_gauss']:
                                            ostr = ostr_format \
                                                % (obs_copy[index],  ri, rj, 
                                                1, 1, t1+1, t2+1, t1+1, t2+1, 
                                                cov, 
                                                0,
                                                0,
                                                0)
                                        else:
                                            ostr = ostr_format \
                                                % (obs_copy[index], ri, rj,
                                                1, 1, t1+1, t2+1, t1+1, t2+1, 
                                                cov, 
                                                0,
                                                0,
                                                0,
                                                0,
                                                0)
                                        olist.append(ostr)
                    else:
                        for s1 in range(len(obs[0,0,:,0,0,0])):
                            for t1 in range(len(obs[0,0,0,:,0,0])):
                                for t2 in range(len(obs[0,0,0,0,:,0])):
                                    for t3 in range(len(obs[0,0,0,0,0,:])):
                                        for i_r1 in range(len(obs[:, 0,0,0,0,0])):
                                            for i_r2 in range(len(obs[0, :, 0, 0,0,0])):
                                                ri = i_r1
                                                rj = i_r2              
                                                cov = obs[i_r1, i_r2, s1, t1, t2, t3]
                                                if not cov_dict['split_gauss']:
                                                    ostr = ostr_format \
                                                        % (obs_copy[index],  ri, rj, 
                                                        s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                        cov, 
                                                        0,
                                                        0,
                                                        0)
                                                else:
                                                    ostr = ostr_format \
                                                        % (obs_copy[index], ri, rj,
                                                        s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                        cov, 
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0)
                                                olist.append(ostr)
                    index += 1

            if not self.save_as_binary:
                if 'terminal' in self.style:
                    print("Writing result to terminal. (Brace yourself...).'")
                    for ostr in olist:
                        print(ostr)
                elif 'list' in self.style:
                    fn = self.filename[self.style.index('list')]
                    with open(fn, 'w') as file:
                        print("Writing '" + fn + "'.")
                        for ostr in olist:
                            file.write("%s\n" % ostr)
        return True

    def __write_cov_list_arbitrary(self,
                                    cov_dict,
                                    obs_dict,
                                    n_tomo_clust,
                                    n_tomo_lens,
                                    sampledim,
                                    read_in_tables,
                                    gauss,
                                    nongauss,
                                    ssc,
                                    fct_args):
        obslist, obsbool = fct_args
        proj_quant_str = 'n\tm\t'

        if not cov_dict['split_gauss']:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%i\ti\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e'
        else:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'

        olist = []
        splitidx = 0
        write_header = True

        olist = []
        splitidx = 0
        write_header = True   
        summary = read_in_tables['arb_summary'] 
        
        gg = True
        if summary['number_summary_gg'] is None:
            gg = False
        gm = True
        if summary['number_summary_gm'] is None:
            gm = False
        mm = True
        if summary['number_summary_mm'] is None:
            mm = False
        for oidx, obs in enumerate(obslist):
            if not obsbool[oidx]:
                splitidx += 3
                continue
            sampledim1 = sampledim
            sampledim2 = sampledim
            if not cov_dict['split_gauss']:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                    write_header = False
                if not isinstance(gauss[oidx], np.ndarray):
                    continue
                else:
                    len_proj_quant1 = len(gauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(gauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(nongauss[oidx], np.ndarray):
                    nongauss[oidx] = np.zeros_like(gauss[oidx])
                else:
                    len_proj_quant1 = len(nongauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(nongauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(ssc[oidx], np.ndarray):
                    ssc[oidx] = np.zeros_like(gauss[oidx])
                else:
                    len_proj_quant1 = len(ssc[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(ssc[oidx][0,:,0,0,0,0,0,0])
                for i_r1 in range(len_proj_quant1):
                    for i_r2 in range(len_proj_quant2):
                        ri = int(np.copy(i_r1))
                        rj = int(np.copy(i_r2))
                        
                        #label ri 
                        if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                            obs_copy = str(summary['gg_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_gg']:
                                obs_copy = str(summary['gg_summary_name'][1])
                                ri -= summary['arb_number_first_summary_gg']
                        if gm and obs in ['gmgm']:
                            obs_copy = str(summary['gm_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_gm']:
                                obs_copy = str(summary['gm_summary_name'][1])
                                ri -= summary['arb_number_first_summary_gm']
                        if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                            obs_copy = str(summary['mmE_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy = str(summary['mmE_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                        if mm and obs in ['gmxim', 'ximxim']:
                            obs_copy = str(summary['mmB_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy = str(summary['mmB_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']

                        #label rj
                        if gg and obs in ['gggg']:
                            if i_r2 >= summary['arb_number_first_summary_gg']:
                                obs_copy += str(summary['gg_summary_name'][1])
                                rj -= summary['arb_number_first_summary_gg']
                            else:
                                obs_copy += str(summary['gg_summary_name'][0])
                        if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                            if i_r2 >= summary['arb_number_first_summary_gm']:
                                obs_copy += str(summary['gm_summary_name'][1])
                                rj -= summary['arb_number_first_summary_gm']
                            else:
                                obs_copy += str(summary['gm_summary_name'][0])
                        if mm and obs in ['xipxip']:
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy += str(summary['mmE_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                obs_copy = str(summary['mmE_summary_name'][0])
                        if mm and obs in ['xipxim', 'ximxim']:
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy += str(summary['mmB_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                obs_copy += str(summary['mmB_summary_name'][0])

                        if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            tomo1 = gauss[oidx].shape[4]
                            if obs == 'gggg':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                            if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                sampledim1 = 1
                                sampledim2 = 1
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo1):
                                        for t4 in range(t3, tomo1):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs == 'gmgm':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                            tomo1 = gauss[oidx].shape[4]
                            tomo2 = gauss[oidx].shape[5]
                            for t1 in range(tomo1):
                                for t2 in range(tomo2):
                                    for t3 in range(tomo1):
                                        for t4 in range(tomo2):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                            tomo1 = gauss[oidx].shape[4]
                            tomo3 = gauss[oidx].shape[6]
                            tomo4 = gauss[oidx].shape[7]
                            if obs == 'gggm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                            if obs in ['mmgm', 'gmxip', 'gmxim']:
                                sampledim1 = 1
                                sampledim2 = sampledim
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo3):
                                        for t4 in range(tomo4):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs in ['ggmm', 'ggxip', 'ggxim']:
                            tomo1 = gauss[oidx].shape[4]
                            tomo2 = gauss[oidx].shape[6]
                            sampledim1 = sampledim
                            sampledim2 = 1
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo2):
                                        for t4 in range(t3, tomo2):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
            else:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                '\tcovg sn\t\tcovng\t\tcovssc')
                    write_header = False
                if not isinstance(gauss[3*oidx], np.ndarray):
                    continue
                else:
                    len_proj_quant1 = len(gauss[3*oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(gauss[3*oidx][0,:,0,0,0,0,0,0])
                if not isinstance(gauss[3*oidx + 1], np.ndarray):
                    gauss[3*oidx + 1] = np.zeros_like(gauss[3*oidx])
                if not isinstance(gauss[3*oidx + 2], np.ndarray):
                    gauss[3*oidx + 2] = np.zeros_like(gauss[3*oidx])
                if not isinstance(nongauss[oidx], np.ndarray):
                    nongauss[oidx] = np.zeros_like(gauss[3*oidx])
                else:
                    len_proj_quant1 = len(nongauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(nongauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(ssc[oidx], np.ndarray):
                    ssc[oidx] = np.zeros_like(gauss[3*oidx])
                else:
                    len_proj_quant1 = len(ssc[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(ssc[oidx][0,:,0,0,0,0,0,0])



                for i_r1 in range(len_proj_quant1):
                    for i_r2 in range(len_proj_quant2):
                        ri = int(np.copy(i_r1))
                        rj = int(np.copy(i_r2))
                        
                        #label ri 
                        if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                            obs_copy = str(summary['gg_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_gg']:
                                obs_copy = str(summary['gg_summary_name'][1])
                                ri -= summary['arb_number_first_summary_gg']
                        if gm and obs in ['gmgm']:
                            obs_copy = str(summary['gm_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_gm']:
                                obs_copy = str(summary['gm_summary_name'][1])
                                ri -= summary['arb_number_first_summary_gm']
                        if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                            obs_copy = str(summary['mmE_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy = str(summary['mmE_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                        if mm and obs in ['gmxim', 'ximxim']:
                            obs_copy = str(summary['mmB_summary_name'][0])
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy = str(summary['mmB_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                
                        #label rj
                        if gg and obs in ['gggg']:
                            if i_r2 >= summary['arb_number_first_summary_gg']:
                                obs_copy += str(summary['gg_summary_name'][1])
                                rj -= summary['arb_number_first_summary_gg']
                            else:
                                obs_copy += str(summary['gg_summary_name'][0])
                        if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                            if i_r2 >= summary['arb_number_first_summary_gm']:
                                obs_copy += str(summary['gm_summary_name'][1])
                                rj -= summary['arb_number_first_summary_gm']
                            else:
                                obs_copy += str(summary['gm_summary_name'][0])
                        if mm and obs in ['xipxip']:
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy += str(summary['mmE_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                obs_copy += str(summary['mmE_summary_name'][0])
                        if mm and obs in ['xipxim', 'ximxim']:
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                obs_copy += str(summary['mmB_summary_name'][1])
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                obs_copy += str(summary['mmB_summary_name'][0])
                        if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            tomo1 = gauss[splitidx].shape[4]
                            if obs == 'gggg':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                            if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                                sampledim1 = 1
                                sampledim2 = 1
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo1):
                                        for t4 in range(t3, tomo1):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs == 'gmgm':
                            sampledim1 = sampledim
                            sampledim2 = sampledim
                            tomo1 = gauss[splitidx].shape[4]
                            tomo2 = gauss[splitidx].shape[5]
                            for t1 in range(tomo1):
                                for t2 in range(tomo2):
                                    for t3 in range(tomo1):
                                        for t4 in range(tomo2):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                            tomo1 = gauss[splitidx].shape[4]
                            tomo3 = gauss[splitidx].shape[6]
                            tomo4 = gauss[splitidx].shape[7]
                            if obs == 'gggm':
                                sampledim1 = sampledim
                                sampledim2 = sampledim
                            if obs in ['mmgm', 'gmxip', 'gmxim']:
                                sampledim1 = 1
                                sampledim2 = sampledim
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo3):
                                        for t4 in range(tomo4):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                        elif obs in ['ggmm', 'ggxip', 'ggxim']:
                            tomo1 = gauss[splitidx].shape[4]
                            tomo2 = gauss[splitidx].shape[6]
                            sampledim1 = sampledim
                            sampledim2 = 1
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo2):
                                        for t4 in range(t3, tomo2):
                                            for i_s1 in range(sampledim1):
                                                for i_s2 in range(sampledim2):
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t1, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                splitidx += 3
        
        index = 0
        if self.has_csmf:
            obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmmE", "csmfmmB"]
            for obs in self.conditional_stellar_mass_function_cov:
                csmf_auto = False
                if index == 0:
                    csmf_auto = True
                if not isinstance(obs, np.ndarray):
                    continue
                if csmf_auto:
                    for i_r1 in range(len(obs[:, 0,0,0])):
                        for i_r2 in range(len(obs[0, :, 0, 0])):
                            for t1 in range(len(obs[0,0,:,0])):
                                for t2 in range(len(obs[0,0,0,:])):
                                    ri = i_r1
                                    rj = i_r2              
                                    cov = obs[i_r1, i_r2, t1, t2]
                                    if not cov_dict['split_gauss']:
                                        ostr = ostr_format \
                                            % (obs_copy[index],  ri, rj, 
                                            1, 1, t1+1, t2+1, t1+1, t2+1, 
                                            cov, 
                                            0,
                                            0,
                                            0)
                                    else:
                                        ostr = ostr_format \
                                            % (obs_copy[index], ri, rj,
                                            1, 1, t1+1, t2+1, t1+1, t2+1, 
                                            cov, 
                                            0,
                                            0,
                                            0,
                                            0,
                                            0)
                                    olist.append(ostr)
                else:
                    for i_r1 in range(len(obs[:, 0,0,0,0,0])):
                        for i_r2 in range(len(obs[0, :, 0, 0,0,0])):
                            for s1 in range(len(obs[0,0,:,0,0,0])):
                                for t1 in range(len(obs[0,0,0,:,0,0])):
                                    for t2 in range(len(obs[0,0,0,0,:,0])):
                                        for t3 in range(len(obs[0,0,0,0,0,:])):
                                            ri = i_r1
                                            rj = i_r2              
                                            cov = obs[i_r1, i_r2, s1, t1, t2, t3]
                                            if not cov_dict['split_gauss']:
                                                ostr = ostr_format \
                                                    % (obs_copy[index],  ri, rj, 
                                                    s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                    cov, 
                                                    0,
                                                    0,
                                                    0)
                                            else:
                                                ostr = ostr_format \
                                                    % (obs_copy[index], ri, rj,
                                                    s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1, 
                                                    cov, 
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0)
                                            olist.append(ostr)
                index += 1
        if not self.save_as_binary:
            if 'terminal' in self.style:
                print("Writing result to terminal. (Brace yourself...).'")
                for ostr in olist:
                    print(ostr)
            elif 'list' in self.style:
                fn = self.filename[self.style.index('list')]
                with open(fn, 'w') as file:
                    print("Writing '" + fn + "'.")
                    for ostr in olist:
                        file.write("%s\n" % ostr)
        return True
    
    def __write_cov_list_arbitrary_cosmosis_style(self,
                                                    cov_dict,
                                                    obs_dict,
                                                    n_tomo_clust,
                                                    n_tomo_lens,
                                                    sampledim,
                                                    read_in_tables,
                                                    gauss,
                                                    nongauss,
                                                    ssc,
                                                    fct_args):
        obslist, obsbool = fct_args
        proj_quant_str = 'n\tm\t'

        if not cov_dict['split_gauss']:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%i\ti\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e'
        else:
            if n_tomo_clust is None and n_tomo_lens is None:
                tomo_str = ''
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'
            else:
                tomo_str = 'tomoi\ttomoj\ttomok\ttomol\t'
                ostr_format = '%s\t%i\t%i\t%i\t%i\t%i\t\t%i\t\t%i\t\t%i\t\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e'

        olist = []
        splitidx = 0
        write_header = True

        olist = []
        splitidx = 0
        write_header = True   
        summary = read_in_tables['arb_summary'] 
        
        gg = True
        if summary['number_summary_gg'] is None:
            gg = False
        gm = True
        if summary['number_summary_gm'] is None:
            gm = False
        mm = True
        if summary['number_summary_mm'] is None:
            mm = False
        for oidx, obs in enumerate(obslist):
            if not obsbool[oidx]:
                splitidx += 3
                continue
            sampledim1 = sampledim
            sampledim2 = sampledim
            if not cov_dict['split_gauss']:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                    write_header = False
                if not isinstance(gauss[oidx], np.ndarray):
                    continue
                else:
                    len_proj_quant1 = len(gauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(gauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(nongauss[oidx], np.ndarray):
                    nongauss[oidx] = np.zeros_like(gauss[oidx])
                else:
                    len_proj_quant1 = len(nongauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(nongauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(ssc[oidx], np.ndarray):
                    ssc[oidx] = np.zeros_like(gauss[oidx])
                else:
                    len_proj_quant1 = len(ssc[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(ssc[oidx][0,:,0,0,0,0,0,0])
                

                if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                    tomo1 = gauss[oidx].shape[4]
                    if obs == 'gggg':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                    if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                        sampledim1 = 1
                        sampledim2 = 1
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo1):
                                for t4 in range(t3, tomo1):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']

                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs == 'gmgm':
                    sampledim1 = sampledim
                    sampledim2 = sampledim
                    tomo1 = gauss[oidx].shape[4]
                    tomo2 = gauss[oidx].shape[5]
                    for t1 in range(tomo1):
                        for t2 in range(tomo2):
                            for t3 in range(tomo1):
                                for t4 in range(tomo2):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']

                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                    tomo1 = gauss[oidx].shape[4]
                    tomo3 = gauss[oidx].shape[6]
                    tomo4 = gauss[oidx].shape[7]
                    if obs == 'gggm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                    if obs in ['mmgm', 'gmxip', 'gmxim']:
                        sampledim1 = 1
                        sampledim2 = sampledim
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo3):
                                for t4 in range(tomo4):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']

                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs in ['ggmm', 'ggxip', 'ggxim']:
                    tomo1 = gauss[oidx].shape[4]
                    tomo2 = gauss[oidx].shape[6]
                    sampledim1 = sampledim
                    sampledim2 = 1
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo2):
                                for t4 in range(t3, tomo2):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']

                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[oidx][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
            else:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                tomo_str + 'cov\t\t\tcovg sva\tcovg mix' +
                                '\tcovg sn\t\tcovng\t\tcovssc')
                    write_header = False
                if not isinstance(gauss[3*oidx], np.ndarray):
                    continue
                else:
                    len_proj_quant1 = len(gauss[3*oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(gauss[3*oidx][0,:,0,0,0,0,0,0])
                if not isinstance(gauss[3*oidx + 1], np.ndarray):
                    gauss[3*oidx + 1] = np.zeros_like(gauss[3*oidx])
                if not isinstance(gauss[3*oidx + 2], np.ndarray):
                    gauss[3*oidx + 2] = np.zeros_like(gauss[3*oidx])
                if not isinstance(nongauss[oidx], np.ndarray):
                    nongauss[oidx] = np.zeros_like(gauss[3*oidx])
                else:
                    len_proj_quant1 = len(nongauss[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(nongauss[oidx][0,:,0,0,0,0,0,0])
                if not isinstance(ssc[oidx], np.ndarray):
                    ssc[oidx] = np.zeros_like(gauss[3*oidx])
                else:
                    len_proj_quant1 = len(ssc[oidx][:,0,0,0,0,0,0,0])
                    len_proj_quant2 = len(ssc[oidx][0,:,0,0,0,0,0,0])

                if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                    tomo1 = gauss[splitidx].shape[4]
                    if obs == 'gggg':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                    if obs in  ['mmmm', 'xipxip', 'xipxim', 'ximxim']:
                        sampledim1 = 1
                        sampledim2 = 1
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo1):
                                for t4 in range(t3, tomo1):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs == 'gmgm':
                    sampledim1 = sampledim
                    sampledim2 = sampledim
                    tomo1 = gauss[splitidx].shape[4]
                    tomo2 = gauss[splitidx].shape[5]
                    for t1 in range(tomo1):
                        for t2 in range(tomo2):
                            for t3 in range(tomo1):
                                for t4 in range(tomo2):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                    tomo1 = gauss[splitidx].shape[4]
                    tomo3 = gauss[splitidx].shape[6]
                    tomo4 = gauss[splitidx].shape[7]
                    if obs == 'gggm':
                        sampledim1 = sampledim
                        sampledim2 = sampledim
                    if obs in ['mmgm', 'gmxip', 'gmxim']:
                        sampledim1 = 1
                        sampledim2 = sampledim
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo3):
                                for t4 in range(tomo4):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                elif obs in ['ggmm', 'ggxip', 'ggxim']:
                    tomo1 = gauss[splitidx].shape[4]
                    tomo2 = gauss[splitidx].shape[6]
                    sampledim1 = sampledim
                    sampledim2 = 1
                    for t1 in range(tomo1):
                        for t2 in range(t1, tomo1):
                            for t3 in range(tomo2):
                                for t4 in range(t3, tomo2):
                                    for i_s1 in range(sampledim1):
                                        for i_s2 in range(sampledim2):
                                            for i_r1 in range(len_proj_quant1):
                                                for i_r2 in range(len_proj_quant2):
                                                    ri = int(np.copy(i_r1))
                                                    rj = int(np.copy(i_r2))
                                                    
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        obs_copy = str(summary['gg_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy = str(summary['gg_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        obs_copy = str(summary['gm_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy = str(summary['gm_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        obs_copy = str(summary['mmE_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        obs_copy = str(summary['mmB_summary_name'][0])
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy = str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            obs_copy += str(summary['gg_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            obs_copy += str(summary['gg_summary_name'][0])
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            obs_copy += str(summary['gm_summary_name'][1])
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            obs_copy += str(summary['gm_summary_name'][0])
                                                    if mm and obs in ['xipxip']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmE_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmE_summary_name'][0])
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            obs_copy += str(summary['mmB_summary_name'][1])
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            obs_copy += str(summary['mmB_summary_name'][0])
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t1, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  ri, rj, 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                splitidx += 3
        index = 0
        if self.has_csmf:
            obs_copy = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmmE", "csmfmmB"]
            for obs in self.conditional_stellar_mass_function_cov:
                csmf_auto = False
                if index == 0:
                    csmf_auto = True
                if not isinstance(obs, np.ndarray):
                    continue
                if csmf_auto:
                    for t1 in range(len(obs[0,0,:,0])):
                        for t2 in range(len(obs[0,0,0,:])):
                            for i_r1 in range(len(obs[:, 0,0,0])):
                                for i_r2 in range(len(obs[0, :, 0, 0])):
                                    ri = i_r1
                                    rj = i_r2
                                    cov = obs[i_r1, i_r2, t1, t2]
                                    if not cov_dict['split_gauss']:
                                        ostr = ostr_format \
                                            % (obs_copy[index],  ri, rj,
                                            1, 1, t1+1, t2+1, t1+1, t2+1,
                                            cov,
                                            0,
                                            0,
                                            0)
                                    else:
                                        ostr = ostr_format \
                                            % (obs_copy[index], ri, rj,
                                            1, 1, t1+1, t2+1, t1+1, t2+1,
                                            cov,
                                            0,
                                            0,
                                            0,
                                            0,
                                            0)
                                    olist.append(ostr)
                else:
                    for s1 in range(len(obs[0,0,:,0,0,0])):
                        for t1 in range(len(obs[0,0,0,:,0,0])):
                            for t2 in range(len(obs[0,0,0,0,:,0])):
                                for t3 in range(len(obs[0,0,0,0,0,:])):
                                    for i_r1 in range(len(obs[:, 0,0,0,0,0])):
                                        for i_r2 in range(len(obs[0, :, 0, 0,0,0])):
                                            ri = i_r1
                                            rj = i_r2
                                            cov = obs[i_r1, i_r2, s1, t1, t2, t3]
                                            if not cov_dict['split_gauss']:
                                                ostr = ostr_format \
                                                    % (obs_copy[index],  ri, rj,
                                                    s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1,
                                                    cov,
                                                    0,
                                                    0,
                                                    0)
                                            else:
                                                ostr = ostr_format \
                                                    % (obs_copy[index], ri, rj,
                                                    s1 + 1, s1 + 1, t1+1, t1+1, t2+1, t3+1,
                                                    cov,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0)
                                            olist.append(ostr)
                index += 1
        
        if not self.save_as_binary:
            if 'terminal' in self.style:
                print("Writing result to terminal. (Brace yourself...).'")
                for ostr in olist:
                    print(ostr)
            elif 'list' in self.style:
                fn = self.filename[self.style.index('list')]
                with open(fn, 'w') as file:
                    print("Writing '" + fn + "'.")
                    for ostr in olist:
                        file.write("%s\n" % ostr)
        return True

    def __create_matrix_csmf(self,covlist, want_diagonal_csmf = False):
        number_m_bins = int(len(covlist[:,0,0,0]))
        number_tomo_bins = int(len(covlist[0,0,:,0]))
        if want_diagonal_csmf:
            data_size = number_m_bins
            covariance = np.zeros((data_size,data_size))
            for i_t in range(number_m_bins):
                for j_t in range(number_m_bins):
                    covariance[i_t,j_t] = covlist[i_t, j_t, i_t, j_t]
        else:
            data_size = number_m_bins * number_tomo_bins
            covariance = np.zeros((data_size,data_size))
            i = 0
            for i_t in range(number_tomo_bins):
                for i_m in range(number_m_bins):
                    j = 0
                    for j_t in range(number_tomo_bins):
                        for j_m in range(number_m_bins):
                            covariance[i,j] = covlist[i_m, j_m, i_t, j_t]
                            j += 1
                    i += 1
        return covariance
    
    def __create_matrix_csmf_cross_LSS(self,covlist,is_i_smaller_j ,want_diagonal_csmf = False):
        number_m_bins = int(len(covlist[0,:,0,0,0,0]))
        number_tomo_bins = int(len(covlist[0,0,0,:,0,0]))
        if is_i_smaller_j:
            data_size_ij = int(len(covlist[0,0,0,0,:,0])*(len(covlist[0,0,0,0,0,:]) + 1)/2*len(covlist[0,0,:,0,0,0])*len(covlist[:,0,0,0,0,0]))
            if want_diagonal_csmf:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for j_tomo in range(i_tomo,len(covlist[0,0,0,0,:, 0])):
                        for i_sample in range(len(covlist[0,0, :, 0,0,0])):
                            for i_theta in range(len(covlist[:,0,0,0,0,0])):
                                j = 0
                                for i_mass_tomo in range(number_tomo_bins):
                                    covariance[i,j] = covlist[i_theta, i_mass_tomo, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                    j += 1
                                i += 1
            else:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0])*len(covlist[0,0,0,:,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for j_tomo in range(i_tomo,len(covlist[0,0,0,0,:, 0])):
                        for i_sample in range(len(covlist[0,0, :, 0,0,0])):
                            for i_theta in range(len(covlist[:,0,0,0,0,0])):
                                j = 0
                                for i_mass_tomo in range(number_tomo_bins):
                                    for i_mass in range(number_m_bins):    
                                        covariance[i,j] = covlist[i_theta, i_mass, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                        j += 1
                                i += 1
        else:
            data_size_ij = int(len(covlist[0,0,0,0,:,0])*len(covlist[0,0,0,0,0,:])*len(covlist[0,0,:,0,0,0])*len(covlist[:,0,0,0,0,0]))
            if want_diagonal_csmf:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for j_tomo in range(len(covlist[0,0,0,0,:, 0])):
                        for i_sample in range(len(covlist[0,0, :, 0,0,0])):
                            for i_theta in range(len(covlist[:,0,0,0,0,0])):
                                j = 0
                                for i_mass_tomo in range(number_tomo_bins):
                                    covariance[i,j] = covlist[i_theta, i_mass_tomo, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                    j += 1
                                i += 1
            else:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0])*len(covlist[0,0,0,:,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for j_tomo in range(len(covlist[0,0,0,0,:, 0])):
                        for i_sample in range(len(covlist[0,0, :, 0,0,0])):
                            for i_theta in range(len(covlist[:,0,0,0,0,0])):
                                j = 0
                                for i_mass_tomo in range(number_tomo_bins):
                                    for i_mass in range(number_m_bins):    
                                        covariance[i,j] = covlist[i_theta, i_mass, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                        j += 1
                                i += 1
        return covariance
    
    def __create_matrix(self,covlist, is_i_smaller_j, is_m_smaller_n):
        if is_i_smaller_j and is_m_smaller_n:
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2)*len(covlist[:,0,0,0,0,0,0,0])*len(covlist[0,0,:,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])*len(covlist[0,0,0,:,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            i = 0
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_sample in range(len(covlist[0 ,0, :, 0,0,0,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for n_tomo in range(m_tomo, len(covlist[0,0,0,0,0,0,0,:])):
                                    for j_sample in range(len(covlist[0,0,0,:,0,0,0,0])):
                                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                            covariance[i,j] = covlist[i_theta,j_theta,i_sample,j_sample,i_tomo,j_tomo,m_tomo,n_tomo]
                                            j += 1
                            i += 1
        if is_i_smaller_j and  not is_m_smaller_n:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2)*len(covlist[:,0,0,0,0,0,0,0])*len(covlist[0,0,:,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[0, :,0,0,0,0,0,0])*len(covlist[0,0,0,:,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_sample in range(len(covlist[0,0, :, 0,0,0,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for n_tomo in range(len(covlist[0,0,0,0,0,0,0,:])):
                                    for j_sample in range(len(covlist[0,0,0,:,0,0,0,0])):
                                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                            covariance[i,j] = covlist[i_theta,j_theta,i_sample,j_sample,i_tomo,j_tomo,m_tomo,n_tomo] 
                                            j += 1
                            i += 1
        if not is_i_smaller_j and is_m_smaller_n:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0])))*len(covlist[:,0,0,0,0,0,0,0])*len(covlist[0,0,:,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])*len(covlist[0,0,:,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(len(covlist[0,0,0,0,0,:,0,0])):
                    for i_sample in range(len(covlist[0 ,0, :, 0,0,0,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for n_tomo in range(m_tomo, len(covlist[0,0,0,0,0,0,0,:])):
                                    for j_sample in range(len(covlist[0,0,0,:,0,0,0,0])):
                                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                            covariance[i,j] = covlist[i_theta,j_theta,i_sample,j_sample,i_tomo,j_tomo,m_tomo,n_tomo] 
                                            j += 1
                            i += 1
        if not is_i_smaller_j and  not is_m_smaller_n:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0])))*len(covlist[:,0,0,0,0,0,0,0])*len(covlist[0,0,:,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[0, :,0,0,0,0,0,0])*len(covlist[0,0,0, :,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))   
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(len(covlist[0,0,0,0,0,:,0,0])):
                    for i_sample in range(len(covlist[0 ,0, :, 0,0,0,0,0])):    
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for n_tomo in range(len(covlist[0,0,0,0,0,0,0,:])):
                                    for j_sample in range(len(covlist[0,0,0,:,0,0,0,0])):
                                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                            covariance[i,j] = covlist[i_theta,j_theta,i_sample,j_sample,i_tomo,j_tomo,m_tomo,n_tomo] 
                                            j += 1
                            i += 1
        return covariance
    
    def __create_matrix_diagonal(self,covlist, diagonal_1, diagonal_2, is_i_smaller_j, is_m_smaller_n):
        if diagonal_1 and diagonal_2:
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]))*len(covlist[0,:,0,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            i = 0
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                    j = 0
                    for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,i_tomo,m_tomo,m_tomo] 
                                j += 1
                    i += 1
        if diagonal_1 and not diagonal_2:
            if is_m_smaller_n:
                data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for n_tomo in range(m_tomo, len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,i_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
            else:
                data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for n_tomo in range(len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,i_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
        if diagonal_2 and not diagonal_1:
            if is_i_smaller_j:
                data_size_mn = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_ij = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for j_tomo in range(i_tomo, len(covlist[0,0,0,0,0,:,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,m_tomo] 
                                    j += 1
                        i += 1
            else:
                data_size_mn = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_ij = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for j_tomo in range(len(covlist[0,0,0,0,0,:,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,m_tomo] 
                                    j += 1
                        i += 1
        return covariance
        
    def __create_matrix_arbitrary(self,covlist, is_i_smaller_j, is_m_smaller_n, probe1, probe2, summary):
        nprobes_probe_1 = summary['number_summary_'+probe1]
        nprobes_probe_2 = summary['number_summary_'+probe2]
        nprobes_firstprobe_probe_1 = summary['arb_number_first_summary_' +probe1]
        nprobes_firstprobe_probe_2 = summary['arb_number_first_summary_' +probe2]
        if nprobes_probe_1 is not None and nprobes_probe_2 is not None:
            covp1p1 = self.__create_matrix(covlist[:nprobes_firstprobe_probe_1,:nprobes_firstprobe_probe_2], is_i_smaller_j, is_m_smaller_n)    
            result = covp1p1
            if nprobes_probe_1 > 1:
                covp2p1 = self.__create_matrix(covlist[nprobes_firstprobe_probe_1:,:nprobes_firstprobe_probe_2], is_i_smaller_j, is_m_smaller_n)   
                result = np.block([[covp1p1],
                                    [covp2p1]])
                if nprobes_probe_2 > 1:
                    covp2p2 = self.__create_matrix(covlist[nprobes_firstprobe_probe_1:,nprobes_firstprobe_probe_2:], is_i_smaller_j, is_m_smaller_n)
                    covp1p2 = self.__create_matrix(covlist[:nprobes_firstprobe_probe_1,nprobes_firstprobe_probe_2:], is_i_smaller_j, is_m_smaller_n)
                    result = np.block([[covp1p1, covp1p2],
                                    [covp2p1, covp2p2]])
                    
            if nprobes_probe_2 > 1 and nprobes_probe_1 == 1 :
                covp1p2 = self.__create_matrix(covlist[:nprobes_firstprobe_probe_1,nprobes_firstprobe_probe_2:], is_i_smaller_j, is_m_smaller_n)
                result = np.block([[covp1p1, covp1p2]])
        return result
    
    def __write_cov_matrix_new(self,
                                obs_dict,
                                cov_dict,
                                n_tomo_clust,
                                n_tomo_lens,
                                sampledim,
                                proj_quant,
                                gauss,
                                nongauss,
                                ssc,
                                fct_args):
        obslist, obsbool, obslength, mult, gg, gm, mm, xipp, xipm, ximm = \
            fct_args
        if obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0:
            if obslength == 6 and mult == 3:
                gauss = [gauss[0]+gauss[1]+gauss[2], 
                        gauss[3]+gauss[4]+gauss[5],
                        gauss[6]+gauss[7]+gauss[8], 
                        gauss[9]+gauss[10]+gauss[11],
                        gauss[12]+gauss[13]+gauss[14], 
                        gauss[15]+gauss[16]+gauss[17]]
            elif obslength == 10 and mult == 3:
                gauss = [gauss[0]+gauss[1]+gauss[2], 
                        gauss[3]+gauss[4]+gauss[5],
                        gauss[6]+gauss[7]+gauss[8], 
                        gauss[9]+gauss[10]+gauss[11],
                        gauss[12]+gauss[13]+gauss[14], 
                        gauss[15]+gauss[16]+gauss[17],
                        gauss[18]+gauss[19]+gauss[20], 
                        gauss[21]+gauss[22]+gauss[23], 
                        gauss[24]+gauss[25]+gauss[26], 
                        gauss[27]+gauss[28]+gauss[29]]

            if obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                xipm = True
                ximm = True
                xipp = True
        
            cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(obslength)]
            cov_diag = []    
            if self.has_csmf:
                covariange_csmf = self.__create_matrix_csmf(self.conditional_stellar_mass_function_cov[0], obs_dict['observables']['csmf_diagonal'])
                covariange_csmfgg = None
                covariange_csmfgm = None
                covariange_csmfmmE = None
                covariange_csmfmmB = None
                if gg:
                    covariange_csmfgg = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[1], True, obs_dict['observables']['csmf_diagonal'])
                if gm:
                    covariange_csmfgm = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[2], False, obs_dict['observables']['csmf_diagonal'])
                if mm:
                    covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[3], True, obs_dict['observables']['csmf_diagonal'])
                if obslength == 10:
                    covariange_csmfmmB = np.zeros_like(covariange_csmfmmE)
            if obslength == 6:
                # 'gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm'
                if self.has_csmf:
                    if covariange_csmfgg is not None:
                        csmf_block = np.block([[covariange_csmfgg]])
                        if covariange_csmfgm is not None:
                            csmf_block = np.block([[covariange_csmfgg],[covariange_csmfgm]])
                            if covariange_csmfmmE is not None:
                                csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm],[ covariange_csmfmmE]])
                    elif covariange_csmfgm is not None:
                        csmf_block = np.block([[covariange_csmfgm]])
                        if covariange_csmfmmE is not None:
                            csmf_block = np.block([[covariange_csmfgm],[ covariange_csmfmmE]])
                    elif covariange_csmfmmE is not None:
                        csmf_block = np.block([[covariange_csmfmmE]])
                if gg:
                    covariance_gggg = self.__create_matrix(cov[0],True,True)

                    cov2d = covariance_gggg
                    cov_diag.append(covariance_gggg)
                    if gm:
                        covariance_gmgm = self.__create_matrix(cov[3],False,False)
                        cov_diag.append(covariance_gmgm)
                        covariance_gggm = self.__create_matrix(cov[1],True,False)
                        cov2d = np.block([[covariance_gggg, covariance_gggm],
                                        [covariance_gggm.T, covariance_gmgm]])
                        if mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_ggmm = self.__create_matrix(cov[2],True,True)
                            covariance_mmgm = self.__create_matrix(cov[4],True,False)
                            cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                            [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                            [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                    elif mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True)
                        cov_diag.append(covariance_mmmm)
                        covariance_ggmm = self.__create_matrix(cov[2],True,True)
                        cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                        [covariance_ggmm.T, covariance_mmmm]])
                elif gm:
                    covariance_gmgm = self.__create_matrix(cov[3],False,False)
                    cov_diag.append(covariance_gmgm)
                    cov2d = covariance_gmgm
                    if mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True)
                        cov_diag.append(covariance_mmmm)
                        covariance_mmgm = self.__create_matrix(cov[4],True,False)
                        cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                        [covariance_mmgm, covariance_mmmm]])
                elif mm:
                    covariance_mmmm = self.__create_matrix(cov[5],True,True)
                    cov_diag.append(covariance_mmmm)
                    cov2d = covariance_mmmm                    
            elif obslength == 10:        
                # 'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 
                # 'ximgt', 'xipxip', 'xipxim', 'ximxim'
                if self.has_csmf:
                    if covariange_csmfgg is not None:
                        csmf_block = np.block([[covariange_csmfgg]])
                        if covariange_csmfgm is not None:
                            csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm]])
                            if covariange_csmfmmE is not None:
                                csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm],[ covariange_csmfmmE]])
                                if covariange_csmfmmB is not None:
                                    csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm],[ covariange_csmfmmE],[ covariange_csmfmmB]])
                    elif covariange_csmfgm is not None:
                        csmf_block = np.block([[covariange_csmfgm]])
                        if covariange_csmfmmE is not None:
                            csmf_block = np.block([[covariange_csmfgm],[ covariange_csmfmmE]])
                            if covariange_csmfmmB is not None:
                                csmf_block = np.block([[covariange_csmfgm],[ covariange_csmfmmE],[ covariange_csmfmmB]])

                    elif covariange_csmfmmE is not None:
                        csmf_block = np.block([[covariange_csmfmmE]])
                        if covariange_csmfmmB is not None:
                            csmf_block = np.block([[covariange_csmfmmE],[ covariange_csmfmmB]])
                if gg:
                    covariance_ww = self.__create_matrix(cov[0],True,True)
                    cov2d = covariance_ww
                    cov_diag.append(covariance_ww)
                    if gm:
                        covariance_gtgt = self.__create_matrix(cov[4],False,False)
                        cov_diag.append(covariance_gtgt)
                        covariance_wgt = self.__create_matrix(cov[1],True,False)
                        cov2d = np.block([[covariance_ww, covariance_wgt],
                                        [covariance_wgt.T, covariance_gtgt]])
                        if xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True)
                            covariance_wxip = self.__create_matrix(cov[2],True,True)
                            covariance_xipgt = self.__create_matrix(cov[5],True,False)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                cov_diag.append(covariance_ximxim)
                                covariance_wxim = self.__create_matrix(cov[3],True,True)
                                covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        
                    elif xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True)
                        covariance_wxip = self.__create_matrix(cov[2],True,True)
                        cov2d = np.block([[covariance_ww, covariance_wxip],
                                        [covariance_wxip.T, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True)
                            covariance_wxim = self.__create_matrix(cov[3],True,True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                elif gm:
                    covariance_gtgt = self.__create_matrix(cov[4],False,False)
                    cov2d = covariance_gtgt
                    cov_diag.append(covariance_gtgt)        
                    if xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True)
                        covariance_xipgt = self.__create_matrix(cov[5],True,False)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                        [covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True)
                            covariance_ximgt = self.__create_matrix(cov[6],True,False)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                elif xipp:
                    covariance_xipxip = self.__create_matrix(cov[7],True,True)
                    cov2d = covariance_xipxip
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix(cov[9],True,True)
                        cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        cov_diag.append(covariance_ximxim)
                        if xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True)
                            cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                            [covariance_xipxim.T, covariance_ximxim]])
                elif ximm:
                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                    cov2d = covariance_xipxip
                    cov_diag.append(covariance_ximxim)
                elif xipm:
                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                    cov2d = covariance_xipxim
            
            if self.has_csmf:
                cov_diag.append(covariange_csmf)
                cov2d = np.block([[cov2d, csmf_block],
                                [csmf_block.T, covariange_csmf]])
            cov2d_total = np.copy(cov2d)
            if cov_dict['split_gauss']:
                cov = [gauss[idx] for idx in range(obslength)]
                cov_diag = []    
                if obslength == 6:
                    # 'gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm'
                    if gg:
                        covariance_gggg = self.__create_matrix(cov[0],True,True)
                        cov2d = covariance_gggg
                        cov_diag.append(covariance_gggg)
                        if gm:
                            covariance_gmgm = self.__create_matrix(cov[3],False,False)
                            cov_diag.append(covariance_gmgm)
                            covariance_gggm = self.__create_matrix(cov[1],True,False)
                            cov2d = np.block([[covariance_gggg, covariance_gggm],
                                            [covariance_gggm.T, covariance_gmgm]])
                            if mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False)
                                cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_ggmm = self.__create_matrix(cov[2],True,True)
                            cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                            [covariance_ggmm.T, covariance_mmmm]])
                    elif gm:
                        covariance_gmgm = self.__create_matrix(cov[3],False,False)
                        cov_diag.append(covariance_gmgm)
                        cov2d = covariance_gmgm
                        if mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_mmgm = self.__create_matrix(cov[4],True,False)
                            cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                            [covariance_mmgm, covariance_mmmm]])
                    elif mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True)
                        cov_diag.append(covariance_mmmm)
                        cov2d = covariance_mmmm
                elif obslength == 10:        
                    # 'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 
                    # 'ximgt', 'xipxip', 'xipxim', 'ximxim'
                    if gg:
                        covariance_ww = self.__create_matrix(cov[0],True,True)
                        cov2d = covariance_ww
                        cov_diag.append(covariance_ww)
                        if gm:
                            covariance_gtgt = self.__create_matrix(cov[4],False,False)
                            cov_diag.append(covariance_gtgt)
                            covariance_wgt = self.__create_matrix(cov[1],True,False)
                            cov2d = np.block([[covariance_ww, covariance_wgt],
                                            [covariance_wgt.T, covariance_gtgt]])
                            if xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                    cov_diag.append(covariance_ximxim)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                            
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True)
                            covariance_wxip = self.__create_matrix(cov[2],True,True)
                            cov2d = np.block([[covariance_ww, covariance_wxip],
                                            [covariance_wxip.T, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                covariance_wxim = self.__create_matrix(cov[3],True,True)
                                cov_diag.append(covariance_ximxim)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                    elif gm:
                        covariance_gtgt = self.__create_matrix(cov[4],False,False)
                        cov2d = covariance_gtgt
                        cov_diag.append(covariance_gtgt)        
                        if xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True)
                            covariance_xipgt = self.__create_matrix(cov[5],True,False)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                            [covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                cov_diag.append(covariance_ximxim)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    elif xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True)
                        cov2d = covariance_xipxip
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True)
                            cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            cov_diag.append(covariance_ximxim)
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                [covariance_xipxim.T, covariance_ximxim]])
                    elif ximm:
                        covariance_ximxim = self.__create_matrix(cov[9],True,True)
                        cov2d = covariance_xipxip
                        cov_diag.append(covariance_ximxim)
                    elif xipm:
                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                        cov2d = covariance_xipxim
                if self.has_csmf:
                    cov_diag.append(covariange_csmf)
                    cov2d = np.block([[cov2d, csmf_block],
                                      [csmf_block.T, covariange_csmf]])

                cov2d_gauss = np.copy(cov2d)
                if self.has_nongauss:
                    cov = [nongauss[idx] for idx in range(obslength)]
                    cov_diag = []    
                    if obslength == 6:
                        # 'gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm'
                        if gg:
                            covariance_gggg = self.__create_matrix(cov[0],True,True)
                            cov2d = covariance_gggg
                            cov_diag.append(covariance_gggg)
                            if gm:
                                covariance_gmgm = self.__create_matrix(cov[3],False,False)
                                cov_diag.append(covariance_gmgm)
                                covariance_gggm = self.__create_matrix(cov[1],True,False)
                                cov2d = np.block([[covariance_gggg, covariance_gggm],
                                                [covariance_gggm.T, covariance_gmgm]])
                                if mm:
                                    covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                    cov_diag.append(covariance_mmmm)
                                    covariance_ggmm = self.__create_matrix(cov[2],True,True)
                                    covariance_mmgm = self.__create_matrix(cov[4],True,False)
                                    cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                    [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                    [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                            elif mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True)
                                cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                                [covariance_ggmm.T, covariance_mmmm]])
                        elif gm:
                            covariance_gmgm = self.__create_matrix(cov[3],False,False)
                            cov_diag.append(covariance_gmgm)
                            cov2d = covariance_gmgm
                            if mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False)
                                cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                                [covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True)
                            cov_diag.append(covariance_mmmm)
                            cov2d = covariance_mmmm
                    elif obslength == 10:        
                        # 'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 
                        # 'ximgt', 'xipxip', 'xipxim', 'ximxim'
                        if gg:
                            covariance_ww = self.__create_matrix(cov[0],True,True)
                            cov2d = covariance_ww
                            cov_diag.append(covariance_ww)
                            if gm:
                                covariance_gtgt = self.__create_matrix(cov[4],False,False)
                                cov_diag.append(covariance_gtgt)
                                covariance_wgt = self.__create_matrix(cov[1],True,False)
                                cov2d = np.block([[covariance_ww, covariance_wgt],
                                                [covariance_wgt.T, covariance_gtgt]])
                                if xipp:
                                    covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                    covariance_wxip = self.__create_matrix(cov[2],True,True)
                                    covariance_xipgt = self.__create_matrix(cov[5],True,False)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                    cov_diag.append(covariance_xipxip)
                                    if ximm:
                                        covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                        cov_diag.append(covariance_ximxim)
                                        covariance_wxim = self.__create_matrix(cov[3],True,True)
                                        covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                        if xipm:
                                            covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                                
                            elif xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True)
                                cov2d = np.block([[covariance_ww, covariance_wxip],
                                                [covariance_wxip.T, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                        elif gm:
                            covariance_gtgt = self.__create_matrix(cov[4],False,False)
                            cov2d = covariance_gtgt
                            cov_diag.append(covariance_gtgt)        
                            if xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                                [covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                cov_diag.append(covariance_ximxim)
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                    [covariance_xipxim.T, covariance_ximxim]])
                        elif ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_ximxim)
                        elif xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True)
                            cov2d = covariance_xipxim

                    cov2d_nongauss = np.copy(cov2d)

                if self.has_ssc:
                    cov = [ssc[idx] for idx in range(obslength)]
                    cov_diag = []    
                    if obslength == 6:
                        # 'gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm'
                        if gg:
                            covariance_gggg = self.__create_matrix(cov[0],True,True)
                            cov2d = covariance_gggg
                            cov_diag.append(covariance_gggg)
                            if gm:
                                covariance_gmgm = self.__create_matrix(cov[3],False,False)
                                cov_diag.append(covariance_gmgm)
                                covariance_gggm = self.__create_matrix(cov[1],True,False)
                                cov2d = np.block([[covariance_gggg, covariance_gggm],
                                                [covariance_gggm.T, covariance_gmgm]])
                                if mm:
                                    covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                    cov_diag.append(covariance_mmmm)
                                    covariance_ggmm = self.__create_matrix(cov[2],True,True)
                                    covariance_mmgm = self.__create_matrix(cov[4],True,False)
                                    cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                    [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                    [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                            elif mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True)
                                cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                                [covariance_ggmm.T, covariance_mmmm]])
                        elif gm:
                            covariance_gmgm = self.__create_matrix(cov[3],False,False)
                            cov_diag.append(covariance_gmgm)
                            cov2d = covariance_gmgm
                            if mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False)
                                cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                                [covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True)
                            cov_diag.append(covariance_mmmm)
                            cov2d = covariance_mmmm
                    elif obslength == 10:        
                        # 'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 
                        # 'ximgt', 'xipxip', 'xipxim', 'ximxim'
                        if gg:
                            covariance_ww = self.__create_matrix(cov[0],True,True)
                            cov2d = covariance_ww
                            cov_diag.append(covariance_ww)
                            if gm:
                                covariance_gtgt = self.__create_matrix(cov[4],False,False)
                                cov_diag.append(covariance_gtgt)
                                covariance_wgt = self.__create_matrix(cov[1],True,False)
                                cov2d = np.block([[covariance_ww, covariance_wgt],
                                                [covariance_wgt.T, covariance_gtgt]])
                                if xipp:
                                    covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                    covariance_wxip = self.__create_matrix(cov[2],True,True)
                                    covariance_xipgt = self.__create_matrix(cov[5],True,False)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                    cov_diag.append(covariance_xipxip)
                                    if ximm:
                                        covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                        cov_diag.append(covariance_ximxim)
                                        covariance_wxim = self.__create_matrix(cov[3],True,True)
                                        covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                        if xipm:
                                            covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                                
                            elif xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True)
                                cov2d = np.block([[covariance_ww, covariance_wxip],
                                                [covariance_wxip.T, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                        elif gm:
                            covariance_gtgt = self.__create_matrix(cov[4],False,False)
                            cov2d = covariance_gtgt
                            cov_diag.append(covariance_gtgt)        
                            if xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                                [covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True)
                                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                cov_diag.append(covariance_ximxim)
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True)
                                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                    [covariance_xipxim.T, covariance_ximxim]])
                        elif ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_ximxim)
                        elif xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True)
                            cov2d = covariance_xipxim

                    cov2d_ssc = np.copy(cov2d)
                

            for i in range(len(cov2d[:,0])):
                for j in range(len(cov2d[:,0])):
                    cov2d_total[j,i] = cov2d_total[i,j]
                    if cov_dict['split_gauss']:
                        cov2d_gauss[j,i] = cov2d_gauss[i,j]
                        if self.has_nongauss:
                            cov2d_nongauss[j,i] = cov2d_nongauss[i,j]
                        if self.has_ssc:
                            cov2d_ssc[j,i] = cov2d_ssc[i,j]
            if len(np.where(np.linalg.eig(cov2d_total)[0] < 0)[0]) > 0:
                print("ALARM: The resulting covariance matrix has negative eigenvalues")
                print("Try to adjust the accuracy settings in the config file:")
                print("For configuration space covariance reduce theta_accuracy and increase integration_intervals, usually a factor of 2 is enough.")
                print("For bandpower covariance reduce bandpower_accuracy.")
                print("For COSEBI covariance reduce En_accuracy.")
            if self.plot:
                self.plot_corrcoeff_matrix(
                    obs_dict, cov2d_total, cov_diag, proj_quant, n_tomo_clust, 
                    n_tomo_lens, sampledim, self.plot ,fct_args)
            if obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                obslist[7] = 'CE_mmCE_mm'
                obslist[9] = 'CB_mmCB_mm'
            if obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['clustering'] == True:
                obslist[0] = 'CE_ggCE_gg'
            if obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obslist[4] = 'CE_gmCE_gm'
            
            hdr_str = 'Covariance matrix with the diagonals in the order: '
            hdr_str += obslist[0]+' ' if obsbool[0] else ''
            if obslength == 6:
                hdr_str += obslist[3]+' ' if obsbool[3] else ''
                hdr_str += obslist[5]+' ' if obsbool[5] else ''
            elif obslength == 10:
                hdr_str += obslist[4]+' ' if obsbool[4] else ''
                hdr_str += obslist[7]+' ' if obsbool[7] else ''
                hdr_str += obslist[9]+' ' if obsbool[9] else ''
            if self.has_csmf:
                hdr_str += " csmf "
            hdr_str += 'with '
            if n_tomo_clust is not None:
                hdr_str += str(n_tomo_clust) + ' tomographic clustering bins and '
            if n_tomo_lens is not None:
                hdr_str += str(n_tomo_lens) + ' tomographic lensing bins and '
            alternative = False
            if self.projected_clust is not None:
                hdr_str += str(len(self.projected_clust)) + ' spatial elements per tomographic clustering bin '
                alternative = True
            if self.projected_lens is not None:
                if not alternative:
                    hdr_str += str(len(self.projected_lens)) + ' spatial elements per tomographic lensing bin '
                else:
                    hdr_str += str(len(self.projected_lens)) + ' spatial elements per tomographic lensing bin and'
                alternative = True
            if not alternative:
                hdr_str += str(len(proj_quant)) + ' spatial elements per tomographic bin'
            
            if 'matrix' in self.style:
                if not cov_dict['split_gauss']:
                    print("Writing matrix output file.")
                    if self.save_as_binary:
                        fn = self.filename[self.style.index('matrix')]
                        name, extension = os.path.splitext(fn)
                        np.save(name, cov2d_total)
                    else:
                        fn = self.filename[self.style.index('matrix')]
                        np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                else:
                    print("Writing matrix output file.")
                    if self.save_as_binary:
                        fn = self.filename[self.style.index('matrix')]
                        name, extension = os.path.splitext(fn)
                        np.save(name, cov2d_total)
                        fn_gauss = name + "_gauss"
                        fn_nongauss = name + "_nongauss"
                        fn_ssc = name + "_SSC"
                        np.save(fn_gauss, cov2d_gauss)
                        if self.has_nongauss:
                            np.save(fn_nongauss, cov2d_nongauss)
                        if self.has_ssc:
                            np.save(fn_ssc, cov2d_ssc)
                    else:
                        fn = self.filename[self.style.index('matrix')]
                        np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                        name, extension = os.path.splitext(fn)
                        fn_gauss = name + "_gauss" + extension
                        fn_nongauss = name + "_nongauss" + extension
                        fn_ssc = name + "_SSC" + extension
                        np.savetxt(fn_gauss, cov2d_gauss, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                        if self.has_nongauss:
                            np.savetxt(fn_nongauss, cov2d_nongauss, fmt='%.6e', delimiter=' ',
                                    newline='\n', header=hdr_str, comments='# ')
                        if self.has_ssc:
                            np.savetxt(fn_ssc, cov2d_ssc, fmt='%.6e', delimiter=' ',
                                    newline='\n', header=hdr_str, comments='# ')
        else:
            gauss = [gauss[0]+gauss[1]+gauss[2],
                     gauss[3]+gauss[4]+gauss[5],
                     gauss[6]+gauss[7]+gauss[8], 
                     gauss[9]+gauss[10]+gauss[11],
                     gauss[12]+gauss[13]+gauss[14], 
                     gauss[15]+gauss[16]+gauss[17],
                     gauss[18]+gauss[19]+gauss[20], 
                     gauss[21]+gauss[22]+gauss[23], 
                     gauss[24]+gauss[25]+gauss[26], 
                     gauss[27]+gauss[28]+gauss[29],
                     gauss[30]+gauss[31]+gauss[32],
                     gauss[33]+gauss[34]+gauss[35],
                     gauss[36]+gauss[37]+gauss[38],
                     gauss[39]+gauss[40]+gauss[41],
                     gauss[42]+gauss[43]+gauss[44],
                     gauss[45]+gauss[46]+gauss[47],
                     gauss[48]+gauss[49]+gauss[50],
                     gauss[51]+gauss[52]+gauss[53],
                     gauss[54]+gauss[55]+gauss[56],
                     gauss[57]+gauss[58]+gauss[59],
                     gauss[60]+gauss[61]+gauss[62],
                     gauss[63]+gauss[64]+gauss[65]]
            """
            gggg_ssss_new, gggg_sssp_new, gggg_sspp_new, \
                gggg_spsp_new, gggg_ppsp_new, gggg_pppp_new, \
                gggm_sssm_new, gggm_sspm_new, gggm_spsm_new, \
                gggm_sppm_new, gggm_ppsm_new, gggm_pppm_new, \
                ggmm_ssmm_new, ggmm_spmm_new, ggmm_ppmm_new, \
                gmgm_smsm_new, gmgm_smpm_new, gmgm_pmsm_new, \
                gmgm_pmpm_new, mmgm_mmsm_new, mmgm_mmpm_new, \
                mmmm_mmmm_new
            """        
            if self.has_nongauss and self.has_ssc:
                cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(len(gauss))]
            if self.has_nongauss and not self.has_ssc:
                cov = [gauss[idx]+nongauss[idx] for idx in range(len(gauss))]
            if not self.has_nongauss and self.has_ssc:
                cov = [gauss[idx]+ssc[idx] for idx in range(len(gauss))]
            if not self.has_nongauss and not self.has_ssc:
                cov = [gauss[idx] for idx in range(len(gauss))]
            

            cov_diag = []
            if gg:
                covariance_gggg_ssss = self.__create_matrix_diagonal(cov[0], True, True, True, True)
                covariance_gggg_sssp = self.__create_matrix_diagonal(cov[1], True, False, True, False)
                covariance_gggg_sspp = self.__create_matrix_diagonal(cov[2], True, False, True, True)
                covariance_gggg_spsp = self.__create_matrix(cov[3],False, False)
                covariance_gggg_ppsp = self.__create_matrix(cov[4],True, False)
                covariance_gggg_pppp = self.__create_matrix(cov[5],True, True)

                cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp],
                                  [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T],
                                  [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp]])
                cov_diag.append(covariance_gggg_ssss)
                cov_diag.append(covariance_gggg_spsp)
                cov_diag.append(covariance_gggg_pppp)
                
                if gm:
                    covariance_gggm_sssm = self.__create_matrix_diagonal(cov[6], True, False, True, False)
                    covariance_gggm_sspm = self.__create_matrix_diagonal(cov[7], True, False, True, False)
                    covariance_gggm_spsm = self.__create_matrix(cov[8], False, False)
                    covariance_gggm_sppm = self.__create_matrix(cov[9],False,False)
                    covariance_gggm_ppsm = self.__create_matrix(cov[10],True,False)
                    covariance_gggm_pppm = self.__create_matrix(cov[11],True,False)
                    covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                    covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                    covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                    covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                    cov_diag.append(covariance_gmgm_smsm)
                    cov_diag.append(covariance_gmgm_pmpm)
                    cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm],
                                      [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm],
                                      [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm],
                                      [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm],
                                      [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                    if mm:
                        covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                        cov_diag.append(covariance_mmmm_mmmm)
                        covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                        covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                        covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                        covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                        covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                        cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm],
                                          [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm],
                                          [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm],
                                          [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                          [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                          [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                elif mm:
                    covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                    cov_diag.append(covariance_mmmm_mmmm)
                    covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                    covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                    covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                    cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_ggmm_ssmm],
                                      [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_ggmm_spmm],
                                      [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_ggmm_ppmm],
                                      [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmmm_mmmm]])
            elif gm:
                covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                cov_diag.append(covariance_gmgm_smsm)
                cov_diag.append(covariance_gmgm_pmpm)
                cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm],
                                  [covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                if mm:
                    covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                    cov_diag.append(covariance_mmmm_mmmm)
                    covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                    covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                    cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                      [covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                      [covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
            elif mm:
                covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                cov_diag.append(covariance_mmmm_mmmm)
                cov2d = covariance_mmmm_mmmm
            cov2d_total = np.copy(cov2d)

            if cov_dict['split_gauss']:
                cov = [gauss[idx] for idx in range(len(gauss))]
                cov_diag = []
                if gg:
                    covariance_gggg_ssss = self.__create_matrix_diagonal(cov[0], True, True, True, True)
                    covariance_gggg_sssp = self.__create_matrix_diagonal(cov[1], True, False, True, False)
                    covariance_gggg_sspp = self.__create_matrix_diagonal(cov[2], True, False, True, True)
                    covariance_gggg_spsp = self.__create_matrix(cov[3],False, False)
                    covariance_gggg_ppsp = self.__create_matrix(cov[4],True, False)
                    covariance_gggg_pppp = self.__create_matrix(cov[5],True, True)

                    cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp],
                                    [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T],
                                    [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp]])
                    cov_diag.append(covariance_gggg_ssss)
                    cov_diag.append(covariance_gggg_spsp)
                    cov_diag.append(covariance_gggg_pppp)
                    
                    if gm:
                        covariance_gggm_sssm = self.__create_matrix_diagonal(cov[6], True, False, True, False)
                        covariance_gggm_sspm = self.__create_matrix_diagonal(cov[7], True, False, True, False)
                        covariance_gggm_spsm = self.__create_matrix(cov[8], False, False)
                        covariance_gggm_sppm = self.__create_matrix(cov[9],False,False)
                        covariance_gggm_ppsm = self.__create_matrix(cov[10],True,False)
                        covariance_gggm_pppm = self.__create_matrix(cov[11],True,False)
                        covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                        covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                        covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                        covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                        cov_diag.append(covariance_gmgm_smsm)
                        cov_diag.append(covariance_gmgm_pmpm)
                        cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm],
                                        [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm],
                                        [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm],
                                        [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm],
                                        [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                        if mm:
                            covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                            cov_diag.append(covariance_mmmm_mmmm)
                            covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                            covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                            covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                            covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                            covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                            cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm],
                                            [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm],
                                            [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm],
                                            [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                            [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                            [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                    elif mm:
                        covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                        cov_diag.append(covariance_mmmm_mmmm)
                        covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                        covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                        covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                        cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_ggmm_ssmm],
                                        [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_ggmm_spmm],
                                        [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_ggmm_ppmm],
                                        [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmmm_mmmm]])
                elif gm:
                    covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                    covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                    covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                    covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                    cov_diag.append(covariance_gmgm_smsm)
                    cov_diag.append(covariance_gmgm_pmpm)
                    cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm],
                                    [covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                    if mm:
                        covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                        cov_diag.append(covariance_mmmm_mmmm)
                        covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                        covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                        cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                        [covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                        [covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                elif mm:
                    covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                    cov_diag.append(covariance_mmmm_mmmm)
                    cov2d = covariance_mmmm_mmmm
                cov2d_gauss = np.copy(cov2d)

                if self.has_ssc:
                    cov = [ssc[idx] for idx in range(len(gauss))]
                    cov_diag = []
                    if gg:
                        covariance_gggg_ssss = self.__create_matrix_diagonal(cov[0], True, True, True, True)
                        covariance_gggg_sssp = self.__create_matrix_diagonal(cov[1], True, False, True, False)
                        covariance_gggg_sspp = self.__create_matrix_diagonal(cov[2], True, False, True, True)
                        covariance_gggg_spsp = self.__create_matrix(cov[3],False, False)
                        covariance_gggg_ppsp = self.__create_matrix(cov[4],True, False)
                        covariance_gggg_pppp = self.__create_matrix(cov[5],True, True)

                        cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp],
                                        [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T],
                                        [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp]])
                        cov_diag.append(covariance_gggg_ssss)
                        cov_diag.append(covariance_gggg_spsp)
                        cov_diag.append(covariance_gggg_pppp)
                        
                        if gm:
                            covariance_gggm_sssm = self.__create_matrix_diagonal(cov[6], True, False, True, False)
                            covariance_gggm_sspm = self.__create_matrix_diagonal(cov[7], True, False, True, False)
                            covariance_gggm_spsm = self.__create_matrix(cov[8], False, False)
                            covariance_gggm_sppm = self.__create_matrix(cov[9],False,False)
                            covariance_gggm_ppsm = self.__create_matrix(cov[10],True,False)
                            covariance_gggm_pppm = self.__create_matrix(cov[11],True,False)
                            covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                            covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                            covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                            covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                            cov_diag.append(covariance_gmgm_smsm)
                            cov_diag.append(covariance_gmgm_pmpm)
                            cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm],
                                            [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm],
                                            [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm],
                                            [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm],
                                            [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                            if mm:
                                covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                                cov_diag.append(covariance_mmmm_mmmm)
                                covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                                covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                                covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                                covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                                covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                                cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm],
                                                [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm],
                                                [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm],
                                                [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                                [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                                [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                        elif mm:
                            covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                            cov_diag.append(covariance_mmmm_mmmm)
                            covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                            covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                            covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                            cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_ggmm_ssmm],
                                            [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_ggmm_spmm],
                                            [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_ggmm_ppmm],
                                            [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmmm_mmmm]])
                    elif gm:
                        covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                        covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                        covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                        covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                        cov_diag.append(covariance_gmgm_smsm)
                        cov_diag.append(covariance_gmgm_pmpm)
                        cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm],
                                        [covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                        if mm:
                            covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                            cov_diag.append(covariance_mmmm_mmmm)
                            covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                            covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                            cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                            [covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                            [covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                    elif mm:
                        covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                        cov_diag.append(covariance_mmmm_mmmm)
                        cov2d = covariance_mmmm_mmmm
                    cov2d_ssc = np.copy(cov2d)
                
                if self.has_nongauss:
                    cov = [nongauss[idx] for idx in range(len(gauss))]
                    cov_diag = []
                    if gg:
                        covariance_gggg_ssss = self.__create_matrix_diagonal(cov[0], True, True, True, True)
                        covariance_gggg_sssp = self.__create_matrix_diagonal(cov[1], True, False, True, False)
                        covariance_gggg_sspp = self.__create_matrix_diagonal(cov[2], True, False, True, True)
                        covariance_gggg_spsp = self.__create_matrix(cov[3],False, False)
                        covariance_gggg_ppsp = self.__create_matrix(cov[4],True, False)
                        covariance_gggg_pppp = self.__create_matrix(cov[5],True, True)

                        cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp],
                                        [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T],
                                        [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp]])
                        cov_diag.append(covariance_gggg_ssss)
                        cov_diag.append(covariance_gggg_spsp)
                        cov_diag.append(covariance_gggg_pppp)
                        
                        if gm:
                            covariance_gggm_sssm = self.__create_matrix_diagonal(cov[6], True, False, True, False)
                            covariance_gggm_sspm = self.__create_matrix_diagonal(cov[7], True, False, True, False)
                            covariance_gggm_spsm = self.__create_matrix(cov[8], False, False)
                            covariance_gggm_sppm = self.__create_matrix(cov[9],False,False)
                            covariance_gggm_ppsm = self.__create_matrix(cov[10],True,False)
                            covariance_gggm_pppm = self.__create_matrix(cov[11],True,False)
                            covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                            covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                            covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                            covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                            cov_diag.append(covariance_gmgm_smsm)
                            cov_diag.append(covariance_gmgm_pmpm)
                            cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm],
                                            [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm],
                                            [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm],
                                            [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm],
                                            [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                            if mm:
                                covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                                cov_diag.append(covariance_mmmm_mmmm)
                                covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                                covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                                covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                                covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                                covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                                cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm],
                                                [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm],
                                                [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm],
                                                [covariance_gggm_sssm.T, covariance_gggm_spsm.T, covariance_gggm_ppsm.T, covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                                [covariance_gggm_sspm.T, covariance_gggm_sppm.T, covariance_gggm_pppm.T, covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                                [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                        elif mm:
                            covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                            cov_diag.append(covariance_mmmm_mmmm)
                            covariance_ggmm_ssmm = self.__create_matrix_diagonal(cov[12],True,False,True,True)
                            covariance_ggmm_spmm = self.__create_matrix(cov[13],False,True)
                            covariance_ggmm_ppmm = self.__create_matrix(cov[14],True,True)
                            cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_ggmm_ssmm],
                                            [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T, covariance_ggmm_spmm],
                                            [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp, covariance_ggmm_ppmm],
                                            [covariance_ggmm_ssmm.T, covariance_ggmm_spmm.T, covariance_ggmm_ppmm.T, covariance_mmmm_mmmm]])
                    elif gm:
                        covariance_gmgm_smsm = self.__create_matrix(cov[15],False,False)
                        covariance_gmgm_smpm = self.__create_matrix(cov[16],False,False)
                        covariance_gmgm_pmsm = self.__create_matrix(cov[17],False,False)
                        covariance_gmgm_pmpm = self.__create_matrix(cov[18],False,False)
                        cov_diag.append(covariance_gmgm_smsm)
                        cov_diag.append(covariance_gmgm_pmpm)
                        cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm],
                                        [covariance_gmgm_pmsm, covariance_gmgm_pmpm]])
                        if mm:
                            covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                            cov_diag.append(covariance_mmmm_mmmm)
                            covariance_mmgm_mmsm = self.__create_matrix(cov[19],True,False)
                            covariance_mmgm_mmpm = self.__create_matrix(cov[20],True,False)
                            cov2d = np.block([[covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm.T],
                                            [covariance_gmgm_pmsm, covariance_gmgm_pmpm, covariance_mmgm_mmpm.T],
                                            [covariance_mmgm_mmsm, covariance_mmgm_mmpm, covariance_mmmm_mmmm]])
                    elif mm:
                        covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                        cov_diag.append(covariance_mmmm_mmmm)
                        cov2d = covariance_mmmm_mmmm
                    cov2d_nongauss = np.copy(cov2d)

                

            for i in range(len(cov2d[:,0])):
                for j in range(len(cov2d[:,0])):
                    cov2d_total[j,i] = cov2d_total[i,j]
                    if cov_dict['split_gauss']:
                        cov2d_gauss[j,i] = cov2d_gauss[i,j]
                        if self.has_nongauss:
                            cov2d_nongauss[j,i] = cov2d_nongauss[i,j]
                        if self.has_ssc:
                            cov2d_ssc[j,i] = cov2d_ssc[i,j]
            if len(np.where(np.linalg.eig(cov2d_total)[0] < 0)[0]) > 0:
                print("ALARM: The resulting covariance matrix has negative eigenvalues")
                print("Try to adjust the accuracy settings in the config file:")
                print("For configuration space covariance reduce theta_accuracy and increase integration_intervals, usually a factor of 2 is enough.")
                print("For bandpower covariance reduce bandpower_accuracy.")
                print("For COSEBI covariance reduce En_accuracy.")
            '''if obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
                obslist[7] = 'CE_mmCE_mm'
                obslist[9] = 'CB_mmCB_mm'
            if obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['clustering'] == True:
                obslist[0] = 'CE_ggCE_gg'
            if obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obslist[4] = 'CE_gmCE_gm
                '''
            
            hdr_str = 'Covariance matrix with the diagonals in the order: '
            '''hdr_str += obslist[0]+' ' if obsbool[0] else ''
            if obslength == 6:
                hdr_str += obslist[3]+' ' if obsbool[3] else ''
                hdr_str += obslist[5]+' ' if obsbool[5] else ''
            elif obslength == 10:
                hdr_str += obslist[4]+' ' if obsbool[4] else ''
                hdr_str += obslist[7]+' ' if obsbool[7] else ''
                hdr_str += obslist[9]+' ' if obsbool[9] else ''
            hdr_str += 'with '
            if n_tomo_clust is not None:
                hdr_str += str(n_tomo_clust) + ' tomographic clustering bins and '
            if n_tomo_lens is not None:
                hdr_str += str(n_tomo_lens) + ' tomographic lensing bins and '
            hdr_str += str(len(proj_quant)) + ' elements per tomographic bin'
            '''
            if 'matrix' in self.style:
                if not cov_dict['split_gauss']:
                    print("Writing matrix output file.")
                    fn = self.filename[self.style.index('matrix')]
                    np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
                else:
                    print("Writing matrix output file.")
                    fn = self.filename[self.style.index('matrix')]
                    np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
                    name, extension = os.path.splitext(fn)
                    fn_gauss = name + "_gauss" + extension
                    fn_nongauss = name + "_nongauss" + extension
                    fn_ssc = name + "_SSC" + extension
                    np.savetxt(fn_gauss, cov2d_gauss, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
                    if self.has_nongauss:
                        np.savetxt(fn_nongauss, cov2d_nongauss, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                    if self.has_ssc:
                        np.savetxt(fn_ssc, cov2d_ssc, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')

    def __write_cov_matrix_arbitrary(self,
                                obs_dict,
                                cov_dict,
                                n_tomo_clust,
                                n_tomo_lens,
                                sampledim,
                                read_in_tables,
                                gauss,
                                nongauss,
                                ssc,
                                fct_args):
        obslist, obsbool, obslength, mult, gg, gm, mm, xipp, xipm, ximm = \
            fct_args
        if obslength == 6 and mult == 3:
            gauss = [gauss[0]+gauss[1]+gauss[2], 
                     gauss[3]+gauss[4]+gauss[5],
                     gauss[6]+gauss[7]+gauss[8], 
                     gauss[9]+gauss[10]+gauss[11],
                     gauss[12]+gauss[13]+gauss[14], 
                     gauss[15]+gauss[16]+gauss[17]]
        elif obslength == 10 and mult == 3:
            gauss = [gauss[0]+gauss[1]+gauss[2], 
                     gauss[3]+gauss[4]+gauss[5],
                     gauss[6]+gauss[7]+gauss[8], 
                     gauss[9]+gauss[10]+gauss[11],
                     gauss[12]+gauss[13]+gauss[14], 
                     gauss[15]+gauss[16]+gauss[17],
                     gauss[18]+gauss[19]+gauss[20], 
                     gauss[21]+gauss[22]+gauss[23], 
                     gauss[24]+gauss[25]+gauss[26], 
                     gauss[27]+gauss[28]+gauss[29]]
        

        if mm:
            xipm = True
            ximm = True
            xipp = True
        else:
            xipm = False,
            ximm = False
            xipp = False
        summary = read_in_tables['arb_summary'] 
        # 'ww', 'wgt', 'wxip', 'wxim', 'gtgt', 'xipgt', 
        # 'ximgt', 'xipxip', 'xipxim', 'ximxim'
        
        
        cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(obslength)]
        cov_diag = []
        
        if self.has_csmf:
            covariange_csmf = self.__create_matrix_csmf(self.conditional_stellar_mass_function_cov[0], obs_dict['observables']['csmf_diagonal'])
            covariange_csmfgg = None
            covariange_csmfgm = None
            covariange_csmfmmE = None
            covariange_csmfmmB = None
            if gg:
                covariange_csmfgg = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[1], True, obs_dict['observables']['csmf_diagonal'])
            if gm:
                covariange_csmfgm = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[2], False, obs_dict['observables']['csmf_diagonal'])
            if mm:
                covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[3], True, obs_dict['observables']['csmf_diagonal'])
            if ximm:
                covariange_csmfmmB = np.zeros_like(covariange_csmfmmE)
            if covariange_csmfgg is not None:
                csmf_block = np.block([[covariange_csmfgg]])
                if covariange_csmfgm is not None:
                    csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm]])
                    if covariange_csmfmmE is not None:
                        csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm],[ covariange_csmfmmE]])
                        if covariange_csmfmmB is not None:
                            csmf_block = np.block([[covariange_csmfgg],[ covariange_csmfgm],[ covariange_csmfmmE],[ covariange_csmfmmB]])
            elif covariange_csmfgm is not None:
                csmf_block = np.block([[covariange_csmfgm]])
                if covariange_csmfmmE is not None:
                    csmf_block = np.block([[covariange_csmfgm],[ covariange_csmfmmE]])
                    if covariange_csmfmmB is not None:
                        csmf_block = np.block([[covariange_csmfgm],[ covariange_csmfmmE],[ covariange_csmfmmB]])

            elif covariange_csmfmmE is not None:
                csmf_block = np.block([[covariange_csmfmmE]])
                if covariange_csmfmmB is not None:
                    csmf_block = np.block([[covariange_csmfmmE],[ covariange_csmfmmB]])

        
        if gg:
            covariance_ww = self.__create_matrix_arbitrary(cov[0],True,True,'gg','gg',summary)
            cov2d = covariance_ww
            cov_diag.append(covariance_ww)
            
            if gm:
                covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False,'gm', 'gm', summary)
                cov_diag.append(covariance_gtgt)
                covariance_wgt = self.__create_matrix_arbitrary(cov[1],True,False,'gg', 'gm', summary)
                cov2d = np.block([[covariance_ww, covariance_wgt],
                                [covariance_wgt.T, covariance_gtgt]])
                
                if xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        cov_diag.append(covariance_ximxim)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary)
                covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary)
                cov2d = np.block([[covariance_ww, covariance_wxip],
                                [covariance_wxip.T, covariance_xipxip]])
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                    covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                    cov_diag.append(covariance_ximxim)
                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
        elif gm:
            covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
            cov2d = covariance_gtgt
            cov_diag.append(covariance_gtgt)        
            if xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary)
                covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                [covariance_xipgt, covariance_xipxip]])
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                    covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                    cov_diag.append(covariance_ximxim)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
        elif xipp:
            covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
            cov2d = covariance_xipxip
            cov_diag.append(covariance_xipxip)
            if ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                cov_diag.append(covariance_ximxim)
                if xipm:
                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                    [covariance_xipxim.T, covariance_ximxim]])
        elif ximm:
            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
            cov2d = covariance_xipxip
            cov_diag.append(covariance_ximxim)
        elif xipm:
            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
            cov2d = covariance_xipxim
        
        if self.has_csmf:
            cov_diag.append(covariange_csmf)
            cov2d = np.block([[cov2d, csmf_block],
                            [csmf_block.T, covariange_csmf]])

        
        cov2d_total = np.copy(cov2d)

        if cov_dict['split_gauss']:
            cov = [gauss[idx] for idx in range(obslength)]
            cov_diag = []
            if gg:
                covariance_ww = self.__create_matrix_arbitrary(cov[0],True,True,'gg','gg',summary)
                cov2d = covariance_ww
                cov_diag.append(covariance_ww)
                
                if gm:
                    covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False,'gm', 'gm', summary)
                    cov_diag.append(covariance_gtgt)
                    covariance_wgt = self.__create_matrix_arbitrary(cov[1],True,False,'gg', 'gm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wgt],
                                    [covariance_wgt.T, covariance_gtgt]])
                    
                    if xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                            cov_diag.append(covariance_ximxim)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wxip],
                                    [covariance_wxip.T, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
            elif gm:
                covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                cov2d = covariance_gtgt
                cov_diag.append(covariance_gtgt)        
                if xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                    [covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                        [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                    cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    cov_diag.append(covariance_ximxim)
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                        cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                        [covariance_xipxim.T, covariance_ximxim]])
            elif ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_ximxim)
            elif xipm:
                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxim
            if self.has_csmf:
                cov_diag.append(covariange_csmf)
                cov2d = np.block([[cov2d, csmf_block],
                                [csmf_block.T, covariange_csmf]])
            cov2d_gauss = np.copy(cov2d)


            cov = [ssc[idx] for idx in range(obslength)]
            cov_diag = []
            if gg:
                covariance_ww = self.__create_matrix_arbitrary(cov[0],True,True,'gg','gg',summary)
                cov2d = covariance_ww
                cov_diag.append(covariance_ww)
                
                if gm:
                    covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False,'gm', 'gm', summary)
                    cov_diag.append(covariance_gtgt)
                    covariance_wgt = self.__create_matrix_arbitrary(cov[1],True,False,'gg', 'gm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wgt],
                                    [covariance_wgt.T, covariance_gtgt]])
                    
                    if xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                            cov_diag.append(covariance_ximxim)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wxip],
                                    [covariance_wxip.T, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
            elif gm:
                covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                cov2d = covariance_gtgt
                cov_diag.append(covariance_gtgt)        
                if xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                    [covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                        [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                    cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    cov_diag.append(covariance_ximxim)
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                        cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                        [covariance_xipxim.T, covariance_ximxim]])
            elif ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_ximxim)
            elif xipm:
                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxim
            cov2d_ssc = np.copy(cov2d)

            cov = [nongauss[idx] for idx in range(obslength)]
            cov_diag = []
            if gg:
                covariance_ww = self.__create_matrix_arbitrary(cov[0],True,True,'gg','gg',summary)
                cov2d = covariance_ww
                cov_diag.append(covariance_ww)
                
                if gm:
                    covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False,'gm', 'gm', summary)
                    cov_diag.append(covariance_gtgt)
                    covariance_wgt = self.__create_matrix_arbitrary(cov[1],True,False,'gg', 'gm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wgt],
                                    [covariance_wgt.T, covariance_gtgt]])
                    
                    if xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                            cov_diag.append(covariance_ximxim)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary)
                    cov2d = np.block([[covariance_ww, covariance_wxip],
                                    [covariance_wxip.T, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
            elif gm:
                covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                cov2d = covariance_gtgt
                cov_diag.append(covariance_gtgt)        
                if xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                    [covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                        [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                    cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    cov_diag.append(covariance_ximxim)
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                        cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                        [covariance_xipxim.T, covariance_ximxim]])
            elif ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_ximxim)
            elif xipm:
                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary)
                cov2d = covariance_xipxim
            cov2d_nongauss = np.copy(cov2d)    
        
        for i in range(len(cov2d[:,0])):
            for j in range(len(cov2d[:,0])):
                cov2d_total[j,i] = cov2d_total[i,j]
                if cov_dict['split_gauss']:
                    cov2d_gauss[j,i] = cov2d_gauss[i,j]
                    cov2d_nongauss[j,i] = cov2d_nongauss[i,j]
                    cov2d_ssc[j,i] = cov2d_ssc[i,j]

        if len(np.where(np.linalg.eig(cov2d_total)[0] < 0)[0]) > 0:
            print("ALARM: The resulting covariance matrix has negative eigenvalues")
            print("Try to adjust the accuracy settings in the config file:")
            print("For configuration space covariance reduce theta_accuracy and increase integration_intervals, usually a factor of 2 is enough.")
            print("For bandpower covariance reduce bandpower_accuracy.")
            print("For COSEBI covariance reduce En_accuracy.")
        if self.plot:
            self.plot_corrcoeff_matrix_arbitrary(
                obs_dict, cov2d_total, cov_diag, summary, n_tomo_clust, 
                    n_tomo_lens, sampledim, self.plot ,fct_args)
        if obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['cosmic_shear'] == True:
            obslist[7] = 'CE_mmCE_mm'
            obslist[9] = 'CB_mmCB_mm'
        if obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['clustering'] == True:
            obslist[0] = 'CE_ggCE_gg'
        if obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
            obslist[4] = 'CE_gmCE_gm'
        
        hdr_str = 'Covariance matrix with the diagonals in the order: '
        hdr_str += obslist[0]+' ' if obsbool[0] else ''
        hdr_str += obslist[4]+' ' if obsbool[4] else ''
        hdr_str += obslist[7]+' ' if obsbool[7] else ''
        hdr_str += obslist[9]+' ' if obsbool[9] else ''
        hdr_str += 'with '
        if n_tomo_clust is not None:
            hdr_str += str(n_tomo_clust) + ' tomographic clustering bins and '
        if n_tomo_lens is not None:
            hdr_str += str(n_tomo_lens) + ' tomographic lensing bins and '
        if obs_dict['observables']['clustering']:
            hdr_str += str(len(summary['WL_gg'])) + ' elements per tomographic bin in gg, '
            hdr_str += str(int(summary['arb_number_first_summary_gg'])) + ' spatial indices for probe 1 '
            if summary['number_summary_gg'] > 1:
                hdr_str += 'and ' + str(int( len(summary['WL_gg']) - summary['arb_number_first_summary_gg'])) + ' spatial indices for probe 2. '
        if obs_dict['observables']['ggl']:
            hdr_str += str(len(summary['WL_gm'])) + ' elements per tomographic bin in gm ' 
            hdr_str += str(int(summary['arb_number_first_summary_gm'])) + ' spatial indices for probe 1 '
            if summary['number_summary_gm'] > 1:
                hdr_str += 'and ' + str(int( len(summary['WL_gm']) - summary['arb_number_first_summary_gm'])) + ' spatial indices for probe 2. '
        if obs_dict['observables']['cosmic_shear']:
            hdr_str += str(len(summary['WL_mmE'])) + ' elements per tomographic bin in mm' 
            hdr_str += str(int(summary['arb_number_first_summary_mm'])) + ' spatial indices for probe 1 '
            if summary['number_summary_mm'] > 1:
                hdr_str += 'and ' + str(int( len(summary['WL_mmE']) - summary['arb_number_first_summary_mm'])) + ' spatial indices for probe 2, both for E and B mode.'
        if 'matrix' in self.style:
            if not cov_dict['split_gauss']:
                print("Writing matrix output file.")
                fn = self.filename[self.style.index('matrix')]
                if self.save_as_binary:
                    name, extension = os.path.splitext(fn)
                    np.save(name, cov2d_total)
                else:
                    np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
            else:
                print("Writing matrix output file.")
                if self.save_as_binary:
                    fn = self.filename[self.style.index('matrix')]
                    name, extension = os.path.splitext(fn)
                    np.save(name, cov2d_total)
                    fn_gauss = name + "_gauss"
                    fn_nongauss = name + "_nongauss"
                    fn_ssc = name + "_SSC"
                    np.save(fn_gauss, cov2d_gauss)
                    if self.has_nongauss:
                        np.save(fn_nongauss, cov2d_nongauss)
                    if self.has_ssc:
                        np.save(fn_ssc, cov2d_ssc)
                else:
                    fn = self.filename[self.style.index('matrix')]
                    np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
                    name, extension = os.path.splitext(fn)
                    fn_gauss = name + "_gauss" + extension
                    fn_nongauss = name + "_nongauss" + extension
                    fn_ssc = name + "_SSC" + extension
                    np.savetxt(fn_gauss, cov2d_gauss, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
                    if self.has_nongauss:
                        np.savetxt(fn_nongauss, cov2d_nongauss, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                    if self.has_ssc:
                        np.savetxt(fn_ssc, cov2d_ssc, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                
        
    def __get_obslist(self, 
                      obs_dict,
                      xipm = False):
            
        if not xipm:
            mm = obs_dict['observables']['cosmic_shear'] 
            gm = obs_dict['observables']['ggl']
            gg = obs_dict['observables']['clustering']
            cross = obs_dict['observables']['cross_terms']

            obslist = ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            obsbool = [gg, gg and gm and cross, gg and mm and cross, gm, 
                    mm and gm and cross, mm]
        else:
            if obs_dict['observables']['cosmic_shear']:
                xipp = obs_dict['THETAspace']['xi_pp']
                xipm = obs_dict['THETAspace']['xi_pm']
                ximm = obs_dict['THETAspace']['xi_mm']
                if obs_dict['observables']['est_shear'] == "cosebi":
                    xipp = True
                    xipm = True
                    ximm = True
            else:
                xipp = False
                xipm = False
                ximm = False
            gm = obs_dict['observables']['ggl']
            gg = obs_dict['observables']['clustering']
            cross = obs_dict['observables']['cross_terms']

            obslist = ['gggg', 'gggm', 'ggxip', 'ggxim', 'gmgm', 'gmxip', 
                       'gmxim', 'xipxip', 'xipxim', 'ximxim']
            obsbool = [gg, gg and gm and cross, gg and xipp and cross, 
                       gg and ximm and cross, gm, gm and xipp and cross, 
                       gm and ximm and cross, xipp, xipm, ximm]


        return obslist, obsbool, len(obslist)

    def __none_to_zero(self,
                       gauss, 
                       nongauss, 
                       ssc):
        for idx in range(len(gauss)):
            if gauss[idx] is None:
                gauss[idx] = 0
        for idx in range(len(nongauss)):
            if nongauss[idx] is None:
                nongauss[idx] = 0
            if ssc[idx] is None:
                ssc[idx] = 0

        return gauss, nongauss, ssc

    def __check_for_empty_input(self, 
                                out, 
                                shape):

        if out is None or type(out) is int:
            out = np.zeros(shape)
        return out

    def __get_sampledim(self,
                        gauss,
                        nongauss,
                        ssc):
        if self.has_gauss:
            sampledim_save = 0
            for idx in range(len(gauss)):
                try:
                    sampledim = (gauss[idx].shape)[2]
                    if(sampledim < sampledim_save):
                        sampledim = sampledim_save
                    sampledim_save = (gauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_nongauss:
            sample_dim_save = 0
            for idx in range(len(nongauss)):
                try:
                    sampledim = (nongauss[idx].shape)[2]
                    if(sampledim < sampledim_save):
                        sampledim = sampledim_save
                    sampledim_save = (nongauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_ssc:
            sampledim_save = 0
            for idx in range(len(ssc)):
                try:
                    sampledim = (ssc[idx].shape)[2]
                    if(sampledim < sampledim_save):
                        sampledim = sampledim_save
                    sampledim_save = (ssc[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if sampledim == -1:
            raise Exception("InputError: Neither of the covariance terms " +
                            "seems to have any values. Please check.")
        
        return sampledim

    def __get_tomodim(self,
                      gauss,
                      nongauss,
                      ssc):
        tomodim = [-1]
        try:
            tomodim = (gauss.shape)[4:]
        except (AttributeError,TypeError):
            ...
        try:
            tomodim = (nongauss.shape)[4:]
        except (AttributeError,TypeError):
            ...
        try:
            tomodim = (ssc.shape)[4:]
        except (AttributeError,TypeError):
            ...
        
        return list(tomodim)

    def __get_idxlist(self, 
                      proj_quant,
                      sampledim):
        idxlist = []
        for idxi, ri in enumerate(proj_quant):
            for idxj in range(len(proj_quant)):
                rj = proj_quant[idxj]
                for s1 in range(sampledim):
                    for s2 in range(sampledim):
                        idxlist.append((idxi, ri, idxj, rj, s1, s2))

        return idxlist

    def project_to_2d_notomo(self, 
                               covmatrix):
        knum = covmatrix.shape[0]
        sampledim = covmatrix.shape[2]
        len2d = knum*sampledim
        cov2d = np.zeros((len2d,len2d))
        idx1, idx2 = 0, 0
        for s1 in range(sampledim):
            for s2 in range(sampledim):
                cov2d[idx1:idx1+knum, idx2:idx2+knum] = \
                    covmatrix[:,:,s1, s2]
                idx1 += knum
                if idx1 == len(cov2d):
                    idx1 = 0
                    idx2 += knum

        return cov2d

    def project_to_2d(self,
                      mode, 
                      covmatrix,
                      tomo1 = None,
                      tomo2 = None):
        ellnum = covmatrix.shape[0]
        sampledim = covmatrix.shape[2]
        if tomo1 is None or tomo2 is None:
            try:
                tomo1, tomo2 = \
                    min(covmatrix.shape[4:]), max(covmatrix.shape[4:])
            except ValueError:
                return self.project_to_2d_notomo(covmatrix)
        if mode in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
            tomo1 = covmatrix.shape[4]
            len2d = int(tomo1*(tomo1+1)/2)*ellnum
            cov2d = np.zeros((len2d,len2d))
            idx1, idx2 = 0, 0
            for s1 in range(sampledim):
                for s2 in range(s1, sampledim):
                    for t1 in range(tomo1):
                        for t3 in range(t1, tomo1):
                            for t2 in range(tomo1):
                                for t4 in range(t2, tomo1):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1, t3, t2, t4]
                                    idx2 += ellnum
                                    if idx2 == len2d:
                                        idx2 = 0
                                        idx1 += ellnum
        elif mode == 'gmgm':
            tomo1 = covmatrix.shape[4]
            tomo2 = covmatrix.shape[5]
            len2d = tomo1*tomo2*ellnum
            cov2d = np.zeros((len2d,len2d))
            idx1, idx2 = 0, 0
            for s1 in range(sampledim):
                for s2 in range(s1, sampledim):
                    for t1 in range(tomo1):
                        for t2 in range(tomo2):
                            for t3 in range(tomo1):
                                for t4 in range(tomo2):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1, t3, t2, t4]
                                    idx2 += ellnum
                                    if idx2 == len2d:
                                        idx2 = 0
                                        idx1 += ellnum
        elif mode in ['gggm', 'mmgm', 'gmxip', 'gmxim']:
            if mode in ['gmxip', 'gmxim']:
                covmatrix = covmatrix.transpose(0,1,2,3,6,7,4,5)
            
            tomo1 = covmatrix.shape[4]
            tomo2 = covmatrix.shape[5]
            tomo3 = covmatrix.shape[6]
            tomo4 = covmatrix.shape[7]
            if mode == 'gggm' or mode == 'mmgm':
                len2d2 = tomo3*tomo4*ellnum
                len2d1 = int(tomo1*(tomo2+1)/2)*ellnum
                cov2d = np.zeros((len2d1,len2d2))

                idx1, idx2 = 0, 0
                for s1 in range(sampledim):
                    for s2 in range(s1, sampledim):
                        for t1 in range(tomo1):
                            for t3 in range(t1,tomo1):
                                for t2 in range(tomo3):
                                    for t4 in range(tomo4):
                                        cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                            covmatrix[:,:,s1, s2, t1, t3, t2, t4]
                                        idx2 += ellnum
                                        if idx2 == len2d2:
                                            idx2 = 0
                                            idx1 += ellnum
            else:             
                len2d1 = tomo1*tomo2*ellnum
                len2d2 = int(tomo3*(tomo4+1)/2)*ellnum
                cov2d = np.zeros((len2d1,len2d2))
                
                idx1, idx2 = 0, 0
                for s1 in range(sampledim):
                    for s2 in range(s1, sampledim):
                        for t1 in range(tomo1):
                            for t3 in range(tomo1):
                                for t2 in range(tomo3):
                                    for t4 in range(t2,tomo4):
                                        cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                            covmatrix[:,:,s1, s2, t1, t3, t2, t4]
                                        idx2 += ellnum
                                        if idx2 == len2d2:
                                            idx2 = 0
                                            idx1 += ellnum
        elif mode in ['ggmm', 'ggxip', 'ggxim']:
            len2d1 = int(tomo1*(tomo1+1)/2)*ellnum
            len2d2 = int(tomo2*(tomo2+1)/2)*ellnum
            cov2d = np.zeros((len2d1,len2d2))
            idx1, idx2 = 0, 0
            for s1 in range(sampledim):
                for s2 in range(s1, sampledim):
                    for t1 in range(tomo1):
                        for t3 in range(t1, tomo1):
                            for t2 in range(tomo2):
                                for t4 in range(t2, tomo2):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1, t3, t2, t4]
                                    idx2 += ellnum
                                    if idx2 == len2d2:
                                        idx2 = 0
                                        idx1 += ellnum
        else:
            raise Exception("mode '" + mode + "' wrong, either gggg, gggm, " +
                "ggmm, ggxip, ggxim, gmgm, mmgm, gmxip, gmxim, mmmm, " +
                "xipxip, xipxim, ximxim")

        return cov2d

    def __mesh_2d_matrix_together(self,
                                  covdiag, 
                                  covoff):
        # covdiag = 'gggg', 'gmgm', 'mmmm' / ..., 'xi_pp', 'xi_mm'
        # covoff  = 'gggm', 'ggmm', 'mmgm'
        # covoff  = 'gggm', 'gg&xipp', 'gg&ximm', 'gm&xipp', 'gm&ximm', 'xipxim'
        len2d = sum([len(covdiag[x]) for x in range(len(covdiag))])
        mesh2d = np.zeros((len2d, len2d))
      
        # block 11
        idx1_s = 0
        idx1_e = idx1_s + len(covdiag[0])
        mesh2d[idx1_s:idx1_e,idx1_s:idx1_e] = covdiag[0]

        # block 22
        if len(covdiag) > 1:
            idx1_s = len(covdiag[0])
            idx1_e = idx1_s + len(covdiag[1])
            mesh2d[idx1_s:idx1_e,idx1_s:idx1_e] = covdiag[1]
        # block 12 and 21
        if len(covoff) > 0:
            idx2_s = 0
            idx2_e = idx2_s + len(covdiag[0])
            try:
                mesh2d[idx1_s:idx1_e,idx2_s:idx2_e] = covoff[0].T
                mesh2d[idx2_s:idx2_e,idx1_s:idx1_e] = covoff[0]
            except ValueError:
                mesh2d[idx1_s:idx1_e,idx2_s:idx2_e] = covoff[0]
                mesh2d[idx2_s:idx2_e,idx1_s:idx1_e] = covoff[0].T

        if len(covdiag) > 2:
            # block 33
            idx1_s = len(covdiag[0]) + len(covdiag[1])
            idx1_e = idx1_s + len(covdiag[2])
            mesh2d[idx1_s:idx1_e, idx1_s:idx1_e] = covdiag[2]

            # block 13 and 31
            if len(covoff) > 1:
                idx2_s = 0
                idx2_e = idx2_s + len(covdiag[0])
                try:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[1].T
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[1]
                except ValueError:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[1]
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[1].T

            # block 23 and 32
            if len(covoff) > 2:
                idx2_s = len(covdiag[0])
                idx2_e = idx2_s + len(covdiag[1])
                add_idx = 0 if len(covoff) == 3 else 1
                try:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[2+add_idx].T
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[2+add_idx]
                except ValueError:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[2+add_idx]
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[2+add_idx].T

        if len(covdiag) > 3:
            # block 44
            idx1_s = len(covdiag[0]) + len(covdiag[1]) + len(covdiag[2])
            idx1_e = idx1_s + len(covdiag[3])
            mesh2d[idx1_s:idx1_e, idx1_s:idx1_e] = covdiag[3]

            # block 14 and 41
            if len(covoff) > 2:
                idx2_s = 0
                idx2_e = idx2_s + len(covdiag[0])
                try:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[2].T
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[2]
                except ValueError:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[2]
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[2].T

            # block 24 and 42
            if len(covoff) > 4:
                idx2_s = len(covdiag[0])
                idx2_e = idx2_s + len(covdiag[1])
                try:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[4].T
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[4]
                except ValueError:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[4]
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[4].T

            # block 34 and 43
            if len(covoff) > 5:
                idx2_s = len(covdiag[0]) + len(covdiag[1])
                idx2_e = idx2_s + len(covdiag[2])
                try:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[5].T
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[5]
                except ValueError:
                    mesh2d[idx1_s:idx1_e, idx2_s:idx2_e] = covoff[5]
                    mesh2d[idx2_s:idx2_e, idx1_s:idx1_e] = covoff[5].T
        
        return mesh2d
    
    def __correlation_matrix(self,
                             cov):
        return cov / np.sqrt( np.diag(cov)[:,None] * np.diag(cov)[None,:] )

    def write_Cells(self,
                    ellrange, 
                    n_tomo_clust,
                    n_tomo_lens,
                    Cells):

        Cell_gg, Cell_gm, Cell_mm = Cells

        if Cell_gg is not None and type(Cell_gg) is not int:
            sampledim = Cell_gg.shape[1]
            ostr_format = '%.10e\t%i\t\t%i\t\t'
            if sampledim == 1:
                Cell_str = 'Cell_gg'     
            else:
                Cell_str = ''
                for sm in range(sampledim):
                    for sn in range(sampledim):
                        Cell_str += 'Cell_g_'+ str(sm+1) +'g_' + str(sn+1) + '\t\t\t\t'
            olist_gg = []
            olist_gg.append("#ell\t\ttomo_i\ttomo_j\t"+Cell_str)
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_clust):
                    for tj in range(n_tomo_clust):
                        ostr = ostr_format \
                                    % (ell, ti+1, tj+1)
                        for i_sample in range(sampledim):
                            for j_sample in range(sampledim):
                                ostr += '%10e\t\t\t' % Cell_gg[ellidx, i_sample, j_sample, ti, tj]
                        olist_gg.append(ostr)
            fname = self.__add_string_to_filename('gg', self.Cellfile)
            with open(fname, 'w') as file:
                print("Writing '" + fname + "'.")
                for ostr in olist_gg:
                    file.write("%s\n" % ostr)

        if Cell_gm is not None and type(Cell_gm) is not int:
            sampledim = Cell_gm.shape[1]
            ostr_format = '%.10e\t%i\t\t%i\t\t'
            if sampledim == 1:
                Cell_str = 'Cell_gkappa'     
            else:
                Cell_str = ''
                for sm in range(sampledim):
                    Cell_str += 'Cell_g_'+ str(sm+1) +'kappa_' + '\t\t\t'
            olist_gm = []
            olist_gm.append("#ell\t\ttomo_i\ttomo_j\t"+Cell_str)
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_clust):
                    for tj in range(n_tomo_lens):
                        ostr = ostr_format \
                            % (ell, ti+1, tj+1)
                        for i_sample in range(sampledim):
                            ostr += '%10e\t\t\t' % Cell_gm[ellidx, i_sample, ti, tj]
                        olist_gm.append(ostr)
            fname = self.__add_string_to_filename('gkappa', self.Cellfile)
            with open(fname, 'w') as file:
                print("Writing '" + fname + "'.")
                for ostr in olist_gm:
                    file.write("%s\n" % ostr)

        if Cell_mm is not None and type(Cell_mm) is not int:
            sampledim = Cell_mm.shape[1]
            ostr_format = '%.10e\t%i\t\t%i\t\t'
            Cell_str = 'Cell_kappakappa'     
            olist_mm = []
            olist_mm.append("#ell\t\ttomo_i\ttomo_j\t"+Cell_str)
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_lens):
                    for tj in range(n_tomo_lens):
                        ostr = ostr_format \
                            % (ell, ti+1, tj+1)
                        ostr += '%10e\t\t\t' % Cell_mm[ellidx, 0, ti, tj]
                        olist_mm.append(ostr)
            fname = self.__add_string_to_filename('kappakappa', self.Cellfile)
            with open(fname, 'w') as file:
                print("Writing '" + fname + "'.")
                for ostr in olist_mm:
                    file.write("%s\n" % ostr)

        return True

    def write_trispectra(self,
                         zet,
                         krange, 
                         sampledim,
                         trispec,
                         tri_bool):

        for idx,tbool in enumerate(tri_bool):
            if not tbool:
                trispec[idx] = \
                    np.zeros((len(krange), len(krange), sampledim, sampledim))

        idxlist = self.__get_idxlist_covk(krange, sampledim)
        olist = []
        olist.append("#log10ki\tlog10kj\t\ts1\ts2\t" +
                     "gggg\t\tgggm\t\tggmm\t\tgmgm\t\tmmgm\t\tmmmm")
        for kdxi, ki, kdxj, kj, m, n in idxlist:
            ostr = "%.4f\t\t%.4f\t\t%i\t%i\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e" \
                 % (ki, kj, m, n,
                    trispec[0][kdxi, kdxj, m, n],
                    trispec[1][kdxi, kdxj, m, n],
                    trispec[2][kdxi, kdxj, m, n], 
                    trispec[3][kdxi, kdxj, m, n],
                    trispec[4][kdxi, kdxj, m, n],
                    trispec[5][kdxi, kdxj, m, n])
            olist.append(ostr)
            
        fname = self.__add_string_to_filename(zet, self.trispecfile)
        with open(fname, 'w') as file:
            print("Writing '" + fname + "'.")
            for ostr in olist:
                file.write("%s\n" % ostr)

        return True
