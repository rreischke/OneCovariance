import enum
import numpy as np

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

    def __init__(self, output_dict):
        self.filename = output_dict['file']
        self.__check_filetype()
        self.style = output_dict['style']
        
        self.plot = output_dict['make_plot']
        self.trispecfile = output_dict['trispec']
        self.Cellfile = output_dict['Cell']
        self.tex = output_dict

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
        gauss, nongauss, ssc = self.__none_to_zero(gauss, nongauss, ssc)

        obslist, obsbool, obslength = self.__get_obslist(obs_dict)
        gg, gm, mm = obsbool[0], obsbool[3], obsbool[5]
        xipp, xipm, ximm = None, None, None
        mult = 1
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
        else:
            raise Exception("OutputError: The gaussian covariance needs at " +
                "least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(nongauss) != obslength:
            raise Exception("OutputError: The nongaussian covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
        if len(ssc) != obslength:
            raise Exception("OutputError: The super-sample covariance needs " +
                "at least 6 entries in the order ['gggg', 'gggm', 'ggmm', " +
                "'gmgm', 'mmgm', 'mmmm'] or ['gggg', 'gggm', 'ggxip', " +
                "'ggxim', 'gmgm', 'gmxip', 'gmxim', 'xipxip', 'xi_pm', " +
                "'xi_mm']. Replacing the respective inputs with 0 or None " +
                "is supported.")
                
        sampledim = self.__get_sampledim(obslength, gauss, nongauss, ssc)

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
                shape = [len(proj_quant)]*2 + [sampledim]*2 + tomodim
            for _ in range(mult):
                gauss[gaussidx] = self.__check_for_empty_input(gauss[gaussidx], shape)
                gaussidx += 1
            nongauss[idx] = self.__check_for_empty_input(nongauss[idx], shape)
            ssc[idx] = self.__check_for_empty_input(ssc[idx], shape)

        if 'terminal' in self.style or 'list' in self.style:
            fct_args = [obslist, obsbool]
        
            self.__write_cov_list(cov_dict, obs_dict, n_tomo_clust, 
                                  n_tomo_lens, sampledim, proj_quant, 
                                  gauss, nongauss, ssc, fct_args)
        if 'matrix' in self.style or self.plot:
            fct_args = [obslist, obsbool, obslength, mult, 
                        gg, gm, mm, xipp, xipm, ximm]
            self.__write_cov_matrix_new(obs_dict, n_tomo_clust,
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
                              filename = None):
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

        ratio = len(covmatrix) / 140
        if self.tex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots(1, 1, figsize=(12,12))

        corr_covmatrix = self.__correlation_matrix(covmatrix)
        
        limit = max(-min(corr_covmatrix.flatten()), max(corr_covmatrix.flatten()))
        cbar = ax.imshow(corr_covmatrix, cmap = 'seismic', 
                         extent = [0, len(corr_covmatrix), 0, len(corr_covmatrix)],
                         vmin=-limit, vmax=limit)
        fig.colorbar(cbar, location='bottom', shrink=.775, aspect=30, pad=0.055).ax.tick_params(axis='x', direction='in')
        ax.text(len(covmatrix)/2, -6*ratio, 'Correlation coefficients', fontsize=24, ha='center', va='center')

        lbls = []
        box_lbls = []
        for idx in range(len(cov_diag)):
            if len(cov_diag[idx]) == len(proj_quant)*sampledim:
                if obs_dict['observables']['est_clust'] == 'k_space':
                    box_lbls.append('$P_\\mathrm{gg}$')
                else:
                    box_lbls.append('gg')
                if obs_dict['observables']['est_ggl'] == 'k_space':
                    box_lbls.append('$P_\\mathrm{gm}$')
                else:
                    box_lbls.append('gm')
                if obs_dict['observables']['est_shear'] == 'k_space':
                    box_lbls.append('$P_\\mathrm{mm}$')
                else:
                    box_lbls.append('mm')
                for s1 in range(sampledim):
                    lbls.append('M'+str(s1+1))
            elif len(cov_diag[idx]) == int(len(proj_quant)*n_tomo_clust*(n_tomo_clust+1)/2) and idx == 0:
                if obs_dict['observables']['est_clust'] == 'C_ell':
                    box_lbls.append('$C_\\mathrm{gg}$')
                elif obs_dict['observables']['est_clust'] == 'w':
                    box_lbls.append('$w$')
                else:
                    box_lbls.append('gg')
                for t1 in range(n_tomo_clust):
                    for t2 in range(t1, n_tomo_clust):
                        lbls.append('L'+str(t1+1)+'-L'+str(t2+1))
            elif len(cov_diag[idx]) == len(proj_quant)*n_tomo_clust*n_tomo_lens:
                if obs_dict['observables']['est_ggl'] == 'C_ell':
                    box_lbls.append('$C_{\\mathrm{g}\\kappa}$')
                elif obs_dict['observables']['est_ggl'] == 'gamma_t':
                    box_lbls.append('$\\gamma_\\mathrm{t}$')
                elif obs_dict['observables']['est_ggl'] == 'bandpowers':
                    box_lbls.append('$\\mathcal{C}_{E,\\mathrm{ggl}}$')
                else:
                    box_lbls.append('gm')
                for t1 in range(n_tomo_clust):
                    for t2 in range(n_tomo_lens):
                        lbls.append('L'+str(t1+1)+'-S'+str(t2+1))
            elif len(cov_diag[idx]) == int(len(proj_quant)*n_tomo_lens*(n_tomo_lens+1)/2):
                if obs_dict['observables']['est_shear'] == 'C_ell':
                    box_lbls.append('$C_{\\kappa\\kappa}$')
                elif obs_dict['observables']['est_shear'] == 'xi_pm':
                    if '$\\xi_+$' not in box_lbls:
                        box_lbls.append('$\\xi_+$')
                    else:
                        box_lbls.append('$\\xi_-$')
                elif obs_dict['observables']['est_shear'] == 'bandpowers':
                    if '$\\mathcal{C}_{E,\\mathrm{mm}}$' not in box_lbls:
                        box_lbls.append('$\\mathcal{C}_{E,\\mathrm{mm}}$')
                    else:
                        box_lbls.append('$\\mathcal{C}_{B,\\mathrm{mm}}$')
                elif obs_dict['observables']['est_shear'] == 'cosebi':
                    if '$E_n$' not in box_lbls:
                        box_lbls.append('$E_n$')
                    else:
                        box_lbls.append('$B_n$')
                else:
                    box_lbls.append('mm')
                for t1 in range(n_tomo_lens):
                    for t2 in range(t1, n_tomo_lens):
                        lbls.append('S'+str(t1+1)+'-S'+str(t2+1))

        idx, offset = -1, 0
        for idx,diag in enumerate(cov_diag[:-1]):
            ax.axvline(x=len(diag)+offset,color='0', lw=1.5, alpha=.7)
            ax.axhline(y=len(covmatrix)-len(diag)-offset,color='0', lw=1.5, alpha=.7)
            ax.text(len(diag)/2+offset, len(covmatrix)+15*ratio, box_lbls[idx], fontsize=24, ha='center', va='center')
            ax.text(-15*ratio, len(covmatrix)-len(diag)/2-offset, box_lbls[idx], fontsize=24, rotation=90, ha='center', va='center')
            offset += len(diag)
        ax.text(len(cov_diag[-1])/2+offset, len(covmatrix)+15*ratio, box_lbls[idx+1], fontsize=24, ha='center', va='center')
        ax.text(-15*ratio, len(covmatrix)-len(cov_diag[-1])/2-offset, box_lbls[idx+1], fontsize=24, rotation=90, ha='center', va='center')
        offset = len(proj_quant)
        while offset < len(corr_covmatrix)-1:
            ax.axvline(x=offset,color='0', lw=1, alpha=.2)
            ax.axhline(y=offset,color='0', lw=1, alpha=.2)
            offset += len(proj_quant)

        tickpos = np.arange(0,len(covmatrix),len(proj_quant))
        ax.xaxis.tick_top()
        ax.set_xticks(tickpos)
        ax.set_yticks(tickpos)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        ax.set_xticklabels([]*len(tickpos))
        ax.set_yticklabels([]*len(tickpos))
        lblpos = tickpos+int(.5*len(proj_quant))
        [ax.text(-6*ratio, pos, lbl, fontsize=14, ha='center', va='center')  for pos, lbl in zip(lblpos, lbls[::-1])]
        [ax.text(pos, len(covmatrix)+6*ratio, lbl, fontsize=14, rotation=270, ha='center', va='center')  for pos, lbl in zip(lblpos, lbls)]

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

        if ('C_ell' in obs_dict['observables'].values() or \
            'bandpowers' in obs_dict['observables'].values()):
            proj_quant_str = 'ell1\tell2\t'
        elif (obs_dict['observables']['est_shear'] == 'xi_pm' or \
              obs_dict['observables']['est_ggl'] == 'gamma_t' or \
              obs_dict['observables']['est_clust'] == 'w'):
            proj_quant_str = 'theta1\ttheta2\t'
        elif 'k_space' in obs_dict['observables'].values():
            proj_quant_str = 'log10k1\t\tlog10k2'
        elif 'cosebis' in obs_dict['observables'].values():
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
            if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'CE_ggCE_mm'
            if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'CE_ggCB_mm'
            if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'CE_gmCE_mm'
            if obs == 'gmxiM' and obs_dict['observables']['est_ggl'] == 'bandpowers' and obs_dict['observables']['est_shear'] == 'bandpowers' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'CE_gmCB_mm'
            

            if obs == 'xipxip' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                obs_copy = 'EEmmmm'
            if obs == 'xipxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                obs_copy = 'EBmmmm'
            if obs == 'ximxim' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['cosmic_shear'] == True:
                obs_copy = 'BBmmmm'
            if obs == 'gggg' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['clustering'] == True:
                obs_copy = 'EEgggg'
            if obs == 'gmgm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EEgmgm'
            if obs == 'gggm' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EEgggm'
            if obs == 'ggxip' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EEggmm'
            if obs == 'ggxim' and obs_dict['observables']['est_clust'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EBggmm'
            if obs == 'gmxip' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EEgmmm'
            if obs == 'gmxiM' and obs_dict['observables']['est_ggl'] == 'cosebi' and obs_dict['observables']['est_shear'] == 'cosebi' and obs_dict['observables']['ggl'] == True:
                obs_copy = 'EBgmmm'
            
            if not obsbool[oidx]:
                splitidx += 3
                continue
            
            if not cov_dict['split_gauss']:
                if write_header:
                    olist.append('#obs\t' +proj_quant_str+ '\t\ts1\ts2\t' +
                                 tomo_str + 'cov\t\t\tcovg\t\tcovng\t\tcovssc')
                    write_header = False
                for idxi, ri, idxj, rj, s1, s2 in idxlist:
                    if n_tomo_clust is None and n_tomo_lens is None:
                        idxs = (idxi, idxj, s1, s2)
                        cov = gauss[oidx][idxs] \
                            + nongauss[oidx][idxs] \
                            + ssc[oidx][idxs]
                        ostr = ostr_format \
                            % (obs_copy,  ri, rj, s1, s2, cov, 
                                gauss[oidx][idxs], 
                                nongauss[oidx][idxs], 
                                ssc[oidx][idxs])
                        olist.append(ostr)
                    else:
                        if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            tomo1 = gauss[oidx].shape[4]
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo1):
                                        for t4 in range(t3, tomo1):
                                            idxs = (idxi, idxj, s1, s2, t1, t2, t3, t4)
                                            cov = gauss[oidx][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                s1, s2, t1+1, t2+1, t3+1, t4+1, 
                                                cov, 
                                                gauss[oidx][idxs],
                                                nongauss[oidx][idxs],
                                                ssc[oidx][idxs])
                                            olist.append(ostr)
                        elif obs == 'gmgm':
                            tomo1 = gauss[oidx].shape[4]
                            tomo2 = gauss[oidx].shape[5]
                            for t1_1 in range(tomo1):
                                for t2_1 in range(tomo2):
                                    for t1_2 in range(tomo1):
                                        for t2_2 in range(tomo2):
                                            idxs = (idxi, idxj, s1, s2, t1_1, t2_1, t1_2, t2_2)
                                            cov = gauss[oidx][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                s1, s2, t1_1+1, t2_1+1, t1_2+1, t2_2+1, 
                                                cov, 
                                                gauss[oidx][idxs],
                                                nongauss[oidx][idxs],
                                                ssc[oidx][idxs])
                                            olist.append(ostr)
                        elif obs == ['gggm', 'mmgm', 'gmxip', 'gmxim']:
                            tomo1 = gauss[oidx].shape[4]
                            tomo3 = gauss[oidx].shape[6]
                            tomo4 = gauss[oidx].shape[7]
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo3):
                                        for t4 in range(tomo4):
                                            idxs = (idxi, idxj, s1, s2, t1, t2, t3, t4)
                                            cov = gauss[oidx][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                   s1, s2, t1+1, t2+1, t3+1, t4+1, 
                                                   cov, 
                                                   gauss[oidx][idxs],
                                                   nongauss[oidx][idxs],
                                                   ssc[oidx][idxs])
                                            olist.append(ostr)
                        elif obs in ['ggmm', 'ggxip', 'ggxim']:
                            tomo1 = gauss[oidx].shape[4]
                            tomo2 = gauss[oidx].shape[6]
                            for t1_1 in range(tomo1):
                                for t1_2 in range(t1_1, tomo1):
                                    for t2_1 in range(tomo2):
                                        for t2_2 in range(t2_1, tomo2):
                                            idxs = (idxi, idxj, s1, s2, t1_1, t1_2, t2_1, t2_2)
                                            cov = gauss[oidx][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                   s1, s2, t1_1+1, t1_2+1, t2_1+1, t2_2+1, 
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
                for idxi, ri, idxj, rj, s1, s2 in idxlist:
                    if n_tomo_clust is None and n_tomo_lens is None:
                        idxs = (idxi, idxj, s1, s2)
                        cov = gauss[splitidx][idxs] \
                            + gauss[splitidx+1][idxs] \
                            + gauss[splitidx+2][idxs] \
                            + nongauss[oidx][idxs] \
                            + ssc[oidx][idxs]
                        ostr = ostr_format \
                            % (obs_copy,  ri, rj, s1, s2, cov,
                                gauss[splitidx][idxs],
                                gauss[splitidx+1][idxs],
                                gauss[splitidx+2][idxs],
                                nongauss[oidx][idxs],
                                ssc[oidx][idxs])
                        olist.append(ostr)
                    else:
                        if obs in ['gggg', 'mmmm', 'xipxip', 'xipxim', 'ximxim']:
                            tomo1 = gauss[splitidx].shape[4]
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo1):
                                        for t4 in range(t3, tomo1):
                                            idxs = (idxi, idxj, s1, s2, t1, t2, t3, t4)
                                            cov = gauss[splitidx][idxs] \
                                                + gauss[splitidx+1][idxs] \
                                                + gauss[splitidx+2][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                s1, s2, t1+1, t2+1, t3+1, t4+1, 
                                                cov, 
                                                gauss[splitidx][idxs],
                                                gauss[splitidx+1][idxs],
                                                gauss[splitidx+2][idxs],
                                                nongauss[oidx][idxs],
                                                ssc[oidx][idxs])
                                            olist.append(ostr)
                        elif obs == 'gmgm':
                            tomo1 = gauss[splitidx].shape[4]
                            tomo2 = gauss[splitidx].shape[5]
                            for t1_1 in range(tomo1):
                                for t2_1 in range(tomo2):
                                    for t1_2 in range(tomo1):
                                        for t2_2 in range(tomo2):
                                            idxs = (idxi, idxj, s1, s2, t1_1, t2_1, t1_2, t2_2)
                                            cov = gauss[splitidx][idxs] \
                                                + gauss[splitidx+1][idxs] \
                                                + gauss[splitidx+2][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                s1, s2, t1_1+1, t2_1+1, t1_2+1, t2_2+1, 
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
                            for t1 in range(tomo1):
                                for t2 in range(t1, tomo1):
                                    for t3 in range(tomo3):
                                        for t4 in range(tomo4):
                                            idxs = (idxi, idxj, s1, s2, t1, t2, t3, t4)
                                            cov = gauss[splitidx][idxs] \
                                                + gauss[splitidx+1][idxs] \
                                                + gauss[splitidx+2][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                   s1, s2, t1+1, t2+1, t3+1, t4+1, 
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
                            for t1_1 in range(tomo1):
                                for t1_2 in range(t1_1, tomo1):
                                    for t2_1 in range(tomo2):
                                        for t2_2 in range(t2_1, tomo2):
                                            idxs = (idxi, idxj, s1, s2, t1_1, t1_2, t2_1, t2_2)
                                            cov = gauss[splitidx][idxs] \
                                                + gauss[splitidx+1][idxs] \
                                                + gauss[splitidx+2][idxs] \
                                                + nongauss[oidx][idxs] \
                                                + ssc[oidx][idxs]
                                            ostr = ostr_format \
                                                % (obs_copy,  ri, rj, 
                                                   s1, s2, t1_1+1, t1_2+1, t2_1+1, t2_2+1, 
                                                   cov, 
                                                   gauss[splitidx][idxs],
                                                   gauss[splitidx+1][idxs],
                                                   gauss[splitidx+2][idxs],
                                                   nongauss[oidx][idxs],
                                                   ssc[oidx][idxs])
                                            olist.append(ostr)
                splitidx += 3

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
    
    def __create_matrix(self,covlist, is_i_smaller_j, is_m_smaller_n):
        if is_i_smaller_j and is_m_smaller_n:
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2)*len(covlist[:,0,0,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[:,0,0,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            i = 0
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for n_tomo in range(m_tomo, len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
        if is_i_smaller_j and  not is_m_smaller_n:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2)*len(covlist[:,0,0,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[:,0,0,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for n_tomo in range(len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
        if not is_i_smaller_j and  not is_m_smaller_n:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0])))*len(covlist[:,0,0,0,0,0,0,0])
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[:,0,0,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))   
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(len(covlist[0,0,0,0,0,:,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for n_tomo in range(len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
        return covariance
    
    def __write_cov_matrix_new(self,
                                obs_dict,
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

        cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(obslength)]
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
                                          [covariance_ggmm.T, covariance_mmgm, covariance_mmgm]])
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

        for i in range(len(cov2d[:,0])):
            for j in range(len(cov2d[:,0])):
                cov2d[j,i] = cov2d[i,j]
        if len(np.where(np.linalg.eig(cov2d)[0] < 0)[0]) > 0:
            print("ALARM: The resulting covariance matrix has negative eigenvalues")
            print("Try to adjust the accuracy settings in the config file:")
            print("For configuration space covariance reduce theta_accuracy and increase integration_intervals, usually a factor of 2 is enough.")
            print("For bandpower covariance reduce reduce bandpower_accuracy.")
            print("For COSEBI covariance reduce reduce En_accuracy.")
        if self.plot:
            self.plot_corrcoeff_matrix(
                obs_dict, cov2d, cov_diag, proj_quant, n_tomo_clust, 
                n_tomo_lens, sampledim, self.plot)
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
        hdr_str += 'with '
        if n_tomo_clust is not None:
            hdr_str += str(n_tomo_clust) + ' tomographic clustering bins and '
        if n_tomo_lens is not None:
            hdr_str += str(n_tomo_lens) + ' tomographic lensing bins and '
        hdr_str += str(len(proj_quant)) + ' elements per tomographic bin'
        if 'matrix' in self.style:
            print("Writing matrix output file.")
            fn = self.filename[self.style.index('matrix')]
            np.savetxt(fn, cov2d, fmt='%.6e', delimiter=' ',
                       newline='\n', header=hdr_str, comments='# ')

    def __write_cov_matrix(self,
                           obs_dict,
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

        cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(obslength)]
        cov_diag, cov_off = [], []
        if obslength == 6:
            # 'gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm'
            if gg:
                cov_diag.append(self.project_to_2d(
                    obslist[0], cov[0], n_tomo_clust, n_tomo_clust))
                if gm:
                    cov_diag.append(self.project_to_2d(
                        obslist[3], cov[3], n_tomo_clust, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[1], cov[1], n_tomo_clust, n_tomo_lens))
                    if mm:
                        cov_diag.append(self.project_to_2d(
                            obslist[5], cov[5], n_tomo_lens, n_tomo_lens))
                        cov_off.append(self.project_to_2d(
                            obslist[2], cov[2], n_tomo_clust, n_tomo_lens))
                        cov_off.append(self.project_to_2d(
                            obslist[4], cov[4], n_tomo_clust, n_tomo_lens))
                elif mm:
                    cov_diag.append(self.project_to_2d(
                        obslist[5], cov[5], n_tomo_lens, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[2], cov[2], n_tomo_clust, n_tomo_lens))
            elif gm:
                cov_diag.append(self.project_to_2d(
                    obslist[3], cov[3], n_tomo_clust, n_tomo_lens))
                if mm:
                    cov_diag.append(self.project_to_2d(
                        obslist[5], cov[5], n_tomo_lens, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[4], cov[4], n_tomo_clust, n_tomo_lens))
            elif mm:
                cov_diag.append(self.project_to_2d(
                    obslist[5], cov[5], n_tomo_lens, n_tomo_lens))
        elif obslength == 10:
            # 'gggg', 'gggm', 'ggxip', 'ggxim', 'gmgm', 'gmxip', 
            # 'gmxim', 'xipxip', 'xipxim', 'ximxim'
            if gg:
                cov_diag.append(self.project_to_2d(
                    obslist[0], cov[0], n_tomo_clust, n_tomo_clust))
                if gm:
                    cov_diag.append(self.project_to_2d(
                        obslist[4], cov[4], n_tomo_clust, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[1], cov[1], n_tomo_clust, n_tomo_lens))
                    if xipp:
                        cov_diag.append(self.project_to_2d(
                            obslist[7], cov[7], n_tomo_lens, n_tomo_lens))
                        cov_off.append(self.project_to_2d(
                            obslist[2], cov[2], n_tomo_clust, n_tomo_lens))
                        if ximm:
                            cov_diag.append(self.project_to_2d(
                                obslist[9], cov[9], n_tomo_lens, n_tomo_lens))
                            cov_off.append(self.project_to_2d(
                                obslist[3], cov[3], n_tomo_clust, n_tomo_lens))
                            cov_off.append(self.project_to_2d(
                                obslist[5], cov[5], n_tomo_clust, n_tomo_lens))
                            cov_off.append(self.project_to_2d(
                                obslist[6], cov[6], n_tomo_clust, n_tomo_lens))
                            if xipm:
                                cov_off.append(self.project_to_2d(
                                    obslist[8], cov[8], n_tomo_lens, n_tomo_lens))
                elif xipp:
                    cov_diag.append(self.project_to_2d(
                        obslist[7], cov[7], n_tomo_lens, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[2], cov[2], n_tomo_clust, n_tomo_lens))
                    if ximm:
                        cov_diag.append(self.project_to_2d(
                            obslist[9], cov[9], n_tomo_lens, n_tomo_lens))
                        cov_off.append(self.project_to_2d(
                            obslist[3], cov[3], n_tomo_clust, n_tomo_lens))
                        if xipm:
                            cov_off.append(self.project_to_2d(
                                obslist[8], cov[8], n_tomo_lens, n_tomo_lens))
            elif gm:
                cov_diag.append(self.project_to_2d(
                    obslist[4], cov[4], n_tomo_clust, n_tomo_lens))
                if xipp:
                    cov_diag.append(self.project_to_2d(
                        obslist[7], cov[7], n_tomo_lens, n_tomo_lens))
                    cov_off.append(self.project_to_2d(
                        obslist[5], cov[5], n_tomo_clust, n_tomo_lens))
                    if ximm:
                        cov_diag.append(self.project_to_2d(
                            obslist[9], cov[9], n_tomo_lens, n_tomo_lens))
                        cov_off.append(self.project_to_2d(
                            obslist[6], cov[6], n_tomo_clust, n_tomo_lens))
                        if xipm:
                            cov_off.append(self.project_to_2d(
                                obslist[8], cov[8], n_tomo_lens, n_tomo_lens))
            elif xipp:
                cov_diag.append(self.project_to_2d(
                    obslist[7], cov[7], n_tomo_lens, n_tomo_lens))
                if ximm:
                    cov_diag.append(self.project_to_2d(
                        obslist[9], cov[9], n_tomo_lens, n_tomo_lens))
                    if xipm:
                        cov_off.append(self.project_to_2d(
                            obslist[8], cov[8], n_tomo_lens, n_tomo_lens))
            elif ximm:
                cov_diag.append(self.project_to_2d(
                    obslist[9], cov[9], n_tomo_lens, n_tomo_lens))
            elif xipm:
                cov_diag.append(self.project_to_2d(
                    obslist[8], cov[8], n_tomo_lens, n_tomo_lens))

        cov2d = self.__mesh_2d_matrix_together(cov_diag, cov_off)

        if self.plot:
            self.plot_corrcoeff_matrix(
                obs_dict, cov2d, cov_diag, proj_quant, n_tomo_clust, 
                n_tomo_lens, sampledim, self.plot)
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
        hdr_str += 'with '
        if n_tomo_clust is not None:
            hdr_str += str(n_tomo_clust) + ' tomographic clustering bins and '
        if n_tomo_lens is not None:
            hdr_str += str(n_tomo_lens) + ' tomographic lensing bins and '
        hdr_str += str(len(proj_quant)) + ' elements per tomographic bin'
        if 'matrix' in self.style:
            print("Writing matrix output file.")
            fn = self.filename[self.style.index('matrix')]
            np.savetxt(fn, cov2d, fmt='%.6e', delimiter=' ',
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
                        obslength,
                        gauss,
                        nongauss,
                        ssc):
        if self.has_gauss:
            for idx in range(len(gauss)):
                try:
                    sampledim = (gauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_nongauss:
            for idx in range(len(nongauss)):
                try:
                    sampledim = (nongauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_ssc:
            for idx in range(len(ssc)):
                try:
                    sampledim = (ssc[idx].shape)[2]
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
                    for s2 in range(s1, sampledim):
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
                    for t1_1 in range(tomo1):
                        for t2_1 in range(t1_1, tomo1):
                            for t1_2 in range(tomo1):
                                for t2_2 in range(t1_2, tomo1):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1_1, t2_1, t1_2, t2_2]
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
                    for t1_1 in range(tomo1):
                        for t2_1 in range(tomo2):
                            for t1_2 in range(tomo1):
                                for t2_2 in range(tomo2):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1_1, t2_1, t1_2, t2_2]
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
                        for t1_1 in range(tomo1):
                            for t2_1 in range(t1_1,tomo1):
                                for t1_2 in range(tomo3):
                                    for t2_2 in range(tomo4):
                                        cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                            covmatrix[:,:,s1, s2, t1_1, t2_1, t1_2, t2_2]
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
                        for t1_1 in range(tomo1):
                            for t2_1 in range(tomo1):
                                for t1_2 in range(tomo3):
                                    for t2_2 in range(t1_2,tomo4):
                                        cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                            covmatrix[:,:,s1, s2, t1_1, t2_1, t1_2, t2_2]
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
                    for t1_1 in range(tomo1):
                        for t2_1 in range(t1_1, tomo1):
                            for t1_2 in range(tomo2):
                                for t2_2 in range(t1_2, tomo2):
                                    cov2d[idx1:idx1+ellnum, idx2:idx2+ellnum] = \
                                        covmatrix[:,:,s1, s2, t1_1, t2_1, t1_2, t2_2]
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
                    Cell_str += 'Cell_gg_' + str(sm+1) + '\t\t\t\t'
            olist_gg = []
            olist_gg.append("#ell\t\ttomo_i\ttomo_j\t"+Cell_str)
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_clust):
                    for tj in range(n_tomo_clust):
                        ostr = ostr_format \
                            % (ell, ti+1, tj+1)
                        for i_sample in range(sampledim):
                            ostr += '%10e\t\t\t' % Cell_gg[ellidx, i_sample, ti, tj]
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
                    Cell_str += 'Cell_gkappa_' + str(sm+1) + '\t\t\t'
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
            if sampledim == 1:
                Cell_str = 'Cell_kappakappa'     
            else:
                Cell_str = ''
                for sm in range(sampledim):
                    Cell_str += 'Cell_kappakappa_' + str(sm+1) + '\t\t'
            olist_mm = []
            olist_mm.append("#ell\t\ttomo_i\ttomo_j\t"+Cell_str)
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_lens):
                    for tj in range(n_tomo_lens):
                        ostr = ostr_format \
                            % (ell, ti+1, tj+1)
                        for i_sample in range(sampledim):
                            ostr += '%10e\t\t\t' % Cell_mm[ellidx, i_sample, ti, tj]
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
