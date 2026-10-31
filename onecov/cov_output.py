import gc
import os

# from astropy.io import fits
#plot stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('font', family='sans-serif')
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

class Output: 
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

    # labels of the C_ell covariance with intrinsic B-modes (E/B layout):
    # E (B) is the E-mode (B-mode) shear spectrum
    CELL_EB_LABELS = {'ggxip': 'ggmmE', 'ggxim': 'ggmmB',
                      'gmxip': 'gmmmE', 'gmxim': 'gmmmB',
                      'xipxip': 'mmEmmE', 'xipxim': 'mmEmmB', 'ximxim': 'mmBmmB'}

    def __init__(self, output_dict, projected_clust = None, projected_lens = None):
        self.filename = output_dict['file']
        self.__check_filetype()
        self.style = output_dict['style']
        self.plot = output_dict['make_plot']
        self.Cellfile = output_dict['Cell']
        self.tex = output_dict['use_tex']
        self.save_as_binary = output_dict['save_as_binary']
        self.list_style_spatial_first = output_dict['list_style_spatial_first']
        self.adjacent = 1
        if output_dict['adjacent_clustering_bins']:
            self.adjacent = 2
        self.projected_lens = projected_lens
        self.projected_clust = projected_clust
        self.N_stellar_mass_bins = None

    
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
    
    def _selected_tomo_combinations(self,
                                    obs_dict,
                                    n_tomo_clust,
                                    n_tomo_lens):
        """
        Returns, per probe (gg, gm, mm), the list of all tomographic bin
        combinations in the order of the covariance matrix and the list of
        the combinations selected in the config file. Probes that are not
        calculated get empty lists. With csmf_diagonal_lenses the clustering
        block only contains the combinations (i, i).
        """
        obs = obs_dict['observables']
        all_gg = [[i, j] for i in range(n_tomo_clust)
                  for j in range(i, n_tomo_clust)]
        if getattr(self, 'csmf_diagonal_lenses', False):
            all_gg = [[i, i] for i in range(n_tomo_clust)]
        all_gm = [[i, j] for i in range(n_tomo_clust)
                  for j in range(n_tomo_lens)]
        all_mm = [[i, j] for i in range(n_tomo_lens)
                  for j in range(i, n_tomo_lens)]
        probes = []
        for flag, key, all_comb in (('clustering', 'combinations_clustering', all_gg),
                                    ('ggl', 'combinations_ggl', all_gm),
                                    ('cosmic_shear', 'combinations_lensing', all_mm)):
            if not obs[flag]:
                probes.append(([], []))
            elif obs[key] is None:
                probes.append((all_comb, all_comb))
            else:
                probes.append((all_comb, obs[key]))
        return probes

    def _sample_factor(self,
                       sampledim):
        """
        Number of stellar mass samples stored per tomographic combination
        (and spatial bin) in the gg and gm blocks. The samples are tied to
        the tomographic bins in the diagonal lens setup.
        """
        if getattr(self, 'csmf_diagonal_lenses', False) or sampledim is None:
            return 1
        return sampledim

    @staticmethod
    def _reduce_covariance(blocks,
                           full_covariance):
        """
        Removes the tomographic bin combinations that are not selected from
        the full matrix, everything after the blocks (the stellar mass
        function) is kept. blocks lists the consecutive blocks of the matrix as tuples
        (probe, all combinations, selected combinations, spatial bins per
        summary, stellar mass samples per combination, None for cosmic shear).
        Within a block the summaries follow each other, each of them ordered
        as (tomographic combination, stellar mass sample, spatial bin).
        """
        indices = []
        offset = 0
        for _, all_comb, selected, n_spatial, n_sample in blocks:
            for n_spat in n_spatial:
                n_spat = n_spat*(n_sample or 1)
                for select in selected:
                    if select in all_comb:
                        start = offset + all_comb.index(select)*n_spat
                        indices.extend(range(start, start + n_spat))
                offset += len(all_comb)*n_spat
        indices.extend(range(offset, len(full_covariance)))
        return full_covariance[np.ix_(indices, indices)]

    @staticmethod
    def _reduced_block_order(probe, combinations, n_spatial, n_sample, diagonal):
        """
        One line of the header of the reduced matrix listing the order of the
        entries of a block: the selected combinations (in the order of the
        config file) outer, sample in the middle, spatial bin inner. With
        diagonal the sample is the first tomographic bin.
        """
        ranges = []
        for t1, t2 in combinations:
            pair = '_{' + str(t1 + 1) + str(t2 + 1) + '}('
            if n_sample is None:
                first, last = probe + pair, probe + pair
            else:
                s_first, s_last = (t1 + 1, t1 + 1) if diagonal else (1, n_sample)
                first = probe + '^{' + str(s_first) + '}' + pair
                last = probe + '^{' + str(s_last) + '}' + pair
            ranges.append(first + '1) ... ' + last + str(n_spatial) + ')')
        return ' '.join(ranges) + '\n'

    def _reduced_header(self,
                        blocks,
                        second_mm,
                        n_entries,
                        obs_dict):
        """
        Header of the matrix with the selected tomographic bin combinations
        for the blocks of _reduce_covariance. second_mm names what a second
        cosmic shear block holds.
        """
        notation = {
            'gg': "galaxy clustering: gg^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n",
            'gm': "galaxy galaxy lensing: gm^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n",
            'mm': "cosmic shear: mm_{tomographic bin i tomographic bin j}(spatial bin k)\n"}
        diagonal = bool(getattr(self, 'csmf_diagonal_lenses', False))
        hdr_str = 'Covariance matrix (' + str(n_entries) + ' x ' + str(n_entries) + \
            ') with only the tomographic bin combinations selected in the config file, in the order given there.\n'
        hdr_str += 'Tomographic bins are counted from 1 here, but from 0 in the config file.\n'
        if self.has_csmf:
            csmf_description, csmf_notation, csmf_order = self._csmf_header(obs_dict)
            hdr_str += csmf_description
        hdr_str += "The structure of the diagonal blocks is, using the following notation\n"
        probes = []
        for probe, all_comb, selected, n_spatial, _ in blocks:
            if probe not in probes and n_spatial and any(c in all_comb for c in selected):
                probes.append(probe)
                hdr_str += notation[probe]
        if self.has_csmf:
            hdr_str += csmf_notation
        repeated = []
        for probe, all_comb, selected, n_spatial, n_sample in blocks:
            if probe not in probes:
                continue
            if probe in repeated:
                hdr_str += "Followed by " + second_mm + "\n"
            repeated.append(probe)
            combinations = [c for c in selected if c in all_comb]
            for i_summary, n_spat in enumerate(n_spatial):
                if i_summary > 0:
                    hdr_str += "Followed by the second " + probe + " summary\n"
                hdr_str += self._reduced_block_order(probe, combinations, n_spat, n_sample, diagonal)
        if self.has_csmf:
            hdr_str += csmf_order
        return hdr_str

    def _write_reduced_matrix(self,
                              covariance,
                              header):
        """
        Writes the matrix with the selected tomographic bin combinations next
        to the full matrix, in the same format.
        """
        name, extension = os.path.splitext(self.filename[self.style.index('matrix')])
        if self.save_as_binary:
            print("Saving covariance with selected tomographic bins in", name + "_reduced.npz")
            np.savez(name + "_reduced", covariance = covariance, header = header)
        else:
            fn_reduced = name + "_reduced" + extension
            print("Saving covariance with selected tomographic bins in", fn_reduced)
            np.savetxt(fn_reduced, covariance, fmt='%.6e', delimiter=' ',
                    newline='\n', header=header, comments='# ')

    def select_tomographic_bins(self,
                                obs_dict,
                                proj_quant,
                                n_tomo_clust,
                                n_tomo_lens,
                                full_covariance,
                                second_shear = False,
                                sampledim = 1):
        """
        Reduced covariance with the tomographic bin combinations of the
        config file and its header. second_shear states that the matrix
        contains a second cosmic shear block (xi_minus or B-modes) after the
        first one, which is reduced in the same way.
        """
        n_spatial_clustering = len(proj_quant)
        n_spatial_ggl = len(proj_quant)
        n_spatial_shear = len(proj_quant)
        if np.any(self.projected_clust):
            n_spatial_clustering = len(self.projected_clust)
            n_spatial_ggl = len(self.projected_clust)
        if np.any(self.projected_lens):
            n_spatial_shear = len(self.projected_lens)

        (all_gg, sel_gg), (all_gm, sel_gm), (all_mm, sel_mm) = \
            self._selected_tomo_combinations(obs_dict, n_tomo_clust, n_tomo_lens)
        n_sample = self._sample_factor(sampledim)
        blocks = [('gg', all_gg, sel_gg, [n_spatial_clustering], n_sample),
                  ('gm', all_gm, sel_gm, [n_spatial_ggl], n_sample),
                  ('mm', all_mm, sel_mm, [n_spatial_shear], None)]
        if second_shear:
            blocks.append(('mm', all_mm, sel_mm, [n_spatial_shear], None))
        reduced = self._reduce_covariance(blocks, full_covariance)
        xi_pm = obs_dict['observables']['est_shear'] == 'xi_pm'
        second_mm = 'the second mm summary (' + ('xi_minus' if xi_pm else 'B-modes') + ')'
        return reduced, self._reduced_header(blocks, second_mm, len(reduced), obs_dict)

    def select_tomographic_bins_arbitrary(self,
                                          obs_dict,
                                          summary,
                                          n_tomo_clust,
                                          n_tomo_lens,
                                          full_covariance,
                                          sampledim = 1):
        """
        Reduced covariance for arbitrary summary statistics and its header.
        The matrix holds gg, gm, the first and the second component of mm
        (E/B); a probe with two summaries has them one after the other.
        """
        n_sample = self._sample_factor(sampledim)

        def spatial_bins(probe, weights):
            if weights is None:
                return []
            n_first = summary['arb_number_first_summary_' + probe]
            if summary['number_summary_' + probe] > 1:
                return [n_first, len(weights) - n_first]
            return [len(weights)]

        (all_gg, sel_gg), (all_gm, sel_gm), (all_mm, sel_mm) = \
            self._selected_tomo_combinations(obs_dict, n_tomo_clust, n_tomo_lens)
        n_mm = spatial_bins('mm', summary['WL_mmE'])
        blocks = [('gg', all_gg, sel_gg, spatial_bins('gg', summary['WL_gg']), n_sample),
                  ('gm', all_gm, sel_gm, spatial_bins('gm', summary['WL_gm']), n_sample),
                  ('mm', all_mm, sel_mm, n_mm, None),
                  ('mm', all_mm, sel_mm, n_mm, None)]
        reduced = self._reduce_covariance(blocks, full_covariance)
        return reduced, self._reduced_header(
            blocks, 'the second component summaries of mm', len(reduced), obs_dict)

    def __split_off_csmf(self,
                         gauss,
                         nongauss,
                         ssc,
                         n_csmf):
        """
        Removes the covariance of the stellar mass function and its
        cross-covariance with the LSS probes (the last n_csmf entries of the
        Gaussian, non-Gaussian and super-sample terms) from the lists. The
        sum is kept in self.conditional_stellar_mass_function_cov, the
        three parts in self.csmf_split for the split output.
        """
        self.csmf_split = {'gauss': list(gauss[-n_csmf:]),
                           'nongauss': list(nongauss[-n_csmf:]),
                           'ssc': list(ssc[-n_csmf:])}
        self.conditional_stellar_mass_function_cov = [
            term_g + term_ng + term_ssc for term_g, term_ng, term_ssc
            in zip(self.csmf_split['gauss'], self.csmf_split['nongauss'], self.csmf_split['ssc'])]
        return gauss[:-n_csmf], nongauss[:-n_csmf], ssc[:-n_csmf]

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
        if obs_dict['ELLspace']['n_spec'] is not None and obs_dict['ELLspace']['n_spec'] != 0:
            self.style = "matrix"
            self.filename = [s for s in self.filename if 'matrix' in s or 'mat' in s]


        obslist, obsbool, obslength = self.__get_obslist(obs_dict)
        gg, gm, mm = obsbool[0], obsbool[3], obsbool[5]
        xipp, xipm, ximm = None, None, None
        mult = 1
        self.conditional_stellar_mass_function_cov = []
        if self.has_csmf:
            gauss, nongauss, ssc = self.__split_off_csmf(gauss, nongauss, ssc, 5)
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
        if obs_dict['ELLspace']['n_spec'] != 0 and obs_dict['ELLspace']['n_spec'] is not None:
            self.style = "matrix"
            self.filename = [s for s in self.filename if 'matrix' in s or 'mat' in s]
            print("For 6x2pt type covariances matrices, only the matrix output is supported.")
            print("The diagonals of the matrix are in the following order: 'ss', 'sp', 'pp', 'sm', 'pm', 'mm'.")
            print("Here 'p','s' and 'm' stand for photometric clustering, spectroscopic clustering and lensing respectively.")
        obslist, obsbool, obslength = self.__get_obslist(obs_dict)
        gg, gm, mm = obsbool[0], obsbool[3], obsbool[5]
        xipp, xipm, ximm = None, None, None
        mult = 1
        self.conditional_stellar_mass_function_cov = []
        if self.has_csmf:
            gauss, nongauss, ssc = self.__split_off_csmf(gauss, nongauss, ssc, 4 if self.is_cell else 5)
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

    @staticmethod
    def _gm_first(obs, *labels):
        """
        Labels (x1, x2, s1, s2, t1, t2, t3, t4) of a list entry. The
        cross-covariance of galaxy-galaxy lensing and cosmic shear of the
        projected estimators ('gmxip', 'gmxim') is stored cosmic shear first;
        its labels are reordered so that galaxy-galaxy lensing comes first,
        as in the name (and as for 'ggxip', 'ggxim').
        """
        if obs in ('gmxip', 'gmxim'):
            x1, x2, s1, s2, t1, t2, t3, t4 = labels
            return x2, x1, s2, s1, t3, t4, t1, t2
        return labels

    @staticmethod
    def _has_negative_eigenvalues(covmatrix):
        """
        Whether the (symmetric) covariance matrix has negative eigenvalues
        beyond rounding. The test uses the correlation matrix, so it does not
        depend on the units of the observables (e.g. dimensionful COSEBIs and
        Psi-stats differ by many orders of magnitude): eigenvalues below
        -1e-12 (an exactly degenerate matrix has eigenvalues of order -1e-16).
        A non-positive diagonal element is always flagged.
        """
        diagonal = np.diag(covmatrix)
        if np.any(diagonal <= 0):
            return bool(np.any(diagonal < 0))
        scale = 1/np.sqrt(diagonal)
        eigenvalues = np.linalg.eigvalsh(covmatrix*scale[:, None]*scale[None, :])
        return bool(np.any(eigenvalues < -1e-12))

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
                         vmin=-limit, vmax=limit)
        fig.colorbar(cbar, location='bottom', shrink=.775, aspect=30, pad=0.055).ax.tick_params(axis='x', direction='in')
        ax.text(len(covmatrix)/2, -6*ratio, 'Correlation coefficients', fontsize=16, ha='center', va='center')

        
        labels_position = []
        labels_position_y = []
        labels_text = []
        position = 0
        old_position = 0
        # with csmf_diagonal_lenses only sample i in clustering bin i (auto
        # correlations only for clustering) enters the matrix
        diagonal_lenses = bool(getattr(self, 'csmf_diagonal_lenses', False))
        if gg:
            if np.any(self.projected_clust):
                proj_quant = self.projected_clust
            sub_position_tomo = 0
            if diagonal_lenses:
                for sub_tomo in range(int(n_tomo_clust)):
                    sub_position_tomo += len(proj_quant)
                    ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                position += len(proj_quant)*n_tomo_clust
            else:
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
            gm_sampledim = 1 if diagonal_lenses else sampledim
            for sub_tomo in range(int(n_tomo_clust*n_tomo_lens)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += len(proj_quant)*gm_sampledim
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                if gm_sampledim > 1:
                    for sub_sample in range(gm_sampledim):
                        sub_position_sample += len(proj_quant)
                        ax.axhline(y=len(covmatrix)-sub_position_sample, color='black', linewidth=.15, ls = ":")
                        ax.axvline(x=sub_position_sample, color='black', linewidth=.15, ls = ":")
            position += len(proj_quant)*gm_sampledim*n_tomo_clust*n_tomo_lens
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

        if self.has_csmf:
            self._plot_csmf_block(ax, covmatrix, position, labels_position, labels_position_y, labels_text)

        ax.xaxis.tick_top()

        plt.yticks(labels_position_y, labels_text)
        plt.xticks(labels_position, labels_text)

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)

        #print("Plotting correlation matrix")
   

    def _plot_csmf_block(self,
                         ax,
                         covmatrix,
                         start,
                         labels_position,
                         labels_position_y,
                         labels_text):
        """
        Labels the block of the stellar mass function (SMF) at the end of the
        correlation matrix, which starts at row start. It has one row per
        stellar mass bin and tomographic bin of the SMF (tomographic bin
        outer, mass bin inner; one row per stellar mass bin with
        csmf_diagonal), the tomographic bins are separated by dashed lines.
        The SMF is the bin-averaged SMF per dex, Phi_mu^i = N_mu^i/(Delta
        log10 M*_mu V_max) (as Phi_mu^i in Dvornik et al. 2023).
        """
        start, end = int(round(start)), len(covmatrix)
        n_tomo = (end - start)//self.N_stellar_mass_bins
        for sub_tomo in range(1, n_tomo):
            sub_position = start + sub_tomo*self.N_stellar_mass_bins
            ax.axhline(y=end - sub_position, color='black', linewidth=.3, ls = "--")
            ax.axvline(x=sub_position, color='black', linewidth=.3, ls = "--")
        labels_position.append(start + (end - start)/2)
        labels_position_y.append(end - start - (end - start)/2)
        labels_text.append(r'$\Phi^i_\mu$' if n_tomo > 1 else r'$\Phi_\mu$')

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
                         vmin=-limit, vmax=limit)
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
                if self.csmf_diagonal_lenses:
                    for sub_tomo in range(int(n_tomo_clust)):
                        sub_position_sample = sub_position_tomo
                        sub_position_tomo += gg_summary_length[i]
                        ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                        ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    position += gg_summary_length[i]*sampledim
                    labels_position.append(position - gg_summary_length[i]*sampledim/2)
                    old_position = position
                    labels_position_y.append(len(covmatrix) - (position - gg_summary_length[i]*sampledim/2))
                else:
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
                if self.csmf_diagonal_lenses:
                    for sub_tomo in range(int(n_tomo_clust*n_tomo_lens)):
                        sub_position_sample = sub_position_tomo
                        sub_position_tomo += gm_summary_length[i]
                        ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                        ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
                    position += gm_summary_length[i]*n_tomo_clust*n_tomo_lens
                    labels_position.append(position - gm_summary_length[i]*n_tomo_clust*n_tomo_lens/2)
                    labels_position_y.append(len(covmatrix) - (position - gm_summary_length[i]*n_tomo_clust*n_tomo_lens/2))
                    old_position = position  
                else:
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
        if self.has_csmf:
            self._plot_csmf_block(ax, covmatrix, position, labels_position, labels_position_y, labels_text)
        ax.xaxis.tick_top()

        plt.yticks(labels_position_y, labels_text)
        plt.xticks(labels_position, labels_text, rotation=90)

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)

        #print("Plotting correlation matrix")

    
    def plot_corrcoeff_matrix_6x2pt(self,
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
                            vmin=-limit, vmax=limit)
        fig.colorbar(cbar, location='bottom', shrink=.775, aspect=30, pad=0.055).ax.tick_params(axis='x', direction='in')
        ax.text(len(covmatrix)/2, -6*ratio, 'Correlation coefficients', fontsize=16, ha='center', va='center')

        
        labels_position = []
        labels_position_y = []
        labels_text = []
        position = 0
        old_position = 0
        if gg:
            sub_position_tomo = 0
            if self.adjacent != 1:
                loopmax = int(2*self.six_times_two_s_tomo_bin -1)
            else:
                loopmax = self.six_times_two_s_tomo_bin
            for sub_tomo in range(loopmax):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_ss_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_ss_spatial_index*loopmax
            old_position = position
            labels_position.append(position/2)
            labels_position_y.append(len(covmatrix) - position/2)
            labels_text.append(r'$C^{\mathrm{ss}}_\mathrm{gg}$')
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        
            for sub_tomo in range(int(self.six_times_two_p_tomo_bin**self.six_times_two_s_tomo_bin)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_ps_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_p_tomo_bin*self.six_times_two_s_tomo_bin*self.six_times_two_ps_spatial_index
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            labels_text.append(r'$C^{\mathrm{sp}}_\mathrm{gg}$')
            old_position = position
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
            
            for sub_tomo in range(int(self.six_times_two_p_tomo_bin*(self.six_times_two_p_tomo_bin + 1)/2)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_pp_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_pp_spatial_index*self.six_times_two_p_tomo_bin*(self.six_times_two_p_tomo_bin + 1)/2
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            labels_text.append(r'$C^{\mathrm{pp}}_\mathrm{gg}$')
            old_position = position
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        
        if gm:
            sub_position_tomo = old_position
            for sub_tomo in range(int(self.six_times_two_s_tomo_bin*self.six_times_two_m_tomo_bin)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_sm_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_ps_spatial_index*self.six_times_two_s_tomo_bin*self.six_times_two_m_tomo_bin
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            labels_text.append(r'$C^{\mathrm{s}}_\mathrm{gm}$')
            old_position = position
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")


            for sub_tomo in range(int(self.six_times_two_p_tomo_bin*self.six_times_two_m_tomo_bin)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_pm_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_pm_spatial_index*self.six_times_two_p_tomo_bin*self.six_times_two_m_tomo_bin
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            labels_text.append(r'$C^{\mathrm{p}}_\mathrm{gm}$')
            old_position = position
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        
        ax.xaxis.tick_top()
        
        if mm:
            sub_position_tomo = old_position
            for sub_tomo in range(int(self.six_times_two_m_tomo_bin*(self.six_times_two_m_tomo_bin+1)/2)):
                sub_position_sample = sub_position_tomo
                sub_position_tomo += self.six_times_two_mm_spatial_index
                ax.axhline(y=len(covmatrix)-sub_position_tomo, color='black', linewidth=.3, ls = "--")
                ax.axvline(x=sub_position_tomo, color='black', linewidth=.3, ls = "--")
            position += self.six_times_two_mm_spatial_index*self.six_times_two_m_tomo_bin*(self.six_times_two_m_tomo_bin+1)/2
            labels_position.append(old_position + (position- old_position)/2)
            labels_position_y.append(len(covmatrix) - old_position - (position- old_position)/2)
            labels_text.append(r'$C_\mathrm{mm}$')
            old_position = position
            ax.axhline(y=len(covmatrix)-position, color='black', linewidth=.5, ls = "-")
            ax.axvline(x=position, color='black', linewidth=.5, ls = "-")
        plt.yticks(labels_position_y, labels_text, rotation=45)
        plt.xticks(labels_position, labels_text,rotation=45,)

        if filename is not None:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)

    #print("Plotting correlation matrix")

    @staticmethod
    def _csmf_bin_values(obs_dict):
        """
        log10 of the stellar masses of the SMF bins (the centres of the bins
        in stellar mass), only used as labels: the SMF is averaged over each
        bin. The stellar masses have the units of the stellar mass parameters
        of the HoD (e.g. dpow_logm0_cen).
        """
        observables = obs_dict['observables']
        if observables['csmf_log10M_bins_upper'] is not None and observables['csmf_log10M_bins_lower'] is not None:
            lower, upper = 10**np.asarray(observables['csmf_log10M_bins_lower']), 10**np.asarray(observables['csmf_log10M_bins_upper'])
        else:
            edges = 10**np.asarray(observables['csmf_log10M_bins'])
            lower, upper = edges[:-1], edges[1:]
        return np.log10(lower + (upper - lower)/2)

    @staticmethod
    def _block_order(probe, n_spatial, n_tomo_1, n_tomo_2, sampledim=None,
                     unique=False, diagonal=False):
        """
        One line of the matrix output header listing the order of the
        entries of a 2-point block, as written by __create_matrix (and its
        diagonal-clustering variant): tomographic bin pair outer, sample
        (sampledim None: no sample index, cosmic shear) in the middle,
        spatial bin inner. unique: only pairs t1 <= t2; diagonal: sample i
        only in lens bin i (csmf_diagonal_lenses), so t1 is the sample and
        for unique pairs only the auto pairs t1 = t2 are kept.
        """
        entries = []
        for t1 in range(1, n_tomo_1 + 1):
            for t2 in range(t1 if unique else 1, n_tomo_2 + 1):
                if diagonal and sampledim is not None:
                    if not unique or t1 == t2:
                        entries.append((t1, t1, t2))
                else:
                    for s in range(1, (sampledim or 1) + 1):
                        entries.append((s if sampledim else None, t1, t2))

        def label(entry, spatial):
            s, t1, t2 = entry
            sample = '' if s is None else '^{' + str(s) + '}'
            return probe + sample + '_{' + str(t1) + str(t2) + '}(' + str(spatial) + ')'

        # first entry, rest of the first pair, rest of the first row of
        # pairs, everything else
        first_pair = [e for e in entries[1:] if e[1:] == entries[0][1:]]
        first_row = [e for e in entries[1 + len(first_pair):] if e[1] == entries[0][1]]
        rest = entries[1 + len(first_pair) + len(first_row):]
        groups = [entries[:1], first_pair, first_row, rest]
        return ' '.join(label(g[0], 1) + ' ... ' + label(g[-1], n_spatial)
                        for g in groups if g) + '\n'

    def _csmf_header(self, obs_dict):
        """
        The lines of the matrix output header for the stellar mass function
        (SMF): its description with units, its notation and the order of its
        entries (tomographic bin outer, stellar mass bin inner; only stellar
        mass bin i in tomographic bin i with csmf_diagonal).
        """
        n_mass = self.N_stellar_mass_bins
        n_tomo = self.conditional_stellar_mass_function_cov[0].shape[2]
        diagonal = obs_dict['observables']['csmf_diagonal']
        description = 'stellar mass function per dex, averaged over the stellar mass bins, ' \
            'phi^i_mu = N^i_mu/(Delta log10 M*_mu V^i_max,mu) in h^3 Mpc^-3 dex^-1 (V_max in (Mpc/h)^3), with ' \
            + str(n_mass) + ' stellar mass bins and ' + str(n_tomo) + (' tomographic bin' if n_tomo == 1 else ' tomographic bins') \
            + (' (only stellar mass bin i in tomographic bin i).\n' if diagonal else '.\n') \
            + 'The SMF auto-covariance is in (h^3 Mpc^-3 dex^-1)^2, its cross-covariance with a 2-point ' \
            'statistic in h^3 Mpc^-3 dex^-1 times the units of that statistic.\n'
        notation = 'stellar mass function: phi^{tomographic bin i}_{stellar mass bin mu}\n'
        if diagonal:
            order = 'phi^{1}_{1} phi^{2}_{2} ... phi^{' + str(n_mass) + '}_{' + str(n_mass) + '}\n'
        elif n_tomo == 1:
            order = 'phi^{1}_{1} ... phi^{1}_{' + str(n_mass) + '}\n'
        else:
            order = 'phi^{1}_{1} ... phi^{1}_{' + str(n_mass) + '} phi^{2}_{1} ... phi^{' + str(n_tomo) \
                + '}_{' + str(n_mass) + '}\n'
        return description, notation, order

    def _csmf_list_rows(self, cov_dict, obs_dict, ostr_format, x_csmf, x_clust, x_lens, arbitrary=False):
        """
        The rows of the list output for the stellar mass function (SMF): its
        auto-covariance and its cross-covariance with clustering,
        galaxy-galaxy lensing and cosmic shear, with all terms of the split
        output (the Gaussian term of the SMF is its shot noise). The columns
        are x1 = log10 of the stellar mass of the SMF bin and x2 = the
        projected quantity of the LSS probe (theta, ell, mode; the indices
        for arbitrary summaries), s1 = 1 (no sample for the SMF), s2 = the
        galaxy sample, tomoi = tomoj = the tomographic bin of the SMF and
        tomok, tomol = the bins of the LSS probe (only tomok <= tomol for
        clustering and cosmic shear, and only sample i in clustering bin i
        with csmf_diagonal_lenses). The SMF is per dex and averaged over the
        stellar mass bins (in h^3 Mpc^-3 dex^-1), x1 (and x2 for csmfcsmf)
        is written with more digits than the other spatial indices, since
        neighbouring narrow bins could not be told apart otherwise.
        """
        mass_format = ostr_format.replace('%.2e', '%.4e', 1)
        mass_mass_format = mass_format.replace('%.2e', '%.4e', 1)
        terms = self.conditional_stellar_mass_function_cov
        parts = getattr(self, 'csmf_split', None)
        if getattr(self, 'is_cell', False):
            names = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmm"]
        elif obs_dict['observables']['est_shear'] == 'xi_pm' and not arbitrary:
            names = ["csmfcsmf", "csmfgg", "csmfgm", "csmfxip", "csmfxim"]
        else:
            names = ["csmfcsmf", "csmfgg", "csmfgm", "csmfmmE", "csmfmmB"]
        diagonal_lenses = bool(obs_dict['observables'].get('csmf_diagonal_lenses'))
        rows = []

        def part(term, k, index):
            if parts is None or not isinstance(parts[term][k], np.ndarray):
                return 0.0
            return parts[term][k][index]

        def row(name, x1, x2, s2, tc1, tc2, t3, t4, k, index):
            cov = terms[k][index]
            gauss, nongauss, ssc = part('gauss', k, index), part('nongauss', k, index), part('ssc', k, index)
            if cov_dict['split_gauss']:
                values = (cov, 0.0, 0.0, gauss, nongauss, ssc)
            else:
                values = (cov, gauss, nongauss, ssc)
            row_format = mass_mass_format if name == "csmfcsmf" else mass_format
            return row_format % ((name, x1, x2, 1, s2, tc1, tc2, t3, t4) + values)

        for k, term in enumerate(terms):
            if not isinstance(term, np.ndarray) or k >= len(names):
                continue
            if k == 0:
                n_mass, n_tomo = term.shape[0], term.shape[2]
                x_mass = np.arange(n_mass) if arbitrary or x_csmf is None else x_csmf
                for tc1 in range(n_tomo):
                    for tc2 in range(n_tomo):
                        for mu1 in range(n_mass):
                            for mu2 in range(n_mass):
                                rows.append(row(names[k], x_mass[mu1], x_mass[mu2], 1, tc1 + 1, tc2 + 1, tc1 + 1, tc2 + 1,
                                                k, (mu1, mu2, tc1, tc2)))
                continue
            n_r, n_mass, n_sample, n_tomo, n_t1, n_t2 = term.shape
            x_mass = np.arange(n_mass) if arbitrary or x_csmf is None else x_csmf
            x_probe = x_lens if k >= 3 else x_clust
            if arbitrary or x_probe is None:
                x_probe = np.arange(n_r)
            symmetric = names[k] not in ("csmfgm",)
            for tc in range(n_tomo):
                for mu in range(n_mass):
                    for i_sample in range(n_sample):
                        for t1 in range(n_t1):
                            for t2 in range(t1 if symmetric else 0, n_t2):
                                if diagonal_lenses and k in (1, 2) and (t1 != i_sample or (k == 1 and t2 != i_sample)):
                                    continue
                                for i_r in range(n_r):
                                    rows.append(row(names[k], x_mass[mu], x_probe[i_r], i_sample + 1, tc + 1, tc + 1, t1 + 1, t2 + 1,
                                                    k, (i_r, mu, i_sample, tc, t1, t2)))
        return rows

    @staticmethod
    def _diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
        """
        Whether an entry (samples i_s1, i_s2, tomographic bins t1..t4) of the
        list output of observable obs is measured with csmf_diagonal_lenses,
        i.e. every galaxy sample is in the clustering bin of the same number
        (both bins of a gg pair, the lens bin of a gm pair). The gm x shear
        arrays (gmxip, gmxim, mmgm) have the shear pair first.
        """
        pairs = {'gggg': ('gg', 'gg'), 'gggm': ('gg', 'gm'), 'ggmm': ('gg', 'mm'),
                 'ggxip': ('gg', 'mm'), 'ggxim': ('gg', 'mm'), 'gmgm': ('gm', 'gm'),
                 'gmxip': ('mm', 'gm'), 'gmxim': ('mm', 'gm'), 'mmgm': ('mm', 'gm')}
        first, second = pairs.get(obs, ('mm', 'mm'))
        for kind, sample, ta, tb in ((first, i_s1, t1, t2), (second, i_s2, t3, t4)):
            if kind != 'mm' and (ta != sample or (kind == 'gg' and tb != sample)):
                return False
        return True

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
        # with csmf_diagonal_lenses only sample i in clustering bin i is
        # measured: the other entries are not written
        diagonal_lenses = bool(obs_dict['observables'].get('csmf_diagonal_lenses')) and \
            (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0)

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
        elif n_tomo_clust is None and n_tomo_lens is None:
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
                obs_copy = str(obs)
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
                if obs_dict['observables']['est_shear'] == 'C_ell' and obs in self.CELL_EB_LABELS:
                    obs_copy = self.CELL_EB_LABELS[obs]
                
                
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
                                                        cov = gauss[oidx][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,
                                                            *self._gm_first(obs, ri, rj, i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
                                                        cov = gauss[splitidx][idxs] \
                                                            + gauss[splitidx+1][idxs] \
                                                            + gauss[splitidx+2][idxs] \
                                                            + nongauss[oidx][idxs] \
                                                            + ssc[oidx][idxs]
                                                        ostr = ostr_format \
                                                            % (obs_copy,
                                                            *self._gm_first(obs, ri, rj, i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
            
            if self.has_csmf:
                olist.extend(self._csmf_list_rows(cov_dict, obs_dict, ostr_format, self._csmf_bin_values(obs_dict),
                                                  self.projected_clust if np.any(self.projected_clust) else proj_quant,
                                                  self.projected_lens if np.any(self.projected_lens) else proj_quant))
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
        # with csmf_diagonal_lenses only sample i in clustering bin i is
        # measured: the other entries are not written
        diagonal_lenses = bool(obs_dict['observables'].get('csmf_diagonal_lenses')) and \
            (obs_dict['ELLspace']['n_spec'] is None or obs_dict['ELLspace']['n_spec'] == 0)
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
        elif n_tomo_clust is None and n_tomo_lens is None:
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
                if obs_dict['observables']['est_shear'] == 'C_ell' and obs in self.CELL_EB_LABELS:
                    obs_copy = self.CELL_EB_LABELS[obs]

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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if np.any(self.projected_clust):
                                                            ri = self.projected_clust[i_r1]
                                                            rj = self.projected_clust[i_r2]
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                            rj = proj_quant[i_r2]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                            % (obs_copy,
                                                            *self._gm_first(obs, ri, rj, i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        if np.any(self.projected_lens):
                                                            rj = self.projected_lens[i_r2]
                                                        else:
                                                            rj = proj_quant[i_r2]
                                                        if np.any(self.projected_clust):
                                                            ri = self.projected_clust[i_r1]
                                                        else:
                                                            ri = proj_quant[i_r1]
                                                        idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
                                                            % (obs_copy,
                                                            *self._gm_first(obs, ri, rj, i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        if diagonal_lenses and not self._diagonal_lens_entry(obs, i_s1, i_s2, t1, t2, t3, t4):
                                                            continue
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
            
            if self.has_csmf:
                olist.extend(self._csmf_list_rows(cov_dict, obs_dict, ostr_format, self._csmf_bin_values(obs_dict),
                                                  self.projected_clust if np.any(self.projected_clust) else proj_quant,
                                                  self.projected_lens if np.any(self.projected_lens) else proj_quant))

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
        elif n_tomo_clust is None and n_tomo_lens is None:
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
                            #obs_copy = str(summary['gg_summary_name'][0])
                            obs_copy = 'gg_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_gg']:
                                #obs_copy = str(summary['gg_summary_name'][1])
                                obs_copy = 'gg_summary_B'
                                ri -= summary['arb_number_first_summary_gg']
                        if gm and obs in ['gmgm']:
                            #obs_copy = str(summary['gm_summary_name'][0])
                            obs_copy = 'gm_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_gm']:
                                #obs_copy = str(summary['gm_summary_name'][1])
                                obs_copy = 'gm_summary_B'
                                ri -= summary['arb_number_first_summary_gm']
                        if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                            #obs_copy = str(summary['mmE_summary_name'][0])
                            obs_copy = 'mmE_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                #obs_copy = str(summary['mmE_summary_name'][1])
                                obs_copy = 'mmE_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                        if mm and obs in ['gmxim', 'ximxim']:
                            #obs_copy = str(summary['mmB_summary_name'][0])
                            obs_copy = 'mmB_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                #obs_copy = str(summary['mmB_summary_name'][1])
                                obs_copy = 'mmB_summary_B'
                                ri -= summary['arb_number_first_summary_mm']

                        #label rj
                        if gg and obs in ['gggg']:
                            if i_r2 >= summary['arb_number_first_summary_gg']:
                                #obs_copy += str(summary['gg_summary_name'][1])
                                obs_copy += '__gg_summary_B'
                                rj -= summary['arb_number_first_summary_gg']
                            else:
                                #obs_copy += str(summary['gg_summary_name'][0])
                                obs_copy += '__gg_summary_A'
                        if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                            if i_r2 >= summary['arb_number_first_summary_gm']:
                                #obs_copy += str(summary['gm_summary_name'][1])
                                obs_copy += '__gm_summary_B'
                                rj -= summary['arb_number_first_summary_gm']
                            else:
                                obs_copy += '__gm_summary_A'
                                #obs_copy += str(summary['gm_summary_name'][0])
                        if mm and obs in ['xipxip', 'ggxip']:
                            if i_r2 >= summary['arb_number_first_summary_mm']:
                                #obs_copy += str(summary['mmE_summary_name'][1])
                                obs_copy += '__mmE_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                #obs_copy = str(summary['mmE_summary_name'][0])
                                obs_copy += '__mmE_summary_A'
                        if mm and obs in ['xipxim', 'ximxim', 'ggxim']:
                            if i_r2 >= summary['arb_number_first_summary_mm']:
                                #obs_copy += str(summary['mmB_summary_name'][1])
                                obs_copy += '__mmB_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                #obs_copy += str(summary['mmB_summary_name'][0])
                                obs_copy += '__mmB_summary_A'


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
                                                        % (obs_copy,
                                                        *self._gm_first(obs, int(ri + 1), int(rj + 1), i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                            #obs_copy = str(summary['gg_summary_name'][0])
                            obs_copy = 'gg_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_gg']:
                                #obs_copy = str(summary['gg_summary_name'][1])
                                obs_copy = 'gg_summary_B'
                                ri -= summary['arb_number_first_summary_gg']
                        if gm and obs in ['gmgm']:
                            #obs_copy = str(summary['gm_summary_name'][0])
                            obs_copy = 'gm_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_gm']:
                                #obs_copy = str(summary['gm_summary_name'][1])
                                obs_copy = 'gm_summary_B'
                                ri -= summary['arb_number_first_summary_gm']
                        if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                            #obs_copy = str(summary['mmE_summary_name'][0])
                            obs_copy = 'mmE_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                #obs_copy = str(summary['mmE_summary_name'][1])
                                obs_copy = 'mmE_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                        if mm and obs in ['gmxim', 'ximxim']:
                            #obs_copy = str(summary['mmB_summary_name'][0])
                            obs_copy = 'mmB_summary_A'
                            if i_r1 >= summary['arb_number_first_summary_mm']:
                                #obs_copy = str(summary['mmB_summary_name'][1])
                                obs_copy = 'mmB_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                
                        #label rj
                        if gg and obs in ['gggg']:
                            if i_r2 >= summary['arb_number_first_summary_gg']:
                                #obs_copy += str(summary['gg_summary_name'][1])
                                obs_copy += '__gg_summary_B'
                                rj -= summary['arb_number_first_summary_gg']
                            else:
                                #obs_copy += str(summary['gg_summary_name'][0])
                                obs_copy += '__gg_summary_A'
                        if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                            if i_r2 >= summary['arb_number_first_summary_gm']:
                                #obs_copy += str(summary['gm_summary_name'][1])
                                obs_copy += '__gm_summary_B'
                                rj -= summary['arb_number_first_summary_gm']
                            else:
                                #obs_copy += str(summary['gm_summary_name'][0])
                                obs_copy += '__gm_summary_A'
                        if mm and obs in ['xipxip']:
                            if i_r2 >= summary['arb_number_first_summary_mm']:
                                #obs_copy += str(summary['mmE_summary_name'][1])
                                obs_copy += '__mmE_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                #obs_copy += str(summary['mmE_summary_name'][0])
                                obs_copy += '__mmE_summary_A'
                        if mm and obs in ['xipxim', 'ximxim', 'ggxim']:
                            if i_r2 >= summary['arb_number_first_summary_mm']:
                                #obs_copy += str(summary['mmB_summary_name'][1])
                                obs_copy += '__mmB_summary_B'
                                ri -= summary['arb_number_first_summary_mm']
                            else:
                                #obs_copy += str(summary['mmB_summary_name'][0])
                                obs_copy += '__mmB_summary_A'
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
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
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
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
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
                                                        % (obs_copy,
                                                        *self._gm_first(obs, int(ri + 1), int(rj + 1), i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1), 
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                splitidx += 3
        
        if self.has_csmf:
            olist.extend(self._csmf_list_rows(cov_dict, obs_dict, ostr_format, None, None, None, arbitrary=True))
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
        elif n_tomo_clust is None and n_tomo_lens is None:
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip', 'ggxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim', 'ggxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
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
                                                    #label ri 
                                                    if gg and obs in ['gggg', 'gggm', 'ggxip', 'ggxim']:
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[oidx][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,
                                                        *self._gm_first(obs, int(ri + 1), int(rj + 1), i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip', 'ggxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim', 'ggxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy, int(ri + 1), int(rj + 1),
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,  int(ri + 1), int(rj + 1),
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy,
                                                        *self._gm_first(obs, int(ri + 1), int(rj + 1), i_s1 + 1, i_s2 + 1, t1+1, t2+1, t3+1, t4+1),
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
                                                        #obs_copy = str(summary['gg_summary_name'][0])
                                                        obs_copy = 'gg_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy = str(summary['gg_summary_name'][1])
                                                            obs_copy = 'gg_summary_B'
                                                            ri -= summary['arb_number_first_summary_gg']
                                                    if gm and obs in ['gmgm']:
                                                        #obs_copy = str(summary['gm_summary_name'][0])
                                                        obs_copy = 'gm_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy = str(summary['gm_summary_name'][1])
                                                            obs_copy = 'gm_summary_B'
                                                            ri -= summary['arb_number_first_summary_gm']
                                                    if mm and obs in [ 'xipxip', 'xipxim', 'gmxip']:
                                                        #obs_copy = str(summary['mmE_summary_name'][0])
                                                        obs_copy = 'mmE_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmE_summary_name'][1])
                                                            obs_copy = 'mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                    if mm and obs in ['gmxim', 'ximxim']:
                                                        #obs_copy = str(summary['mmB_summary_name'][0])
                                                        obs_copy = 'mmB_summary_A'
                                                        if i_r1 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy = str(summary['mmB_summary_name'][1])
                                                            obs_copy = 'mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                            
                                                    #label rj
                                                    if gg and obs in ['gggg']:
                                                        if i_r2 >= summary['arb_number_first_summary_gg']:
                                                            #obs_copy += str(summary['gg_summary_name'][1])
                                                            obs_copy += '__gg_summary_B'
                                                            rj -= summary['arb_number_first_summary_gg']
                                                        else:
                                                            #obs_copy += str(summary['gg_summary_name'][0])
                                                            obs_copy += '__gg_summary_A'
                                                    if gm and obs in ['gmgm', 'gggm', 'gmxip', 'gmxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_gm']:
                                                            #obs_copy += str(summary['gm_summary_name'][1])
                                                            obs_copy += '__gm_summary_B'
                                                            rj -= summary['arb_number_first_summary_gm']
                                                        else:
                                                            #obs_copy += str(summary['gm_summary_name'][0])
                                                            obs_copy += '__gm_summary_A'
                                                    if mm and obs in ['xipxip', 'ggxip']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmE_summary_name'][1])
                                                            obs_copy += '__mmE_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmE_summary_name'][0])
                                                            obs_copy += '__mmE_summary_A'
                                                    if mm and obs in ['xipxim', 'ximxim', 'ggxim']:
                                                        if i_r2 >= summary['arb_number_first_summary_mm']:
                                                            #obs_copy += str(summary['mmB_summary_name'][1])
                                                            obs_copy += '__mmB_summary_B'
                                                            ri -= summary['arb_number_first_summary_mm']
                                                        else:
                                                            #obs_copy += str(summary['mmB_summary_name'][0])
                                                            obs_copy += '__mmB_summary_A'
                                                    idxs = (i_r1, i_r2, i_s1, i_s2, t1, t2, t3, t4)
                                                    cov = gauss[splitidx][idxs] \
                                                        + gauss[splitidx+1][idxs] \
                                                        + gauss[splitidx+2][idxs] \
                                                        + nongauss[oidx][idxs] \
                                                        + ssc[oidx][idxs]
                                                    ostr = ostr_format \
                                                        % (obs_copy, int(ri + 1), int(rj + 1),
                                                        i_s1+1, i_s2+1, t1+1, t2+1, t3+1, t4+1, 
                                                        cov, 
                                                        gauss[splitidx][idxs],
                                                        gauss[splitidx+1][idxs],
                                                        gauss[splitidx+2][idxs],
                                                        nongauss[oidx][idxs],
                                                        ssc[oidx][idxs])
                                                    olist.append(ostr)
                splitidx += 3
        if self.has_csmf:
            olist.extend(self._csmf_list_rows(cov_dict, obs_dict, ostr_format, None, None, None, arbitrary=True))
        
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
        if self.N_stellar_mass_bins is None:
            self.N_stellar_mass_bins = number_m_bins
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

    

    def __create_matrix_csmf_cross_LSS_arbitrary(self,covlist, probe, summary, is_i_smaller_j, want_diagonal_csmf = False):
        nprobes_probe_1 = summary['number_summary_'+probe]
        nprobes_firstprobe_probe_1 = summary['arb_number_first_summary_' +probe]
        if nprobes_probe_1 is not None:
            covp1p1 = self.__create_matrix_csmf_cross_LSS(covlist[:nprobes_firstprobe_probe_1], is_i_smaller_j, want_diagonal_csmf)    
            result = covp1p1
            if nprobes_probe_1 > 1:
                covp2p1 = self.__create_matrix_csmf_cross_LSS(covlist[nprobes_firstprobe_probe_1:], is_i_smaller_j, want_diagonal_csmf)   
                result = np.block([[covp1p1],
                                    [covp2p1]])
            return result        
        
    def __create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(self,covlist, probe, summary, is_i_smaller_j, want_diagonal_csmf = False, is_ij_lensing = False):
        nprobes_probe_1 = summary['number_summary_'+probe]
        nprobes_firstprobe_probe_1 = summary['arb_number_first_summary_' +probe]
        if nprobes_probe_1 is not None:
            covp1p1 = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(covlist[:nprobes_firstprobe_probe_1], is_i_smaller_j, want_diagonal_csmf, is_ij_lensing)    
            result = covp1p1
            if nprobes_probe_1 > 1:
                covp2p1 = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(covlist[nprobes_firstprobe_probe_1:], is_i_smaller_j, want_diagonal_csmf, is_ij_lensing)   
                result = np.block([[covp1p1],
                                    [covp2p1]])
            return result     
    
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
                    for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:])):
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
                    for j_tomo in range(i_tomo,len(covlist[0,0,0,0,0,:])):
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
                    for j_tomo in range(len(covlist[0,0,0,0,0, :])):
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
                    for j_tomo in range(len(covlist[0,0,0,0,0, :])):
                        for i_sample in range(len(covlist[0,0, :, 0,0,0])):
                            for i_theta in range(len(covlist[:,0,0,0,0,0])):
                                j = 0
                                for i_mass_tomo in range(number_tomo_bins):
                                    for i_mass in range(number_m_bins):    
                                        covariance[i,j] = covlist[i_theta, i_mass, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                        j += 1
                                i += 1
        return covariance
    
    def __create_matrix_csmf_cross_LSS_diagonal_clustering(self,covlist, is_ij_clustering ,want_diagonal_csmf = False, is_ij_lensing = False):
        number_m_bins = int(len(covlist[0,:,0,0,0,0]))
        number_tomo_bins = int(len(covlist[0,0,0,:,0,0]))
        if is_ij_clustering and not is_ij_lensing:
            data_size_ij = int(len(covlist[0,0,0,0,:,0])*len(covlist[:,0,0,0,0,0]))
            if want_diagonal_csmf:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0])):
                        j = 0
                        for i_mass_tomo in range(number_tomo_bins):
                            covariance[i,j] = covlist[i_theta, i_mass_tomo, i_tomo, i_mass_tomo, i_tomo, i_tomo]
                            j += 1
                        i += 1
            else:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0])*len(covlist[0,0,0,:,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0])):
                        j = 0
                        for i_mass_tomo in range(number_tomo_bins):
                            for i_mass in range(number_m_bins):    
                                covariance[i,j] = covlist[i_theta, i_mass, i_tomo, i_mass_tomo, i_tomo, i_tomo]
                                j += 1
                        i += 1
        else:
            if is_ij_lensing:
                data_size_ij = int(len(covlist[0,0,0,0,:,0])*(len(covlist[0,0,0,0,0,:]) + 1)/2*len(covlist[:,0,0,0,0,0]))
            else:
                data_size_ij = int(len(covlist[0,0,0,0,:,0])*len(covlist[0,0,0,0,0,:])*len(covlist[:,0,0,0,0,0]))
            if want_diagonal_csmf:
                i = 0
                data_size_mn = int(len(covlist[0,:,0,0,0,0]))
                covariance = np.zeros((data_size_ij,data_size_mn))
                for i_tomo in range(len(covlist[0,0,0,0,:,0])):
                    # galaxy sample i_tomo in clustering bin i_tomo for ggl,
                    # a single sample for cosmic shear
                    i_sample = 0 if is_ij_lensing else i_tomo
                    j_start = 0
                    if is_ij_lensing:
                        j_start = i_tomo
                    for j_tomo in range(j_start,len(covlist[0,0,0,0,0, :])):
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
                    # galaxy sample i_tomo in clustering bin i_tomo for ggl,
                    # a single sample for cosmic shear
                    i_sample = 0 if is_ij_lensing else i_tomo
                    j_start = 0
                    if is_ij_lensing:
                        j_start = i_tomo
                    for j_tomo in range(j_start,len(covlist[0,0,0,0,0, :])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0])):
                            j = 0
                            for i_mass_tomo in range(number_tomo_bins):
                                for i_mass in range(number_m_bins):    
                                    covariance[i,j] = covlist[i_theta, i_mass, i_sample, i_mass_tomo, i_tomo, j_tomo]
                                    j += 1
                            i += 1
        return covariance
    
    def __create_matrix(self,covlist, is_i_smaller_j, is_m_smaller_n, is_ij_lensing = False, is_mn_lensing = False):
        if self.csmf_diagonal_lenses:
            return self.__create_matrix_diagonal_clustering(covlist, is_i_smaller_j, is_m_smaller_n, is_ij_lensing, is_mn_lensing)
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
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])*len(covlist[0,0,0,:,0,0,0,0])
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
            if self.adjacent != 1:
                data_size_ij = int(2*len(covlist[0,0,0,0,:,0,0,0]) - 1)*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(2*len(covlist[0,0,0,0,0,0,:,0]) - 1)*len(covlist[0,:,0,0,0,0,0,0])
            else:
                data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]))*len(covlist[0,:,0,0,0,0,0,0])
            covariance = np.zeros((data_size_ij,data_size_mn))
            i = 0
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for j_tomo in range(i_tomo, i_tomo + self.adjacent):
                    if j_tomo == len(covlist[0,0,0,0,0,:,0,0]):
                        continue
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for k_tomo in range(m_tomo, m_tomo + self.adjacent):
                                if k_tomo == len(covlist[0,0,0,0,0,0,0,:]):
                                    continue
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,k_tomo] 
                                    j += 1
                        i += 1
        if diagonal_1 and not diagonal_2:
            if is_m_smaller_n:
                if self.adjacent != 1:
                    data_size_ij = int(2*len(covlist[0,0,0,0,:,0,0,0]) - 1)*len(covlist[:,0,0,0,0,0,0,0])
                else:        
                    data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for j_tomo in range(i_tomo, i_tomo + self.adjacent):
                        if j_tomo == len(covlist[0,0,0,0,0,:,0,0]):
                            continue
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for n_tomo in range(m_tomo, len(covlist[0,0,0,0,0,0,0,:])):
                                    for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                        covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,i_tomo,m_tomo,n_tomo] 
                                        j += 1
                            i += 1
            else:
                if self.adjacent != 1:
                    data_size_ij = int(2*len(covlist[0,0,0,0,:,0,0,0]) - 1)*len(covlist[:,0,0,0,0,0,0,0])
                else:        
                    data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:])))*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for j_tomo in range(i_tomo, i_tomo + self.adjacent):
                        if j_tomo == len(covlist[0,0,0,0,0,:,0,0]):
                            continue
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
                if self.adjacent != 1:
                    data_size_mn = int(2*len(covlist[0,0,0,0,:,0,0,0]) -1)*len(covlist[:,0,0,0,0,0,0,0])
                else:        
                    data_size_mn = int(len(covlist[0,0,0,0,:,0,0,0]))*len(covlist[:,0,0,0,0,0,0,0])
                data_size_ij = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2)*len(covlist[0, :,0,0,0,0,0,0])
                covariance = np.zeros((data_size_ij,data_size_mn))
                i = 0
                for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                    for j_tomo in range(i_tomo, len(covlist[0,0,0,0,0,:,0,0])):
                        for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                            j = 0
                            for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                                for k_tomo in range(m_tomo, m_tomo + self.adjacent):
                                    if k_tomo == len(covlist[0,0,0,0,0,0,0,:]):
                                        continue
                                
                                    for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                        covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,m_tomo] 
                                        j += 1
                        i += 1
            else:
                if self.adjacent != 1:
                    data_size_mn = int(2*len(covlist[0,0,0,0,:,0,0,0]) -1)*len(covlist[:,0,0,0,0,0,0,0])
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
                                for k_tomo in range(m_tomo, m_tomo + self.adjacent):
                                    if k_tomo == len(covlist[0,0,0,0,0,0,0,:]):
                                        continue
                                    for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                        covariance[i,j] = covlist[i_theta,j_theta,0,0,i_tomo,j_tomo,m_tomo,m_tomo] 
                                        j += 1
                        i += 1
        return covariance
    

    def __create_matrix_diagonal_clustering(self,covlist, is_ij_clustering, is_mn_clustering, is_ij_lensing = False, is_mn_lensing = False):
        if is_ij_clustering and is_mn_clustering and not is_ij_lensing and not is_mn_lensing:
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0])*len(covlist[:,0,0,0,0,0,0,0]))
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0])*len(covlist[0, :,0,0,0,0,0,0]))
            covariance = np.zeros((data_size_ij,data_size_mn))
            i = 0
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                    j = 0
                    for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                        for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                            covariance[i,j] = covlist[i_theta,j_theta,i_tomo,m_tomo,i_tomo,i_tomo,m_tomo,m_tomo]
                            j += 1
                    i += 1
        if is_ij_clustering and not is_ij_lensing and not is_mn_clustering:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0])*len(covlist[:,0,0,0,0,0,0,0]))
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]))*len(covlist[0, :,0,0,0,0,0,0]))
            if is_mn_lensing:
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2*len(covlist[0, :,0,0,0,0,0,0]))
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                    j = 0
                    for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                        n_start = 0
                        m_sample = np.copy(m_tomo)
                        if is_mn_lensing:
                            n_start = m_tomo
                            m_sample = 0
                        for n_tomo in range(n_start,len(covlist[0,0,0,0,0,0,0,:])):
                            for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                covariance[i,j] = covlist[i_theta,j_theta,i_tomo,m_sample,i_tomo,i_tomo,m_tomo,n_tomo] 
                                j += 1
                    i += 1
        if not is_ij_clustering and is_mn_clustering and not is_mn_lensing:
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]))*len(covlist[:,0,0,0,0,0,0,0]))
            if is_ij_lensing:
                data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2*len(covlist[:,0,0,0,0,0,0,0]))
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0])*len(covlist[0, :,0,0,0,0,0,0]))
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                j_start = 0
                i_sample = np.copy(i_tomo)
                if is_ij_lensing:
                    j_start = i_tomo
                    i_sample = 0
                for j_tomo in range(j_start,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                covariance[i,j] = covlist[i_theta,j_theta,i_sample,m_tomo,i_tomo,j_tomo,m_tomo,m_tomo] 
                                j += 1
                        i += 1
        if (not is_ij_clustering and not is_mn_clustering) or ((is_ij_clustering and is_ij_lensing) or (is_mn_clustering and is_mn_lensing)):
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]))*len(covlist[:,0,0,0,0,0,0,0]))
            if is_ij_lensing:
                data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0]) * (len(covlist[0,0,0,0,0,:,0,0]) + 1)/2*len(covlist[:,0,0,0,0,0,0,0]))
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]))*len(covlist[0, :,0,0,0,0,0,0]))
            if is_mn_lensing:
                data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2*len(covlist[0, :,0,0,0,0,0,0]))
            covariance = np.zeros((data_size_ij,data_size_mn))   
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                j_start = 0
                i_sample = np.copy(i_tomo)
                if is_ij_lensing:
                    j_start = i_tomo
                    i_sample = 0
                for j_tomo in range(j_start,len(covlist[0,0,0,0,0,:,0,0])):
                    for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                        j = 0
                        for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                            n_start = 0
                            m_sample = np.copy(m_tomo)
                            if is_mn_lensing:
                                n_start = m_tomo
                                m_sample = 0
                            for n_tomo in range(n_start,len(covlist[0,0,0,0,0,0,0,:])):
                                for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                    covariance[i,j] = covlist[i_theta,j_theta,i_sample,m_sample,i_tomo,j_tomo,m_tomo,n_tomo] 
                                    j += 1
                        i += 1
        if is_ij_clustering and not is_ij_lensing and (is_mn_clustering and is_mn_lensing):
            i = 0
            data_size_ij = int(len(covlist[0,0,0,0,:,0,0,0])*len(covlist[:,0,0,0,0,0,0,0]))
            data_size_mn = int(len(covlist[0,0,0,0,0,0,:,0]) * (len(covlist[0,0,0,0,0,0,0,:]) + 1)/2*len(covlist[0, :,0,0,0,0,0,0]))
            covariance = np.zeros((data_size_ij,data_size_mn))
            for i_tomo in range(len(covlist[0,0,0,0,:,0,0,0])):
                for i_theta in range(len(covlist[:,0,0,0,0,0,0,0])):
                    j = 0
                    for m_tomo in range(len(covlist[0,0,0,0,0,0,:,0])):
                        n_start = 0
                        m_sample = np.copy(m_tomo)
                        if is_mn_lensing:
                            n_start = m_tomo
                            m_sample = 0
                        for n_tomo in range(n_start,len(covlist[0,0,0,0,0,0,0,:])):
                            for j_theta in range(len(covlist[0,:,0,0,0,0,0,0])):
                                covariance[i,j] = covlist[i_theta,j_theta,i_tomo,m_sample,i_tomo,i_tomo,m_tomo,n_tomo] 
                                j += 1
                    i += 1
        return covariance


        
    def __create_matrix_arbitrary(self,covlist, is_i_smaller_j, is_m_smaller_n, probe1, probe2, summary, is_ij_lensing = False, is_mn_lensing = False):
        if self.csmf_diagonal_lenses:
            return self.__create_matrix_arbitrary_diagonal_clustering(covlist, is_i_smaller_j, is_m_smaller_n, probe1, probe2, summary, is_ij_lensing, is_mn_lensing)
        else:
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
    
    def __create_matrix_arbitrary_diagonal_clustering(self,covlist, is_ij_clustering, is_mn_clustering, probe1, probe2, summary, is_ij_lensing = False, is_mn_lensing = False):
        nprobes_probe_1 = summary['number_summary_'+probe1]
        nprobes_probe_2 = summary['number_summary_'+probe2]
        nprobes_firstprobe_probe_1 = summary['arb_number_first_summary_' +probe1]
        nprobes_firstprobe_probe_2 = summary['arb_number_first_summary_' +probe2]
        if nprobes_probe_1 is not None and nprobes_probe_2 is not None:
            covp1p1 = self.__create_matrix_diagonal_clustering(covlist[:nprobes_firstprobe_probe_1,:nprobes_firstprobe_probe_2], is_ij_clustering, is_mn_clustering, is_ij_lensing, is_mn_lensing)    
            result = covp1p1
            if nprobes_probe_1 > 1:
                covp2p1 = self.__create_matrix_diagonal_clustering(covlist[nprobes_firstprobe_probe_1:,:nprobes_firstprobe_probe_2], is_ij_clustering, is_mn_clustering, is_ij_lensing, is_mn_lensing)   
                result = np.block([[covp1p1],
                                    [covp2p1]])
                if nprobes_probe_2 > 1:
                    covp2p2 = self.__create_matrix_diagonal_clustering(covlist[nprobes_firstprobe_probe_1:,nprobes_firstprobe_probe_2:], is_ij_clustering, is_mn_clustering, is_ij_lensing, is_mn_lensing)
                    covp1p2 = self.__create_matrix_diagonal_clustering(covlist[:nprobes_firstprobe_probe_1,nprobes_firstprobe_probe_2:], is_ij_clustering, is_mn_clustering, is_ij_lensing, is_mn_lensing)
                    result = np.block([[covp1p1, covp1p2],
                                    [covp2p1, covp2p2]])
                    
            if nprobes_probe_2 > 1 and nprobes_probe_1 == 1 :
                covp1p2 = self.__create_matrix_diagonal_clustering(covlist[:nprobes_firstprobe_probe_1,nprobes_firstprobe_probe_2:], is_ij_clustering, is_mn_clustering, is_ij_lensing, is_mn_lensing)
                result = np.block([[covp1p1, covp1p2]])
        return result
    
    def __csmf_matrix_blocks(self,
                             csmf_cov,
                             obs_dict,
                             gg,
                             gm,
                             mm,
                             second_mm,
                             summary=None):
        """
        Matrices of the covariance of the stellar mass function and of its
        cross-covariance with the LSS probes (in the order of the LSS
        probes) from the list csmf_cov (auto-covariance, cross-covariance
        with gg, gm, the first and, if second_mm, the second cosmic shear
        statistic), for arbitrary summary statistics if summary is given.

        Returns
        -------
        covariance_csmf, csmf_block : arrays
        """
        diagonal = obs_dict['observables']['csmf_diagonal']
        covariance_csmf = self.__create_matrix_csmf(csmf_cov[0], diagonal)

        def cross(idx, probe, flag, lensing=False):
            if summary is None:
                if self.csmf_diagonal_lenses:
                    return self.__create_matrix_csmf_cross_LSS_diagonal_clustering(csmf_cov[idx], flag and not lensing, diagonal, is_ij_lensing=lensing)
                return self.__create_matrix_csmf_cross_LSS(csmf_cov[idx], flag, diagonal)
            if self.csmf_diagonal_lenses:
                return self.__create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(csmf_cov[idx], probe, summary, flag, want_diagonal_csmf=diagonal, is_ij_lensing=lensing)
            return self.__create_matrix_csmf_cross_LSS_arbitrary(csmf_cov[idx], probe, summary, flag, diagonal)

        blocks = []
        if gg:
            blocks.append(cross(1, 'gg', True))
        if gm:
            blocks.append(cross(2, 'gm', False))
        if mm:
            for idx in ([3, 4] if second_mm else [3]):
                if idx < len(csmf_cov) and isinstance(csmf_cov[idx], np.ndarray):
                    blocks.append(cross(idx, 'mm', True, lensing=True))
                else:
                    # not calculated (e.g. B-modes)
                    blocks.append(np.zeros_like(blocks[-1]))
        return covariance_csmf, np.block([[block] for block in blocks])

    def __add_csmf_rows(self,
                        cov2d,
                        term,
                        obs_dict,
                        gg,
                        gm,
                        mm,
                        second_mm,
                        summary=None):
        """
        Adds the rows and columns of the stellar mass function to the
        covariance matrix cov2d of the LSS probes, for the term 'total',
        'gauss', 'nongauss' or 'ssc'.
        """
        csmf_cov = self.conditional_stellar_mass_function_cov if term == 'total' else self.csmf_split[term]
        covariance_csmf, csmf_block = self.__csmf_matrix_blocks(csmf_cov, obs_dict, gg, gm, mm, second_mm, summary)
        return np.block([[cov2d, csmf_block],
                         [csmf_block.T, covariance_csmf]])

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
        self.csmf_diagonal_lenses = obs_dict['observables']['csmf_diagonal_lenses']
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

            if obs_dict['observables']['est_shear'] in ('cosebi', 'C_ell') and obs_dict['observables']['cosmic_shear'] == True:
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
                    if self.csmf_diagonal_lenses:
                        covariange_csmfgg = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(self.conditional_stellar_mass_function_cov[1], True, obs_dict['observables']['csmf_diagonal'])
                    else:    
                        covariange_csmfgg = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[1], True, obs_dict['observables']['csmf_diagonal'])
                if gm:
                    if self.csmf_diagonal_lenses:
                        covariange_csmfgm = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(self.conditional_stellar_mass_function_cov[2], False, obs_dict['observables']['csmf_diagonal'])
                    else:
                        covariange_csmfgm = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[2], False, obs_dict['observables']['csmf_diagonal'])
                
                if mm:
                    if self.csmf_diagonal_lenses:
                        covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(self.conditional_stellar_mass_function_cov[3], False, obs_dict['observables']['csmf_diagonal'], is_ij_lensing=True)
                    else:
                        covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[3], True, obs_dict['observables']['csmf_diagonal'])
                if obslength == 10:
                    # second cosmic shear block (e.g. xi_minus), zero if it is
                    # not calculated (B-modes)
                    if len(self.conditional_stellar_mass_function_cov) > 4 and isinstance(self.conditional_stellar_mass_function_cov[4], np.ndarray):
                        if self.csmf_diagonal_lenses:
                            covariange_csmfmmB = self.__create_matrix_csmf_cross_LSS_diagonal_clustering(self.conditional_stellar_mass_function_cov[4], False, obs_dict['observables']['csmf_diagonal'], is_ij_lensing=True)
                        else:
                            covariange_csmfmmB = self.__create_matrix_csmf_cross_LSS(self.conditional_stellar_mass_function_cov[4], True, obs_dict['observables']['csmf_diagonal'])
                    else:
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
                            covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                            covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                            cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                            [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                            [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                    elif mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                        cov_diag.append(covariance_mmmm)
                        covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                        cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                        [covariance_ggmm.T, covariance_mmmm]])
                elif gm:
                    covariance_gmgm = self.__create_matrix(cov[3],False,False)
                    cov_diag.append(covariance_gmgm)
                    cov2d = covariance_gmgm
                    if mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                        cov_diag.append(covariance_mmmm)
                        covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                        cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                        [covariance_mmgm, covariance_mmmm]])
                elif mm:
                    covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
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
                            covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                            covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                            covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                cov_diag.append(covariance_ximxim)
                                covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        
                    elif xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                        covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wxip],
                                        [covariance_wxip.T, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                            covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                elif gm:
                    covariance_gtgt = self.__create_matrix(cov[4],False,False)
                    cov2d = covariance_gtgt
                    cov_diag.append(covariance_gtgt)        
                    if xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                        covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                        [covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                            covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                elif xipp:
                    covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                    cov2d = covariance_xipxip
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                        cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        cov_diag.append(covariance_ximxim)
                        if xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                            cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                            [covariance_xipxim.T, covariance_ximxim]])
                elif ximm:
                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                    cov2d = covariance_ximxim
                    cov_diag.append(covariance_ximxim)
                elif xipm:
                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                    cov2d = covariance_xipxim
            
            if self.has_csmf:
                cov2d = self.__add_csmf_rows(cov2d, 'total', obs_dict, gg, gm, mm, obslength == 10)
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
                                covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                            cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                            [covariance_ggmm.T, covariance_mmmm]])
                    elif gm:
                        covariance_gmgm = self.__create_matrix(cov[3],False,False)
                        cov_diag.append(covariance_gmgm)
                        cov2d = covariance_gmgm
                        if mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                            cov_diag.append(covariance_mmmm)
                            covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                            cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                            [covariance_mmgm, covariance_mmmm]])
                    elif mm:
                        covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
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
                                covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                    cov_diag.append(covariance_ximxim)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                            
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                            covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wxip],
                                            [covariance_wxip.T, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                cov_diag.append(covariance_ximxim)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                    elif gm:
                        covariance_gtgt = self.__create_matrix(cov[4],False,False)
                        cov2d = covariance_gtgt
                        cov_diag.append(covariance_gtgt)        
                        if xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                            covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                            [covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                cov_diag.append(covariance_ximxim)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    elif xipp:
                        covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                        cov2d = covariance_xipxip
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                            cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            cov_diag.append(covariance_ximxim)
                            if xipm:
                                covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                [covariance_xipxim.T, covariance_ximxim]])
                    elif ximm:
                        covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                        cov2d = covariance_ximxim
                        cov_diag.append(covariance_ximxim)
                    elif xipm:
                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                        cov2d = covariance_xipxim
                if self.has_csmf:
                    cov2d = self.__add_csmf_rows(cov2d, 'gauss', obs_dict, gg, gm, mm, obslength == 10)

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
                                    covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                    cov_diag.append(covariance_mmmm)
                                    covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                    covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                                    cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                    [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                    [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                            elif mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                                [covariance_ggmm.T, covariance_mmmm]])
                        elif gm:
                            covariance_gmgm = self.__create_matrix(cov[3],False,False)
                            cov_diag.append(covariance_gmgm)
                            cov2d = covariance_gmgm
                            if mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                                [covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
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
                                    covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                    covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                    covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                    cov_diag.append(covariance_xipxip)
                                    if ximm:
                                        covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                        cov_diag.append(covariance_ximxim)
                                        covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                        covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                        if xipm:
                                            covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                                
                            elif xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wxip],
                                                [covariance_wxip.T, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                        elif gm:
                            covariance_gtgt = self.__create_matrix(cov[4],False,False)
                            cov2d = covariance_gtgt
                            cov_diag.append(covariance_gtgt)        
                            if xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                                [covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                cov_diag.append(covariance_ximxim)
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                    [covariance_xipxim.T, covariance_ximxim]])
                        elif ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                            cov2d = covariance_ximxim
                            cov_diag.append(covariance_ximxim)
                        elif xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                            cov2d = covariance_xipxim

                    if self.has_csmf:
                        cov2d = self.__add_csmf_rows(cov2d, 'nongauss', obs_dict, gg, gm, mm, obslength == 10)
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
                                    covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                    cov_diag.append(covariance_mmmm)
                                    covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                    covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                                    cov2d = np.block([[covariance_gggg, covariance_gggm, covariance_ggmm],
                                                    [covariance_gggm.T, covariance_gmgm, covariance_mmgm.T],
                                                    [covariance_ggmm.T, covariance_mmgm, covariance_mmmm]])
                            elif mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_ggmm = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                cov2d = np.block([[covariance_gggg, covariance_ggmm],
                                                [covariance_ggmm.T, covariance_mmmm]])
                        elif gm:
                            covariance_gmgm = self.__create_matrix(cov[3],False,False)
                            cov_diag.append(covariance_gmgm)
                            cov2d = covariance_gmgm
                            if mm:
                                covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
                                cov_diag.append(covariance_mmmm)
                                covariance_mmgm = self.__create_matrix(cov[4],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_gmgm, covariance_mmgm.T],
                                                [covariance_mmgm, covariance_mmmm]])
                        elif mm:
                            covariance_mmmm = self.__create_matrix(cov[5],True,True,True,True)
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
                                    covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                    covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                    covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                                    cov_diag.append(covariance_xipxip)
                                    if ximm:
                                        covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                        cov_diag.append(covariance_ximxim)
                                        covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                        covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                        if xipm:
                                            covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                                
                            elif xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                covariance_wxip = self.__create_matrix(cov[2],True,True,is_mn_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wxip],
                                                [covariance_wxip.T, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                    covariance_wxim = self.__create_matrix(cov[3],True,True,is_mn_lensing=True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                        elif gm:
                            covariance_gtgt = self.__create_matrix(cov[4],False,False)
                            cov2d = covariance_gtgt
                            cov_diag.append(covariance_gtgt)        
                            if xipp:
                                covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                                covariance_xipgt = self.__create_matrix(cov[5],True,False,is_ij_lensing=True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                                [covariance_xipgt, covariance_xipxip]])
                                cov_diag.append(covariance_xipxip)
                                if ximm:
                                    covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                    covariance_ximgt = self.__create_matrix(cov[6],True,False,is_ij_lensing=True)
                                    cov_diag.append(covariance_ximxim)
                                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                    if xipm:
                                        covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        elif xipp:
                            covariance_xipxip = self.__create_matrix(cov[7],True,True,True,True)
                            cov2d = covariance_xipxip
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                                cov_diag.append(covariance_ximxim)
                                if xipm:
                                    covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                                    [covariance_xipxim.T, covariance_ximxim]])
                        elif ximm:
                            covariance_ximxim = self.__create_matrix(cov[9],True,True,True,True)
                            cov2d = covariance_ximxim
                            cov_diag.append(covariance_ximxim)
                        elif xipm:
                            covariance_xipxim = self.__create_matrix(cov[8],True,True,True,True)
                            cov2d = covariance_xipxim

                    if self.has_csmf:
                        cov2d = self.__add_csmf_rows(cov2d, 'ssc', obs_dict, gg, gm, mm, obslength == 10)
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
            if self._has_negative_eigenvalues(cov2d_total):
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
            
            hdr_str = 'Covariance matrix for the following probes (in this order) :\n'
            if gg:
                hdr_str += 'galaxy clustering with ' + str(n_tomo_clust) + ' tomographic bins (only the unique bin combinations) and ' + str(len(self.projected_clust)) + ' spatial bins and ' + str(sampledim) + ' stellar mass bins. \n'
            if gm:  
                hdr_str += 'galaxy galaxy lensing ' + str(n_tomo_clust) + ' tomographic lens bins and ' + str(n_tomo_lens) + ' tomographic source bins (all bin combinations) and ' + str(len(self.projected_clust)) + ' spatial bins and ' + str(sampledim) + ' stellar mass bins.\n'
            if mm:
                hdr_str += 'cosmic shear with ' + str(n_tomo_lens) + ' tomographic bins (only the unique bin combinations) and ' + str(len(self.projected_lens)) + ' spatial bins.\n'
            if self.has_csmf:
                csmf_description, csmf_notation, csmf_order = self._csmf_header(obs_dict)
                hdr_str += csmf_description

            hdr_str += "The structure of the diagonal blocks is, using the following notation\n"
            if gg:
                hdr_str += "galaxy clustering: gg^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if gm:
                hdr_str += "galaxy galaxy lensing: gm^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if mm:
                hdr_str += "cosmic shear: mm_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if self.has_csmf:
                hdr_str += csmf_notation
            if gg:
                hdr_str += self._block_order('gg', len(self.projected_clust), n_tomo_clust, n_tomo_clust,
                                             sampledim, unique=True, diagonal=self.csmf_diagonal_lenses)
            if gm:
                hdr_str += self._block_order('gm', len(self.projected_clust), n_tomo_clust, n_tomo_lens,
                                             sampledim, diagonal=self.csmf_diagonal_lenses)
            if mm:
                mm_order = self._block_order('mm', len(self.projected_lens), n_tomo_lens, n_tomo_lens, unique=True)
                hdr_str += mm_order
                # the matrix only contains a second cosmic shear block
                # (xi_minus or B-modes) if both auto blocks were assembled
                if obslength == 10 and xipp and ximm:
                    if obs_dict['observables']['est_shear'] == 'xi_pm':
                        hdr_str += "Followed by the second mm summary (xi_minus)\n"
                    else:
                        hdr_str += "Followed by the second mm summary (B-modes)\n"
                    hdr_str += mm_order
            if self.has_csmf:
                hdr_str += csmf_order

            if 'matrix' in self.style:
                if not cov_dict['split_gauss']:
                    print("Writing matrix output file.")
                    if self.save_as_binary:
                        fn = self.filename[self.style.index('matrix')]
                        name, extension = os.path.splitext(fn)
                        np.savez(name, covariance = cov2d_total, header = hdr_str)
                    else:
                        fn = self.filename[self.style.index('matrix')]
                        np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                else:
                    print("Writing matrix output file.")
                    if self.save_as_binary:
                        fn = self.filename[self.style.index('matrix')]
                        name, extension = os.path.splitext(fn)
                        np.savez(name, covariance = cov2d_total, header = hdr_str)
                        fn_gauss = name + "_gauss"
                        fn_nongauss = name + "_nongauss"
                        fn_ssc = name + "_SSC"
                        np.savez(fn_gauss, covariance = cov2d_gauss, header = hdr_str)
                        if self.has_nongauss:
                            np.savez(fn_nongauss, covariance = cov2d_nongauss, header = hdr_str)
                        if self.has_ssc:
                            np.savez(fn_ssc, covariance = cov2d_ssc, header = hdr_str)
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
                if obs_dict['observables']['combinations_clustering'] is not None or obs_dict['observables']['combinations_ggl'] is not None or obs_dict['observables']['combinations_lensing'] is not None:
                    cov2d_total_reduced, hdr_reduced = self.select_tomographic_bins(obs_dict, proj_quant, n_tomo_clust, n_tomo_lens, cov2d_total,
                                                                                    obslength == 10 and xipp and ximm,
                                                                                    sampledim)
                    self._write_reduced_matrix(cov2d_total_reduced, hdr_reduced)

        else:
            self.csmf_diagonal_lenses = False
            if mult == 3:
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
            else:
                ...
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
                if self.has_gauss:
                    cov = [gauss[idx]+nongauss[idx]+ssc[idx] for idx in range(len(gauss))]
                else:
                    cov = [nongauss[idx]+ssc[idx] for idx in range(len(gauss))]
            if self.has_nongauss and not self.has_ssc:
                if self.has_gauss:
                    cov = [gauss[idx]+nongauss[idx] for idx in range(len(gauss))]
                else:
                    cov = [nongauss[idx] for idx in range(len(gauss))]
            if not self.has_nongauss and self.has_ssc:
                if self.has_gauss:
                    cov = [gauss[idx]+ssc[idx] for idx in range(len(gauss))]
                else:
                    cov = [ssc[idx] for idx in range(len(gauss))]
            if not self.has_nongauss and not self.has_ssc:
                cov = [gauss[idx] for idx in range(len(gauss))]
            
            self.six_times_two_ss_spatial_index = None
            self.six_times_two_ps_spatial_index = None
            self.six_times_two_pp_spatial_index = None
            self.six_times_two_sm_spatial_index = None
            self.six_times_two_pm_spatial_index = None
            self.six_times_two_mm_spatial_index = None
            self.six_times_two_s_tomo_bin = None
            self.six_times_two_p_tomo_bin = None
            self.six_times_two_m_tomo_bin = None
            cov_diag = []
            if gg:
                covariance_gggg_ssss = self.__create_matrix_diagonal(cov[0], True, True, True, True)
                covariance_gggg_sssp = self.__create_matrix_diagonal(cov[1], True, False, True, False)
                covariance_gggg_sspp = self.__create_matrix_diagonal(cov[2], True, False, True, True)
                covariance_gggg_spsp = self.__create_matrix(cov[3],False, False)
                covariance_gggg_ppsp = self.__create_matrix(cov[4],True, False)
                covariance_gggg_pppp = self.__create_matrix(cov[5],True, True)
                self.six_times_two_ss_spatial_index = len(cov[0][:,0,0,0,0,0,0,0])
                self.six_times_two_ps_spatial_index = len(cov[1][0,:,0,0,0,0,0,0])
                self.six_times_two_pp_spatial_index = len(cov[2][0,:,0,0,0,0,0,0])
                self.six_times_two_s_tomo_bin = len(cov[0][0,0,0,0,:,0,0,0])
                self.six_times_two_p_tomo_bin = len(cov[1][0,0,0,0,0,0,0,:])
                cov2d = np.block([[covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp],
                                  [covariance_gggg_sssp.T, covariance_gggg_spsp, covariance_gggg_ppsp.T],
                                  [covariance_gggg_sspp.T, covariance_gggg_ppsp, covariance_gggg_pppp]])
                cov_diag.append(covariance_gggg_ssss.shape)
                cov_diag.append(covariance_gggg_spsp.shape)
                cov_diag.append(covariance_gggg_pppp.shape)
                
                
                
                if gm:
                    covariance_gggm_sssm = self.__create_matrix_diagonal(cov[6], True, False, True, False)
                    covariance_gggm_sspm = self.__create_matrix_diagonal(cov[7], True, False, True, False)
                    self.six_times_two_sm_spatial_index = len(cov[6][0,:,0,0,0,0,0,0])
                    self.six_times_two_pm_spatial_index = len(cov[7][0,:,0,0,0,0,0,0])
                    self.six_times_two_m_tomo_bin = len(cov[6][0,0,0,0,0,0,0,:])

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
                        self.six_times_two_mm_spatial_index = len(cov[21][0,:,0,0,0,0,0,0])
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
                        
                        
                        del cov
                        del covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm
                        del covariance_gggg_spsp, covariance_gggg_ppsp, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm
                        del covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm
                        del covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm
                        del covariance_gmgm_pmpm, covariance_mmgm_mmpm
                        del covariance_mmmm_mmmm
                        gc.collect()
                elif mm:
                    covariance_mmmm_mmmm = self.__create_matrix(cov[21],True,True)
                    self.six_times_two_m_tomo_bin = len(cov[21][0,0,0,0,0,0,0,:])
                    self.six_times_two_mm_spatial_index = len(cov[21][0,:,0,0,0,0,0,0])

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
                self.six_times_two_sm_spatial_index = len(cov[15][0,:,0,0,0,0,0,0])
                self.six_times_two_pm_spatial_index = len(cov[16][0,:,0,0,0,0,0,0])
                self.six_times_two_m_tomo_bin = len(cov[6][0,0,0,0,0,0,0,:])

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
                self.six_times_two_m_tomo_bin = len(cov[21][0,0,0,0,0,0,0,:])
                cov_diag.append(covariance_mmmm_mmmm)
                cov2d = covariance_mmmm_mmmm
            cov2d_total = np.copy(cov2d)
            print("Collected Gaussian covariance")

            if cov_dict['split_gauss'] and (self.has_ssc or self.has_nongauss):
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
                            del cov
                            del covariance_gggg_ssss, covariance_gggg_sssp, covariance_gggg_sspp, covariance_gggm_sssm, covariance_gggm_sspm, covariance_ggmm_ssmm
                            del covariance_gggg_spsp, covariance_gggg_ppsp, covariance_gggm_spsm, covariance_gggm_sppm, covariance_ggmm_spmm
                            del covariance_gggg_pppp, covariance_gggm_ppsm, covariance_gggm_pppm, covariance_ggmm_ppmm
                            del covariance_gmgm_smsm, covariance_gmgm_smpm, covariance_mmgm_mmsm
                            del covariance_gmgm_pmpm, covariance_mmgm_mmpm
                            del covariance_mmmm_mmmm
                            gc.collect()
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
            else:
                cov2d_gauss = np.copy(cov2d_total)

            print("Collected non-Gaussian and/or SSC covariance")
                
            for i in range(len(cov2d[:,0])):
                for j in range(len(cov2d[:,0])):
                    cov2d_total[j,i] = cov2d_total[i,j]
                    if cov_dict['split_gauss']:
                        cov2d_gauss[j,i] = cov2d_gauss[i,j]
                        if self.has_nongauss:
                            cov2d_nongauss[j,i] = cov2d_nongauss[i,j]
                        if self.has_ssc:
                            cov2d_ssc[j,i] = cov2d_ssc[i,j]
            if self._has_negative_eigenvalues(cov2d_total):
                print("ALARM: The resulting covariance matrix has negative eigenvalues")
                print("Try to adjust the accuracy settings in the config file:")
                print("For configuration space covariance reduce theta_accuracy and increase integration_intervals, usually a factor of 2 is enough.")
                print("For bandpower covariance reduce bandpower_accuracy.")
                print("For COSEBI covariance reduce En_accuracy.")
            
            hdr_str = 'Covariance matrix with the diagonals in the order:\n'
            if gg:
                hdr_str += 'spec_spec with '
                hdr_str += str(self.six_times_two_ss_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_s_tomo_bin) + ' tomographic bins. '
                if self.adjacent:
                    hdr_str += 'I am considering adjacent tomographic bins.\n'
                else:
                    hdr_str += 'I am considering only auto tomographic bins.\n'
                hdr_str += 'spec_phot with '
                hdr_str += str(self.six_times_two_ps_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_s_tomo_bin) + ' and ' + str(self.six_times_two_p_tomo_bin) + ' tomographic bins respectively\n'
                hdr_str += 'phot_phot with '
                hdr_str += str(self.six_times_two_pp_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_p_tomo_bin) + ' tomographic bins\n'
                
            if gm:
                hdr_str += 'spec_shear with '
                hdr_str += str(self.six_times_two_sm_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_s_tomo_bin) + ' and ' + str(self.six_times_two_m_tomo_bin) + ' tomographic bins respectively\n'

                hdr_str += 'phot_shear with '
                hdr_str += str(self.six_times_two_pm_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_p_tomo_bin) + ' and ' + str(self.six_times_two_m_tomo_bin) + ' tomographic bins respectively\n'
                
            if mm:
                hdr_str += 'shear_shear with '
                hdr_str += str(self.six_times_two_mm_spatial_index) + ' spatial indices and '
                hdr_str += str(self.six_times_two_m_tomo_bin) + ' tomographic bins\n'
            hdr_str += "The structure of the diagonal blocks is, using the following notation\n"
            if gg:
                hdr_str += "spec-spec clustering: ss_{tomographic bin i tomographic bin j}(spatial bin k)\n"
                hdr_str += "spec-phot clustering: sp_{tomographic bin i tomographic bin j}(spatial bin k)\n"
                hdr_str += "phot-phot clustering: pp_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if gm:
                hdr_str += "spec galaxy galaxy lensing: sm_{tomographic bin i tomographic bin j}(spatial bin k)\n"
                hdr_str += "phot galaxy galaxy lensing: pm_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if mm:
                hdr_str += "cosmic shear: mm_{tomographic bin i tomographic bin j}(spatial bin k)\n"
            if gg:
                hdr_str += "ss_{11}(1) ... ss_{11}(" + str(self.six_times_two_ss_spatial_index)+ ")"
                if self.adjacent:
                    hdr_str += " ss_{12}(1) ... ss_{12}(" + str(self.six_times_two_ss_spatial_index)+ ") ss_{22}(1) ... ss_{" +str(self.six_times_two_s_tomo_bin) + str(self.six_times_two_s_tomo_bin) + "}(" + str(self.six_times_two_ss_spatial_index) +")" 
                else:
                    hdr_str += " ss_{22}(1) ... ss_{" +str(self.six_times_two_s_tomo_bin) + str(self.six_times_two_s_tomo_bin) + "}(" + str(self.six_times_two_ss_spatial_index) +")" 
                hdr_str += "\n"
                hdr_str += "sp_{11}(1) ... sp_{11}(" + str(self.six_times_two_ps_spatial_index)+ ")"
                hdr_str += " sp_{12}(1) ... sp_{1"+ str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_ps_spatial_index)+ ")"
                hdr_str += " sp_{21}(1) ... sp_{2"+ str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_ps_spatial_index)+ ")... sp_{"+ str(self.six_times_two_s_tomo_bin) + str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_ps_spatial_index)+ ")"
                hdr_str += "\n"
                hdr_str += "pp_{11}(1) ... pp_{11}(" + str(self.six_times_two_pp_spatial_index)+ ")"
                hdr_str += " pp_{12}(1) ... pp_{1"+ str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_pp_spatial_index)+ ")"
                hdr_str += " pp_{22}(1) ... pp_{2"+ str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_pp_spatial_index)+ ")... pp_{"+ str(self.six_times_two_p_tomo_bin) + str(self.six_times_two_p_tomo_bin) + "}(" + str(self.six_times_two_pp_spatial_index)+ ")"
                hdr_str += "\n"
            if gm:
                hdr_str += "sm_{11}(1) ... sm_{11}(" + str(self.six_times_two_sm_spatial_index)+ ")"
                hdr_str += " sm_{12}(1) ... sm_{1"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_sm_spatial_index)+ ")"
                hdr_str += " sm_{21}(1) ... sm_{2"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_sm_spatial_index)+ ")... sp_{"+ str(self.six_times_two_s_tomo_bin) + str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_sm_spatial_index)+ ")"
                hdr_str += "\n"
                hdr_str += "pm_{11}(1) ... pm_{11}(" + str(self.six_times_two_pm_spatial_index)+ ")"
                hdr_str += " pm_{12}(1) ... pm_{1"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_pm_spatial_index)+ ")"
                hdr_str += " pm_{21}(1) ... pm_{2"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_pm_spatial_index)+ ")... sp_{"+ str(self.six_times_two_p_tomo_bin) + str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_pm_spatial_index)+ ")"
                hdr_str += "\n"
            if mm:
                hdr_str += "mm_{11}(1) ... mm_{11}(" + str(self.six_times_two_mm_spatial_index)+ ")"
                hdr_str += " mm_{12}(1) ... mm_{1"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_mm_spatial_index)+ ")"
                hdr_str += " mm_{22}(1) ... mm_{2"+ str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_mm_spatial_index)+ ")... mm_{"+ str(self.six_times_two_m_tomo_bin) + str(self.six_times_two_m_tomo_bin) + "}(" + str(self.six_times_two_mm_spatial_index)+ ")"
                hdr_str += "\n"


            if self.plot and len(cov2d_total[:,0]) < 1000:
                self.plot_corrcoeff_matrix_6x2pt(obs_dict, cov2d_total, cov_diag, proj_quant, n_tomo_clust, 
                    n_tomo_lens, sampledim, self.plot ,fct_args)
            else:
                print("This is a rather large covariance matrix, will not produce a plot, please do it yourself.")
        


            

            if 'matrix' in self.style:
                if not cov_dict['split_gauss']:
                    print("Writing matrix output file.")
                    fn = self.filename[self.style.index('matrix')]
                    if self.save_as_binary:
                        name, extension = os.path.splitext(fn)
                        np.savez(name, covariance = cov2d_total, header = hdr_str)
                    else:
                        np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                                newline='\n', header=hdr_str, comments='# ')
                else:
                    print("Writing matrix output file.")
                    if self.save_as_binary:
                        fn = self.filename[self.style.index('matrix')]
                        name, extension = os.path.splitext(fn)
                        np.savez(name, covariance = cov2d_total, header = hdr_str)
                        fn_gauss = name + "_gauss"
                        fn_nongauss = name + "_nongauss"
                        fn_ssc = name + "_SSC"
                        np.savez(fn_gauss, covariance = cov2d_gauss, header = hdr_str)
                        if self.has_nongauss:
                            np.savez(fn_nongauss, covariance = cov2d_nongauss, header = hdr_str)
                        if self.has_ssc:
                            np.savez(fn_ssc, covariance = cov2d_ssc, header = hdr_str)
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
                '''if not cov_dict['split_gauss']:
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
                                newline='\n', header=hdr_str, comments='# ')'''

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
        self.csmf_diagonal_lenses = obs_dict['observables']['csmf_diagonal_lenses']
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
                if self.csmf_diagonal_lenses:
                    covariange_csmfgg = self.__create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(self.conditional_stellar_mass_function_cov[1],"gg", summary, True,obs_dict['observables']['csmf_diagonal'])           
                else:
                    covariange_csmfgg = self.__create_matrix_csmf_cross_LSS_arbitrary(self.conditional_stellar_mass_function_cov[1],"gg", summary, True,obs_dict['observables']['csmf_diagonal'])   
            if gm:
                if self.csmf_diagonal_lenses:
                    covariange_csmfgm = self.__create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(self.conditional_stellar_mass_function_cov[2],"gm", summary, False, obs_dict['observables']['csmf_diagonal'])                
                else:
                    covariange_csmfgm = self.__create_matrix_csmf_cross_LSS_arbitrary(self.conditional_stellar_mass_function_cov[2],"gm", summary, False, obs_dict['observables']['csmf_diagonal'])
            if mm:
                if self.csmf_diagonal_lenses:
                    covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(self.conditional_stellar_mass_function_cov[3],"mm", summary, True, want_diagonal_csmf=obs_dict['observables']['csmf_diagonal'],is_ij_lensing=True)                
                else:
                    covariange_csmfmmE = self.__create_matrix_csmf_cross_LSS_arbitrary(self.conditional_stellar_mass_function_cov[3],"mm", summary, True, obs_dict['observables']['csmf_diagonal'])
            if ximm:
                # second set of cosmic shear summaries, zero if it is not calculated
                if len(self.conditional_stellar_mass_function_cov) > 4 and isinstance(self.conditional_stellar_mass_function_cov[4], np.ndarray):
                    if self.csmf_diagonal_lenses:
                        covariange_csmfmmB = self.__create_matrix_csmf_cross_LSS_diagonal_clustering_arbitrary(self.conditional_stellar_mass_function_cov[4],"mm", summary, True, want_diagonal_csmf=obs_dict['observables']['csmf_diagonal'],is_ij_lensing=True)
                    else:
                        covariange_csmfmmB = self.__create_matrix_csmf_cross_LSS_arbitrary(self.conditional_stellar_mass_function_cov[4],"mm", summary, True, obs_dict['observables']['csmf_diagonal'])
                else:
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
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary, is_mn_lensing=True)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov_diag.append(covariance_ximxim)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                cov2d = np.block([[covariance_ww, covariance_wxip],
                                [covariance_wxip.T, covariance_xipxip]])
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                    cov_diag.append(covariance_ximxim)
                    cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                    [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                        [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
        elif gm:
            covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary, is_mn_lensing=True)
            cov2d = covariance_gtgt
            cov_diag.append(covariance_gtgt)        
            if xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                [covariance_xipgt, covariance_xipxip]])
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                    cov_diag.append(covariance_ximxim)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                    [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                    [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                    [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
        elif xipp:
            covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
            cov2d = covariance_xipxip
            cov_diag.append(covariance_xipxip)
            if ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                cov_diag.append(covariance_ximxim)
                if xipm:
                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                    [covariance_xipxim.T, covariance_ximxim]])
        elif ximm:
            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
            cov2d = covariance_ximxim
            cov_diag.append(covariance_ximxim)
        elif xipm:
            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
            cov2d = covariance_xipxim
        
        if self.has_csmf:
            cov2d = self.__add_csmf_rows(cov2d, 'total', obs_dict, gg, gm, mm, ximm, summary)

        
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
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary, is_mn_lensing=True)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                        [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                        [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                    
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                    cov2d = np.block([[covariance_ww, covariance_wxip],
                                    [covariance_wxip.T, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                        [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                            [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
            elif gm:
                covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                cov2d = covariance_gtgt
                cov_diag.append(covariance_gtgt)        
                if xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                    cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                    [covariance_xipgt, covariance_xipxip]])
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                        cov_diag.append(covariance_ximxim)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                        [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                        [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
            elif xipp:
                covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                cov2d = covariance_xipxip
                cov_diag.append(covariance_xipxip)
                if ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                    [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                    cov_diag.append(covariance_ximxim)
                    if xipm:
                        covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                        [covariance_xipxim.T, covariance_ximxim]])
            elif ximm:
                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                cov2d = covariance_ximxim
                cov_diag.append(covariance_ximxim)
            elif xipm:
                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                cov2d = covariance_xipxim
            if self.has_csmf:
                cov2d = self.__add_csmf_rows(cov2d, 'gauss', obs_dict, gg, gm, mm, ximm, summary)
            cov2d_gauss = np.copy(cov2d)

            if self.has_ssc:
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
                            covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary, is_mn_lensing=True)
                            covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov_diag.append(covariance_ximxim)
                                covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                                covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        
                    elif xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wxip],
                                        [covariance_wxip.T, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                elif gm:
                    covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                    cov2d = covariance_gtgt
                    cov_diag.append(covariance_gtgt)        
                    if xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                        [covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_xipxip
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        cov_diag.append(covariance_ximxim)
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                            [covariance_xipxim.T, covariance_ximxim]])
                elif ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_ximxim
                    cov_diag.append(covariance_ximxim)
                elif xipm:
                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_xipxim
                if self.has_csmf:
                    cov2d = self.__add_csmf_rows(cov2d, 'ssc', obs_dict, gg, gm, mm, ximm, summary)
                cov2d_ssc = np.copy(cov2d)
            else:
                cov2d_ssc = np.zeros_like(cov2d_gauss)

            if self.has_nongauss:
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
                            covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True,'gg', 'mm', summary, is_mn_lensing=True)
                            covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                            cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip],
                                            [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T],
                                            [covariance_wxip.T, covariance_xipgt, covariance_xipxip]])
                            cov_diag.append(covariance_xipxip)
                            if ximm:
                                covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov_diag.append(covariance_ximxim)
                                covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                                covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                [covariance_wxip.T, covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                                [covariance_wxim.T, covariance_ximgt, np.zeros_like(covariance_ximxim), covariance_ximxim]])
                                if xipm:
                                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                    cov2d = np.block([[covariance_ww, covariance_wgt, covariance_wxip, covariance_wxim],
                                                    [covariance_wgt.T, covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                                    [covariance_wxip.T, covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                                    [covariance_wxim.T, covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                        
                    elif xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True ,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_wxip = self.__create_matrix_arbitrary(cov[2],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                        cov2d = np.block([[covariance_ww, covariance_wxip],
                                        [covariance_wxip.T, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_wxim = self.__create_matrix_arbitrary(cov[3],True,True, 'gg', 'mm', summary, is_mn_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                            [covariance_wxip.T, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_wxim.T, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary,is_ij_lensing=True, is_mn_lensing=True)
                                cov2d = np.block([[covariance_ww, covariance_wxip,covariance_wxim],
                                                [covariance_wxip.T, covariance_xipxip, covariance_xipxim],
                                                [covariance_wxim.T, covariance_xipxim.T, covariance_ximxim]])
                elif gm:
                    covariance_gtgt = self.__create_matrix_arbitrary(cov[4],False,False, 'gm', 'gm', summary)
                    cov2d = covariance_gtgt
                    cov_diag.append(covariance_gtgt)        
                    if xipp:
                        covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        covariance_xipgt = self.__create_matrix_arbitrary(cov[5],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                        cov2d = np.block([[covariance_gtgt, covariance_xipgt.T],
                                        [covariance_xipgt, covariance_xipxip]])
                        cov_diag.append(covariance_xipxip)
                        if ximm:
                            covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            covariance_ximgt = self.__create_matrix_arbitrary(cov[6],True,False, 'mm', 'gm', summary, is_ij_lensing=True)
                            cov_diag.append(covariance_ximxim)
                            cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                            [covariance_ximgt, np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                            if xipm:
                                covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                                cov2d = np.block([[covariance_gtgt, covariance_xipgt.T, covariance_ximgt.T],
                                            [covariance_xipgt, covariance_xipxip, covariance_xipxim],
                                            [covariance_ximgt, covariance_xipxim.T, covariance_ximxim]])
                elif xipp:
                    covariance_xipxip = self.__create_matrix_arbitrary(cov[7],True,True,'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_xipxip
                    cov_diag.append(covariance_xipxip)
                    if ximm:
                        covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                        cov2d = np.block([[covariance_xipxip, np.zeros_like(covariance_ximxim)],
                                        [np.zeros_like(covariance_ximxim).T, covariance_ximxim]])
                        cov_diag.append(covariance_ximxim)
                        if xipm:
                            covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                            cov2d = np.block([[covariance_xipxip, covariance_xipxim],
                                            [covariance_xipxim.T, covariance_ximxim]])
                elif ximm:
                    covariance_ximxim = self.__create_matrix_arbitrary(cov[9],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_ximxim
                    cov_diag.append(covariance_ximxim)
                elif xipm:
                    covariance_xipxim = self.__create_matrix_arbitrary(cov[8],True,True, 'mm', 'mm', summary, is_ij_lensing=True, is_mn_lensing=True)
                    cov2d = covariance_xipxim
                if self.has_csmf:
                    cov2d = self.__add_csmf_rows(cov2d, 'nongauss', obs_dict, gg, gm, mm, ximm, summary)
                cov2d_nongauss = np.copy(cov2d)
            else:
                cov2d_nongauss = np.zeros_like(cov2d_gauss)

        for i in range(len(cov2d[:,0])):
            for j in range(len(cov2d[:,0])):
                cov2d_total[j,i] = cov2d_total[i,j]
                if cov_dict['split_gauss']:
                    cov2d_gauss[j,i] = cov2d_gauss[i,j]
                    cov2d_nongauss[j,i] = cov2d_nongauss[i,j]
                    cov2d_ssc[j,i] = cov2d_ssc[i,j]

        if self._has_negative_eigenvalues(cov2d_total):
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
        
        two_gg = False
        two_gm = False
        two_mm = False
        if summary['number_summary_gg'] is not None:
            if summary['number_summary_gg'] > 1:
                two_gg = True
            spatial_first_probe_gg = summary['arb_number_first_summary_gg']
            spatial_second_probe_gg = len(summary['WL_gg']) - summary['arb_number_first_summary_gg']
        
        if summary['number_summary_gm'] is not None:
            if summary['number_summary_gm'] > 1:
                two_gm = True
            spatial_first_probe_gm = summary['arb_number_first_summary_gm']
            spatial_second_probe_gm = len(summary['WL_gm']) - summary['arb_number_first_summary_gm']
        
        if summary['number_summary_mm'] is not None:
            if summary['number_summary_mm'] > 1:
                two_mm = True
            spatial_first_probe_mm = summary['arb_number_first_summary_mm']
            spatial_second_probe_mm = len(summary['WL_mmE']) - summary['arb_number_first_summary_mm']

        hdr_str = 'Covariance matrix for the following probes (in this order) :\n'
        if gg:
            hdr_str += 'galaxy clustering with ' + str(n_tomo_clust) + ' tomographic bins (only the unique bin combinations) and ' + str(spatial_first_probe_gg) + ' spatial bins for the first probe and '
            if two_gg:
                hdr_str += str(spatial_second_probe_gg) + " for the second probe and "
            hdr_str += str(sampledim) + ' stellar mass bins. \n'
        if gm:  
            hdr_str += 'galaxy galaxy lensing with ' + str(n_tomo_clust) + ' tomographic lens bins and ' + str(n_tomo_lens) + ' tomographic source bins (all bin combinations) and ' + str(spatial_first_probe_gm) + ' spatial bins for the first probe and '
            if two_gm:
                hdr_str += str(spatial_second_probe_gm) + " for the second probe and "
            hdr_str += str(sampledim) + ' stellar mass bins. \n'
        if mm:
            hdr_str += 'cosmic shear with ' + str(n_tomo_lens) + ' tomographic bins (only the unique bin combinations) and ' + str(spatial_first_probe_mm) + ' spatial bins for the first probe'
            if two_mm:
                hdr_str += " and " + str(spatial_second_probe_mm) + " for the second probe"
            hdr_str += '.\n'
        if self.has_csmf:
            csmf_description, csmf_notation, csmf_order = self._csmf_header(obs_dict)
            hdr_str += csmf_description

        hdr_str += "The structure of the diagonal blocks is, using the following notation\n"
        if gg:
            hdr_str += "galaxy clustering: gg^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n"
        if gm:
            hdr_str += "galaxy galaxy lensing: gm^{stellar mass sample bin}_{tomographic bin i tomographic bin j}(spatial bin k)\n"
        if mm:
            hdr_str += "cosmic shear: mm_{tomographic bin i tomographic bin j}(spatial bin k)\n"
        if self.has_csmf:
            hdr_str += csmf_notation
        diagonal = bool(getattr(self, 'csmf_diagonal_lenses', False))
        if gg:
            hdr_str += self._block_order('gg', spatial_first_probe_gg, n_tomo_clust, n_tomo_clust,
                                         sampledim, unique=True, diagonal=diagonal)
            if two_gg:
                hdr_str += "Followed by the second gg summary\n"
                hdr_str += self._block_order('gg', spatial_second_probe_gg, n_tomo_clust, n_tomo_clust,
                                             sampledim, unique=True, diagonal=diagonal)
        if gm:
            hdr_str += self._block_order('gm', spatial_first_probe_gm, n_tomo_clust, n_tomo_lens,
                                         sampledim, diagonal=diagonal)
            if two_gm:
                hdr_str += "Followed by the second gm summary\n"
                hdr_str += self._block_order('gm', spatial_second_probe_gm, n_tomo_clust, n_tomo_lens,
                                             sampledim, diagonal=diagonal)
        if mm:
            hdr_str += self._block_order('mm', spatial_first_probe_mm, n_tomo_lens, n_tomo_lens, unique=True)
            if two_mm:
                hdr_str += "Followed by the second mm summary\n"
                hdr_str += self._block_order('mm', spatial_second_probe_mm, n_tomo_lens, n_tomo_lens, unique=True)
            hdr_str += "The same block is then repeated for the second component summaries of mm.\n"
        if self.has_csmf:
            hdr_str += csmf_order


        if 'matrix' in self.style:
            if not cov_dict['split_gauss']:
                print("Writing matrix output file.")
                fn = self.filename[self.style.index('matrix')]
                if self.save_as_binary:
                    name, extension = os.path.splitext(fn)
                    np.savez(name, covariance = cov2d_total, header = hdr_str)
                else:
                    np.savetxt(fn, cov2d_total, fmt='%.6e', delimiter=' ',
                            newline='\n', header=hdr_str, comments='# ')
            else:
                print("Writing matrix output file.")
                if self.save_as_binary:
                    fn = self.filename[self.style.index('matrix')]
                    name, extension = os.path.splitext(fn)
                    np.savez(name, covariance = cov2d_total, header = hdr_str)
                    fn_gauss = name + "_gauss"
                    fn_nongauss = name + "_nongauss"
                    fn_ssc = name + "_SSC"
                    np.savez(fn_gauss, covariance = cov2d_gauss, header = hdr_str)
                    if self.has_nongauss:
                        np.savez(fn_nongauss, covariance = cov2d_nongauss, header = hdr_str)
                    if self.has_ssc:
                        np.savez(fn_ssc, covariance = cov2d_ssc, header = hdr_str)
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
            if obs_dict['observables']['combinations_clustering'] is not None or obs_dict['observables']['combinations_ggl'] is not None or obs_dict['observables']['combinations_lensing'] is not None:
                cov2d_total_reduced, hdr_reduced = self.select_tomographic_bins_arbitrary(obs_dict, summary, n_tomo_clust, n_tomo_lens, cov2d_total,
                                                                                          sampledim)
                self._write_reduced_matrix(cov2d_total_reduced, hdr_reduced)
                
        
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
                if obs_dict['observables']['est_shear'] in ("cosebi", "C_ell"):
                    # E- and B-modes (C_ell: with intrinsic B-modes)
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
                    sampledim = max(sampledim, sampledim_save)
                    sampledim_save = (gauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_nongauss:
            sample_dim_save = 0
            for idx in range(len(nongauss)):
                try:
                    sampledim = (nongauss[idx].shape)[2]
                    sampledim = max(sampledim, sampledim_save)
                    sampledim_save = (nongauss[idx].shape)[2]
                    break
                except (AttributeError,TypeError):
                    sampledim = -1
        if self.has_ssc:
            sampledim_save = 0
            for idx in range(len(ssc)):
                try:
                    sampledim = (ssc[idx].shape)[2]
                    sampledim = max(sampledim, sampledim_save)
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

        Cell_gg, Cell_gm, Cell_mm = Cells[:3]
        # the (tabulated) intrinsic B-mode spectrum of cosmic shear, if any
        Cell_mmB = Cells[3] if len(Cells) > 3 else None

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

        if Cell_mmB is not None and type(Cell_mmB) is not int:
            ostr_format = '%.10e\t%i\t\t%i\t\t'
            olist_mmB = ["#ell\t\ttomo_i\ttomo_j\tCell_kappakappa_BB"]
            for ellidx, ell in enumerate(ellrange):
                for ti in range(n_tomo_lens):
                    for tj in range(n_tomo_lens):
                        ostr = ostr_format % (ell, ti+1, tj+1)
                        ostr += '%10e\t\t\t' % Cell_mmB[ellidx, 0, ti, tj]
                        olist_mmB.append(ostr)
            fname = self.__add_string_to_filename('kappakappa_BB', self.Cellfile)
            with open(fname, 'w') as file:
                print("Writing '" + fname + "'.")
                for ostr in olist_mmB:
                    file.write("%s\n" % ostr)

        return True
