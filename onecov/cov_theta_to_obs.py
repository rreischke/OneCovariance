import copy
import time

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from onecov.cov_bandpowers import bandpower_real_filters, hann_window
    from onecov.cov_cosebis import cosebi_filters, psi_Qn, psi_Un
    from onecov.cov_theta_space import CovTHETASpace
except:
    from cov_bandpowers import bandpower_real_filters, hann_window
    from cov_cosebis import cosebi_filters, psi_Qn, psi_Un
    from cov_theta_space import CovTHETASpace

ARCMIN = np.pi/180/60
# default number of fine theta bins over the full angular range of a tracer
# group and their spacing (fine_theta_edges) for the filters (if not
# specified in [covTHETAspace settings]: theta_to_obs_bins, theta_to_obs_type);
# chosen for a discretisation error of about 1e-3 of the Gaussian covariance
# (compared with the direct projection from ell space)
DEFAULT_FINE_BINS = {'cosebi': (500, 'log'),      # COSEBIs (logarithmic filters)
                     'psi': (400, 'lin'),         # Psi statistics (polynomials in theta)
                     'bandpowers': (1000, 'loglin'),
                     'arbitrary': (1000, 'loglin'),
                     # rebinning to the correlation functions in coarser bins is exact for
                     # all fine bins, by default the coarse bins themselves
                     'theta': (1, 'log')}
# Gauss-Legendre nodes per fine bin for the bin-integrated filters
NODES_PER_BIN = 8
# estimated relative discretisation error of the summaries (from the
# comparison with half as many fine bins) above which a warning is printed
DISCRETISATION_WARNING = 1e-3
# the summary families of the ten covariance blocks (left, right)
BLOCK_FAMILIES = [('gg', 'gg'), ('gg', 'gm'), ('gg', 'mm'), ('gg', 'mm'), ('gm', 'gm'),
                  ('mm', 'gm'), ('mm', 'gm'), ('mm', 'mm'), ('mm', 'mm'), ('mm', 'mm')]
BLOCK_NAMES = ['gggg', 'gggm', 'ggmmE', 'ggmmB', 'gmgm', 'mmEgm', 'mmBgm', 'mmEmmE', 'mmEmmB', 'mmBmmB']


class RealSpaceFilter:
    """
    A real-space filter R(theta) of a summary statistic, in 1/rad^2, such
    that the summary is int dtheta theta R(theta) xi(theta) with theta in
    radians.

    Parameters
    ----------
    func : callable
        R as a function of theta in arcmin (vectorised), zero outside its
        support.
    edges : array
        angles in arcmin at which R or its derivatives may be discontinuous,
        at least the ends of the support. The fine theta bins have bin
        edges there.
    average : bool
        whether the summary is an average of the correlation function
        (e.g. a top-hat bin): its filter matrix is then normalised to unit
        sum, so that a constant correlation function is reproduced exactly
        also without bin averaging.
    """

    def __init__(self, func, edges, average=False):
        self.func = func
        self.edges = np.atleast_1d(np.asarray(edges, dtype=float))
        self.average = average

    def __call__(self, theta):
        return self.func(np.asarray(theta, dtype=float))

    def scaled(self, factor):
        return RealSpaceFilter(lambda theta: factor*self.func(theta), self.edges, self.average)

    @classmethod
    def tabulated(cls, theta, R, kind='cubic', edges=None):
        """
        Filter tabulated at theta (in arcmin), interpolated with a cubic
        spline in ln(theta) ('cubic', for smooth filters) or linearly
        ('linear', e.g. for filters read from files), zero outside the
        tabulated range.
        """
        theta = np.asarray(theta, dtype=float)
        R = np.asarray(R, dtype=float)
        order = np.argsort(theta)
        theta, R = theta[order], R[order]
        lo, hi = theta[0]*(1 - 1e-10), theta[-1]*(1 + 1e-10)
        if kind == 'cubic':
            spline = CubicSpline(np.log(theta), R)

            def func(t):
                result = np.zeros_like(t)
                inside = (t >= lo) & (t <= hi)
                result[inside] = spline(np.log(t[inside]))
                return result
        else:
            def func(t):
                return np.interp(t, theta, R, left=0.0, right=0.0)
        if edges is None:
            edges = [theta[0], theta[-1]]
            nonzero = np.nonzero(R)[0]
            if len(nonzero) > 0:
                edges += [theta[nonzero[0]], theta[nonzero[-1]]]
        return cls(func, edges)


class RealSpaceFilters:
    """
    The real-space filters of all summaries, per tracer:
        gg : list of RealSpaceFilter, O = int dtheta theta R w
        gm : list of RealSpaceFilter, O = int dtheta theta R gamma_t
        mmE, mmB : lists of pairs (R_+, R_-) of RealSpaceFilter (or None
            for a vanishing filter), O = int dtheta theta [R_+ xi_+ + R_- xi_-]
    For E/B summaries the 'B' pairs are (R_+, -R_-) of the 'E' pairs.
    """

    def __init__(self):
        self.gg = []
        self.gm = []
        self.mmE = []
        self.mmB = []

    def edges(self, which):
        result = []
        for name in which:
            for entry in getattr(self, name):
                for f in (entry if isinstance(entry, tuple) else (entry,)):
                    if f is not None:
                        result.extend(f.edges)
        return np.array(result)


def eb_pairs(R_plus, R_minus):
    """
    The 'E' and 'B' pairs of filters of summaries
    E/B = int dtheta theta [R_+ xi_+ +/- R_- xi_-].
    """
    E = [(p, m) for p, m in zip(R_plus, R_minus, strict=True)]
    B = [(p, None if m is None else m.scaled(-1.0)) for p, m in zip(R_plus, R_minus, strict=True)]
    return E, B


def fine_theta_edges(breakpoints, n_bins, spacing='log'):
    """
    Fine bin edges between the smallest and the largest breakpoint (in
    arcmin) with all breakpoints as bin edges. The bins are equally spaced
    in u(theta) over the full range,
        'log':    u = ln(theta),
        'lin':    u = theta,
        'loglin': u = ln(theta)/ln(theta_max/theta_min)
                      + (theta - theta_min)/(theta_max - theta_min),
    i.e. logarithmic at small and linear at large angles. Between two
    neighbouring breakpoints the spacing is the closest one to that of
    n_bins bins over the full range (at least one bin).
    """
    points = np.sort(np.unique(np.asarray(breakpoints, dtype=float)))
    points = points[points > 0]
    keep = np.concatenate(([True], np.diff(np.log(points)) > 1e-8))
    points = points[keep]
    tmin, tmax = points[0], points[-1]
    if spacing == 'log':
        def u(theta):
            return np.log(theta)
    elif spacing == 'lin':
        def u(theta):
            return theta
    elif spacing == 'loglin':
        def u(theta):
            return np.log(theta/tmin)/np.log(tmax/tmin) + (theta - tmin)/(tmax - tmin)
    else:
        raise Exception("ConfigError: theta_to_obs_type must be 'log', 'lin' or 'loglin', not " +
                        str(spacing) + ".")
    total = u(tmax) - u(tmin)
    edges = [points[0]]
    for lo, hi in zip(points[:-1], points[1:], strict=True):
        n = max(1, int(np.ceil(n_bins*(u(hi) - u(lo))/total - 1e-6)))
        # invert u (monotonic) on a dense grid within the segment
        dense = np.geomspace(lo, hi, 64*n + 1)
        segment = np.interp(np.linspace(u(lo), u(hi), n + 1), u(dense), dense)
        segment[-1] = hi
        edges.extend(segment[1:])
    return np.array(edges)


def filter_matrix(filters, edges, bin_averaging):
    """
    The matrix A[m, i] that maps the correlation function in the fine bins
    i (bin edges edges, in arcmin) to the summaries m,
        A[m, i] = int_{bin i} dtheta theta R_m(theta)   (theta in radians)
    with bin averaging, which is exact if the correlation function is
    constant within the fine bins (the measured correlation functions are
    the pair-weighted averages over the bins). Without bin averaging the
    correlation functions are evaluated at the (logarithmic) bin centres and
    A[m, i] = theta_i^2 R_m(theta_i) Delta ln(theta)_i.

    Returns None if all filters vanish.
    """
    if len(filters) == 0 or all(f is None for f in filters):
        return None
    lo, hi = np.log(edges[:-1]), np.log(edges[1:])
    if bin_averaging:
        x, w = np.polynomial.legendre.leggauss(NODES_PER_BIN)
        ln_theta = .5*(hi + lo)[:, None] + .5*(hi - lo)[:, None]*x[None, :]
        weight = .5*(hi - lo)[:, None]*w[None, :]
    else:
        ln_theta = .5*(hi + lo)[:, None]
        weight = (hi - lo)[:, None]
    theta = np.exp(ln_theta)
    # dtheta theta = theta^2 dln(theta), theta in radians
    weight = weight*(theta*ARCMIN)**2
    A = np.zeros((len(filters), len(edges) - 1))
    for m, f in enumerate(filters):
        if f is not None:
            A[m] = np.sum(f(theta)*weight, axis=1)
            if f.average:
                A[m] /= np.sum(A[m])
    return A


def bin_weights(edges, bin_averaging):
    """
    Relative number of pairs in the bins for a uniform pair density,
    int_bin dtheta theta (or theta^2 Delta ln theta at the logarithmic bin
    centre without bin averaging).
    """
    if bin_averaging:
        return edges[1:]**2 - edges[:-1]**2
    return edges[1:]*edges[:-1]*np.log(edges[1:]/edges[:-1])


def merged_filter_matrix(filters, edges, breakpoints, bin_averaging):
    """
    The filter matrix of the summaries measured from half as many bins:
    neighbouring pairs of fine bins (between two breakpoints) are merged,
    the correlation function in a merged bin being the pair-weighted average
    of the fine bins (for a uniform pair density). As a map of the fine
    bins, A_half[m, i] = A_coarse[m, J(i)] w_i/w_J(i) with the coarse bin J(i)
    of the fine bin i and the pair weights w (bin_weights).
    """
    if len(filters) == 0 or all(f is None for f in filters):
        return None
    is_break = np.any(np.isclose(edges[:, None], np.asarray(breakpoints)[None, :], rtol=1e-9, atol=0), axis=1)
    keep = np.zeros(len(edges), dtype=bool)
    keep[0], keep[-1] = True, True
    count = 0
    for k in range(1, len(edges) - 1):
        if is_break[k]:
            keep[k], count = True, 0
        else:
            count += 1
            keep[k] = count % 2 == 0
    coarse = edges[keep]
    J = np.searchsorted(coarse, np.sqrt(edges[1:]*edges[:-1])) - 1
    w = bin_weights(edges, bin_averaging)
    w_coarse = np.bincount(J, w, minlength=len(coarse) - 1)
    A_coarse = filter_matrix(filters, coarse, bin_averaging)
    return A_coarse[:, J]*(w/w_coarse[J])[None, :]


def _contract(A_left, cov, A_right):
    """
    sum_ij A_left[m, i] cov[i, j, ...] A_right[n, j], or 0 if one of them
    vanishes.
    """
    if A_left is None or A_right is None or not isinstance(cov, np.ndarray):
        return 0
    tmp = np.tensordot(A_left, cov, axes=(1, 0))
    return np.moveaxis(np.tensordot(A_right, tmp, axes=(1, 1)), 0, 1)


def _contract_first(A, cov):
    """
    sum_i A[m, i] cov[i, ...], or 0 if one of them vanishes.
    """
    if A is None or not isinstance(cov, np.ndarray):
        return 0
    return np.tensordot(A, cov, axes=(1, 0))


# for each real-space covariance block, the pairs of filter matrices it is
# contracted with ('gg', 'gm': clustering and galaxy-galaxy lensing filters;
# 'PE', 'ME', 'PB', 'MB': the xi_+ (P) and xi_- (M) parts of the E and B shear
# summaries), see CovTHETAtoOBS._theta_block
CONTRACTIONS = {'ww': [('gg', 'gg')],
                'wgt': [('gg', 'gm')],
                'wxip': [('gg', 'PE'), ('gg', 'PB')],
                'wxim': [('gg', 'ME'), ('gg', 'MB')],
                'gtgt': [('gm', 'gm')],
                'xipgt': [('PE', 'gm'), ('PB', 'gm')],
                'ximgt': [('ME', 'gm'), ('MB', 'gm')],
                'xipxip': [('PE', 'PE'), ('PE', 'PB'), ('PB', 'PB')],
                'xipxim': [('PE', 'ME'), ('PE', 'MB'), ('PB', 'ME'), ('PB', 'MB')],
                'ximxim': [('ME', 'ME'), ('ME', 'MB'), ('MB', 'MB')]}


# number of array elements of the entries buffered by a _ContractedBlock
# before they are mapped in one matrix product
CONTRACTION_BUFFER_SIZE = 8_000_000


class _ContractedBlock:
    """
    Stands in for a block of the real-space covariance in the fine bins
    (CovTHETASpace._theta_target): the entries block[(i, j)] = value set by
    the projections are mapped to the summaries right away (buffered and
    mapped in matrix products), so that the block in the fine bins is never
    stored. Holds sum_ij A_L[m, i] A_R[n, j] block[i, j] for all pairs of
    filter matrices (A_L, A_R) of the block, stacked along the first axis
    (as CovTHETAtoOBS._theta_block). The entries are added up, so every
    entry must be set once; setting the entry just set again overwrites it
    (the mirror entries of the projections, i = j).
    """
    accumulates = True

    def __init__(self, pairs, sizes, rest_shape):
        self.pairs = pairs
        self.rest_shape = tuple(rest_shape)
        self.out = np.zeros((len(pairs),) + tuple(sizes) + self.rest_shape)
        self.capacity = max(1, CONTRACTION_BUFFER_SIZE//max(int(np.prod(self.rest_shape)), 1))
        self.rows, self.cols, self.values = [], [], []
        self.last = (None, 0)

    def __setitem__(self, index, value):
        if index == self.last[0]:
            # the entry just set is overwritten
            if value is self.last[1]:
                return
            previous, self.last = self.last[1], (index, value)
            value = value - previous
        else:
            self.last = (index, value)
        if not isinstance(value, np.ndarray):
            if value != 0:
                raise ValueError("_ContractedBlock: only arrays (or 0) can be set.")
            return
        self.rows.append(index[0])
        self.cols.append(index[1])
        self.values.append(value)
        if len(self.values) >= self.capacity:
            self.flush()

    def __getitem__(self, index):
        # only the entry just set can be read (the mirror entries of the projections)
        if index != self.last[0]:
            raise KeyError("_ContractedBlock: only the entry just set can be read.")
        return self.last[1]

    def flush(self):
        if not self.values:
            return
        rows, cols = np.array(self.rows), np.array(self.cols)
        values = np.reshape(np.array(self.values), (len(rows), -1))
        for k, (A_L, A_R) in enumerate(self.pairs):
            if A_L is None or A_R is None:
                continue
            weights = (A_L[:, rows][:, None, :]*A_R[:, cols][None, :, :]).reshape(-1, len(rows))
            self.out[k] += np.reshape(weights @ values, self.out.shape[1:])
        self.rows, self.cols, self.values = [], [], []

    def add_diagonal(self, diagonal):
        """
        Adds a block which is diagonal in the fine bins, given its diagonal
        with shape (fine bins, ...).
        """
        flat = np.reshape(diagonal, (len(diagonal), -1))
        for k, (A_L, A_R) in enumerate(self.pairs):
            if A_L is None or A_R is None:
                continue
            weights = (A_L[:, None, :]*A_R[None, :, :]).reshape(-1, len(diagonal))
            self.out[k] += np.reshape(weights @ flat, self.out.shape[1:])

    def add_product(self, matrix, left, right):
        """
        Adds matrix x left(theta_1) x right(theta_2) (see
        CovTHETASpace._theta_add_product): the signals are mapped first.
        """
        for k, (A_L, A_R) in enumerate(self.pairs):
            if A_L is None or A_R is None:
                continue
            left_A = np.tensordot(A_L, left, axes=(1, 0))
            right_A = np.tensordot(A_R, right, axes=(1, 0))
            self.out[k] += matrix[None, None]*(left_A[:, None, :, None, :, :, None, None]
                                               *right_A[None, :, None, :, None, None, :, :])

    def add_full(self, block):
        """
        Adds a block given in the fine bins (e.g. replaced terms).
        """
        for k, (A_L, A_R) in enumerate(self.pairs):
            part = _contract(A_L, block, A_R)
            if isinstance(part, np.ndarray):
                self.out[k] += part

    def result(self):
        self.flush()
        return self.out


def _part(contracted, k):
    """
    The k-th contraction of a contracted block (or 0).
    """
    return contracted[k] if isinstance(contracted, np.ndarray) else 0


def _swap(block):
    """
    The covariance block of the exchanged observables, i.e. with the
    angular, sample and tomographic indices of the two observables
    exchanged (blocks with shape (x1, x2, s1, s2, t1, t2, t3, t4)).
    """
    if not isinstance(block, np.ndarray):
        return block
    return block.transpose(1, 0, 3, 2, 6, 7, 4, 5)


class CovTHETAtoOBS(CovTHETASpace):
    """
    This class calculates the covariance of summary statistics that are
    linear in the real-space correlation functions,
        O_m = int dtheta theta R_m(theta) w(theta),
        O_m = int dtheta theta R_m(theta) gamma_t(theta),
        O_m = int dtheta theta [R_+m(theta) xi_+(theta) + R_-m(theta) xi_-(theta)],
    directly from the real-space covariance (CovTHETASpace) in fine angular
    bins. The summaries are the measured ones, i.e. they are computed from
    the correlation functions in the fine bins, so that
        Cov(O_m, O_n) = sum_ij A_mi Cov(xi_i, xi_j) A_nj
    with A_mi = int_{bin i} dtheta theta R_m(theta) (see filter_matrix).
    Every term of the real-space covariance (Gaussian sample variance, mixed
    term and shot noise, non-Gaussian, super-sample, stellar mass function
    cross terms) is mapped this way, independently of how it was obtained.

    The filters are
        - COSEBIs (shear) and Psi statistics (clustering, galaxy-galaxy
          lensing) for the estimator 'cosebi' ([covCOSEBI settings]),
        - bandpowers for the estimator 'bandpowers' ([covbandpowers
          settings]),
        - the correlation functions in the bins of [covTHETAspace settings]
          for the estimators 'xi_pm', 'gamma_t' and 'w', i.e. a rebinning of
          the fine bins (pair-weighted for a uniform pair density),
        - the real space filters of the arbitrary summary statistics
          (arb_real_filter_*_file) if do_arbitrary_obs is set.

    The fine bins are set separately for clustering and galaxy-galaxy
    lensing (which share them) and for cosmic shear, with all edges of the
    filters (support, apodisation, top-hats) as bin edges. Their number over
    the full angular range and their spacing (log, lin or loglin, see
    fine_theta_edges) are set by '[covTHETAspace settings]:
    theta_to_obs_bins' and 'theta_to_obs_type' (or their _clustering and
    _lensing versions), with defaults for the filters (DEFAULT_FINE_BINS).
    The summaries from the fine bins differ from those with the continuous
    filters by an error proportional to the square of the bin width; it is
    estimated from the summaries measured from half as many bins and
    printed (self.discretisation_error).

    Every real-space block is mapped as soon as it is complete
    (_theta_block), so that the blocks in the fine bins are not all kept at
    the same time. Terms of the real-space covariance can be replaced by
    overriding _theta_block (before calling this class's version).

    Parameters
    ----------
    see CovTHETASpace

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_theta_to_obs import CovTHETAtoOBS
    inp = Input()
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput(bias)
    read_in_tables = fileinp.read_input(inp.config_name)
    covobs = CovTHETAtoOBS(covterms, observables, output, cosmo, bias,
        iA, hod, survey_params, prec, read_in_tables)
    gauss, nongauss, ssc = covobs.calc_covTHETAtoOBS(observables, output,
        bias, hod, survey_params, prec, read_in_tables)
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
        observables = obs_dict['observables']
        do_gg = bool(observables['clustering'])
        do_gm = bool(observables['ggl'])
        do_mm = bool(observables['cosmic_shear'])
        self.arbitrary_filters = bool(obs_dict['arbitrary_summary']['do_arbitrary_summary'])
        self.filters = self.__get_filters(obs_dict, read_in_tables, do_gg, do_gm, do_mm)
        self.theta_obs_dict = self.__theta_space_obs_dict(obs_dict, do_gg, do_gm, do_mm)
        # map every real-space block to the summaries as soon as it is
        # complete (_theta_block); False keeps the full real-space blocks and
        # maps them at the end (same result, much more memory)
        self.contract_blocks = True
        self.A_gg, self.A_gm = None, None
        self.A_mmE, self.A_mmB = (None, None), (None, None)
        CovTHETASpace.__init__(self,
                               cov_dict,
                               self.theta_obs_dict,
                               output_dict,
                               cosmo_dict,
                               bias_dict,
                               iA_dict,
                               hod_dict,
                               survey_params_dict,
                               prec,
                               read_in_tables)
        self.__get_filter_matrices()
        # the projected quantity of the output (summary labels)
        self.obs_proj_quant = self.obs_bins_lensing if self.obs_bins_lensing is not None \
            else self.obs_bins_clustering

    def __get_filters(self, obs_dict, read_in_tables, do_gg, do_gm, do_mm):
        """
        Collects the real-space filters of all summaries (RealSpaceFilters)
        and the labels of the summaries for the output.
        """
        filters = RealSpaceFilters()
        self.obs_bins_clustering = None
        self.obs_bins_lensing = None
        # number and spacing of the fine bins suited to the filters (DEFAULT_FINE_BINS)
        self.default_fine_bins = {'clustering': DEFAULT_FINE_BINS['theta'], 'lensing': DEFAULT_FINE_BINS['theta']}
        if self.arbitrary_filters:
            # real-space filters of the arbitrary summary statistics (theta in
            # arcmin, R in 1/rad^2), all summaries of a tracer in turn; the
            # shear filters give E and B summaries as documented in config.ini
            tabs = read_in_tables['arb_summary']

            def tabulated(thetas, filters_R):
                result = []
                for theta, R in zip(thetas, filters_R, strict=True):
                    f = RealSpaceFilter.tabulated(theta, R, kind='linear')
                    # top-hats (R or theta R constant on their support) are
                    # correlation functions in bins (as in CovARBsummary):
                    # averages, whatever the normalisation of the file
                    R_values = np.asarray(R)
                    nonzero = R_values != 0
                    for values in ((np.asarray(theta)*R_values)[nonzero][1:-1], R_values[nonzero][1:-1]):
                        if len(values) > 1 and np.all(np.round(values/values[0], 4) == 1.):
                            f.average = True
                    result.append(f)
                return result

            def shear_pairs(R_plus, R_minus):
                # E/B summaries (config.ini), except for correlation functions in
                # bins: then the 'E' summary is xi_+ and the 'B' summary xi_-
                # (as for their Fourier filters K_0 and K_4)
                E, B = eb_pairs(R_plus, R_minus)
                for i, (p, m) in enumerate(zip(R_plus, R_minus, strict=True)):
                    if p.average and m.average:
                        E[i], B[i] = (p, None), (None, m)
                return E, B

            # unknown filters: fine at small and at large angles
            self.default_fine_bins = {'clustering': DEFAULT_FINE_BINS['arbitrary'],
                                      'lensing': DEFAULT_FINE_BINS['arbitrary']}
            if do_gg:
                filters.gg = tabulated(tabs['theta_gg'], tabs['RL_gg'])
            if do_gm:
                filters.gm = tabulated(tabs['theta_gm'], tabs['RL_gm'])
            if do_gg or do_gm:
                self.obs_bins_clustering = np.arange(1, max(len(filters.gg), len(filters.gm)) + 1)
            if do_mm:
                filters.mmE, filters.mmB = shear_pairs(tabulated(tabs['theta_mm_p'], tabs['RL_mm_p']),
                                                       tabulated(tabs['theta_mm_m'], tabs['RL_mm_m']))
                self.obs_bins_lensing = np.arange(1, len(filters.mmE) + 1)
            return filters
        est = obs_dict['observables']
        if do_gg or do_gm:
            # the estimators of clustering and galaxy-galaxy lensing must be of the same kind
            kinds = set()
            if do_gg:
                kinds.add({'w': 'theta', 'cosebi': 'cosebi', 'bandpowers': 'bandpowers'}.get(est['est_clust']))
            if do_gm:
                kinds.add({'gamma_t': 'theta', 'cosebi': 'cosebi', 'bandpowers': 'bandpowers'}.get(est['est_ggl']))
            if len(kinds) != 1 or None in kinds:
                raise Exception("ConfigError: theta_to_obs requires the same kind of estimator for clustering "
                                "and galaxy-galaxy lensing (w/gamma_t, cosebi or bandpowers).")
            kind = kinds.pop()
            if kind == 'theta':
                edges = self.__coarse_theta_edges(obs_dict['THETAspace'], 'clustering')
                tophats = self.__tophat_filters(edges)
                self.obs_bins_clustering = np.exp(.5*(np.log(edges[1:]) + np.log(edges[:-1])))
                if do_gg:
                    filters.gg = tophats
                if do_gm:
                    filters.gm = tophats
            elif kind == 'cosebi':
                # U_n and Q_n are polynomials in theta (not ln theta)
                self.default_fine_bins['clustering'] = DEFAULT_FINE_BINS['psi']
                settings = obs_dict['COSEBIs']
                tmin, tmax = settings['theta_min_clustering'], settings['theta_max_clustering']
                n_modes = settings['En_modes_clustering']
                theta = np.geomspace(tmin*ARCMIN, tmax*ARCMIN, 10000)
                Un = psi_Un(tmin*ARCMIN, tmax*ARCMIN, theta, n_modes)
                Qn = psi_Qn(Un, theta)
                self.obs_bins_clustering = np.arange(1, n_modes + 1)
                if do_gg:
                    filters.gg = [RealSpaceFilter.tabulated(theta/ARCMIN, Un[n], edges=[tmin, tmax])
                                  for n in range(n_modes)]
                if do_gm:
                    filters.gm = [RealSpaceFilter.tabulated(theta/ARCMIN, Qn[n], edges=[tmin, tmax])
                                  for n in range(n_modes)]
            else:
                # the bandpower filters oscillate with ell theta
                self.default_fine_bins['clustering'] = DEFAULT_FINE_BINS['bandpowers']
                bp = self.__bandpower_filters(obs_dict['bandpowers'], 'clustering')
                self.obs_bins_clustering = bp['ell_bins']
                if do_gg:
                    filters.gg = bp['gg']
                if do_gm:
                    filters.gm = bp['gm']
        if do_mm:
            if est['est_shear'] == 'xi_pm':
                edges = self.__coarse_theta_edges(obs_dict['THETAspace'], 'lensing')
                tophats = self.__tophat_filters(edges)
                self.obs_bins_lensing = np.exp(.5*(np.log(edges[1:]) + np.log(edges[:-1])))
                # the 'E' summaries are xi_+, the 'B' summaries xi_-
                filters.mmE = [(f, None) for f in tophats]
                filters.mmB = [(None, f) for f in tophats]
            elif est['est_shear'] == 'cosebi':
                self.default_fine_bins['lensing'] = DEFAULT_FINE_BINS['cosebi']
                settings = obs_dict['COSEBIs']
                tmin, tmax = settings['theta_min_lensing'], settings['theta_max_lensing']
                n_modes = settings['En_modes_lensing']
                theta, Tp, Tm = cosebi_filters(tmin*ARCMIN, tmax*ARCMIN, n_modes,
                                               settings['dimensionless_cosebi'])
                self.obs_bins_lensing = np.arange(1, n_modes + 1)
                # E_n/B_n = 1/2 int dtheta theta [T_+n xi_+ +/- T_-n xi_-]
                filters.mmE, filters.mmB = eb_pairs(
                    [RealSpaceFilter.tabulated(theta/ARCMIN, Tp[n]/2, edges=[tmin, tmax]) for n in range(n_modes)],
                    [RealSpaceFilter.tabulated(theta/ARCMIN, Tm[n]/2, edges=[tmin, tmax]) for n in range(n_modes)])
            elif est['est_shear'] == 'bandpowers':
                self.default_fine_bins['lensing'] = DEFAULT_FINE_BINS['bandpowers']
                bp = self.__bandpower_filters(obs_dict['bandpowers'], 'lensing')
                self.obs_bins_lensing = bp['ell_bins']
                filters.mmE, filters.mmB = eb_pairs(bp['mm_plus'], bp['mm_minus'])
            else:
                raise Exception("ConfigError: theta_to_obs is not available for the cosmic shear estimator " +
                                str(est['est_shear']) + ", only for xi_pm, cosebi and bandpowers.")
        return filters

    @staticmethod
    def __coarse_theta_edges(settings, tracer):
        """
        The bin edges (in arcmin) of the correlation functions as specified
        in [covTHETAspace settings] for tracer 'clustering' or 'lensing'.
        """
        kind = settings['theta_type_' + tracer]
        if kind == 'list':
            return np.array(settings['theta_list_' + tracer], dtype=float)
        lo, hi, n = settings['theta_min_' + tracer], settings['theta_max_' + tracer], settings['theta_bins_' + tracer]
        if kind == 'lin':
            return np.linspace(lo, hi, n + 1)
        return np.geomspace(lo, hi, n + 1)

    @staticmethod
    def __tophat_filters(edges):
        """
        Filters of the correlation functions in the bins with edges (in
        arcmin): R = 1/int_bin dtheta theta on the bin, i.e. the
        average over the bin weighted with theta (the number of pairs for a
        uniform pair density).
        """
        filters = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            norm = 2.0/((hi*ARCMIN)**2 - (lo*ARCMIN)**2)
            filters.append(RealSpaceFilter(
                lambda theta, lo=lo, hi=hi, norm=norm: np.where((theta >= lo) & (theta < hi), norm, 0.0),
                [lo, hi], average=True))
        return filters

    @staticmethod
    def __bandpower_filters(settings, tracer):
        """
        The real-space filters of the bandpowers (bandpower_real_filters)
        with the Hann apodisation of [covbandpowers settings] for tracer
        'clustering' or 'lensing'.
        """
        width = settings['apodisation_log_width_' + tracer]
        theta_lo, theta_up = settings['theta_lo_' + tracer], settings['theta_up_' + tracer]
        if settings['ell_type_' + tracer] == 'lin':
            ell_ul = np.linspace(settings['ell_min_' + tracer], settings['ell_max_' + tracer],
                                 settings['ell_bins_' + tracer] + 1)
            ell_bins = .5*(ell_ul[1:] + ell_ul[:-1])
        else:
            ell_ul = np.geomspace(settings['ell_min_' + tracer], settings['ell_max_' + tracer],
                                  settings['ell_bins_' + tracer] + 1)
            ell_bins = np.exp(.5*(np.log(ell_ul[1:]) + np.log(ell_ul[:-1])))
        edges = [theta_lo*np.exp(-width/2), theta_lo*np.exp(width/2),
                 theta_up*np.exp(-width/2), theta_up*np.exp(width/2)]

        def make(ell_lo, ell_up, kind, prefactor):
            def func(theta):
                T = hann_window(theta, theta_lo, theta_up, width)
                result = np.zeros_like(theta)
                inside = T > 0
                g_plus, g_minus, h = bandpower_real_filters(theta[inside]*ARCMIN, ell_lo, ell_up)
                result[inside] = prefactor*T[inside]*{'plus': g_plus, 'minus': g_minus, 'h': h}[kind]
                return result
            return RealSpaceFilter(func, edges)

        result = {'ell_bins': ell_bins, 'gg': [], 'gm': [], 'mm_plus': [], 'mm_minus': []}
        for ell_lo, ell_up in zip(ell_ul[:-1], ell_ul[1:], strict=True):
            norm = np.log(ell_up/ell_lo)
            result['gg'].append(make(ell_lo, ell_up, 'plus', 2*np.pi/norm))
            result['gm'].append(make(ell_lo, ell_up, 'h', 2*np.pi/norm))
            result['mm_plus'].append(make(ell_lo, ell_up, 'plus', np.pi/norm))
            result['mm_minus'].append(make(ell_lo, ell_up, 'minus', np.pi/norm))
        return result

    def __theta_space_obs_dict(self, obs_dict, do_gg, do_gm, do_mm):
        """
        The observables dictionary for the real-space covariance in the
        fine bins: correlation functions as estimators, fine bin edges
        including all filter edges.
        """
        theta_obs = copy.deepcopy(obs_dict)
        theta_obs['observables']['est_shear'] = 'xi_pm'
        theta_obs['observables']['est_ggl'] = 'gamma_t'
        theta_obs['observables']['est_clust'] = 'w'
        theta_obs['arbitrary_summary']['do_arbitrary_summary'] = False
        settings = theta_obs['THETAspace']
        settings['xi_pp'] = True
        settings['xi_pm'] = True
        settings['xi_mm'] = True
        if settings.get('theta_acc') is None:
            settings['theta_acc'] = 1e-5
        if settings.get('bin_averaging') is None:
            settings['bin_averaging'] = True
        if settings.get('integration_intervals') is None:
            settings['integration_intervals'] = 40
        self.fine_edges_clustering = None
        self.fine_edges_lensing = None
        for tracer, active, names in (('clustering', do_gg or do_gm, ('gg', 'gm')),
                                      ('lensing', do_mm, ('mmE', 'mmB'))):
            if not active:
                continue
            n_bins = settings.get('theta_to_obs_bins_' + tracer) or settings.get('theta_to_obs_bins') \
                or self.default_fine_bins[tracer][0]
            spacing = settings.get('theta_to_obs_type_' + tracer) or settings.get('theta_to_obs_type') \
                or self.default_fine_bins[tracer][1]
            edges = fine_theta_edges(self.filters.edges(names), n_bins, spacing)
            setattr(self, 'fine_edges_' + tracer, edges)
            settings['theta_type_' + tracer] = 'list'
            settings['theta_list_' + tracer] = edges
            settings['theta_min_' + tracer] = edges[0]
            settings['theta_max_' + tracer] = edges[-1]
            settings['theta_bins_' + tracer] = len(edges) - 1
        # the common bins are used for the triplet mixed term (cosmic shear)
        edges = self.fine_edges_lensing if do_mm else self.fine_edges_clustering
        settings['theta_type'] = 'list'
        settings['theta_list'] = edges
        settings['theta_min'] = edges[0]
        settings['theta_max'] = edges[-1]
        settings['theta_bins'] = len(edges) - 1
        return theta_obs

    def __get_filter_matrices(self):
        """
        The filter matrices (filter_matrix) of all summaries on the fine
        bins.
        """
        bin_averaging = self.theta_obs_dict['THETAspace']['bin_averaging']
        # the summaries from half as many bins are appended as further rows
        # (the estimate of the discretisation error, calc_covTHETAtoOBS)
        self.n_summaries = {'gg': len(self.filters.gg), 'gm': len(self.filters.gm), 'mm': len(self.filters.mmE)}

        def matrix(filters, edges, breakpoints):
            A = filter_matrix(filters, edges, bin_averaging)
            if A is None:
                return None
            return np.vstack((A, merged_filter_matrix(filters, edges, breakpoints, bin_averaging)))

        if self.gg or self.gm:
            breakpoints = self.filters.edges(('gg', 'gm'))
            self.A_gg = matrix(self.filters.gg, self.theta_ul_bins_clustering, breakpoints)
            self.A_gm = matrix(self.filters.gm, self.theta_ul_bins_clustering, breakpoints)
        if self.mm:
            breakpoints = self.filters.edges(('mmE', 'mmB'))
            for name in ('mmE', 'mmB'):
                pairs = getattr(self.filters, name)
                setattr(self, 'A_' + name,
                        (matrix([p for p, _ in pairs], self.theta_ul_bins_lensing, breakpoints),
                         matrix([m for _, m in pairs], self.theta_ul_bins_lensing, breakpoints)))
        print("Mapping the real-space covariance in " +
              ", ".join(str(len(edges) - 1) + " fine " + tracer + " bins"
                        for tracer, edges in (('clustering', self.fine_edges_clustering),
                                              ('lensing', self.fine_edges_lensing))
                        if edges is not None) +
              " to the summary statistics.")

    def __contracted_block(self, name, rest_shape):
        """
        An empty _ContractedBlock for the real-space block name with the
        pairs of filter matrices of CONTRACTIONS[name].
        """
        matrices = {'gg': self.A_gg, 'gm': self.A_gm,
                    'PE': self.A_mmE[0], 'ME': self.A_mmE[1], 'PB': self.A_mmB[0], 'MB': self.A_mmB[1]}
        # (with the rows of the summaries from half as many bins)
        sizes = {'gg': 2*self.n_summaries['gg'], 'gm': 2*self.n_summaries['gm'],
                 'PE': 2*self.n_summaries['mm'], 'ME': 2*self.n_summaries['mm'],
                 'PB': 2*self.n_summaries['mm'], 'MB': 2*self.n_summaries['mm']}
        left, right = CONTRACTIONS[name][0]
        return _ContractedBlock([(matrices[L], matrices[R]) for L, R in CONTRACTIONS[name]],
                                (sizes[left], sizes[right]), rest_shape)

    def _theta_target(self, name, shape):
        """
        A _ContractedBlock instead of the real-space block in the fine bins
        (see CovTHETASpace._theta_target).
        """
        if not self.contract_blocks:
            return CovTHETASpace._theta_target(self, name, shape)
        return self.__contracted_block(name, shape[2:])

    def _theta_diagonal(self, name, diagonal):
        if not self.contract_blocks:
            return CovTHETASpace._theta_diagonal(self, name, diagonal)
        block = self.__contracted_block(name, diagonal.shape[1:])
        block.add_diagonal(diagonal)
        return block

    def _theta_add_product(self, block, matrix, left, right):
        if isinstance(block, _ContractedBlock):
            block.add_product(matrix, left, right)
            return block
        return CovTHETASpace._theta_add_product(self, block, matrix, left, right)

    def _theta_block(self, name, block):
        """
        The mapped real-space covariance block (see
        CovTHETASpace._theta_block): the contractions A_L block A_R^T for all
        pairs (L, R) of CONTRACTIONS[name], stacked along a new first axis.
        Blocks given in the fine bins (e.g. terms replaced by overriding this
        method) are mapped here.
        """
        if not self.contract_blocks:
            return block
        if isinstance(block, _ContractedBlock):
            return block.result()
        if not isinstance(block, np.ndarray):
            return block
        contracted = self.__contracted_block(name, block.shape[2:])
        contracted.add_full(block)
        return contracted.result()

    def __shear_shear(self, xipxip, xipxim, ximxim, left, right):
        """
        Covariance of the shear summaries with filter matrices
        left = (P_L, M_L) and right = (P_R, M_R) for xi_+ and xi_-:
        P_L C_++ P_R^T + P_L C_+- M_R^T + M_L C_-+ P_R^T + M_L C_-- M_R^T.
        """
        (P_L, M_L), (P_R, M_R) = left, right
        return _contract(P_L, xipxip, P_R) + _contract(P_L, xipxim, M_R) \
            + _swap(_contract(P_R, xipxim, M_L)) + _contract(M_L, ximxim, M_R)

    def project_blocks(self, blocks):
        """
        Maps the ten real-space covariance blocks
            ['ww', 'wgt', 'wxi+', 'wxi-', 'gtgt', 'xi+gt', 'xi-gt',
             'xi+xi+', 'xi+xi-', 'xi-xi-']
        to the ten blocks of the summaries
            ['gggg', 'gggm', 'ggmmE', 'ggmmB', 'gmgm', 'mmEgm', 'mmBgm',
             'mmEmmE', 'mmEmmB', 'mmBmmB']
        (the layout of CovCOSEBI, CovBandPowers and CovARBsummary). The
        blocks are contracted already (_theta_block) unless
        self.contract_blocks is False.
        """
        ww, wgt, wxip, wxim, gtgt, xipgt, ximgt, xipxip, xipxim, ximxim = blocks
        if self.contract_blocks:
            # E = P + M, B = P' + M' (the xi_+ and xi_- parts); the xi_- xi_+
            # terms are the exchanged xi_+ xi_- contractions
            return [_part(ww, 0),
                    _part(wgt, 0),
                    _part(wxip, 0) + _part(wxim, 0),
                    _part(wxip, 1) + _part(wxim, 1),
                    _part(gtgt, 0),
                    _part(xipgt, 0) + _part(ximgt, 0),
                    _part(xipgt, 1) + _part(ximgt, 1),
                    _part(xipxip, 0) + _part(xipxim, 0) + _swap(_part(xipxim, 0)) + _part(ximxim, 0),
                    _part(xipxip, 1) + _part(xipxim, 1) + _swap(_part(xipxim, 2)) + _part(ximxim, 1),
                    _part(xipxip, 2) + _part(xipxim, 3) + _swap(_part(xipxim, 3)) + _part(ximxim, 2)]
        (P_E, M_E), (P_B, M_B) = self.A_mmE, self.A_mmB
        return [_contract(self.A_gg, ww, self.A_gg),
                _contract(self.A_gg, wgt, self.A_gm),
                _contract(self.A_gg, wxip, P_E) + _contract(self.A_gg, wxim, M_E),
                _contract(self.A_gg, wxip, P_B) + _contract(self.A_gg, wxim, M_B),
                _contract(self.A_gm, gtgt, self.A_gm),
                _contract(P_E, xipgt, self.A_gm) + _contract(M_E, ximgt, self.A_gm),
                _contract(P_B, xipgt, self.A_gm) + _contract(M_B, ximgt, self.A_gm),
                self.__shear_shear(xipxip, xipxim, ximxim, self.A_mmE, self.A_mmE),
                self.__shear_shear(xipxip, xipxim, ximxim, self.A_mmE, self.A_mmB),
                self.__shear_shear(xipxip, xipxim, ximxim, self.A_mmB, self.A_mmB)]

    def project_csmf(self, terms):
        """
        Maps the stellar mass function terms [auto, w, gt, xi+, xi-] (the
        cross terms with the real-space angle as first axis) to
        [auto, gg, gm, mmE, mmB].
        """
        auto, w, gt, xip, xim = terms
        (P_E, M_E), (P_B, M_B) = self.A_mmE, self.A_mmB
        return [auto,
                _contract_first(self.A_gg, w),
                _contract_first(self.A_gm, gt),
                _contract_first(P_E, xip) + _contract_first(M_E, xim),
                _contract_first(P_B, xip) + _contract_first(M_B, xim)]

    def project_terms(self, terms):
        """
        Maps a list of real-space covariance terms as returned by
        calc_covTHETA (10 blocks, or 30 with the split Gaussian term, each
        block's sample variance, mixed and shot noise term in turn,
        followed by the 5 stellar mass function terms if requested) to the
        summaries.
        """
        terms = list(terms)
        n_main = len(terms) - (5 if self.csmf else 0)
        main, tail = terms[:n_main], terms[n_main:]
        if n_main == 30:
            parts = [self.project_blocks(main[k::3]) for k in range(3)]
            main = [parts[k][i] for i in range(10) for k in range(3)]
        elif n_main == 10:
            main = self.project_blocks(main)
        else:
            raise Exception("Unexpected number of real-space covariance terms: " + str(len(terms)))
        if tail:
            tail = self.project_csmf(tail)
        return main + tail

    def __split_check(self, terms):
        """
        Splits mapped terms (project_terms) into the summaries from the fine
        bins and those from half as many bins (the rows appended to the
        filter matrices).
        """
        n = self.n_summaries
        n_main = len(terms) - (5 if self.csmf else 0)
        main, check = [], []
        for i, term in enumerate(terms):
            if not isinstance(term, np.ndarray):
                main.append(term)
                check.append(term)
            elif i < n_main:
                left, right = BLOCK_FAMILIES[i//3 if n_main == 30 else i]
                main.append(term[:n[left], :n[right]])
                check.append(term[n[left]:, n[right]:])
            elif i == n_main:
                # the auto-covariance of the stellar mass function
                main.append(term)
                check.append(term)
            else:
                family = ['gg', 'gm', 'mm', 'mm'][i - n_main - 1]
                main.append(term[:n[family]])
                check.append(term[n[family]:])
        return main, check

    def __discretisation_error(self, main, check):
        """
        Estimated discretisation error of the ten blocks (all terms
        together): the summaries from the fine bins differ from the
        continuous ones by about a third of their difference to those from
        half as many bins (the error decreases with the square of the bin
        width), relative to sqrt(max|C_LL| max|C_RR|) with the
        auto-covariances of the two summaries of the block (i.e. on the
        scale of the correlation coefficients, also for blocks which almost
        vanish, e.g. E x B).
        """
        totals = []
        for i in range(len(BLOCK_NAMES)):
            total_main, total_check = 0, 0
            for terms_main, terms_check in zip(main, check, strict=True):
                n_main = len(terms_main) - (5 if self.csmf else 0)
                idx = [3*i, 3*i + 1, 3*i + 2] if n_main == 30 else [i]
                for j in idx:
                    total_main = total_main + terms_main[j]
                    total_check = total_check + terms_check[j]
            totals.append((total_main, total_check))
        # the auto-covariance blocks of gg, gm, mmE and mmB
        auto = {'gg': 0, 'gm': 4, 'mmE': 7, 'mmB': 9}
        sides = [('gg', 'gg'), ('gg', 'gm'), ('gg', 'mmE'), ('gg', 'mmB'), ('gm', 'gm'),
                 ('mmE', 'gm'), ('mmB', 'gm'), ('mmE', 'mmE'), ('mmE', 'mmB'), ('mmB', 'mmB')]
        scale = {key: np.max(np.abs(totals[i][0])) if isinstance(totals[i][0], np.ndarray) else 0
                 for key, i in auto.items()}
        errors = {}
        for name, (total_main, total_check), (left, right) in zip(BLOCK_NAMES, totals, sides, strict=True):
            norm = np.sqrt(scale[left]*scale[right])
            if isinstance(total_main, np.ndarray) and norm > 0:
                errors[name] = np.max(np.abs(total_main - total_check))/3/norm
        return errors

    def calc_covTHETAtoOBS(self,
                           obs_dict,
                           output_dict,
                           bias_dict,
                           hod_dict,
                           survey_params_dict,
                           prec,
                           read_in_tables):
        """
        Calculates the covariance of the summary statistics from the
        real-space covariance in the fine bins (see the class description).

        Returns
        -------
        gauss, nongauss, ssc : lists of arrays
            in the layout of CovCOSEBI.calc_covCOSEBI, i.e. each with the
            10 blocks (or 30 with the split Gaussian term)
                ['gggg', 'gggm', 'ggmmE', 'ggmmB', 'gmgm', 'mmEgm',
                 'mmBgm', 'mmEmmE', 'mmEmmB', 'mmBmmB']
            and the stellar mass function terms if requested, each block
            with shape (summaries, summaries, sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)
            For the estimators xi_pm, 'mmE' is xi_+ and 'mmB' xi_-.
        """
        gauss, nongauss, ssc = self.calc_covTHETA(self.theta_obs_dict, output_dict, bias_dict,
                                                  hod_dict, survey_params_dict, prec, read_in_tables)
        t0 = time.time()
        main, check = [], []
        for terms in (gauss, nongauss, ssc):
            terms_main, terms_check = self.__split_check(self.project_terms(terms))
            main.append(terms_main)
            check.append(terms_check)
        print("Mapped the real-space covariance to the summary statistics in " +
              str(round(time.time() - t0, 1)) + "s.")
        self.discretisation_error = self.__discretisation_error(main, check)
        print("Estimated relative discretisation error of the summaries (from the comparison with half as "
              "many fine bins): " + ", ".join(f"{name} {error:.1e}" for name, error in self.discretisation_error.items()))
        if any(error > DISCRETISATION_WARNING for error in self.discretisation_error.values()):
            print("THETAtoOBSWarning: the discretisation error is larger than " + str(DISCRETISATION_WARNING) +
                  ", consider more fine bins ([covTHETAspace settings]: theta_to_obs_bins) or a different "
                  "spacing (theta_to_obs_type).")
        return tuple(main)
