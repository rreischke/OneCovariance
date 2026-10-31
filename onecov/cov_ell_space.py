import hashlib
import logging
import time

import camb
import healpy as hp
import levin
import numpy as np
from camb import model
from scipy.fft import fht, fhtoffset
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import Akima1DInterpolator, RegularGridInterpolator, UnivariateSpline, interp1d
from scipy.linalg import block_diag
from scipy.special import spherical_jn
from scipy.sparse import csr_matrix

DEG2TORAD2 = (180 / np.pi) ** 2
ARCMIN2TORAD2 = (60 * 60) * DEG2TORAD2


try:
    from onecov.cov_halo_model import camb_transfer_functions, simpson_weights
    from onecov.cov_output import Output
    from onecov.cov_polyspectra import PolySpectra
except:
    from cov_halo_model import camb_transfer_functions, simpson_weights
    from cov_output import Output
    from cov_polyspectra import PolySpectra

# maximum wavenumber in h/Mpc up to which CAMB calculates the transfer
# functions for the non-linear matter power spectra
CAMB_KMAX_NONLINEAR = 100.
# number of array elements (8 bytes each) per temporary array in the
# quadrature projections; the work is done in blocks of this size (several
# such arrays exist at a time), so that the memory does not grow with the
# number of tomographic bins or of the estimator modes
PROJECTION_BLOCK_SIZE = 4_000_000
# Gauss-Legendre nodes (in log10 M*) for the averages over the stellar mass
# bins of the stellar mass function: CSMF_MIN_NODES plus CSMF_NODES_PER_DEX
# per dex of bin width (the SMF falls exponentially at the high-mass end)
CSMF_MIN_NODES = 3
CSMF_NODES_PER_DEX = 40
# non-Limber C_ell (_NonLimberProjector): log-uniform comoving distance grid
# (Mpc/h) of the FFTLog transforms, its size and the FFTLog bias for Bessel
# orders >= 2 (negative: stable at low k); lensing-type kernels are
# tapered below NONLIMBER_TAPER_CHI. The correction is computed for
# increasing ell until it stays below NONLIMBER_TOL (relative to
# sqrt(C_aa C_bb)) for NONLIMBER_N_CONVERGED multipoles, at most up to
# NONLIMBER_ELL_MAX.
NONLIMBER_CHI_RANGE = (0.05, 2.0e4)
NONLIMBER_N_CHI = 2**16
NONLIMBER_BIAS = -1.0
NONLIMBER_TAPER_CHI = 5.0
NONLIMBER_TOL = 1e-4
NONLIMBER_N_CONVERGED = 3
NONLIMBER_ELL_MAX = 1000


class LazyWeightsLevin:
    """
    A levin.Levin integrator that passes the weights of init_w_ell on only
    when an integration over them is requested (the cquad methods, all with
    'well' in their name). With projection_method = quadrature the
    projections use the registered weights instead
    (CovELLSpace._register_projection_weights), and the copy of the weights
    in levin, several times their size, is large for many weights (e.g. fine
    real-space bins). All other methods are passed on directly.
    """

    def __init__(self, levin_int):
        self._levin_int = levin_int
        self._pending_weights = None

    def init_w_ell(self, ell, weights):
        # references only, the arrays are not modified afterwards
        self._pending_weights = (ell, weights)

    def __getattr__(self, name):
        if 'well' in name and self._pending_weights is not None:
            ell, weights = self._pending_weights
            self._pending_weights = None
            self._levin_int.init_w_ell(ell, np.ascontiguousarray(weights))
        return getattr(self._levin_int, name)


class EllCovColumns(np.ndarray):
    """
    Four-point ell-space covariance of the projections stored only for the
    tomographic combinations that are calculated: an array (ell, ell,
    column) with tomo_shape, the full shape (sample, sample, tomo, tomo,
    tomo, tomo) of the covariance, and columns, the flat indices (in
    tomo_shape) of the stored columns. All other combinations are zero.
    """

    def __new__(cls, n_ell, tomo_shape, columns):
        obj = np.zeros((n_ell, n_ell, len(columns))).view(cls)
        obj.tomo_shape, obj.columns = tuple(tomo_shape), np.asarray(columns)
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.tomo_shape = getattr(obj, 'tomo_shape', None)
            self.columns = getattr(obj, 'columns', None)


class _LineOfSightIntegrals:
    """
    Line-of-sight integrals target[:, :, *index] = sum_chi weight(chi)
    los_weights(chi) integrand(chi, ell_1, ell_2) for many tomographic
    combinations. Consecutive integrals with the same integrand are collected
    and done as one matrix product (in blocks of PROJECTION_BLOCK_SIZE).
    """

    def __init__(self, target, los_weights):
        self.target = target
        self.los_weights = los_weights
        self.integrand, self.indices, self.weights = None, [], []
        # position of each stored tomographic column of an EllCovColumns
        self.position = None
        if isinstance(target, EllCovColumns):
            self.position = {column: i for i, column in enumerate(target.columns.tolist())}

    @staticmethod
    def _same(a, b):
        # the same data (views are created anew for every call); while it is
        # referenced here, its memory can not be reused by another array
        return a is b or (a.__array_interface__['data'][0] == b.__array_interface__['data'][0]
                          and a.shape == b.shape and a.strides == b.strides)

    def add(self, index, weight, integrand):
        if self.integrand is not None and not self._same(self.integrand, integrand):
            self.flush()
        self.integrand = integrand
        self.indices.append(tuple(index))
        self.weights.append(weight*self.los_weights)

    def _store(self, index, value):
        if self.position is None:
            self.target[(slice(None), slice(None)) + index] = value
        else:
            column = self.position[int(np.ravel_multi_index(index, self.target.tomo_shape))]
            self.target[:, :, column] = value

    def flush(self):
        if self.integrand is None:
            return
        flat = np.reshape(self.integrand, (self.integrand.shape[0], -1))
        weights = np.array(self.weights)
        for rows in CovELLSpace._blocks(len(self.indices), flat.shape[1]):
            for index, value in zip(self.indices[rows], weights[rows] @ flat):
                self._store(index, np.reshape(value, self.integrand.shape[1:]))
        self.integrand, self.indices, self.weights = None, [], []


class _FactorisedLineOfSightIntegrals(_LineOfSightIntegrals):
    """
    Line-of-sight integrals of factorised integrands (super-sample
    covariance), target[:, :, *index] = sum_chi weight(chi) los_weights(chi)
    variance(chi) response_1(chi, ell_1) response_2(chi, ell_2)
    = response_1^T diag(v) response_2. The (chi, ell, ell) integrand is never
    formed; consecutive integrals with the same responses are done as one
    matrix product (in blocks of PROJECTION_BLOCK_SIZE), also if the survey
    variance differs between them.
    """

    def add(self, index, weight, response_1, response_2, variance):
        if self.integrand is not None and not (self._same(self.integrand[0], response_1)
                                               and self._same(self.integrand[1], response_2)):
            self.flush()
        self.integrand = (response_1, response_2)
        self.indices.append(tuple(index))
        self.weights.append(weight*variance*self.los_weights)

    def flush(self):
        if self.integrand is None:
            return
        response_1, response_2 = self.integrand
        response_1_T = np.ascontiguousarray(response_1.T)
        n_ell_1, n_chi = response_1_T.shape
        weights = np.array(self.weights)
        for rows in CovELLSpace._blocks(len(self.indices), n_ell_1*n_chi):
            scaled = response_1_T[None, :, :]*weights[rows, None, :]
            values = (scaled.reshape(-1, n_chi) @ response_2).reshape(len(scaled), n_ell_1, -1)
            for index, value in zip(self.indices[rows], values):
                self._store(index, value)
        self.integrand, self.indices, self.weights = None, [], []


class _NonLimberProjector:
    """
    Exact (non-Limber) and Limber angular power spectra of linear fields
    with separable growth, P(k, chi_1, chi_2) = D(chi_1) D(chi_2) P_ref(k):
    C_ab(ell) = 2/pi int dk k^2 P_ref(k) I_a(k) I_b(k) with
    I_a(k) = int dchi f_a(chi) j_ell(k chi) for density-like kernels and
    I_a(k) = int dchi f_a(chi) j_ell(k chi)/(k chi)^2 for lensing-like
    kernels (the growth D and all ell-dependent prefactors are part of the
    kernels or applied by the caller). The Bessel integrals are FFTLog
    Hankel transforms (scipy.fft.fht) of all kernels at once on a
    log-uniform chi grid; k is the reciprocal log-uniform grid (the same
    for all ell), on which the k integral is a trapezoidal sum in ln k.
    Lensing kernels use j_ell(x)/x^2 = j_{ell-2}/((2ell-1)(2ell+1))
    + j_ell [1/((2ell-1)(2ell+1)) + 1/((2ell+1)(2ell+3))]
    + j_{ell+2}/((2ell+1)(2ell+3)), dividing by (k chi)^2 directly would
    amplify the FFTLog errors at low k.
    """

    def __init__(self, P_ref, chi_range=NONLIMBER_CHI_RANGE,
                 n_chi=NONLIMBER_N_CHI, bias=NONLIMBER_BIAS):
        self.P_ref = P_ref
        self.chi = np.geomspace(chi_range[0], chi_range[1], n_chi)
        self.dln = np.log(self.chi[1]/self.chi[0])
        self.bias = bias
        self.k = self.chi/(self.chi[0]*self.chi[-1])
        self.weights = np.full(n_chi, self.dln)
        self.weights[[0, -1]] *= 0.5

    @staticmethod
    def _x_low(order):
        """
        Largest x below which |j_order| < 1e-12 max|j_order|: I(k)
        vanishes for k chi_max < x_low (chi_max the end of the support).
        """
        x = np.geomspace(1e-6, order + 10, 4000)
        j = np.abs(spherical_jn(int(order), x))
        return x[np.argmax(j > 1e-12*j.max())]

    def _bessel(self, f, order):
        """
        int dchi f(chi) j_order(k chi) for the kernels f (n, n_chi). Below
        the k where j_order is negligible on the whole support of a kernel
        I is set to zero (FFTLog has a small absolute error there).
        """
        bias = 0.0 if order <= 1 else self.bias
        I = np.sqrt(np.pi/2)*self.k**-1.5*fht(f/np.sqrt(self.chi), self.dln,
                                               order + 0.5, bias=bias)
        nonzero = np.abs(f) > 1e-12*np.max(np.abs(f), axis=1, keepdims=True)
        chi_max = np.where(nonzero.any(axis=1),
                           self.chi[len(self.chi) - 1 - np.argmax(nonzero[:, ::-1], axis=1)],
                           self.chi[-1])
        I[self.k[None, :] < self._x_low(order)/chi_max[:, None]] = 0.0
        return I

    def transform(self, f, ell, lensing=False):
        """
        I_a(k) of the kernels f (shape (n, n_chi), on self.chi), returns
        the k grid and I with shape (n, n_chi). Lensing kernels need
        ell >= 2 (zero otherwise).
        """
        f = np.atleast_2d(f)
        ell = int(round(ell))
        if not lensing:
            return self.k, self._bessel(f, ell)
        if ell < 2:
            return self.k, np.zeros_like(f)
        lower = 1/((2*ell - 1)*(2*ell + 1))
        upper = 1/((2*ell + 1)*(2*ell + 3))
        return self.k, lower*self._bessel(f, ell - 2) \
            + (lower + upper)*self._bessel(f, ell) + upper*self._bessel(f, ell + 2)

    def exact(self, I_a, I_b, k):
        return (I_a*(2/np.pi*k**3*self.P_ref(k)*self.weights)) @ I_b.T

    def limber(self, K_a, K_b, ell):
        """
        int dchi/chi^2 K_a K_b P_ref((ell + 1/2)/chi) for kernels K
        (shape (n, n_chi), including the growth and the ell factors).
        """
        weight = self.P_ref((ell + 0.5)/self.chi)/self.chi*self.weights
        return (K_a*weight) @ K_b.T


class CovELLSpace(PolySpectra):
    """
    This class calculates the ell-space covariance for angular power
    spectra estimators (kappa-kappa, tracer-tracer and tracer-kappa).
    Inherits the functionality of the PolySpectra class.

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
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
    cosmo_dict : dictionary
        Specifies all cosmological parameters. To be passed from the
        read_input method of the Input class.
    bias_dict : dictionary
        Specifies all the information about the bias model. To be passed
        from the read_input method of the Input class.
    iA_dict: dictionary
        Specifies all the information about the intrinsic alignment model.
        To be passed from the read_input method in the Input class.
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
        'survey_area_clust' : dictionary
            Contains information about the survey footprint that is read
            from the fits file of the survey (processed by healpy).
            Possible keys: 'area', 'ell', 'a_lm'
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

    Attributes
    ----------
        see CovKSpace class

        ellrange : array
            Multipoles at which the covariance is calculated

        deg2torad2 : float
            Conversion factor from degrees to radian squared

        arcmin2torad2 : float
            Conversion factor from arcminutes to radian squared

        spline_zclust : list
            List of spline objects for the redshift distribution for each
            tomographic bin of the clustering analysis

        spline_zlens : list
            List of spline objects for the redshift distribution for each
            tomographic bin of the lensing analysis

        los_integration_chi : array
            Values of the comoving distance at which the line of sight
            integration is carried out.

        spline_z_of_chi : spline_object
            ...

        spline_lensweight : list
            List of spline objects for the lensing weight for each
            tomographic bin of the lensing analysis

        Cell_gg : array
            Array storing the C_ell of the clustering analysis with shape
            (# ell modes, # sample bins, # tomographic bins,
            # tomographic bins)

        Cell_gm, Cell_gkappa : array
            Array storing the C_ell of the galaxy-galaxy lensing analysis
            with shape (# ell modes, # sample bins, # tomographic bins,
            # tomographic bins)

        Cell_mm, Cell_kappakappa : array
            Array storing the C_ell of the cosmic shear analysis with shape
            (# ell modes, # tomographic bins, # tomographic bins)

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_ell_space import CovELLSpace
    inp = Input()
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    covell = CovELLSpace(covterms, observables, output, cosmo, bias,
        iA, hod, survey_params, prec, read_in_tables)

    """

    @staticmethod
    def _simpson_weights_nonuniform(x):
        """
        Weights w with sum(w*y) == scipy.integrate.simpson(y, x=x) for any y.
        """
        n = len(x)
        w = np.zeros(n)
        h = np.diff(x)
        m = n if n % 2 == 1 else n - 1
        h0, h1 = h[0:m - 1:2], h[1:m - 1:2]
        hs, hr = h0 + h1, h1/h0
        np.add.at(w, np.arange(0, m - 2, 2), hs/6*(2 - hr))
        np.add.at(w, np.arange(1, m - 1, 2), hs/6*hs*hs/(h0*h1))
        np.add.at(w, np.arange(2, m, 2), hs/6*(2 - 1/hr))
        if n % 2 == 0:
            h0, h1 = h[-2], h[-1]
            w[-1] += (2*h1**2 + 3*h0*h1)/(6*(h0 + h1))
            w[-2] += (h1**2 + 3*h0*h1)/(6*h0)
            w[-3] -= h1**3/(6*h0*(h0 + h1))
        return w

    @staticmethod
    def _steffen_slopes(x, y):
        """
        Derivatives at the nodes of the Steffen spline (as
        gsl_interp_steffen) through (x, y[:, column]).
        """
        h = np.diff(x)
        s = np.diff(y, axis=0)/h[:, None]
        yp = np.empty_like(y)
        yp[0], yp[-1] = s[0], s[-1]
        p = (s[:-1]*h[1:, None] + s[1:]*h[:-1, None])/(h[:-1, None] + h[1:, None])
        yp[1:-1] = (np.sign(s[:-1]) + np.sign(s[1:]))*np.minimum(np.minimum(np.abs(s[:-1]), np.abs(s[1:])), 0.5*np.abs(p))
        return yp

    @staticmethod
    def _hermite_matrices(x, x_eval, inside, side, derivative):
        """
        Matrices A and B such that A @ y + B @ yp is the piecewise cubic
        Hermite interpolant through (x, y) with derivatives yp at the nodes
        (or its derivative) at x_eval, in the rows where inside is True (the
        other rows are zero). At the nodes the cubic piece to the given side
        is used.
        """
        A = np.zeros((len(x_eval), len(x)))
        B = np.zeros_like(A)
        rows = np.nonzero(inside)[0]
        i = np.clip(np.searchsorted(x, x_eval[rows], side=side) - 1, 0, len(x) - 2)
        h = x[i + 1] - x[i]
        t = (x_eval[rows] - x[i])/h
        if derivative:
            A[rows, i], A[rows, i + 1] = 6*(t**2 - t)/h, 6*(t - t**2)/h
            B[rows, i], B[rows, i + 1] = 3*t**2 - 4*t + 1, 3*t**2 - 2*t
        else:
            A[rows, i], A[rows, i + 1] = 2*t**3 - 3*t**2 + 1, 3*t**2 - 2*t**3
            B[rows, i], B[rows, i + 1] = h*(t**3 - 2*t**2 + t), h*(t**3 - t**2)
        return A, B

    def _integrand_at(self, x_tab, y_tab, x_eval, logy=True, derivative=False):
        """
        The integrands y_tab[:, column] tabulated at x_tab evaluated at x_eval
        as in Levin.init_integral: Steffen spline in ln x of ln y (of y if a
        column is not positive), extrapolated linearly in ln x beyond the
        table (with a non-positive slope at the upper end). With derivative,
        also returns the derivatives with respect to ln x from the right and
        from the left (they differ at the nodes of x_tab, where the spline
        is only once differentiable, and where the extrapolation starts).
        The table is prepared here; the evaluation is done in parallel by
        levin.evaluate_hermite (num_cores threads), or as matrix products in
        blocks of x_eval if levin does not provide it (older build).
        """
        lx, le = np.log(x_tab), np.log(x_eval)
        positive = np.all(y_tab > 0, axis=0) if logy else np.zeros(y_tab.shape[1], dtype=bool)
        Y = np.where(positive, np.log(np.where(y_tab > 0, y_tab, 1.0)), y_tab)
        YP = self._steffen_slopes(lx, Y)
        slope = np.minimum((Y[-1] - Y[-2])/(lx[-1] - lx[-2]), 0.0)
        slope0 = (Y[1] - Y[0])/(lx[1] - lx[0])
        if hasattr(levin, 'evaluate_hermite'):
            result = levin.evaluate_hermite(lx, Y, YP, slope, slope0, positive, le, derivative,
                                            int(getattr(self, 'num_cores', 1)))
            return result if derivative else result[0]
        values = np.empty((len(le), y_tab.shape[1]))
        if derivative:
            d_right, d_left = np.empty_like(values), np.empty_like(values)
        for rows in self._blocks(len(le), len(lx)):
            e = le[rows]
            above, below = e >= lx[-1], e < lx[0]
            A, B = self._hermite_matrices(lx, e, ~above & ~below, 'right', False)
            out = A @ Y + B @ YP
            out[above] = Y[-1] + slope*(e[above] - lx[-1])[:, None]
            out[below] = Y[0] + slope0*(e[below] - lx[0])[:, None]
            values[rows] = np.where(positive, np.exp(out), out)
            if not derivative:
                continue
            above, below = e >= lx[-1], e < lx[0]
            A, B = self._hermite_matrices(lx, e, ~above & ~below, 'right', True)
            dout = A @ Y + B @ YP
            dout[above] = slope
            dout[below] = slope0
            d_right[rows] = np.where(positive, values[rows]*dout, dout)
            d_left[rows] = d_right[rows]
            at_node = np.isin(e, lx)
            if np.any(at_node):
                e = e[at_node]
                above, below = e > lx[-1], e <= lx[0]
                A, B = self._hermite_matrices(lx, e, ~above & ~below, 'left', True)
                dout = A @ Y + B @ YP
                dout[above] = slope
                dout[below] = slope0
                d_left[rows][at_node] = np.where(positive, values[rows][at_node]*dout, dout)
        if not derivative:
            return values
        return values, d_right, d_left

    def _register_projection_weights(self, levin_int, ell, weights, accuracy):
        """
        Registers the weights handed to levin_int.init_w_ell (shape (modes,
        len(ell))) for the quadrature projections (projection_method =
        quadrature). The accuracy of Simpson's rule on the grid of the weights
        is estimated by comparing with every second node; if it is worse than
        a tenth of the requested accuracy, the weights are refined with Akima
        interpolation (as used by levin) until it is (and below 1e-5 in any
        case). If that does not succeed, the projections with this integrator
        use cquad.
        """
        registry = self.__dict__.setdefault('_projection_weights_of', {})
        if getattr(self, 'projection_method', 'quadrature') != 'quadrature':
            return
        ell = np.array(ell, dtype=float)
        # no copy of the (possibly large) weights, they are not modified (C
        # order for levin.gauss_hermite_projection, usually the case already)
        weights = np.ascontiguousarray(weights, dtype=float)
        tol = max(min(0.1*accuracy, 1e-5), 1e-10)
        for _ in range(6):
            error = self._weight_quadrature_error(ell, weights)
            if error <= tol:
                registry[id(levin_int)] = (ell, weights, levin_int, tol, {})
                return
            ell_mid = np.sqrt(ell[:-1]*ell[1:])
            weights = np.concatenate((weights, Akima1DInterpolator(ell, weights, axis=1)(ell_mid)), axis=1)
            ell = np.concatenate((ell, ell_mid))
            order = np.argsort(ell)
            ell, weights = ell[order], np.take(weights, order, axis=1)
        print("ProjectionWarning: the weight functions are not resolved well enough for the "
              "quadrature projection (estimated error " + str(error) + "), using cquad instead.")

    def _weight_quadrature_error(self, ell, weights):
        """
        Estimated relative error of Simpson's rule for int dell W(ell)^2/ell
        on the grid ell (from the comparison with every second node).
        """
        idx = np.arange(0, len(ell), 2)
        if idx[-1] != len(ell) - 1:
            idx = np.append(idx, len(ell) - 1)
        w_full = self._simpson_weights_nonuniform(ell)/ell
        w_half = self._simpson_weights_nonuniform(ell[idx])/ell[idx]
        full = np.zeros(len(weights))
        half = np.zeros(len(weights))
        # in blocks of rows, many weights on a fine grid are large
        rows = max(1, PROJECTION_BLOCK_SIZE//max(len(ell), 1))
        for start in range(0, len(weights), rows):
            block = weights[start:start + rows]
            full[start:start + rows] = block**2 @ w_full
            half[start:start + rows] = block[:, idx]**2 @ w_half
        ok = full > 0
        if not np.any(ok):
            return 0.0
        return float(np.max(np.abs(full[ok] - half[ok])/15/full[ok]))

    def _use_quadrature_projection(self, levin_int):
        registered = getattr(self, '_projection_weights_of', {}).get(id(levin_int))
        return (getattr(self, 'projection_method', 'quadrature') == 'quadrature'
                and registered is not None and registered[2] is levin_int)

    def _project_single(self, levin_int, ell_limits, modes, integrand):
        """
        int dell W_m(ell) integrand(ell) over ell_limits[m] for all modes m and
        all columns of the integrand (tabulated on self.ellrange, interpolated
        as in Levin.init_integral). With the quadrature projection this is
        Simpson's rule on the nodes of the registered weights (in blocks of
        modes), otherwise cquad.

        Returns
        -------
        result : array
            with shape (len(modes), integrand.shape[1])
        """
        integrand = np.asarray(integrand, dtype=float)
        result = np.zeros((len(modes), integrand.shape[1]))
        if not self._use_quadrature_projection(levin_int):
            levin_int.init_integral(self.ellrange, integrand, True, True)
            for i, mode in enumerate(modes):
                result[i] = levin_int.cquad_integrate_single_well(ell_limits[mode], mode)
            return result
        groups = {}
        for i, mode in enumerate(modes):
            groups.setdefault((ell_limits[mode][0], ell_limits[mode][-1]), []).append(i)
        n_nodes = len(self._projection_weights_of[id(levin_int)][0])
        for (lo, hi), idx in groups.items():
            idx = np.array(idx)
            values = None
            for rows in self._blocks(len(idx), n_nodes):
                x, Wx = self._weights_on_nodes(levin_int, [modes[i] for i in idx[rows]], lo, hi)
                if values is None:
                    values = self._integrand_at(self.ellrange, integrand, x, True)*self._simpson_weights_nonuniform(x)[:, None]
                result[idx[rows]] = Wx @ values
        return result

    def _weights_on_nodes(self, levin_int, mode_list, lo, hi):
        """
        Nodes of the weight grid within [lo, hi] (including the end points)
        and the weights of mode_list there (Akima interpolated at the ends).
        """
        x, inner, W_lo, W_hi = self._weight_nodes(levin_int, lo, hi)
        W = self._projection_weights_of[id(levin_int)][1]
        mode_list = np.asarray(mode_list, dtype=int)
        return x, np.concatenate((W_lo[mode_list, None], W[mode_list, inner], W_hi[mode_list, None]), axis=1)

    def _weight_nodes(self, levin_int, lo, hi):
        """
        The nodes x of the weight grid within [lo, hi] (including the end
        points), the slice of the registered weight nodes strictly inside
        and the weights of all modes at lo and hi (Akima interpolated),
        cached per (lo, hi) with the registered weights.
        """
        ell_w, W, _, _, cache = self._projection_weights_of[id(levin_int)]
        if (lo, hi) not in cache:
            # ell_w is ascending, the nodes inside are a slice
            inner = slice(np.searchsorted(ell_w, lo, side='right'), np.searchsorted(ell_w, hi, side='left'))
            x = np.concatenate(([lo], ell_w[inner], [hi]))
            ends = []
            for end in (lo, hi):
                i = np.searchsorted(ell_w, end)
                window = slice(max(i - 6, 0), min(i + 6, len(ell_w)))
                ends.append(Akima1DInterpolator(ell_w[window], W[:, window], axis=1)([end])[:, 0])
            cache[(lo, hi)] = (x, inner) + tuple(ends)
        return cache[(lo, hi)]

    def _project_gauss_quadrature(self, levin_int, integrand, keys, lo, hi):
        """
        int_lo^hi dell W_m(ell) W_n(ell) integrand(ell) for all (m, n) in
        keys and all columns of the integrand (_project_gauss_quadrature_keys).
        The keys are split into groups with at most about 8
        PROJECTION_BLOCK_SIZE/(number of weight nodes) distinct weights each,
        so that the copies of the weights on their nodes stay bounded for
        many weights (e.g. fine real-space bins). Not with
        levin.gauss_hermite_projection, which makes no copies.
        """
        modes = sorted(set(m for key in keys for m in key))
        n_nodes = len(self._projection_weights_of[id(levin_int)][0])
        half_block = max(4, 4*PROJECTION_BLOCK_SIZE//max(n_nodes, 1))
        if len(modes) <= 2*half_block or hasattr(levin, 'gauss_hermite_projection'):
            return self._project_gauss_quadrature_keys(levin_int, integrand, keys, lo, hi)
        block_of = {mode: i//half_block for i, mode in enumerate(modes)}
        groups = {}
        for i, (m, n) in enumerate(keys):
            groups.setdefault((block_of[m], block_of[n]), []).append(i)
        result = np.empty((len(keys), integrand.shape[1]))
        for idx in groups.values():
            result[idx] = self._project_gauss_quadrature_keys(levin_int, integrand, [keys[i] for i in idx], lo, hi)
        return result

    def _project_gauss_quadrature_keys(self, levin_int, integrand, keys, lo, hi):
        """
        int_lo^hi dell W_m(ell) W_n(ell) integrand(ell) for all (m, n) in
        keys and all columns of the integrand (tabulated on self.ellrange).
        The integrands are represented by piecewise cubic Hermite polynomials
        on a sampling grid (on which they are accurate to a hundredth of the
        tolerance of the weights, relative to the local value), integrated
        against W_m W_n with Simpson's rule on the nodes of the weights.
        Columns that would need a much finer grid than most (or one not much
        coarser than the weights) are evaluated on the nodes of the weights
        directly.
        """
        modes = sorted(set(m for key in keys for m in key))
        position = {mode: i for i, mode in enumerate(modes)}
        kernel = hasattr(levin, 'gauss_hermite_projection')
        if kernel:
            # the kernel reads the registered weights, no copies on the nodes
            x = self._weight_nodes(levin_int, lo, hi)[0]
        else:
            x, Wx = self._weights_on_nodes(levin_int, modes, lo, hi)
            w = self._simpson_weights_nonuniform(x)
        first = np.array([position[m] for m, _ in keys], dtype=int)
        second = np.array([position[n] for _, n in keys], dtype=int)
        tol = 0.01*self._projection_weights_of[id(levin_int)][3]
        grids, level = [], np.full(integrand.shape[1], -1)
        todo = np.arange(integrand.shape[1])
        for grid in self._refined_grids(lo, hi, len(x)//4):
            ok = self._hermite_error(integrand[:, todo], grid, True, local=True, per_column=True) <= tol
            level[todo[ok]] = len(grids)
            grids.append(grid)
            todo = todo[~ok]
            if len(todo) == 0:
                break
        n_col = integrand.shape[1]
        if kernel:
            # per key: the moments on the nodes and the contraction per segment
            # and column (on the nodes for the columns evaluated directly)
            best, cost_best = None, 4*len(x)*(n_col + 1)
            for i_grid, grid in enumerate(grids):
                n_ok = np.sum((level >= 0) & (level <= i_grid))
                cost = 4*len(x) + 4*len(grid)*n_ok + (4*len(x)*(n_col - n_ok + 1) if n_ok < n_col else 0)
                if cost < cost_best:
                    best, cost_best = i_grid, cost
        else:
            best, cost_best = None, (len(keys) + 400)*len(x)*n_col
            for i_grid, grid in enumerate(grids):
                n_ok = np.sum((level >= 0) & (level <= i_grid))
                cost = (20*len(keys)*len(x) + (len(keys) + 1200)*len(grid)*n_ok
                        + (len(keys) + 400)*len(x)*(n_col - n_ok))
                if cost < cost_best:
                    best, cost_best = i_grid, cost
        first_modes, first = np.unique(first, return_inverse=True)
        second_modes, second = np.unique(second, return_inverse=True)
        pair = first*len(second_modes) + second
        if kernel:
            columns = np.nonzero((level >= 0) & (level <= best))[0] if best is not None else np.arange(0)
            todo = np.setdiff1d(np.arange(n_col), columns)
            modes = np.array(modes, dtype=np.int64)
            return self._project_gauss_kernel(levin_int, integrand, lo, hi, grids[best] if best is not None else None,
                                              columns, todo, modes[first_modes], modes[second_modes], pair)
        W_first, W_second = Wx[first_modes], Wx[second_modes]*w
        result = np.empty((len(keys), integrand.shape[1]))
        todo = np.arange(integrand.shape[1])
        if best is not None:
            grid = grids[best]
            columns = np.nonzero((level >= 0) & (level <= best))[0]
            todo = np.nonzero((level < 0) | (level > best))[0]
            k, coefficients = self._hermite_coefficients(grid, x)
            starts = np.flatnonzero(np.diff(k, prepend=-1))
            stops = np.append(starts[1:], len(x))
            # with many keys (most of the pairs of the modes, e.g. fine real-space
            # bins) all pairs are accumulated and the keys picked at the end: no
            # gather per segment and larger matrix products (which thread well)
            n_pairs = len(first_modes)*len(second_modes)
            all_pairs = len(keys) >= n_pairs//2
            n_rows = n_pairs if all_pairs else len(keys)
            for cols in self._blocks(len(columns), 3*len(grid)):
                values, d_right, d_left = self._integrand_at(self.ellrange, integrand[:, columns[cols]], grid, True,
                                                             derivative=True)
                part = np.zeros((n_rows, cols.stop - cols.start))
                # (a quarter of the usual block size per interval: longer products)
                for segs in self._blocks(len(starts), n_rows):
                    G = np.empty((4, segs.stop - segs.start, n_rows))
                    for i, (start, stop) in enumerate(zip(starts[segs], stops[segs])):
                        A = (coefficients[:, None, start:stop]*W_first[None, :, start:stop]).reshape(-1, stop - start)
                        products = (A @ W_second[:, start:stop].T).reshape(4, -1)
                        G[:, i] = products if all_pairs else products[:, pair]
                    k_seg = k[starts[segs]]
                    part += (G[0].T @ values[k_seg] + G[1].T @ values[k_seg + 1]
                             + G[2].T @ d_right[k_seg] + G[3].T @ d_left[k_seg + 1])
                result[:, columns[cols]] = part[pair] if all_pairs else part
        for cols in self._blocks(len(todo), len(x)):
            F = self._integrand_at(self.ellrange, integrand[:, todo[cols]], x)
            if len(keys)*len(x) <= PROJECTION_BLOCK_SIZE:
                result[:, todo[cols]] = (W_first[first]*W_second[second]) @ F
                continue
            for j, column in enumerate(todo[cols]):
                gram = np.zeros((len(first_modes), len(second_modes)))
                for nodes in self._blocks(len(x), len(second_modes)):
                    gram += W_first[:, nodes] @ (W_second[:, nodes]*F[nodes, j]).T
                result[:, column] = gram.ravel()[pair]
        return result

    def _project_gauss_kernel(self, levin_int, integrand, lo, hi, grid, columns, todo, first_modes, second_modes,
                              pair):
        """
        The projections of _project_gauss_quadrature_keys with
        levin.gauss_hermite_projection (num_cores threads, reading the
        registered weights without copies): the columns of the integrand on
        their piecewise cubic Hermite representation on grid, the columns
        todo on the nodes of the weights directly (segments of one node). The
        keys are the pairs (first_modes[i], second_modes[l]) with pair =
        i*len(second_modes) + l.
        """
        x, inner, W_lo, W_hi = self._weight_nodes(levin_int, lo, hi)
        W = self._projection_weights_of[id(levin_int)][1][:, inner]
        w = self._simpson_weights_nonuniform(x)
        row_of = np.full(len(first_modes)*len(second_modes), -1, dtype=np.int64)
        row_of[pair] = np.arange(len(pair))
        n_threads = int(getattr(self, 'num_cores', 1))
        result = np.empty((len(pair), integrand.shape[1]))

        def project(coefficients, starts, stops, k, values, d_right, d_left):
            return levin.gauss_hermite_projection(W, W_lo, W_hi, w, first_modes, second_modes, row_of, len(pair),
                                                  coefficients, starts, stops, k, values, d_right, d_left, n_threads)

        # at most 256 columns per call (bounded accumulators of the threads)
        if len(columns):
            k, coefficients = self._hermite_coefficients(grid, x)
            starts = np.flatnonzero(np.diff(k, prepend=-1))
            stops = np.append(starts[1:], len(x))
            for cols in self._blocks(len(columns), max(3*len(grid), PROJECTION_BLOCK_SIZE//256)):
                values, d_right, d_left = self._integrand_at(self.ellrange, integrand[:, columns[cols]], grid, True,
                                                             derivative=True)
                result[:, columns[cols]] = project(coefficients, starts, stops, k[starts], values, d_right, d_left)
        if len(todo):
            nodes = np.arange(len(x))
            coefficients = np.zeros((4, len(x)))
            coefficients[0] = 1.0
            for cols in self._blocks(len(todo), max(len(x), PROJECTION_BLOCK_SIZE//256)):
                F = self._integrand_at(self.ellrange, integrand[:, todo[cols]], x)
                F = np.concatenate((F, np.zeros((1, F.shape[1]))))
                result[:, todo[cols]] = project(coefficients, nodes, nodes + 1, nodes, F, F, F)
        return result

    @staticmethod
    def _hermite_coefficients(grid, x):
        """
        The interval k of grid that contains each x (ascending) and the
        coefficients (4, len(x)) of the piecewise cubic Hermite interpolant
        (in u = ln ell) at x: of the values at grid[k] and grid[k + 1], of the
        derivative from the right at grid[k] and of the derivative from the
        left at grid[k + 1].
        """
        u_grid, u = np.log(grid), np.clip(np.log(x), np.log(grid[0]), np.log(grid[-1]))
        k = np.clip(np.searchsorted(u_grid, u, side='right') - 1, 0, len(u_grid) - 2)
        h = u_grid[k + 1] - u_grid[k]
        t = (u - u_grid[k])/h
        return k, np.array((2*t**3 - 3*t**2 + 1, 3*t**2 - 2*t**3, h*(t**3 - 2*t**2 + t), h*(t**3 - t**2)))

    @property
    def _compact_ell_cov(self):
        """
        Whether the ell-space covariances are only calculated as needed by
        the projections (in the classes of the projected estimators): the
        Gaussian ones only on the diagonal (shape (ell, 1, ...)), the
        four-point ones only for the calculated tomographic combinations
        (EllCovColumns). Not if the multiplicative shear bias terms of the
        C_ell estimator are added to the four-point ones.
        """
        return type(self) is not CovELLSpace and not ((self.mm or self.gm) and self.est_shear == 'C_ell')

    @staticmethod
    def _linear_weights(grid, x):
        """
        Interval indices i (in [0, len(grid) - 2]) and weights t such that
        the linear interpolation (and extrapolation from the outermost
        intervals) of f on grid at x is (1 - t) f[i] + t f[i + 1], as in
        RegularGridInterpolator.
        """
        index = np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2)
        return index, (x - grid[index])/(grid[index + 1] - grid[index])

    @classmethod
    def _trilinear_on_mesh(cls, interpolator, chi, log_k):
        """
        The linear RegularGridInterpolator (chi, ln k_1, ln k_2) of a
        trispectrum evaluated on the mesh (chi, ln k(chi, ell_1), ln k(chi,
        ell_2)), with log_k of shape (len(chi), ell), as sums of gathered
        table values: the same as calling the interpolator on all points,
        but without its search and weight arrays for every point.
        """
        grid_chi, grid_k, table = interpolator.grid[0], interpolator.grid[1], interpolator.values
        i_chi, t_chi = cls._linear_weights(grid_chi, chi)
        i_k, t_k = cls._linear_weights(grid_k, log_k)
        result = np.zeros((len(chi),) + (log_k.shape[1],)*2)
        for shift_chi, weight_chi in ((0, 1 - t_chi), (1, t_chi)):
            for shift_1, weight_1 in ((0, 1 - t_k), (1, t_k)):
                for shift_2, weight_2 in ((0, 1 - t_k), (1, t_k)):
                    result += weight_chi[:, None, None]*weight_1[:, :, None]*weight_2[:, None, :] \
                        * table[(i_chi + shift_chi)[:, None, None], (i_k + shift_1)[:, :, None], (i_k + shift_2)[:, None, :]]
        return result

    def _diagonal_lens_mask(self, probes):
        """
        With csmf_diagonal_lenses only galaxy sample i in clustering bin i
        is measured. Returns 1 for these combinations and 0 otherwise, with
        shape (sample, sample, tomo, tomo, tomo, tomo) of the covariance of
        probes ('gggg', 'gggm', 'ggmm', 'gmgm' or 'mmgm'): both bins of a gg
        pair and the lens bin of a gm pair are the bin of the sample.
        """
        n_clust, n_lens = self.n_tomo_clust, self.n_tomo_lens
        in_bin = np.eye(self.sample_dim, n_clust)
        if probes == 'gggg':
            return in_bin[:, None, :, None, None, None]*in_bin[:, None, None, :, None, None] \
                * in_bin[None, :, None, None, :, None]*in_bin[None, :, None, None, None, :]
        if probes == 'gggm':
            return in_bin[:, None, :, None, None, None]*in_bin[:, None, None, :, None, None] \
                * in_bin[None, :, None, None, :, None]*np.ones(n_lens)[None, None, None, None, None, :]
        if probes == 'ggmm':
            return in_bin[:, None, :, None, None, None]*in_bin[:, None, None, :, None, None] \
                * np.ones((1, 1, 1, 1, n_lens, n_lens))
        if probes == 'gmgm':
            return in_bin[:, None, :, None, None, None]*in_bin[None, :, None, None, :, None] \
                * np.ones((1, 1, 1, n_lens, 1, n_lens))
        if probes == 'mmgm':
            return in_bin[None, :, None, None, :, None]*np.ones((1, 1, n_lens, n_lens, 1, n_lens))
        raise Exception("WildError: no clustering sample in " + probes + ".")

    def _ell_4pt_array(self, probes, tomo_shape, symmetric_12, symmetric_34, compact=True):
        """
        Zero array for a four-point ell-space covariance of the probes
        (e.g. 'gggm') with the tomographic shape (sample, sample, tomo,
        tomo, tomo, tomo). With _compact_ell_cov (and compact) an
        EllCovColumns with the combinations t1 <= t2 (if symmetric_12) and
        t3 <= t4 (if symmetric_34). With csmf_diagonal_lenses only the
        combinations with galaxy sample i in tomographic clustering bin i
        are stored, as only those are calculated.
        """
        if not (self._compact_ell_cov and compact):
            return np.zeros((len(self.ellrange), len(self.ellrange)) + tuple(tomo_shape))
        s1, s2, t1, t2, t3, t4 = np.indices(tomo_shape).reshape(6, -1)
        keep = np.ones(len(s1), dtype=bool)
        if symmetric_12:
            keep &= t2 >= t1
        if symmetric_34:
            keep &= t4 >= t3
        if self.csmf_diagonal_lenses:
            if probes[:2] == 'gg':
                keep &= (t1 == s1) & (t2 == s1)
            if probes[:2] == 'gm':
                keep &= t1 == s1
            if probes[2:] == 'gg':
                keep &= (t3 == s2) & (t4 == s2)
            if probes[2:] == 'gm':
                keep &= t3 == s2
        return EllCovColumns(len(self.ellrange), tomo_shape, np.nonzero(keep)[0])

    def _ell_cov_flat(self, ell_cov, flat_length):
        """
        An ell-space covariance as (ell, ell, column), or (ell, 1, column) if
        only its diagonal is stored; EllCovColumns as they are.
        """
        if isinstance(ell_cov, EllCovColumns):
            return ell_cov
        return np.reshape(ell_cov, (len(self.ellrange), -1, flat_length))

    @staticmethod
    def _tomo_shape(ell_cov):
        """
        The shape of an ell-space covariance without its two ell axes.
        """
        if isinstance(ell_cov, EllCovColumns):
            return ell_cov.tomo_shape
        return ell_cov[0, 0].shape

    @staticmethod
    def _expand_columns(values, columns, original_shape):
        """
        Results for the stored columns of an EllCovColumns (columns, or None
        for all columns) in the full original_shape (zero elsewhere).
        """
        if columns is None:
            return np.reshape(values, original_shape)
        full = np.zeros(int(np.prod(original_shape)))
        full[columns] = values
        return np.reshape(full, original_shape)

    def _pixel_weights(self, ell_cov):
        """
        The pixel window weights (ell, ell) shaped to multiply an ell-space
        covariance (only their diagonal if only that is stored).
        """
        weights = self.pixelweight_matrix
        if np.ndim(ell_cov) < 2:
            # terms that are not calculated (0) stay a number
            return 1.0
        if np.shape(ell_cov)[1] == 1 and len(self.ellrange) > 1:
            weights = np.diagonal(weights)[:, None]
        return np.reshape(weights, np.shape(weights) + (1,)*(np.ndim(ell_cov) - 2))

    @staticmethod
    def _blocks(n, size):
        """
        Slices covering range(n) in blocks such that a block times size
        elements stay within PROJECTION_BLOCK_SIZE (at least one per block).
        """
        step = max(1, PROJECTION_BLOCK_SIZE//max(size, 1))
        return [slice(start, min(start + step, n)) for start in range(0, n, step)]

    def _hermite_error(self, integrand, grid, logy, local=False, per_column=False):
        """
        Maximum error (relative to the maximum of each column on grid, or
        with local to the larger absolute value at the ends of each
        interval, but at least 1e-10 of the maximum) of the piecewise cubic
        Hermite representation of the interpolated integrands on grid, at
        the midpoints of the intervals; for every column with per_column.
        """
        mid = np.sqrt(grid[:-1]*grid[1:])
        h = np.diff(np.log(grid))[:, None]
        error = np.zeros(integrand.shape[1])
        for cols in self._blocks(integrand.shape[1], len(grid)):
            part = integrand[:, cols]
            values, d_right, d_left = self._integrand_at(self.ellrange, part, grid, logy, derivative=True)
            hermite = 0.5*(values[:-1] + values[1:]) + h/8*(d_right[:-1] - d_left[1:])
            hermite -= self._integrand_at(self.ellrange, part, mid, logy)
            if local:
                # where a column is negligible (deep tails), relative to 1e-10
                # of its maximum
                scale = np.maximum(np.maximum(np.abs(values[:-1]), np.abs(values[1:])),
                                   1e-10*np.max(np.abs(values), axis=0))
                scale[scale == 0] = 1.0
            else:
                scale = np.max(np.abs(values), axis=0)
                scale[scale == 0] = 1.0
            error[cols] = np.max(np.abs(hermite)/scale, axis=0)
        if per_column:
            return error
        return float(np.max(error, initial=0.0))

    def _refined_grids(self, lo, hi, max_points=2**18):
        """
        Grids on [lo, hi] for the piecewise cubic Hermite representation (in
        ln ell, with exact values and derivatives) of the integrands as
        interpolated in _integrand_at, successively refined by factors of
        two (up to max_points points). They contain the nodes of
        self.ellrange, where the Steffen spline is only once differentiable,
        and the start of the extrapolation, so that every interval lies within
        one cubic piece: exact for integrands interpolated in y.
        """
        nodes = self.ellrange[(self.ellrange > lo) & (self.ellrange < hi)]
        dlog = np.log(self.ellrange[-1]/self.ellrange[0])/max(len(self.ellrange) - 1, 1)
        base = [np.array([lo]), nodes]
        if hi > self.ellrange[-1]:
            start = max(lo, self.ellrange[-1])
            n_tail = int(np.ceil(np.log(hi/start)/dlog))
            base.append(np.geomspace(start, hi, n_tail + 1)[1:-1])
        base = np.unique(np.concatenate(base + [np.array([hi])]))
        lnbase = np.log(base)
        for refine in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if len(base)*refine > max_points:
                return
            fractions = np.arange(refine)/refine
            lngrid = np.append((lnbase[:-1, None] + np.diff(lnbase)[:, None]*fractions[None, :]).ravel(), lnbase[-1])
            grid = np.exp(lngrid)
            grid[0], grid[-1] = lo, hi
            yield grid

    def _sampling_grid(self, lo, hi, integrand, logy, tol):
        """
        The first grid of _refined_grids on which the representation of the
        integrands is accurate to tol (relative to the maximum of each
        column). None if that needs more than 2^18 points.
        """
        for grid in self._refined_grids(lo, hi):
            if self._hermite_error(integrand, grid, logy) <= tol:
                return grid
        return None

    def _quadrature_vectors(self, levin_int, grid, modes, ell_limits):
        """
        For every mode the vectors (K, D_r, D_l) with
        K @ f + D_r @ df_r + D_l @ df_l = int dell W(ell) p(ell)
        over the integration range of the mode, where p is the piecewise
        cubic Hermite interpolant (in u = ln ell) of the values f and the
        derivatives d f/d u on grid, taken from the right (df_r, at the start
        of each interval) and from the left (df_l, at its end). The integral
        is done with Simpson's rule on the nodes of the weights.
        """
        u_grid = np.log(grid)
        result = {}
        # the weights with the same range share their nodes, so the Hermite
        # basis on the nodes is the same: a sparse matrix (nodes x grid), the
        # vectors of all weights of the range follow from one product (in
        # blocks of weights)
        by_range = {}
        for mode in modes:
            by_range.setdefault((ell_limits[mode][0], ell_limits[mode][-1]), []).append(mode)
        for (lo, hi), range_modes in by_range.items():
            n_nodes = None
            for rows in self._blocks(len(range_modes), len(self._projection_weights_of[id(levin_int)][0])//4):
                block = range_modes[rows]
                x, Wx = self._weights_on_nodes(levin_int, block, lo, hi)
                if n_nodes is None:
                    n_nodes = len(x)
                    u = np.clip(np.log(x), u_grid[0], u_grid[-1])
                    k = np.clip(np.searchsorted(u_grid, u, side='right') - 1, 0, len(u_grid) - 2)
                    h = u_grid[k + 1] - u_grid[k]
                    t = (u - u_grid[k])/h
                    nodes = np.arange(n_nodes)
                    shape = (n_nodes, len(grid))
                    basis_K = csr_matrix((np.concatenate((2*t**3 - 3*t**2 + 1, -2*t**3 + 3*t**2)),
                                          (np.concatenate((nodes, nodes)), np.concatenate((k, k + 1)))), shape=shape)
                    basis_Dr = csr_matrix((h*(t**3 - 2*t**2 + t), (nodes, k)), shape=shape)
                    basis_Dl = csr_matrix((h*(t**3 - t**2), (nodes, k + 1)), shape=shape)
                    simpson_x = self._simpson_weights_nonuniform(x)
                ww = Wx*simpson_x[None, :]
                K, D_r, D_l = (np.asarray((basis.T @ ww.T).T) for basis in (basis_K, basis_Dr, basis_Dl))
                for i, mode in enumerate(block):
                    result[mode] = (K[i], D_r[i], D_l[i])
        return result

    def _project_4pt_quadrature(self, levin_int, ell_limits, ell_cov, live, original_shape, outer_logy):
        """
        Four-point projection of _project_4pt_jobs with fixed quadratures:
        inner integrals over ell_2 for all ell_1 and all columns, then the
        outer integrals over ell_1, both as matrix products with the vectors
        of _quadrature_vectors. Returns False (nothing done) if the sampling
        of the integrands can not be made accurate enough.
        """
        tol = self._projection_weights_of[id(levin_int)][3]
        n_ell = len(self.ellrange)
        columns = getattr(ell_cov, 'columns', None)
        flat = np.reshape(np.asarray(ell_cov), (n_ell, n_ell, -1))
        unique_idx, inverse = self._unique_columns_blocked(np.reshape(flat, (n_ell*n_ell, -1)))
        n_col = len(unique_idx)
        inner_modes = sorted(set(self._weight_key(job[1]) for job in live))
        outer_modes = sorted(set(self._weight_key(job[0]) for job in live))
        lo = min(ell_limits[m][0] for m in inner_modes + outer_modes)
        hi = max(ell_limits[m][-1] for m in inner_modes + outer_modes)
        # inner integrals, sampling checked on a few rows
        rows = np.unique(np.linspace(0, n_ell - 1, 8).astype(int))
        check = np.moveaxis(flat[rows][:, :, unique_idx]*self.ellrange[None, :, None], 1, 0).reshape(n_ell, -1)
        grid = self._sampling_grid(lo, hi, check, True, tol)
        del check
        if grid is None:
            return False
        K = self._quadrature_vectors(levin_int, grid, inner_modes, ell_limits)
        K_inner = np.array([K[m][0] for m in inner_modes])
        Dr_inner = np.array([K[m][1] for m in inner_modes])
        Dl_inner = np.array([K[m][2] for m in inner_modes])
        inner = np.zeros((n_ell, len(inner_modes), n_col))
        # blocks of columns and of ell_1 rows
        for cols in self._blocks(n_col, len(grid)):
            n_cols = cols.stop - cols.start
            for ell_1 in self._blocks(n_ell, len(grid)*n_cols):
                block = flat[ell_1][:, :, unique_idx[cols]]*self.ellrange[None, :, None]
                n_rows = block.shape[0]
                block = np.moveaxis(block, 1, 0).reshape(n_ell, -1)
                values, d_right, d_left = self._integrand_at(self.ellrange, block, grid, True, derivative=True)
                inner[ell_1, :, cols] = np.moveaxis((K_inner @ values + Dr_inner @ d_right + Dl_inner @ d_left)
                                                    .reshape(len(inner_modes), n_rows, n_cols), 1, 0)
        # outer integrals, on one grid (fine enough for all inner weights)
        inner *= self.ellrange[:, None, None]
        integrand = np.reshape(inner, (n_ell, -1))
        grid = self._sampling_grid(lo, hi, integrand, outer_logy, tol)
        if grid is None:
            return False
        K = self._quadrature_vectors(levin_int, grid, outer_modes, ell_limits)
        K_outer = np.array([K[m][0] for m in outer_modes])
        Dr_outer = np.array([K[m][1] for m in outer_modes])
        Dl_outer = np.array([K[m][2] for m in outer_modes])
        position_outer = {mode: i for i, mode in enumerate(outer_modes)}
        position_inner = {mode: i for i, mode in enumerate(inner_modes)}
        # the outer integrals for chunks of the weights: the jobs with their
        # outer and inner weights in a pair of chunks (a, b) and (b, a) at a
        # time, so that a job and its mirror (averaged in _assign_4pt_results)
        # are done together and the memory does not grow with the square of
        # the number of weights (many weights, e.g. fine real-space bins)
        all_modes = sorted(set(outer_modes) | set(inner_modes))
        size = max(n_col, int(np.prod(original_shape)))
        chunk_size = max(1, int(np.sqrt(16*PROJECTION_BLOCK_SIZE/size)))
        chunk_of = {mode: i//chunk_size for i, mode in enumerate(all_modes)}
        pairs = {}
        for job in live:
            key = (self._weight_key(job[0]), self._weight_key(job[1]))
            a, b = chunk_of[key[0]], chunk_of[key[1]]
            pairs.setdefault((min(a, b), max(a, b)), []).append(job)
        inner_3d = np.reshape(inner, (n_ell, len(inner_modes), n_col))
        for jobs in pairs.values():
            keys = sorted(set((self._weight_key(job[0]), self._weight_key(job[1])) for job in jobs))
            results = {}
            # the two orientations (outer in chunk a, inner in chunk b and vice versa)
            for orientation in set((chunk_of[key[0]], chunk_of[key[1]]) for key in keys):
                part_keys = [key for key in keys if (chunk_of[key[0]], chunk_of[key[1]]) == orientation]
                outer_part = sorted(set(key[0] for key in part_keys))
                inner_part = sorted(set(key[1] for key in part_keys))
                rows = [position_outer[m] for m in outer_part]
                integrand_part = np.reshape(inner_3d[:, [position_inner[m] for m in inner_part], :], (n_ell, -1))
                projected = np.empty((len(rows), integrand_part.shape[1]))
                for cols in self._blocks(integrand_part.shape[1], len(grid)):
                    values, d_right, d_left = self._integrand_at(self.ellrange, integrand_part[:, cols], grid, outer_logy,
                                                                 derivative=True)
                    projected[:, cols] = K_outer[rows] @ values + Dr_outer[rows] @ d_right + Dl_outer[rows] @ d_left
                projected = np.reshape(projected, (len(rows), len(inner_part), n_col))
                row_of = {mode: i for i, mode in enumerate(outer_part)}
                column_of = {mode: i for i, mode in enumerate(inner_part)}
                for key in part_keys:
                    results[key] = self._expand_columns(projected[row_of[key[0]], column_of[key[1]]][inverse],
                                                        columns, original_shape)
            self._assign_4pt_results(jobs, [results[(self._weight_key(job[0]), self._weight_key(job[1]))] for job in jobs])
        return True

    @staticmethod
    def _n_areas(areas):
        """
        Number of survey areas given for a probe (zero if the probe has no
        survey area, i.e. it is not calculated).
        """
        return 0 if areas is None else len(areas)

    @staticmethod
    def _max_area(area_12, area_34):
        """
        The larger of two survey areas as a float. The areas can be passed
        as numbers or as the (one-element) arrays of survey_params_dict.
        """
        return float(max(np.max(area_12), np.max(area_34)))

    def _sample_overlap_sn(self,
                           shot_noise,
                           galaxy_pairs):
        """
        Shot noise of two galaxy samples, which can share galaxies, from
        the shot noise of each sample (with the sample and the two
        tomographic bins as its last three axes, ~ 1/n_1^i n_1^j for
        clustering or 1/n_1^i for ggl): rho^i rho^j sqrt(SN_1 SN_2) for
        galaxy pairs and rho^i sqrt(SN_1 SN_2) for ggl, with rho the overlap
        of the samples in tomographic bin i (see calc_Cells_Limber). For
        disjoint samples this is delta_12 SN_1.

        Returns
        -------
        shot_noise_12 : array
            with the axes (..., sample, sample, tomo_i, tomo_j)
        """
        rho = getattr(self, 'sample_overlap', None)
        if rho is None:
            rho = np.eye(self.sample_dim)[:, :, None]*np.ones(shot_noise.shape[-2])[None, None, :]
        product = np.sqrt(np.abs(shot_noise[..., :, None, :, :]*shot_noise[..., None, :, :, :]))
        factor = rho[:, :, :, None]*rho[:, :, None, :] if galaxy_pairs else rho[:, :, :, None]
        result = factor*product
        # a single sample exactly as before
        for i_sample in range(self.sample_dim):
            result[..., i_sample, i_sample, :, :] = shot_noise[..., i_sample, :, :]
        return result

    def _split_csmf_terms(self,
                          gauss,
                          nongauss,
                          ssc):
        """
        Splits the covariance of the stellar mass function (SMF) and its
        cross-covariance with the LSS probes, which the calc_cov methods
        collect at the end of the Gaussian terms, into their Gaussian (shot
        noise of the SMF, and its sample variance if the SSC is not
        requested), non-Gaussian (three-point cross term) and super-sample
        parts, which are appended to the respective lists in the
        same order (the auto-covariance first, then the cross-covariances).

        Parameters
        ----------
        gauss, nongauss, ssc : lists
            covariance terms as returned by the calc_cov methods

        Returns
        -------
        gauss, nongauss, ssc : lists
        """
        if not self.csmf:
            return gauss, nongauss, ssc
        n_csmf = 4 if type(self) is CovELLSpace else 5
        auto, cross = gauss[-n_csmf], gauss[-n_csmf + 1:]
        gauss = gauss[:-n_csmf]
        if type(self) is CovELLSpace:
            cross_ssc = list(self.csmf_parts['cross_ssc'])
        else:
            cross_ssc = getattr(self, 'csmf_projected_ssc', None)
        if cross_ssc is None or not self.cov_dict['ssc']:
            cross_ssc = [0*term for term in cross]
        cross_ssc = cross_ssc + [0*term for term in cross[len(cross_ssc):]]
        auto_sn, auto_ssc = self.csmf_parts['auto_sn'], self.csmf_parts['auto_ssc']
        if not self.cov_dict['ssc']:
            # the sample variance of the SMF is always included, without the
            # SSC terms in the Gaussian part
            auto_sn, auto_ssc = auto_sn + auto_ssc, 0*auto_ssc
        gauss = gauss + [auto_sn] + [0*term for term in cross]
        nongauss = list(nongauss) + [0*auto] + [term - term_ssc for term, term_ssc in zip(cross, cross_ssc)]
        ssc = list(ssc) + [auto_ssc] + cross_ssc
        return gauss, nongauss, ssc

    def _n_eff_clust_per_sample(self, survey_params_dict):
        """
        Effective number density of the clustering galaxies in 1/arcmin^2
        with shape (n_tomo_clust, sample_dim). 'n_eff_clust' is either one
        value per tomographic bin and galaxy sample (all samples of the first
        tomographic bin, then those of the second, ...), or one value per
        tomographic bin for a single sample or with csmf_diagonal_lenses
        (sample i in tomographic bin i). Otherwise the number density is
        estimated from the HoD.
        """
        n_eff = survey_params_dict['n_eff_clust']
        if n_eff is not None:
            n_eff = np.asarray(n_eff, dtype=float).ravel()
            if len(n_eff) == self.n_tomo_clust*self.sample_dim:
                return n_eff.reshape((self.n_tomo_clust, self.sample_dim))
            if len(n_eff) == self.n_tomo_clust and self.csmf_diagonal_lenses:
                return np.repeat(n_eff[:, None], self.sample_dim, axis=1)
        print("ConfigWarning: [survey specs]: 'n_eff_clust' does not give one "
              "value per tomographic bin and galaxy sample, the number density "
              "of the clustering galaxies is estimated from the HoD.")
        return self.Ngal.T/self.arcmin2torad2

    @property
    def _los_weights(self):
        """
        Simpson weights for line-of-sight integrals over
        self.los_integration_chi, simpson(y, x=chi, axis=0) = w @ y.
        """
        return simpson_weights(self.los_integration_chi)

    @staticmethod
    def _chi_ell_points(chi, ell, log_k=True):
        """
        Points (chi, k) with k = (ell + 1/2)/chi for all combinations of
        chi and ell, flattened with chi varying slowest (as needed for the
        line-of-sight integrals). Returns the chi values and log(k) (or k
        if not log_k).
        """
        k = (ell[None, :] + .5)/chi[:, None]
        return np.repeat(chi, len(ell)), (np.log(k) if log_k else k).ravel()

    @staticmethod
    def _unique_columns(integrand):
        """
        Finds the bitwise identical columns of the integrand (shape
        (len(ellrange), number of integrals)), which only have to be
        integrated once. Returns the indices of the unique columns and the
        index of the unique column for every column.
        """
        columns = np.ascontiguousarray(np.transpose(integrand))
        keys = columns.view(np.dtype((np.void, columns.itemsize*columns.shape[1]))).ravel()
        _, unique_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
        return unique_idx.tolist(), inverse

    @staticmethod
    def _unique_columns_blocked(array):
        """
        As _unique_columns for a large array (shape (rows, columns)), in
        blocks of PROJECTION_BLOCK_SIZE elements: columns are grouped by a
        fingerprint (their product with a fixed random vector) and only
        merged if they are bitwise identical to the first column of their
        group. Returns the indices of the unique columns and the index of
        the unique column for every column.
        """
        n_rows, n_col = array.shape
        probe = np.random.default_rng(1).standard_normal(n_rows)
        bits = array.view(np.uint64)
        fingerprint = np.empty(n_col)
        for cols in CovELLSpace._blocks(n_col, n_rows):
            # summed row by row: the same operations for every column (a
            # matrix product does not guarantee that)
            fingerprint[cols] = np.sum(probe[:, None]*array[:, cols], axis=0)
        fingerprint = fingerprint.view(np.uint64)
        representative = np.arange(n_col)
        # columns that differ from the first column with the same fingerprint
        # are grouped again among themselves
        todo = np.arange(n_col)
        while len(todo) > 0:
            _, first, group = np.unique(fingerprint[todo], return_index=True, return_inverse=True)
            candidate = todo[first[np.ravel(group)]]
            same = np.empty(len(todo), dtype=bool)
            for part in CovELLSpace._blocks(len(todo), n_rows):
                same[part] = np.all(bits[:, todo[part]] == bits[:, candidate[part]], axis=0)
            representative[todo[same]] = candidate[same]
            todo = todo[~same]
        unique_idx, inverse = np.unique(representative, return_inverse=True)
        return unique_idx, inverse

    @staticmethod
    def _ell_limits_double(ell_limits, m_mode, n_mode):
        """
        Integration limits for an integrand with the product of the weights
        m_mode and n_mode: the limits of the more oscillatory weight, i.e.
        the longer list.
        """
        if len(ell_limits[m_mode]) < len(ell_limits[n_mode]):
            return ell_limits[n_mode]
        return ell_limits[m_mode]

    def _gauss_todo(self,
                    sva_flat,
                    mix_flat,
                    sva_jobs,
                    mix_jobs):
        """
        Pairs the Gaussian ell-space covariance terms with their projection
        jobs: sva and mix separately if the Gaussian term is split, their sum
        with the sva_jobs otherwise. mix_flat can be None if there is no
        mixed term.
        """
        if mix_flat is None:
            return [(sva_flat, sva_jobs)]
        if self.cov_dict['split_gauss']:
            return [(sva_flat, sva_jobs), (mix_flat, mix_jobs)]
        return [(sva_flat + mix_flat, sva_jobs)]

    @staticmethod
    def _weight_aliases(weights):
        """
        For every weight (row of weights) returns the index of the first
        bitwise identical weight, or -1 if the weight vanishes identically
        (e.g. B-mode filters without B-modes in the C_ell). Projections with
        identical weights only have to be calculated once and projections
        with a vanishing weight are zero.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        keys = weights.view(np.dtype((np.void, weights.itemsize*weights.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        alias = first[inverse]
        alias[~np.any(weights != 0, axis=1)] = -1
        return alias.tolist()

    def _weight_key(self, mode):
        """
        Representative of the weight 'mode' according to self.weight_alias
        (set by the classes via _weight_aliases), the mode itself otherwise.
        """
        alias = getattr(self, 'weight_alias', None)
        return mode if alias is None else alias[mode]

    def _project_gauss_jobs(self,
                            levin_int,
                            todo,
                            original_shape,
                            name):
        """
        Projects Gaussian (diagonal in ell) ell-space covariances with two
        weights, target_array[target_index] =
            prefac * int dell ell W_m(ell) W_n(ell) Cov(ell, ell).

        Parameters
        ----------
        levin_int : levin.Levin
            integrator with the weights initialised via init_w_ell
        todo : list
            of (ell_cov_flat, jobs) with ell_cov_flat of shape
            (len(ellrange), len(ellrange), number of integrals), or
            (len(ellrange), 1, number of integrals) if only the diagonal is
            stored, and jobs
            (ell_limits, m_mode, n_mode, target_array, target_index, prefac[,
            mirror_index]). If mirror_index is given, the result is also
            stored there (blocks symmetric in m_mode and n_mode).

        Note
        ----
        The integrator is initialised once per ell-space covariance and all
        its jobs are carried out in one call; only the unique integrand
        columns and each distinct pair of weights are integrated, jobs with
        a vanishing weight are zero.
        """
        t0 = time.time()
        for ell_cov_flat, jobs in todo:
            live = {}
            for job in jobs:
                key = (self._weight_key(job[1]), self._weight_key(job[2]))
                if min(key) < 0:
                    job[3][job[4]] = 0
                    if len(job) > 6:
                        job[3][job[6]] = 0
                else:
                    live.setdefault(key, []).append(job)
            if len(live) == 0:
                continue
            if ell_cov_flat.shape[1] == 1 and len(self.ellrange) > 1:
                # only the diagonal is stored
                integrand = ell_cov_flat[:, 0, :]*self.ellrange[:, None]
            else:
                integrand = np.moveaxis(np.diagonal(ell_cov_flat)*self.ellrange,0,-1)
            if self._use_quadrature_projection(levin_int):
                by_range = {}
                for key, key_jobs in live.items():
                    limits = key_jobs[0][0]
                    by_range.setdefault((limits[0], limits[-1]), []).append(key)
                # identical columns (symmetries of the covariance) only once
                unique_idx, inverse = self._unique_columns(integrand)
                for (lo, hi), keys in by_range.items():
                    results = self._project_gauss_quadrature(levin_int, integrand[:, unique_idx], keys, lo, hi)
                    self._assign_gauss_results(live, keys, results, inverse, original_shape)
                continue
            levin_int.init_integral(self.ellrange, integrand, True, True)
            unique_idx, inverse = self._unique_columns(integrand)
            representatives = [key_jobs[0] for key_jobs in live.values()]
            results = levin_int.cquad_integrate_double_well_subset(
                [job[0] for job in representatives], [job[1] for job in representatives],
                [job[2] for job in representatives], unique_idx)
            self._assign_gauss_results(live, list(live.keys()), np.array(results), inverse, original_shape)
        print('\rProjection of the Gaussian term for the ' + name + ' done in ' +
              str(round((time.time()-t0)/60, 1)) + 'min', end="")

    @staticmethod
    def _assign_gauss_results(live, keys, results, inverse, original_shape):
        """
        target_array[target_index] = prefac*result (and at mirror_index) for
        the jobs of _project_gauss_jobs, with results[i] the unique columns
        (expanded with inverse) for keys[i]. Targets with set_many
        (CovTHETAtoOBS) get all their entries at once, in blocks (mirror
        entries only if they differ from the index; an entry set again right
        after it is overwritten, as with item assignment).
        """
        batched, scalar = {}, {}
        for i, key in enumerate(keys):
            result = None
            for job in live[key]:
                target = job[3]
                entry = batched.get(id(target))
                if entry is None:
                    entry = batched[id(target)] = (target, [], [], []) if hasattr(target, 'set_many') else False
                # (prefactors are numbers or arrays of size one, e.g. 1/area)
                prefac = scalar.get(id(job[5])) if entry else None
                if entry and prefac is None and np.size(job[5]) == 1:
                    prefac = scalar[id(job[5])] = float(np.reshape(job[5], -1)[0])
                if prefac is not None:
                    rows, indices, prefacs = entry[1:]
                    for index in ((job[4], job[6]) if len(job) > 6 and job[6] != job[4] else (job[4],)):
                        if indices and indices[-1] == index:
                            rows[-1], prefacs[-1] = i, prefac
                            continue
                        rows.append(i)
                        indices.append(index)
                        prefacs.append(prefac)
                    continue
                if result is None:
                    result = np.reshape(results[i][inverse], original_shape)
                target[job[4]] = job[5] * result
                if len(job) > 6:
                    target[job[6]] = target[job[4]]
        for entry in batched.values():
            if not entry or not entry[1]:
                continue
            target, rows, indices, prefacs = entry
            rows, prefacs = np.array(rows), np.array(prefacs, dtype=float)
            step = max(1, PROJECTION_BLOCK_SIZE//max(results.shape[1], 1))
            for start in range(0, len(rows), step):
                part = slice(start, start + step)
                target.set_many(indices[part], prefacs[part, None]*results[rows[part]], inverse)

    def _project_gauss_jobs_added(self,
                                  levin_int,
                                  todo,
                                  original_shape,
                                  name):
        """
        As _project_gauss_jobs, but the results are added to the target
        arrays, target_array[target_index] += prefac * int dell ell W_m(ell)
        W_n(ell) Cov(ell, ell). Used for the contribution of the intrinsic
        B-modes (with their own pair of weights) to covariance blocks which
        already hold the E-mode contribution.
        """
        buffers = {}
        added = []
        for ell_cov_flat, jobs in todo:
            buffered_jobs = []
            for job in jobs:
                if getattr(job[3], 'accumulates', False):
                    # a target that adds what is set (CovTHETAtoOBS) needs no buffer
                    buffered_jobs.append(job)
                    continue
                if id(job[3]) not in buffers:
                    buffers[id(job[3])] = (job[3], np.zeros_like(job[3]))
                buffered_jobs.append(tuple(job[:3]) + (buffers[id(job[3])][1],) + tuple(job[4:]))
            added.append((ell_cov_flat, buffered_jobs))
        self._project_gauss_jobs(levin_int, added, original_shape, name)
        for target, buffer in buffers.values():
            target += buffer

    def _Cell_mmB_projected(self,
                            levin_int,
                            ell_limits,
                            modes,
                            prefactors,
                            pixel_weight=None):
        """
        Projections of the intrinsic B-mode spectrum C_BB with the weights
        'modes' of levin_int, prefactor * int dell ell W_mode(ell) C_BB(ell),
        for the signals of the projected cosmic shear estimators.

        Parameters
        ----------
        levin_int : levin.Levin
            integrator with the weights initialised via init_w_ell
        ell_limits : list
            of the integration limits of all weights
        modes : list
            of the weights to be used
        prefactors : list
            of the prefactors for each weight (e.g. including a sign)
        pixel_weight : array
            optional, multiplied twice to the spectrum

        Returns
        -------
        result : array
            with shape (len(modes), 1, n_tomo_lens, n_tomo_lens)
        """
        result = np.zeros((len(modes), 1, self.n_tomo_lens, self.n_tomo_lens))
        if self.Cell_mmB is None:
            return result
        original_shape = self.Cell_mmB[0].shape
        integrand = np.reshape(self.Cell_mmB, (len(self.ellrange), -1))*self.ellrange[:, None]
        if pixel_weight is not None:
            integrand = integrand*(pixel_weight**2)[:, None]
        levin_int.init_integral(self.ellrange, integrand, True, True)
        for i_mode, (mode, prefactor) in enumerate(zip(modes, prefactors)):
            result[i_mode] = prefactor*np.reshape(np.array(
                levin_int.cquad_integrate_single_well(ell_limits[mode], mode)), original_shape)
        return result

    def _assign_4pt_results(self, live, values):
        """
        target_array[target_index] = prefac*value (divided by the divisor if
        not None) for the jobs of _project_4pt_jobs. A job and its mirror,
        which writes the same target array at the exchanged index with the
        outer and inner weights exchanged (the same summary statistics on
        both sides), give the same covariance with its two sides exchanged:
        both are set to their average (it is symmetric then). They differ by
        the different treatment of the two multipoles, mostly the
        extrapolation of the ell-space covariance beyond the ell range
        (T(ell_1, ell_2) is extrapolated in ell_2 for the inner integrals,
        the inner integrals in ell_1 for the outer ones).
        """
        final = []
        for job, value in zip(live, values):
            value = job[4]*value
            if job[5] is not None:
                value = value/job[5]
            final.append(value)
        position = {(id(job[2]), tuple(np.atleast_1d(job[3]))): i for i, job in enumerate(live)}
        for i, job in enumerate(live):
            index = tuple(np.atleast_1d(job[3]))
            j = position.get((id(job[2]), index[::-1])) if len(index) == 2 else None
            if j is None or j < i:
                continue
            mirror = live[j]
            if (self._weight_key(mirror[0]), self._weight_key(mirror[1])) != \
                    (self._weight_key(job[1]), self._weight_key(job[0])):
                continue
            a, b = final[i], final[j]
            if np.ndim(a) != 6:
                continue
            # (sample_1, sample_2, tomo_i, tomo_j, tomo_k, tomo_l) of one side
            # is (sample_2, sample_1, tomo_k, tomo_l, tomo_i, tomo_j) of the other
            b_t = np.transpose(b, (1, 0, 4, 5, 2, 3))
            if b_t.shape != np.shape(a):
                continue
            mean = 0.5*(a + b_t)
            final[i], final[j] = mean, np.transpose(mean, (1, 0, 4, 5, 2, 3))
        # targets with set_many (CovTHETAtoOBS) get all their entries at once,
        # in blocks (an entry set again right after it is overwritten)
        batched = {}
        for job, value in zip(live, final):
            if not (hasattr(job[2], 'set_many') and isinstance(value, np.ndarray)):
                job[2][job[3]] = value
                continue
            indices, values = batched.setdefault(id(job[2]), (job[2], [], []))[1:]
            if indices and indices[-1] == job[3]:
                values[-1] = value
                continue
            indices.append(job[3])
            values.append(value)
        for target, indices, values in batched.values():
            step = max(1, PROJECTION_BLOCK_SIZE//max(values[0].size, 1))
            for start in range(0, len(values), step):
                target.set_many(indices[start:start + step], np.array(values[start:start + step]))

    def _project_4pt_jobs(self,
                          levin_int,
                          ell_limits,
                          ell_cov,
                          jobs,
                          original_shape,
                          name,
                          outer_logy=True):
        """
        Projects a four-point (non-Gaussian or super-sample) ell-space
        covariance T(ell1, ell2),
        target_array[target_index] = prefac * int dell1 ell1 W_outer(ell1)
                                     int dell2 ell2 W_inner(ell2) T(ell1, ell2)
        and divided by divisor if it is not None.

        Parameters
        ----------
        levin_int : levin.Levin
            integrator with the weights initialised via init_w_ell
        ell_limits : list
            integration limits for each weight
        ell_cov : array
            with shape (len(ellrange), len(ellrange), ...), or an
            EllCovColumns (the results are expanded to original_shape)
        jobs : list
            of jobs (outer_mode, inner_mode, target_array, target_index,
            prefac, divisor)
        outer_logy : bool
            whether the outer integrand is interpolated in log space

        Note
        ----
        The inner integral does not depend on the outer mode, it is
        calculated once per (distinct) inner weight for all ell1 and used for
        all jobs with this inner weight. Only the unique integrand columns
        and distinct weights are integrated, jobs with a vanishing weight are
        zero.
        """
        t0 = time.time()
        live = []
        for job in jobs:
            if min(self._weight_key(job[0]), self._weight_key(job[1])) < 0:
                job[2][job[3]] = 0
            else:
                live.append(job)
        if len(live) == 0:
            return
        if self._use_quadrature_projection(levin_int):
            if self._project_4pt_quadrature(levin_int, ell_limits, ell_cov, live, original_shape, outer_logy):
                print('\rProjection of the four-point term for the ' + name + ' done in ' +
                      str(round((time.time()-t0)/60, 1)) + 'min', end="")
                return
            print("ProjectionWarning: the integrands of the four-point projection for the " + name +
                  " are not sampled accurately enough for the quadrature, using cquad instead.")
        columns = getattr(ell_cov, 'columns', None)
        ell_cov_flat = np.reshape(np.asarray(ell_cov), (len(self.ellrange), len(self.ellrange), -1))
        inner_modes = sorted(set(self._weight_key(job[1]) for job in live))
        inner = {n_mode: np.zeros((len(self.ellrange), ell_cov_flat.shape[2])) for n_mode in inner_modes}
        projected = {}
        for i_ell in range(len(self.ellrange)):
            integrand = ell_cov_flat[i_ell, :, :]*self.ellrange[:, None]
            levin_int.init_integral(self.ellrange, integrand, True, True)
            unique_idx, inverse = self._unique_columns(integrand)
            results = levin_int.cquad_integrate_single_well_subset(
                [ell_limits[n_mode][:] for n_mode in inner_modes], inner_modes, unique_idx)
            for n_mode, result in zip(inner_modes, results):
                inner[n_mode][i_ell, :] = np.array(result)[inverse]
            print('\rProjection of the four-point term for the ' + name + ' at ' +
                  str(round((i_ell + 1)/len(self.ellrange)*100, 1)) + '% in ' +
                  str(round((time.time()-t0)/60, 1)) + 'min', end="")
        for n_mode in inner_modes:
            outer_jobs = [job for job in live if self._weight_key(job[1]) == n_mode]
            outer_modes = sorted(set(self._weight_key(job[0]) for job in outer_jobs))
            integrand = inner[n_mode]*self.ellrange[:, None]
            levin_int.init_integral(self.ellrange, integrand, True, outer_logy)
            unique_idx, inverse = self._unique_columns(integrand)
            results = levin_int.cquad_integrate_single_well_subset(
                [ell_limits[m_mode][:] for m_mode in outer_modes], outer_modes, unique_idx)
            result_of = {m_mode: self._expand_columns(np.array(result)[inverse], columns, original_shape)
                         for m_mode, result in zip(outer_modes, results)}
            for job in outer_jobs:
                projected[id(job)] = result_of[self._weight_key(job[0])]
        self._assign_4pt_results(live, [projected[id(job)] for job in live])

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
        PolySpectra.__init__(self,
                             0,
                             cov_dict,
                             obs_dict,
                             cosmo_dict,
                             bias_dict,
                             hod_dict,
                             survey_params_dict,
                             prec,
                             read_in_tables)
        self.do_not_update_ell = False
        self.checked_input_cells = False
        self.do_arbitrary_radial_mm = False
        self.do_arbitrary_radial_gg = False
        self.do_arbitrary_radial = False
        self.add_to_matter = False
        self.add_to_galaxy = False
        self.n_spec = None
        if obs_dict['ELLspace']['n_spec'] is not None and obs_dict['ELLspace']['n_spec'] != 0:
            self.n_spec = int(obs_dict['ELLspace']['n_spec'])
        self.number_radial_weights_mm = 0
        if read_in_tables['arb_radial']['do_arbitrary_radial_weights']:
            self.do_arbitrary_radial = read_in_tables['arb_radial']['do_arbitrary_radial_weights']
            if read_in_tables['arb_radial']['number_radial_weights_gg']:
                self.do_arbitrary_radial_gg = True
                if read_in_tables['arb_radial']['add_to_galaxy']:
                    self.add_to_galaxy = True
            if read_in_tables['arb_radial']['number_radial_weights_mm']:
                self.do_arbitrary_radial_mm = True
                self.number_radial_weights_mm = read_in_tables['arb_radial']['number_radial_weights_mm']
                if read_in_tables['arb_radial']['add_to_matter']:
                    self.add_to_matter = True

        self.ellrange_spec = None
        self.ellrange_photo = None
        self.ellrange_spec_photo = None

        self.est_shear = obs_dict['observables']['est_shear']
        self.est_ggl = obs_dict['observables']['est_ggl']
        self.est_clust = obs_dict['observables']['est_clust']
        self.clustering_z = obs_dict['observables']['clustering_z']
        if obs_dict['arbitrary_summary']['do_arbitrary_summary'] is not None:
            self.do_arbitrary = obs_dict['arbitrary_summary']['do_arbitrary_summary']
        else:
            self.do_arbitrary = False
        self.csmf = obs_dict['observables']['csmf']
        self.csmf_diagonal_lenses = False   
        self.csmf_diagonal = False
        self.csmf_auto_only = False
        if self.csmf:
            if obs_dict['observables']['csmf_log10M_bins_upper'] is not None and obs_dict['observables']['csmf_log10M_bins_lower'] is not None:
                csmf_lower = np.asarray(obs_dict['observables']['csmf_log10M_bins_lower'], dtype=float)
                csmf_upper = np.asarray(obs_dict['observables']['csmf_log10M_bins_upper'], dtype=float)
            else:
                csmf_lower = np.asarray(obs_dict['observables']['csmf_log10M_bins'][:-1], dtype=float)
                csmf_upper = np.asarray(obs_dict['observables']['csmf_log10M_bins'][1:], dtype=float)
            # centres of the bins in stellar mass (only used as labels)
            self.log10csmf_mass_bins = np.log10((10**csmf_lower + 10**csmf_upper)/2)
            # the SMF is measured per dex, Phi_mu = N_mu/(Delta log10 M*_mu V_max):
            # all quantities linear in the SMF are averaged over the bins with
            # the quadrature of _csmf_bin_quadrature
            self.deltalog10M_csmf = csmf_upper - csmf_lower
            self.csmf_quad_nodes, self.csmf_quad_weights = \
                self._csmf_bin_quadrature(csmf_lower, csmf_upper, per_dex=True)
            self.Vmax = read_in_tables['csmf']['V_max']
            self.csmf_V_max_zmin = read_in_tables['csmf']['V_max_zmin']
            # overlapping stellar mass bins of the SMF (bins given via
            # csmf_log10M_bins_lower/upper) share galaxies: (bin 1, bin 2),
            # with the quadrature of the integral over the overlap
            self.csmf_bin_overlaps = []
            overlap_lower, overlap_upper = [], []
            for i_mass in range(len(csmf_lower)):
                for j_mass in range(i_mass + 1, len(csmf_lower)):
                    low, high = max(csmf_lower[i_mass], csmf_lower[j_mass]), min(csmf_upper[i_mass], csmf_upper[j_mass])
                    if high > low:
                        self.csmf_bin_overlaps.append((i_mass, j_mass))
                        overlap_lower.append(low)
                        overlap_upper.append(high)
            self.csmf_overlap_nodes, self.csmf_overlap_weights = \
                self._csmf_bin_quadrature(overlap_lower, overlap_upper, per_dex=False)
            self.csmf_number_mass_bins = len(self.log10csmf_mass_bins)
            self.csmf_diagonal = obs_dict['observables']['csmf_diagonal']
            self.csmf_auto_only = obs_dict['observables']['csmf_auto_only']
            if self.csmf_diagonal and (self.n_tomo_csmf != self.csmf_number_mass_bins):
                raise Exception("ConfigError: You ask for the diagonal-only mode in the stellar mass function, i.e. considering only bins i with tomographic bin j. However, "
                                "the number of tomographic bins is " + str(self.n_tomo_csmf) + ", while the number of mass bins is " + str(self.csmf_number_mass_bins) +
                                ". Please fix this in the config so that they are equal, or deactivate 'csmf_diagonal'.") 
        self.csmf_diagonal_lenses = bool(obs_dict['observables']['csmf_diagonal_lenses']) and self.sample_dim > 1
        if self.csmf_diagonal_lenses and (self.n_tomo_clust != self.sample_dim):
            raise Exception("ConfigError: You ask for the diagonal-only mode for the clustering measurement, i.e. considering only stellar mass bins i with tomographic bin i. However, "
                            "the number of tomographic bins for clustering is " + str(self.n_tomo_clust) + ", while the number of stellar mass bins for clustering is " + str(self.sample_dim) +
                            ". Please fix this in the config so that they are equal, or deactivate 'csmf_diagonal_lenses'.")
        self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
        self.Cells, self.tab_bools = self.__check_for_tabulated_Cells(read_in_tables['Cxy'])
        if (self.mm and self.est_shear == 'C_ell') and ((self.gm and self.est_ggl == 'C_ell') or (self.gg and self.est_clust == 'C_ell')):
            if (self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is None) or (self.ellrange_clustering_ul is None and self.ellrange_lensing_ul is not None):
                if self.ellrange_spec_ul is None and self.ellrange_photo_ul is None:
                    raise Exception("ConfigError: You require the C_ell covariance for lensing and clustering or ggl. However, you only specified the ellrange for lensing and not for clustering/GGL or vice versa. Please fix this in the config file under covELLspace settings.")     
        self.integration_intervals = obs_dict['THETAspace']['integration_intervals']
        self.projection_method = obs_dict['ELLspace'].get('projection_method') or 'quadrature'
        self.deg2torad2 = DEG2TORAD2
        self.arcmin2torad2 = ARCMIN2TORAD2
        # Handle redshift-dependent bias if enabled
        if self.redshift_dep_bias:
            self.bias_of_zet = []
            for i_tomo in range(self.n_tomo_clust):
                self.bias_of_zet.append(
                    UnivariateSpline(
                        read_in_tables['zclust']['z'],
                        read_in_tables['zet_dep_bias'][i_tomo, :],
                        k=3, s=0, ext=1
                    )
                )
        # Set up redshift distribution splines
        self.__set_redshift_distribution_splines(obs_dict['ELLspace'], read_in_tables)
        if self.csmf:
            self.__set_csmf_windows(obs_dict['ELLspace'], survey_params_dict, read_in_tables)
            self.__print_csmf_settings(read_in_tables)
        # Check k-range support for ell calculations
        self.__check_krange_support(
            obs_dict, cosmo_dict, bias_dict, hod_dict, prec)
        # Pixel window function and weights
        if obs_dict['ELLspace'].get('pixelised_cell', False):
            pixel_Nside = obs_dict['ELLspace']['pixel_Nside']
            obs_dict['ELLspace']['ellmax'] = int(3 * pixel_Nside + 1)
            self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
            integer_ell = np.copy(self.ellrange.astype(int))
            # Healpy pixel window function
            self.pixel_weight = hp.sphtfunc.pixwin(pixel_Nside)[integer_ell]
            self.pixelweight_matrix = (
                self.pixel_weight[:, None] ** 2 * self.pixel_weight[None, :] ** 2
            )
            self.Cells, self.tab_bools = None, [False, False, False]
        else:
            self.pixel_weight = np.ones_like(self.ellrange)
        # Set up CAMB parameters for matter power spectrum
        self.camb_pars.set_matter_power(kmax=self.mass_func.k[-1], redshifts=[0])
        # sigma_8 from the transfer functions already calculated for the mass function (same parameters)
        results = camb_transfer_functions(self.camb_pars_new)
        results.calc_power_spectra()
        # Update initial power spectrum parameters
        self.camb_pars_new.InitPower.set_params(
            ns=cosmo_dict['ns'],
            As=1.8e-9 / results.get_sigma8()[0] ** 2 * cosmo_dict['sigma8'] ** 2
        )
        self.num_cores_save = self.num_cores
        # Calculate survey area
        self.calc_survey_area(survey_params_dict)
        # Compute Cells (angular power spectra)
        self.get_Cells(
            obs_dict, output_dict,
            bias_dict, iA_dict, hod_dict, prec, read_in_tables
        )
        
       
    
    def __check_krange_support(self,
                               obs_dict,
                               cosmo_dict,
                               bias_dict,
                               hod_dict,
                               prec):
        r"""
        Performs consistency checks whether the k-range support is big enough
        to house a given ell_range calculations for the Cells. If not, the ranges
        are updated accordingly and the user is notified.

        Parameters
        ----------
        obs_dict : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
            'observables' : dictionary
                Specifies which observables (cosmic shear, galaxy-galaxy
                lensing and/or clustering) should be calculated. Also,
                indicates whether cross-terms are evaluated.
            'ELLspace' : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.
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

        Returns:
        ---------
        True

        """
        update_massfunc, update_ellrange, kmin, kmax, ellmin, ellmax, ellbins = \
            self.consistency_checks_for_Cell_calculation(
                obs_dict, cosmo_dict, prec['powspec'],
                self.ellrange, self.los_integration_chi, self.do_not_update_ell)
        if update_massfunc:
            self.mass_func.update(lnk_min=np.log(10**kmin),
                                  lnk_max=np.log(10**kmax))
            self.update_mass_func(
                self.mass_func.z, bias_dict, hod_dict, prec)

        if update_ellrange:
            obs_dict['ELLspace']['ell_min'] = ellmin
            obs_dict['ELLspace']['ell_max'] = ellmax
            obs_dict['ELLspace']['ell_bins'] = ellbins
            obs_dict['ELLspace']['ell_type'] = 'log'
            self.ellrange = self.__set_multipoles(obs_dict['ELLspace'])
        return True

    def __set_multipoles(self,
                         covELLspacesettings):
        r"""
        Calculates the multioples at which the covariance is calculated.
        Also sets the multipoles for the binned ell-space covariance for
        the spectroscopic and photometric samples.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the multipoles at which the covariance is
            evaluated.

        Returns
        -------
        ellrange : array
            with shape (ell bins)

        """
        self.ellrange_clustering_ul = None
        self.ellrange_clustering = None
        if covELLspacesettings['ell_type_clustering'] is not None and covELLspacesettings['ell_bins_clustering'] is not None and covELLspacesettings['ell_max_clustering'] is not None and covELLspacesettings['ell_min_clustering'] is not None:
            if (self.est_clust == "C_ell" and self.gg) or (self.est_ggl == "C_ell" and self.gm): 
                if covELLspacesettings['ell_type_clustering'] == 'lin':
                    self.ellrange_clustering_ul = np.linspace(
                        covELLspacesettings['ell_min_clustering'],
                        covELLspacesettings['ell_max_clustering'],
                        covELLspacesettings['ell_bins_clustering'] + 1).astype(int)
                    self.ellrange_clustering = .5 * (self.ellrange_clustering_ul[1:] + self.ellrange_clustering_ul[:-1])
                else:
                    self.ellrange_clustering_ul = np.unique(np.geomspace(covELLspacesettings['ell_min_clustering'], covELLspacesettings['ell_max_clustering'], covELLspacesettings['ell_bins_clustering'] + 1).astype(int))
                    if len(self.ellrange_clustering_ul) != covELLspacesettings['ell_bins_clustering'] + 1:
                        print("InputWarning: you required", covELLspacesettings['ell_bins_clustering'], "logarithmically-spaced clustering bins.",
                              "However, the specified clustering ell-range from ", covELLspacesettings['ell_min_clustering'],  "to", covELLspacesettings['ell_max_clustering'], "does not support that many unique bins.",
                              "Adjusting the number of bins to", len(self.ellrange_clustering_ul) - 1, ". The bin boundaries are:")
                    self.ellrange_clustering = np.exp(.5 * (np.log(self.ellrange_clustering_ul[1:])
                                        + np.log(self.ellrange_clustering_ul[:-1])))
        self.ellrange_lensing_ul = None
        self.ellrange_lensing = None
        if covELLspacesettings['ell_type_lensing'] is not None and covELLspacesettings['ell_bins_lensing'] is not None and covELLspacesettings['ell_max_lensing'] is not None and covELLspacesettings['ell_min_lensing'] is not None:
            if (self.est_shear == "C_ell" and self.mm): 
                if covELLspacesettings['ell_type_lensing'] == 'lin':
                    self.ellrange_lensing_ul = np.linspace(
                        covELLspacesettings['ell_min_lensing'],
                        covELLspacesettings['ell_max_lensing'],
                        covELLspacesettings['ell_bins_lensing'] + 1).astype(int)
                    self.ellrange_lensing = .5 * (self.ellrange_lensing_ul[1:] + self.ellrange_lensing_ul[:-1])
                else:
                    self.ellrange_lensing_ul = np.unique(np.geomspace(covELLspacesettings['ell_min_lensing'], covELLspacesettings['ell_max_lensing'], covELLspacesettings['ell_bins_lensing'] + 1).astype(int))
                    if len(self.ellrange_lensing_ul) != covELLspacesettings['ell_bins_lensing'] + 1:
                        print("InputWarning: you required", covELLspacesettings['ell_bins_lensing'],"logarithmically-spaced lensing bins.",
                              "However, the specified lensing ell-range from", covELLspacesettings['ell_min_lensing'],  "to", covELLspacesettings['ell_max_lensing'], "does not support that many unique bins.", 
                              "Adjusting the number of bins to", len(self.ellrange_lensing_ul) - 1, ". The bin boundaries are:") 
                    self.ellrange_lensing = np.exp(.5 * (np.log(self.ellrange_lensing_ul[1:])
                                        + np.log(self.ellrange_lensing_ul[:-1])))


        self.ellrange_spec_ul = None
        self.ellrange_photo_ul = None
        self.ellrange_spec_photo_ul = None
        if covELLspacesettings['n_spec'] is not None and covELLspacesettings['n_spec'] != 0:
            if covELLspacesettings['ell_spec_type'] == 'lin':
                self.ellrange_spec_ul = np.linspace(
                    covELLspacesettings['ell_spec_min'],
                    covELLspacesettings['ell_spec_max'],
                    covELLspacesettings['ell_spec_bins'] + 1).astype(int)
                self.ellrange_spec = .5 * (self.ellrange_spec_ul[1:] + self.ellrange_spec_ul[:-1])
            else:
                self.ellrange_spec_ul = np.geomspace(covELLspacesettings['ell_spec_min'], covELLspacesettings['ell_spec_max'], covELLspacesettings['ell_spec_bins'] + 1).astype(int)
                self.ellrange_spec = np.exp(.5 * (np.log(self.ellrange_spec_ul[1:])
                                    + np.log(self.ellrange_spec_ul[:-1])))
            if covELLspacesettings['ell_photo_type'] == 'lin':
                self.ellrange_photo_ul = np.linspace(
                    covELLspacesettings['ell_photo_min'],
                    covELLspacesettings['ell_photo_max'],
                    covELLspacesettings['ell_photo_bins'] + 1).astype(int)
                self.ellrange_photo = .5 * (self.ellrange_photo_ul[1:] + self.ellrange_photo_ul[:-1])
            else:
                self.ellrange_photo_ul = np.geomspace(covELLspacesettings['ell_photo_min'], covELLspacesettings['ell_photo_max'], covELLspacesettings['ell_photo_bins'] + 1).astype(int)
                self.ellrange_photo = np.exp(.5 * (np.log(self.ellrange_photo_ul[1:])
                                    + np.log(self.ellrange_photo_ul[:-1])))
            if covELLspacesettings['ell_spec_photo_type'] == 'lin':
                self.ellrange_spec_photo_ul = np.linspace(
                    covELLspacesettings['ell_spec_photo_min'],
                    covELLspacesettings['ell_spec_photo_max'],
                    covELLspacesettings['ell_spec_photo_bins'] + 1).astype(int)
                self.ellrange_spec_photo = .5 * (self.ellrange_spec_photo_ul[1:] + self.ellrange_spec_photo_ul[:-1])
            else:
                self.ellrange_spec_photo_ul = np.geomspace(covELLspacesettings['ell_spec_photo_min'], covELLspacesettings['ell_spec_photo_max'], covELLspacesettings['ell_spec_photo_bins'] + 1).astype(int)
                self.ellrange_spec_photo = np.exp(.5 * (np.log(self.ellrange_spec_photo_ul[1:])
                                    + np.log(self.ellrange_spec_photo_ul[:-1])))
        if covELLspacesettings['ell_type'] == 'lin':
            return np.linspace(
                covELLspacesettings['ell_min'],
                covELLspacesettings['ell_max'],
                covELLspacesettings['ell_bins'])
        elif covELLspacesettings['limber']:
            return np.geomspace(covELLspacesettings['ell_min'], covELLspacesettings['ell_max'], covELLspacesettings['ell_bins'])
        else:
            return np.unique(np.geomspace(covELLspacesettings['ell_min'], covELLspacesettings['ell_max'], covELLspacesettings['ell_bins']).astype(int)).astype(float)
        
        

    def __set_redshift_distribution_splines(self,
                                            covELLspacesettings,
                                            read_in_tables):
        r"""
        Sets the splines the redshift distributions for later integration along
        the line-of-sight.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the order of the polynomial for the interpolation
            of redshift distributions

        """
        los_integration_z = np.linspace(
            0, self.zet_max, covELLspacesettings['integration_steps'])
        self.spline_z_of_chi = UnivariateSpline(
            self.cosmology.comoving_distance(los_integration_z).value
            * self.cosmology.h,
            los_integration_z,
            k=1, s=0, ext=0)

        self.chimin = 10
        self.chimax = self.cosmology.comoving_distance(self.zet_max).value \
            * self.cosmology.h
        self.los_integration_chi = np.geomspace(
            self.chimin, self.chimax, covELLspacesettings['integration_steps'])
        # the line-of-sight grid of all tracers (the C_ell calculations
        # narrow self.los_integration_chi to the range of each bin pair)
        self.los_integration_chi_full = np.copy(self.los_integration_chi)

        arb_gg = read_in_tables['arb_radial']['do_arbitrary_radial_weights'] and \
            read_in_tables['arb_radial']['number_radial_weights_gg'] is not None and \
            not read_in_tables['arb_radial']['add_to_galaxy']
        self.magnification_active = bool(self.magnification) and (self.gg or self.gm) \
            and 's' in self.zet_clust and not arb_gg
        if self.magnification and not self.magnification_active and (self.gg or self.gm):
            print("ConfigWarning: [magnification] is ignored: no s(z) given or the clustering kernels " +
                  "are replaced by arbitrary radial weights.")
        self.spline_zclust = []
        # clustering kernels without the redshift dependent bias and the
        # slope of the number counts, for the magnification bias
        self.spline_zclust_unbiased = []
        self.spline_mag_s = []
        self.spline_zlens = []
        self.spline_zcsmf = []
        self.chi_min_clust = np.zeros(self.n_tomo_clust)
        self.chi_max_clust = np.zeros(self.n_tomo_clust)
        if self.gm or self.gg:
            dzdchi_clust = self.cosmology.efunc(self.zet_clust['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        if self.gm or self.mm:
            dzdchi_lens = self.cosmology.efunc(self.zet_lens['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        if self.csmf:
            dzdchi_csmf = self.cosmology.efunc(self.zet_csmf['z']) \
                / self.cosmology.hubble_distance.value  \
                / self.cosmology.h
        subtract = 0
        
        if self.add_to_galaxy:
            subtract = read_in_tables['arb_radial']['number_radial_weights_gg']
        for tomo in range(self.n_tomo_clust - subtract):
            norm = simpson(self.zet_clust['nz'][tomo], x = self.zet_clust['z'])
            self.zet_clust['nz'][tomo] /= norm
            if covELLspacesettings['nz_polyorder'] == 0:
                self.spline_zclust.append(interp1d(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust,
                    kind='nearest-up', fill_value=0))
            else:
                self.spline_zclust.append(UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust,
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
            min_check = 0
            for i_z, zet in enumerate(self.los_integration_chi, start = 1):     
                cdf = simpson(
                    self.spline_zclust[tomo](self.los_integration_chi[:i_z]),  x = self.los_integration_chi[:i_z])
                if cdf > 0. and min_check == 0:
                    if i_z > 1:
                        self.chi_min_clust[tomo] = self.los_integration_chi[i_z]
                    else:
                        self.chi_min_clust[tomo] = self.los_integration_chi[0]
                    min_check = 1
                if cdf >= 0.9999 or i_z == len(self.los_integration_chi) - 1:
                    if i_z < len(self.los_integration_chi) - 1:
                        self.chi_max_clust[tomo] = self.los_integration_chi[i_z]
                    else:
                        self.chi_max_clust[tomo] = self.los_integration_chi[-1]
                    break
            self.spline_zclust_unbiased.append(self.spline_zclust[tomo])
            if self.magnification_active:
                chi_of_z = self.cosmology.comoving_distance(self.zet_clust['z']).value \
                    * self.cosmology.h
                self.spline_mag_s.append(interp1d(
                    chi_of_z, self.zet_clust['s'][tomo], bounds_error=False,
                    fill_value=(self.zet_clust['s'][tomo][0], self.zet_clust['s'][tomo][-1])))
            if self.redshift_dep_bias:
                self.spline_zclust[tomo] = UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_clust['z']).value
                    * self.cosmology.h,
                    self.zet_clust['nz'][tomo]*dzdchi_clust*read_in_tables['zet_dep_bias'][tomo,:],
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1)
            
            
        
        if read_in_tables['arb_radial']['do_arbitrary_radial_weights']:
            if read_in_tables['arb_radial']['number_radial_weights_gg'] is not None:
                if self.gm or self.gg:
                    if not read_in_tables['arb_radial']['add_to_galaxy']:
                        self.spline_zclust = []
                        max_arb = self.n_tomo_clust
                    else:
                        max_arb = read_in_tables['arb_radial']['number_radial_weights_gg']
                    for tomo in range(max_arb):
                        if covELLspacesettings['nz_polyorder'] == 0:
                            self.spline_zclust.append(interp1d(
                                self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_gg']).value
                                * self.cosmology.h,
                                read_in_tables['arb_radial']['wz_gg'][tomo],
                                kind='nearest-up', fill_value=0))
                        else:
                            self.spline_zclust.append(UnivariateSpline(
                                self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_gg']).value
                                * self.cosmology.h,
                                read_in_tables['arb_radial']['wz_gg'][tomo],
                                k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
                        min_check = 0
                        norm = simpson(read_in_tables['arb_radial']['wz_gg'][tomo], x = read_in_tables['arb_radial']['z_gg'])
                        add_idx = 0
                        if read_in_tables['arb_radial']['add_to_galaxy']:
                            add_idx += read_in_tables['arb_radial']['number_radial_weights_gg']
                        for i_z, zet in enumerate(self.los_integration_chi, start = 1):     
                            cdf = simpson(
                                self.spline_zclust[tomo + add_idx](self.los_integration_chi[:i_z]),  x = self.los_integration_chi[:i_z])/norm
                            if cdf > 0. and min_check == 0:
                                if i_z > 1:
                                    self.chi_min_clust[tomo + add_idx] = self.los_integration_chi[i_z -2]
                                else:
                                    self.chi_min_clust[tomo + add_idx] = self.los_integration_chi[0]
                                min_check = 1
                            if cdf >= 0.9999 or i_z == len(self.los_integration_chi) - 1:
                                if i_z < len(self.los_integration_chi) - 1:
                                    self.chi_max_clust[tomo + add_idx] = self.los_integration_chi[i_z +1]
                                else:
                                    self.chi_max_clust[tomo + add_idx] = self.los_integration_chi[-1]
                                break
            if self.gm or self.mm:
                if read_in_tables['arb_radial']['number_radial_weights_mm'] is not None:
                    if not self.add_to_matter:
                        self.spline_lensweight = []
                        for tomo in range(self.n_tomo_lens):
                            if covELLspacesettings['nz_polyorder'] == 0:
                                self.spline_lensweight.append(interp1d(
                                    self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_mm']).value
                                    * self.cosmology.h,
                                    read_in_tables['arb_radial']['wz_mm'][tomo],
                                    kind='nearest-up', fill_value=0))
                            else:
                                self.spline_lensweight.append(UnivariateSpline(
                                    self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_mm']).value
                                    * self.cosmology.h,
                                    read_in_tables['arb_radial']['wz_mm'][tomo],
                                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
                    else:
                        self.aux_spline_lensweight = []
                        for tomo in range(self.number_radial_weights_mm):
                            if covELLspacesettings['nz_polyorder'] == 0:
                                self.aux_spline_lensweight.append(interp1d(
                                    self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_mm']).value
                                    * self.cosmology.h,
                                    read_in_tables['arb_radial']['wz_mm'][tomo],
                                    kind='nearest-up', fill_value=0))
                            else:
                                self.aux_spline_lensweight.append(UnivariateSpline(
                                    self.cosmology.comoving_distance(read_in_tables['arb_radial']['z_mm']).value
                                    * self.cosmology.h,
                                    read_in_tables['arb_radial']['wz_mm'][tomo],
                                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
        

        if self.csmf:
            for tomo in range(self.n_tomo_csmf):
                norm = simpson(self.zet_csmf['pz'][tomo], x = self.zet_csmf['z'])
                self.zet_csmf['pz'][tomo] /= norm
                self.spline_zcsmf.append(UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_csmf['z']).value
                    * self.cosmology.h,
                    self.zet_csmf['pz'][tomo]*dzdchi_csmf,
                    s=0, ext=1))
                norm = simpson(self.spline_zcsmf[tomo](self.los_integration_chi), x = self.los_integration_chi)
                self.spline_zcsmf[tomo] = UnivariateSpline(
                    self.cosmology.comoving_distance(self.zet_csmf['z']).value
                    * self.cosmology.h,
                    self.zet_csmf['pz'][tomo]*dzdchi_csmf/norm,
                    s=0, ext=1)

            
        for tomo in range(self.n_tomo_lens - self.number_radial_weights_mm):
            if covELLspacesettings['nz_polyorder'] == 0:
                self.spline_zlens.append(interp1d(self.cosmology.comoving_distance(
                    self.zet_lens['z']).value*self.cosmology.h,
                    self.zet_lens['photoz'][tomo]*dzdchi_lens,
                    kind='nearest-up', fill_value=0))
            else:
                self.spline_zlens.append(UnivariateSpline(
                    self.cosmology.comoving_distance(
                        self.zet_lens['z']).value*self.cosmology.h,
                    self.zet_lens['photoz'][tomo]*dzdchi_lens,
                    k=covELLspacesettings['nz_polyorder'], s=0, ext=1))
        
        
        
    def __set_csmf_windows(self,
                           covELLspacesettings,
                           survey_params_dict,
                           read_in_tables):
        r"""
        Sets the normalised line-of-sight distributions p_mu^i(chi) of the
        galaxies which enter stellar mass bin mu and tomographic bin i of
        the stellar mass function (SMF) per dex, Phi_mu^i = N_mu^i/(Delta log10 M*_mu V_max).
        They are either read in per stellar mass bin ('zcsmf_mass_file'),
        or they are top-hats in comoving volume defined by V_max,
        p_mu^i(chi) = 3 chi^2 / (chi_u^3 - chi_l^3) for chi_l < chi < chi_u,
        with V_max = A/3 (chi_u^3 - chi_l^3), as for volume limited
        samples. The lower bound chi_l is set by 'V_max_zmin' or, if not
        given, by the lower edge (0.1 per cent quantile) of the redshift
        distribution of the tomographic bin ('zcsmf_file').

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            here the polynomial order of the n(z) interpolation.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant value is the survey area of the clustering
            measurement. To be passed from the read_input method of the
            Input class.
        read_in_tables : dictionary
            Look-up tables, here the redshift distributions per stellar
            mass bin of the SMF. To be passed from the read_input method
            of the FileInput class.

        Returns
        -------
        Sets the private variables 'self.csmf_window_chi_range' with shape
        (csmf mass bins, csmf tomographic bins, 2) and
        'self.spline_csmf_window' (None for the V_max top-hats).

        """
        n_mass = len(self.log10csmf_mass_bins)
        self.csmf_window_chi_range = np.zeros((n_mass, self.n_tomo_csmf, 2))
        self.spline_csmf_window = None
        if read_in_tables['zcsmf']['pz_mass'] is not None:
            zet = read_in_tables['zcsmf']['z_mass']
            chi = self.cosmology.comoving_distance(zet).value*self.cosmology.h
            dzdchi = self.cosmology.efunc(zet) \
                / self.cosmology.hubble_distance.value \
                / self.cosmology.h
            self.spline_csmf_window = []
            for i_mass in range(n_mass):
                self.spline_csmf_window.append([])
                for i_tomo in range(self.n_tomo_csmf):
                    p_chi = read_in_tables['zcsmf']['pz_mass'][i_tomo, i_mass]*dzdchi
                    p_chi = p_chi/simpson(p_chi, x = chi)
                    support = np.where(p_chi > 0)[0]
                    self.csmf_window_chi_range[i_mass, i_tomo, 0] = chi[support[0]]
                    self.csmf_window_chi_range[i_mass, i_tomo, 1] = chi[support[-1]]
                    self.spline_csmf_window[i_mass].append(UnivariateSpline(
                        chi, p_chi, k=max(1, covELLspacesettings['nz_polyorder']), s=0, ext=1))
            return True

        area = survey_params_dict['survey_area_clust'][0]/self.deg2torad2
        chi = self.cosmology.comoving_distance(self.zet_csmf['z']).value*self.cosmology.h
        for i_tomo in range(self.n_tomo_csmf):
            cdf = cumulative_trapezoid(self.spline_zcsmf[i_tomo](chi), chi, initial = 0)
            cdf /= cdf[-1]
            if self.csmf_V_max_zmin is not None:
                chi_low = self.cosmology.comoving_distance(self.csmf_V_max_zmin[i_tomo]).value*self.cosmology.h
            else:
                chi_low = np.interp(1e-3, cdf, chi)
            chi_n_of_z_max = chi[np.where(self.spline_zcsmf[i_tomo](chi) > 0)[0][-1]]
            for i_mass in range(n_mass):
                chi_high = (chi_low**3 + 3.0*self.Vmax[i_mass, i_tomo]/area)**(1/3)
                if chi_high > chi_n_of_z_max:
                    print("ConfigWarning: The V_max of stellar mass bin " + str(i_mass + 1) +
                          " in tomographic bin " + str(i_tomo + 1) + " of the stellar mass " +
                          "function extends to z = " + str(round(float(self.__z_of_chi(chi_high)), 3)) +
                          " beyond the redshift distribution of that bin (z = " +
                          str(round(float(self.__z_of_chi(chi_n_of_z_max)), 3)) + "). Please check " +
                          "'V_max_file', 'V_max_zmin' and 'zcsmf_file'.")
                self.csmf_window_chi_range[i_mass, i_tomo, 0] = chi_low
                self.csmf_window_chi_range[i_mass, i_tomo, 1] = chi_high
        return True

    def __z_of_chi(self,
                   chi):
        """
        Redshift at comoving distance chi (in Mpc/h).
        """
        zet = np.linspace(0, 10, 10000)
        return np.interp(chi, self.cosmology.comoving_distance(zet).value*self.cosmology.h, zet)

    @staticmethod
    def _csmf_bin_quadrature(log10_lower,
                             log10_upper,
                             per_dex=True):
        """
        Gauss-Legendre quadrature in log10 M* over stellar mass bins,
        sum_n weights[mu, n] f(10**nodes[n]) = int_mu dM* f(M*), divided by
        the width of bin mu in dex if per_dex. Applied to the SMF dn/dM*,
        this is the SMF per dex averaged over the bin, N_mu/(Delta log10 M*
        V), and in the same way for everything linear in the SMF (Phi~, the
        count bispectra). The number of nodes grows with the width of the
        bin (CSMF_MIN_NODES, CSMF_NODES_PER_DEX).

        Parameters
        ----------
        log10_lower, log10_upper : arrays
            log10 of the lower and upper bounds of the bins

        Returns
        -------
        nodes : array
            log10 M* of the nodes of all bins, shape (nodes,)
        weights : array
            with shape (bins, nodes)
        """
        nodes, weights = [], []
        for low, high in zip(log10_lower, log10_upper):
            x, w = np.polynomial.legendre.leggauss(
                CSMF_MIN_NODES + int(np.ceil((high - low)*CSMF_NODES_PER_DEX)))
            log10M = low + (x + 1)*(high - low)/2
            # dM* = ln(10) M* dlog10 M*
            weight = w*(high - low)/2*np.log(10)*10**log10M
            nodes.append(log10M)
            weights.append(weight/(high - low) if per_dex else weight)
        if not nodes:
            return np.zeros(0), np.zeros((0, 0))
        return np.concatenate(nodes), block_diag(*[w[None, :] for w in weights])

    def _csmf_window(self,
                     i_mass,
                     i_tomo,
                     chi):
        """
        Normalised line-of-sight distribution p_mu^i(chi) of the galaxies in
        stellar mass bin i_mass and tomographic bin i_tomo of the stellar
        mass function (see __set_csmf_windows), zero outside its support.

        Parameters
        ----------
        i_mass, i_tomo : int
            stellar mass bin and tomographic bin of the SMF
        chi : array
            comoving distances in Mpc/h

        Returns
        -------
        p_chi : array
            with unit h/Mpc
            with shape (len(chi))
        """
        chi_low, chi_high = self.csmf_window_chi_range[i_mass, i_tomo]
        inside = (chi >= chi_low) & (chi <= chi_high)
        if self.spline_csmf_window is None:
            return np.where(inside, 3.0*chi**2/(chi_high**3 - chi_low**3), 0.0)
        return np.where(inside, self.spline_csmf_window[i_mass][i_tomo](chi), 0.0)

    def _csmf_los_chi(self,
                      i_mass,
                      i_tomo,
                      chi_low,
                      chi_high,
                      covELLspacesettings):
        """
        Line-of-sight integration points on the overlap of [chi_low,
        chi_high] with the support of the SMF window of stellar mass bin
        i_mass and tomographic bin i_tomo, None if they do not overlap.
        """
        chi_low = max(chi_low, self.csmf_window_chi_range[i_mass, i_tomo, 0])
        chi_high = min(chi_high, self.csmf_window_chi_range[i_mass, i_tomo, 1])
        if chi_low >= chi_high:
            return None
        return self.__get_updated_los_integration_chi(chi_low, chi_high, covELLspacesettings)

    def __set_lensweight_splines(self,
                                 covELLspacesettings,
                                 iA_dict):
        r"""
        Calculates and splines the lensing weight for later integration
        along the line-of-sight.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        """
        linear_growth_factor = np.interp(self.los_integration_chi,self.los_chi,(self.power_mm_lin_z[:,int(len(self.mass_func.k)/2)]/self.power_mm_lin_z[0,int(len(self.mass_func.k)/2)])**0.5)
        self.spline_lensweight = []
        # intrinsic alignment part of spline_lensweight (for the non-Limber
        # projection, where it has a different ell dependence)
        self.spline_lensweight_IA = []
        for tomo in range(self.n_tomo_lens - self.number_radial_weights_mm):
            norm = simpson(self.spline_zlens[tomo](
                self.los_integration_chi), x = self.los_integration_chi)
            aux_integral = np.zeros_like(self.los_integration_chi)
            for zetidx in range(covELLspacesettings['integration_steps']):
                aux_chi = self.los_integration_chi[zetidx]
                chi = np.geomspace(
                    aux_chi, self.chimax, covELLspacesettings['integration_steps'])
                aux_integral[zetidx] = simpson(self.spline_zlens[tomo](chi)
                                                * (chi - aux_chi)/chi,
                                                x = chi)/norm
            aux_integral = aux_integral * 3/2 \
                / self.cosmology.hubble_distance.value**2 \
                / self.cosmology.h**2 \
                * self.cosmology.Om0 \
                * self.los_integration_chi \
                / self.cosmology.scale_factor(
                    self.spline_z_of_chi(self.los_integration_chi))
            aux_W_iA = -iA_dict['A_IA']*((1.0 + self.spline_z_of_chi(self.los_integration_chi))/(1+iA_dict['z_pivot_IA']))**(iA_dict['eta_IA'])*0.0134*self.cosmology.Om0/linear_growth_factor*self.spline_zlens[tomo](
                self.los_integration_chi)/norm
            self.spline_lensweight.append(UnivariateSpline(self.los_integration_chi,
                                                           aux_integral + aux_W_iA,
                                                           k=1, s=0, ext=0))
            self.spline_lensweight_IA.append(UnivariateSpline(self.los_integration_chi,
                                                              aux_W_iA, k=1, s=0, ext=0))
        if self.add_to_matter:
            for tomo in range(self.number_radial_weights_mm):
                self.spline_lensweight.append(self.aux_spline_lensweight[tomo])


    def __set_magweight_splines(self,
                                covELLspacesettings):
        r"""
        Calculates and splines the magnification kernel of each clustering
        bin for the line-of-sight integration,
        q_mu^i(chi) = 3/2 Om (H0/c)^2 chi/a(chi)
                      int_chi^chimax dchi' p_i(chi') (5 s_i(chi') - 2) (chi' - chi)/chi',
        with p_i the clustering redshift distribution without the redshift
        dependent bias and s_i the slope of the number counts. The
        magnification contribution to the galaxy overdensity is then the
        convergence-like field with kernel q_mu^i.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        """
        self.spline_magweight = []
        prefactor = 3/2 / self.cosmology.hubble_distance.value**2 \
            / self.cosmology.h**2 * self.cosmology.Om0 \
            * self.los_integration_chi \
            / self.cosmology.scale_factor(
                self.spline_z_of_chi(self.los_integration_chi))
        for tomo in range(self.n_tomo_clust):
            if tomo >= len(self.spline_zclust_unbiased):
                # additional arbitrary radial weights are not magnified
                self.spline_magweight.append(UnivariateSpline(
                    self.los_integration_chi, np.zeros_like(self.los_integration_chi),
                    k=1, s=0, ext=1))
                continue
            p_chi = self.spline_zclust_unbiased[tomo]
            s_chi = self.spline_mag_s[tomo]
            norm = simpson(p_chi(self.los_integration_chi), x=self.los_integration_chi)
            aux_integral = np.zeros_like(self.los_integration_chi)
            for zetidx in range(covELLspacesettings['integration_steps']):
                aux_chi = self.los_integration_chi[zetidx]
                chi = np.geomspace(
                    aux_chi, self.chimax, covELLspacesettings['integration_steps'])
                aux_integral[zetidx] = simpson(
                    p_chi(chi) * (5*s_chi(chi) - 2) * (chi - aux_chi)/chi,
                    x=chi)/norm
            self.spline_magweight.append(UnivariateSpline(
                self.los_integration_chi, aux_integral*prefactor, k=1, s=0, ext=0))

    def __magnification_Cells(self,
                              spline_Pgm,
                              spline_Pmm,
                              covELLspacesettings):
        r"""
        Magnification contributions to the Limber angular power spectra,
        on the full line-of-sight range (the lensing kernel of the
        magnification is not compact),
        dC_gg^ij = int dchi/chi^2 [W_i q_j P_gm + q_i W_j P_gm + q_i q_j P_mm],
        dC_gm^ij = sqrt(fijl) int dchi/chi^2 q_i L_j P_mm,
        with W the clustering kernels, q the magnification kernels and L the
        lensing efficiencies.

        Parameters
        ----------
        spline_Pgm : list
            Splines of log10 P_gm for each galaxy sample.
        spline_Pmm : RegularGridInterpolator
            Spline of log10 P_mm.
        covELLspacesettings : dictionary
            Specifies the number of line-of-sight integration steps.

        Returns
        ----------
        dCell_gg : array of shape (n_ell, sample_dim, sample_dim, n_tomo_clust, n_tomo_clust)
        dCell_gm : array of shape (n_ell, sample_dim, n_tomo_clust, n_tomo_lens), or
            None if no lensing kernels exist.
        """
        n_ell = len(self.ellrange)
        n_t = self.n_tomo_clust

        def power_grid(chi):
            chi_grid, ell_grid = np.meshgrid(chi, self.ellrange, indexing='ij')
            return np.stack([chi_grid.ravel(),
                             np.log10((ell_grid.ravel() + 0.5) / chi_grid.ravel())]).T

        # magnification kernels are not compact: matter-matter terms on the
        # full range
        chi = self.los_integration_chi_full
        x_values = power_grid(chi)
        P_mm = 10**spline_Pmm(x_values).reshape((len(chi), n_ell))
        q = np.array([self.spline_magweight[t](chi) for t in range(n_t)])
        auto = np.zeros((n_t, n_t, n_ell))
        for tomo_i in range(n_t):
            for tomo_j in range(tomo_i, n_t):
                auto[tomo_i, tomo_j] = simpson(
                    P_mm*(q[tomo_i]*q[tomo_j]/chi**2)[:, None], x=chi, axis=0)
                auto[tomo_j, tomo_i] = auto[tomo_i, tomo_j]
        # galaxy-magnification terms W_i q_j P_gm: the support of W_i is
        # compact, integrate over the range of clustering bin i
        cross = np.zeros((self.sample_dim, n_t, n_t, n_ell))
        for tomo_i in range(n_t):
            if not self.chi_max_clust[tomo_i] > self.chi_min_clust[tomo_i]:
                continue
            chi_i = self.__get_updated_los_integration_chi(
                self.chi_min_clust[tomo_i], self.chi_max_clust[tomo_i], covELLspacesettings)
            x_i = power_grid(chi_i)
            W_i = self.spline_zclust[tomo_i](chi_i)
            for i_sample in range(self.sample_dim):
                P_gm = 10**spline_Pgm[i_sample](x_i).reshape((len(chi_i), n_ell))
                for tomo_j in range(n_t):
                    cross[i_sample, tomo_i, tomo_j] = simpson(
                        P_gm*(W_i*self.spline_magweight[tomo_j](chi_i)/chi_i**2)[:, None],
                        x=chi_i, axis=0)
        dCell_gg = np.zeros((n_ell, self.sample_dim, self.sample_dim, n_t, n_t))
        for i_sample in range(self.sample_dim):
            for j_sample in range(self.sample_dim):
                dCell_gg[:, i_sample, j_sample] = np.moveaxis(
                    cross[i_sample] + np.swapaxes(cross[j_sample], 0, 1) + auto, -1, 0)
        dCell_gm = None
        if self.n_tomo_lens > 0 and hasattr(self, 'spline_lensweight') \
                and len(self.spline_lensweight) > 0:
            dCell_gm = np.zeros((n_ell, self.sample_dim, n_t, self.n_tomo_lens))
            for tomo_i in range(n_t):
                for tomo_j in range(self.n_tomo_lens):
                    dCell_gm[:, :, tomo_i, tomo_j] = np.sqrt(self.fijl)[:, None] * simpson(
                        P_mm*(q[tomo_i]*self.spline_lensweight[tomo_j](chi)/chi**2)[:, None],
                        x=chi, axis=0)[:, None]
        return dCell_gg, dCell_gm

    def __check_for_tabulated_Cells(self,
                                    Cxy_tab):
        r"""
        Checks wether previously caclulated Cells are stored in read in files
        as provided in the input

        Parameters
        ----------
        Cxy_tab : dictionary
            default : None
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer, optional).

        Returns
        ----------
        [Cgg, Cgm, Cmm]: list of arrays
            Contain the read in files for the Cells
            as previously specified. If none are specified, returns 'None'
        [gg_tab_bool, gm_tab_bool, mm_tab_bool] : boolean
            Flags whether the input files have been specified and read

        """
        if not self.checked_input_cells:
            ellrange_clustering = None
            ellrange_lensing = None
            if Cxy_tab['ell'] is None:
                self.do_not_update_ell = False 
                return None, [False, False, False]
            self.do_not_update_ell = True
            if abs(self.ellrange[0] - Cxy_tab['ell'][0]) > 1e-3 or \
            abs(self.ellrange[-1] - Cxy_tab['ell'][-1]) > 1e-3 or \
            len(self.ellrange) != len(Cxy_tab['ell']):
                print("FileInputWarning: The tabulated angular modes (ells) from " +
                    "the angular power spectra file(s) do not match the ones " +
                    "currently in use, the ells will now be overwritten to " +
                    "the tabulated ones.")
              
            self.ellrange = Cxy_tab['ell']
            gg_tab_bool, gm_tab_bool, mm_tab_bool = False, False, False
            if (self.gg or self.gm) and Cxy_tab['gg'] is not None:
                gg_tab_bool = True
                ellrange_clustering = Cxy_tab['ell_clust']
            if (self.mm or self.gm) and Cxy_tab['mm'] is not None:
                mm_tab_bool = True
                ellrange_lensing = Cxy_tab['ell_lens']
            if (self.gm or (self.gg and self.mm and self.cross_terms)) and \
            Cxy_tab['gm'] is not None:
                ellrange_clustering = Cxy_tab['ell_clust']
                gm_tab_bool = True
                if not gg_tab_bool or not mm_tab_bool:
                    print("FileInputWarning: To calculate the cross " +
                        "galaxy-matter covariance the three angular power " +
                        "spectra 'galaxy-galaxy', 'galaxy-kappa' and " +
                        "'kappa-kappa' are needed. The missing C_ells will be " +
                        "calculated which may cause a biased result.")
            Cgg, Cgm, Cmm = None, None, None
            if gg_tab_bool:
                Cgg = Cxy_tab['gg']
            if mm_tab_bool:
                Cmm = Cxy_tab['mm']
            if gm_tab_bool:
                Cgm = Cxy_tab['gm']
                if gg_tab_bool:
                    Cgg = Cxy_tab['gg']
                if mm_tab_bool:
                    Cmm = Cxy_tab['mm']
            if ellrange_clustering is not None and ellrange_lensing is not None:
                if not np.array_equal(ellrange_clustering, ellrange_lensing):
                    ellmin = ellrange_clustering[0]
                    ellmax = ellrange_clustering[-1]
                    if ellrange_lensing[0] < ellmin and ellrange_lensing[-1] > ellmax :
                        ellmin = ellrange_lensing[0]
                        ellmax = ellrange_lensing[-1]
                        if Cgg is not None:
                            Cgg_aux = np.zeros((len(ellrange_lensing), len(Cgg[0,:,0,0,0]), len(Cgg[0,0, :,0,0]), len(Cgg[0,0, 0, :,0]), len(Cgg[0,0,0,0,:])))
                            for i_sample in range(len(Cgg[0,:,0,0,0])):
                                for j_sample in range(len(Cgg[0,0, :, 0,0])):
                                    for i_tomo in range(len(Cgg[0,0,0,:,0])):
                                        for j_tomo in range(len(Cgg[0,0,0,0,:])):
                                            Cggspline = UnivariateSpline(ellrange_clustering, Cgg[:,i_sample, j_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                            Cgg_aux[:,i_sample, j_sample, i_tomo,j_tomo] = Cggspline(ellrange_lensing)
                            Cgg = Cgg_aux
                        if Cgm is not None:
                            Cgm_aux = np.zeros((len(ellrange_lensing), len(Cgm[0, :,0,0]), len(Cgm[0, 0, :,0]), len(Cgm[0,0,0,:])))
                            for i_sample in range(len(Cgm[0,:,0,0])):
                                for i_tomo in range(len(Cgm[0,0,:,0])):
                                    for j_tomo in range(len(Cgm[0,0,0,:])):
                                            Cgmspline = UnivariateSpline(ellrange_clustering, Cgm[:,i_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                            Cgm_aux[:,i_sample, i_tomo,j_tomo] = Cgmspline(ellrange_lensing)
                            Cgm = Cgm_aux
                        self.ellrange = ellrange_lensing
                    else:
                        Cmm_aux = np.zeros((len(ellrange_clustering), len(Cmm[0, :,0,0]), len(Cmm[0, 0, :,0]), len(Cmm[0,0,0,:])))
                        for i_sample in range(len(Cmm[0,:,0,0])):
                            for i_tomo in range(len(Cmm[0,0,:,0])):
                                for j_tomo in range(len(Cmm[0,0,0,:])):
                                        Cmmspline = UnivariateSpline(ellrange_lensing, Cmm[:,i_sample, i_tomo, j_tomo], k = 3, s = 0, ext = 0)
                                        Cmm_aux[:,i_sample, i_tomo,j_tomo] = Cmmspline(ellrange_clustering)
                        Cmm = Cmm_aux
                        self.ellrange = ellrange_clustering
                else:
                    self.ellrange = ellrange_lensing
            elif ellrange_clustering is None:
                self.ellrange = ellrange_lensing
            else:
                self.ellrange = ellrange_clustering
            self.checked_input_cells = True

            
            return [Cgg, Cgm, Cmm], [gg_tab_bool, gm_tab_bool, mm_tab_bool]
        else:
            self.checked_input_cells = True
        
    def __check_for_tabulated_Tells(self,
                                    Tuvxy_tab):
        r"""
        Checks wether previously caclulated trispectra are stored in read in files
        as provided in the input

        Parameters
        ----------
        Tuvxy_tab : dictionary
            default : None
            Look-up table for the T_ell projected tri-spectra (gggg, gggm, ggmm, gmgm , mmgm, mmmm, optional).

        Returns
        ----------
        [Tgggg, Tgggm, Tggmm, Tgmgm, Tmmgm, Tmmmm]: list of arrays
            Contain the read in files for the Cells
            as previously specified. If none are specified, returns 'None'
        """
        if Tuvxy_tab['ell'] is None:
            return None
        if abs(self.ellrange[0] - Tuvxy_tab['ell'][0]) > 1e-3 or \
           abs(self.ellrange[-1] - Tuvxy_tab['ell'][-1]) > 1e-3 or \
           len(self.ellrange) != len(Tuvxy_tab['ell']):
            if np.any(self.tab_bools):
                raise Exception("InputError: The tabulated angular modes (ells) from the trispectrum files does not match the ones provided in the tabulated C_ells")
            else:
                print("FileInputWarning: The tabulated angular modes (ells) from " +
                  "the angular power spectra file(s) do not match the ones " +
                  "currently in use, the ells will now be overwritten to " +
                  "the tabulated ones.")


    def __update_los_integration_chi(self, chi_min, chi_max, covELLspacesettings):
        r"""
        Function to update the line-of-sight integration range.

        Parameters
        ----------
        chimin, chimax : float
            minimum and maximum comoving distance of the integration range
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.

        Returns:
        ------------
        Updates the private variable 'self.los_integration_chi'

        """
        self.los_integration_chi = np.geomspace(
            chi_min, chi_max, covELLspacesettings['integration_steps'])

    def __get_updated_los_integration_chi(self, chi_min, chi_max, covELLspacesettings):
        r"""
        Returns an updated version of the line-of-sight integration range.

        Parameters
        ----------
        chimin, chimax : float
            minimum and maximum comoving distance of the integration range
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.

        Returns:
        ------------
        self.los_integration_chi : array
            Integration range for line-of-sight integration

        """
        return np.geomspace(chi_min, chi_max, covELLspacesettings['integration_steps'])

    def get_Cells(self,
                  obs_dict,
                  output_dict,
                  bias_dict,
                  iA_dict,
                  hod_dict,
                  prec, read_in_tables):
        r"""
        Calculates the angular power spectra either with the Limber
        or full non-Limber projection.

        Parameters
        ----------
        obs_dict : dictionary
        with the following keys (To be passed from the read_input method
        of the Input class.)
        'observables' : dictionary
            Specifies which observables (cosmic shear, galaxy-galaxy
            lensing and/or clustering) should be calculated. Also,
            indicates whether cross-terms are evaluated.
        'ELLspace' : dictionary
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
        output_dict : dictionary
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        iA_dict: dictionary
            Specifies all the information about the intrinsic alignment model.
            To be passed from the read_input method in the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
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
            'survey_area_clust' : dictionary
                Contains information about the survey footprint that is read
                from the fits file of the survey (processed by healpy).
                Possible keys: 'area', 'ell', 'a_lm'
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

        Returns:
        ------------
        None :  sets private variables 'self.Cell_gg', 'self.Cell_gm' and 'self.Cell_mm'
            Used later to calculate the Gaussian covariance matrix. See class documentation.

        """
        self.Cell_mmB = None
        if self.cov_dict['gauss']:
            self.Cell_gg, self.Cell_gm, self.Cell_mm = \
                self.calc_Cells_Limber(obs_dict['ELLspace'],
                                    bias_dict,
                                    iA_dict,
                                    hod_dict, prec,
                                    read_in_tables['Cxy'])
            # cosmic shear has a single sample axis (also for tabulated input)
            if isinstance(self.Cell_mm, np.ndarray) and self.Cell_mm.ndim == 4:
                self.Cell_mm = np.ascontiguousarray(self.Cell_mm[:, :1])
            self.Cell_mmB = self._intrinsic_Bmode_Cell(read_in_tables['Cxy'])
            if not obs_dict['ELLspace']['limber']:
                self.Cell_gg_limber = np.copy(self.Cell_gg)
                self.calc_Cells_nonLimber(obs_dict['ELLspace'],
                                        read_in_tables['Cxy'])
            if output_dict['Cell']:
                out = Output(output_dict)
                Cells = [self.Cell_gg, self.Cell_gm, self.Cell_mm, self.Cell_mmB]
                out.write_Cells(self.ellrange, self.n_tomo_clust, self.n_tomo_lens,
                                Cells)
                
    def _intrinsic_Bmode_Cell(self,
                              Cxy_tab):
        """
        The intrinsic cosmic shear B-mode angular power spectrum C_BB from
        the tabulated input (CmmB_file) on self.ellrange. The code does not
        predict B-modes itself, they only enter if they are passed. They
        enter the Gaussian covariance of all cosmic shear estimators (and
        their signals) but not the non-Gaussian and super-sample terms, for
        which no B-mode model is available.

        Parameters
        ----------
        Cxy_tab : dictionary
            Look-up table for the C_ell projected power spectra with the
            keys 'ell_mmB' and 'mmB' for the B-modes.

        Returns
        -------
        Cell_mmB : array or None
            with shape (len(self.ellrange), 1, n_tomo_lens, n_tomo_lens),
            None without a CmmB_file. Outside of the tabulated ell range it
            is set to zero.
        """
        if Cxy_tab is None or Cxy_tab.get('mmB') is None:
            return None
        if not self.mm:
            print("InputWarning: An intrinsic B-mode angular power spectrum " +
                  "(CmmB_file) is passed, but cosmic shear is not requested. " +
                  "It is ignored.")
            return None
        if self.n_spec is not None and self.n_spec != 0:
            raise Exception("ConfigError: Intrinsic B-modes (CmmB_file) are " +
                            "not supported together with spectroscopic " +
                            "clustering bins (n_spec > 0).")
        ell_tab = np.asarray(Cxy_tab['ell_mmB'], dtype=float)
        CmmB_tab = np.asarray(Cxy_tab['mmB'], dtype=float)
        if len(ell_tab) == len(self.ellrange) and np.allclose(ell_tab, self.ellrange, rtol=1e-10, atol=0):
            Cell_mmB = np.copy(CmmB_tab)
        else:
            if self.ellrange[0] < ell_tab[0]*(1 - 1e-6) or self.ellrange[-1] > ell_tab[-1]*(1 + 1e-6):
                print("FileInputWarning: The tabulated intrinsic B-mode " +
                      "angular power spectrum (CmmB_file) covers the multipoles " +
                      str(ell_tab[0]) + " to " + str(ell_tab[-1]) + ", but the " +
                      "covariance needs " + str(self.ellrange[0]) + " to " +
                      str(self.ellrange[-1]) + ". It is set to zero outside " +
                      "of the tabulated range.")
            # C_BB can be zero or negative (cross-spectra): interpolate
            # linearly in the spectrum, cubically in ln(ell)
            Cell_mmB = interp1d(np.log(ell_tab), CmmB_tab, kind='cubic', axis=0,
                                bounds_error=False, fill_value=0.0)(np.log(self.ellrange))
        Cell_mmB = np.ascontiguousarray(Cell_mmB.reshape(
            (len(self.ellrange), 1, self.n_tomo_lens, self.n_tomo_lens)))
        print("Including the tabulated intrinsic B-mode angular power " +
              "spectrum in the Gaussian covariance of cosmic shear.")
        return Cell_mmB

    def calc_Cells_nonLimber(self,
                             covELLspacesettings,
                             Cxy_tab):
        r"""
        Replaces the Limber angular power spectra at low multipoles by
        their non-Limber version.  The correction of each block (gg, gm, mm) 
        is added for increasing ell until it is below NONLIMBER_TOL relative
        to sqrt(C_aa C_bb) for NONLIMBER_N_CONVERGED multipoles in a row 
        (at most up to NONLIMBER_ELL_MAX); tabulated C_ell are not changed.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        Cxy_tab : dictionary
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer).

        Returns
        ----------
        None :
            Adds the non-Limber correction to self.Cell_gg, self.Cell_gm and
            self.Cell_mm and sets self.ell_max_nonlimber ({'gg', 'gm', 'mm':
            the largest multipole that was corrected}).
        """
        self.ell_max_nonlimber = None
        do_gg = (self.gg or self.gm) and not self.tab_bools[0] \
            and isinstance(self.Cell_gg, np.ndarray)
        do_gm = (self.gm or (self.gg and self.mm and self.cross_terms)) \
            and not self.tab_bools[1] and isinstance(self.Cell_gm, np.ndarray)
        do_mm = (self.mm or self.gm) and not self.tab_bools[2] \
            and isinstance(self.Cell_mm, np.ndarray)
        if not (do_gg or do_gm or do_mm):
            return
        if self.do_arbitrary_radial_gg or self.do_arbitrary_radial_mm:
            print("ConfigWarning: The non-Limber angular power spectra are not available " +
                  "with arbitrary radial weights, the Limber ones are used.")
            return
        print("Calculating the non-Limber correction of the angular power spectra (C_ell's).")
        chi_0 = self.los_chi[0]

        def P_ref(k):
            return np.exp(self.power_mm_lin_spline(
                np.stack([np.full_like(k, chi_0), np.log(k)], axis=-1)))

        proj = _NonLimberProjector(P_ref)
        chi = proj.chi
        chi_eval = np.clip(chi, self.los_chi[0], self.los_chi[-1])
        k_ref = min(max(0.01, 2*self.mass_func.k[0]), 0.5*self.mass_func.k[-1])
        lnP_lin = self.power_mm_lin_spline(
            np.stack([chi_eval, np.full_like(chi, np.log(k_ref))], axis=-1))
        growth = np.exp(0.5*(lnP_lin - np.log(P_ref(np.array([k_ref]))[0])))
        in_los = (chi >= self.chimin) & (chi <= self.chimax)
        taper = (1 - np.exp(-(chi/NONLIMBER_TAPER_CHI)**4))*in_los

        def on_grid(spline, lo, hi):
            values = np.zeros_like(chi)
            inside = (chi >= lo) & (chi <= hi)
            values[inside] = spline(chi[inside])
            return values

        n_chi = len(chi)
        mag = self.magnification_active and (do_gg or do_gm)
        if do_gg or do_gm:
            # linear bias of each sample from the power spectra at k_ref
            bias = np.array([np.sqrt(10**self.spline_Pgg[i_sample*self.sample_dim + i_sample](
                np.stack([chi_eval, np.full_like(chi, np.log10(k_ref))], axis=-1)) / np.exp(lnP_lin))
                for i_sample in range(self.sample_dim)])
            W = np.array([on_grid(self.spline_zclust[tomo], self.chi_min_clust[tomo], self.chi_max_clust[tomo])
                          for tomo in range(self.n_tomo_clust)])
            # legs ordered (sample, tomographic bin)
            gal_density = (bias[:, None, :]*W[None, :, :]*growth).reshape((-1, n_chi))
            if mag:
                gal_kappa = np.array([self.spline_magweight[tomo](chi) for tomo in range(self.n_tomo_clust)]) \
                    * growth*taper
        if do_gm or do_mm:
            L_IA = np.array([on_grid(self.spline_lensweight_IA[tomo], self.chimin, self.chimax)
                             for tomo in range(self.n_tomo_lens)])*growth
            L_lens = np.array([self.spline_lensweight[tomo](chi) for tomo in range(self.n_tomo_lens)]) \
                * growth*taper - L_IA*taper
        n_t, n_s = self.n_tomo_clust, self.sample_dim
        # each block is corrected until its own correction has converged
        n_converged = {'gg': 0 if do_gg else NONLIMBER_N_CONVERGED,
                       'gm': 0 if do_gm else NONLIMBER_N_CONVERGED,
                       'mm': 0 if do_mm else NONLIMBER_N_CONVERGED}
        self.ell_max_nonlimber = {}
        for eidx, ell in enumerate(self.ellrange):
            active = [key for key, n in n_converged.items() if n < NONLIMBER_N_CONVERGED]
            if ell > NONLIMBER_ELL_MAX or not active:
                break
            do_gg, do_gm, do_mm = ('gg' in active), ('gm' in active), ('mm' in active)
            if do_gg or do_gm:
                k, I_gal = proj.transform(gal_density, ell)
                K_gal = gal_density
                if mag:
                    _, I_kappa = proj.transform(gal_kappa, ell, lensing=True)
                    I_gal = I_gal + ell*(ell + 1)*np.tile(I_kappa, (n_s, 1))
                    K_gal = K_gal + np.tile(gal_kappa, (n_s, 1))
                limber_gal = np.diag(proj.limber(K_gal, K_gal, ell))
            if do_gm or do_mm:
                sqrt_factorial = np.sqrt(max((ell + 2)*(ell + 1)*ell*(ell - 1), 0.0))
                k, I_lens = proj.transform(L_lens, ell, lensing=True)
                _, I_IA = proj.transform(L_IA, ell)
                I_shear = sqrt_factorial*(I_lens + I_IA/(ell + 0.5)**2)
                K_shear = np.sqrt(self.fijl[eidx])*(L_lens + L_IA)
                limber_shear = np.diag(proj.limber(K_shear, K_shear, ell))

            def correction(I_a, I_b, K_a, K_b, norm_a, norm_b):
                delta = proj.exact(I_a, I_b, k) - proj.limber(K_a, K_b, ell)
                norm = np.sqrt(np.abs(np.outer(norm_a, norm_b)))
                return delta, np.max(np.abs(delta)[norm > 0]/norm[norm > 0], initial=0.0)

            relative = {}
            if do_gg:
                delta, relative['gg'] = correction(I_gal, I_gal, K_gal, K_gal, limber_gal, limber_gal)
                self.Cell_gg[eidx] += delta.reshape((n_s, n_t, n_s, n_t)).transpose(0, 2, 1, 3)
            if do_gm and ell >= 2:
                delta, relative['gm'] = correction(I_gal, I_shear, K_gal, K_shear, limber_gal, limber_shear)
                self.Cell_gm[eidx] += delta.reshape((n_s, n_t, self.n_tomo_lens))
            if do_mm and ell >= 2:
                delta, relative['mm'] = correction(I_shear, I_shear, K_shear, K_shear, limber_shear, limber_shear)
                self.Cell_mm[eidx] += delta.reshape(self.Cell_mm[eidx].shape)
            for key, rel in relative.items():
                self.ell_max_nonlimber[key] = ell
                n_converged[key] = n_converged[key] + 1 if rel < NONLIMBER_TOL else 0
        print("Non-Limber correction applied up to ell = " +
              ", ".join(key + ": " + str(int(ell)) for key, ell in self.ell_max_nonlimber.items()) + ".")

    def calc_Cells_Limber(self,
                          covELLspacesettings,
                          bias_dict,
                          iA_dict,
                          hod_dict,
                          prec,
                          Cxy_tab):
        r"""
        Calculates the Limber projection of the power spectrum for all
        tracers and tomographic bin combinations over the ell range
        specified in covELLspacesettings

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be
            passedfrom the read_input method of the Input class.
        iA_dict: dictionary
            Specifies all the information about the intrinsic alignment model.
            To be passed from the read_input method in the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the
            differentgalaxy populations. To be passed from the
            read_input method of the Input class.
        prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        Cxy_tab : dictionary
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer).

        Returns
        ----------
        Cell_gg, Cell_gm, Cell_mm : list of arrays
            Angular power spectra for the different tracers. They each
            have the shape (len(self.ellrange), self.sample_dim,
            self.n_tomo_clust, self.n_tomo_clust)

        """
        
        self.los_interpolation_sampling = int(
            (self.zet_max - 0) / covELLspacesettings['delta_z'])
        self.los_interpolation_sampling = max(self.los_interpolation_sampling, 3)
        
        self.los_interpolation_sampling = min(self.los_interpolation_sampling, 250)
        aux_gg = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k),
                           self.sample_dim, self.sample_dim))
        aux_gm = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k),
                           self.sample_dim))
        aux_mm = np.zeros((self.los_interpolation_sampling,
                           len(self.mass_func.k)))
        self.power_mm_lin_z = np.zeros_like(aux_mm)
        self.los_z = np.linspace(
            0, self.zet_max, self.los_interpolation_sampling)
        if self.los_interpolation_sampling == 250:
            self.los_z = np.geomspace(self.spline_z_of_chi(self.chimin), self.zet_max, self.los_interpolation_sampling)
        self.los_chi = self.cosmology.comoving_distance(
            self.los_z).value * self.cosmology.h
        aux_ngal = np.zeros((self.los_interpolation_sampling,self.sample_dim))
        grid = self.hod.Mbins
        self.samples_overlap = any(min(grid[i, -1], grid[j, -1]) > max(grid[i, 0], grid[j, 0])
                                   for i in range(len(grid)) for j in range(len(grid)) if i != j)
        if self.samples_overlap:
            aux_ngal_overlap = np.zeros((self.los_interpolation_sampling, self.sample_dim, self.sample_dim))
        mag = self.magnification_active
        if (self.mm or self.gm or self.unbiased_clustering or mag) and prec['hm']['transfer_model'] == 'CAMB' and self.Pxy_tab['mm'] is None:
            self.camb_pars_new.set_matter_power(redshifts = self.los_z[::-1], kmax=min(self.mass_func.k[-1], CAMB_KMAX_NONLINEAR))
            self.camb_pars_new.NonLinear = model.NonLinear_pk
            if prec['powspec']['nl_model'] == "mead2020_feedback":
                self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'], HMCode_logT_AGN = prec['powspec']['HMCode_logT_AGN'])
            elif prec['powspec']['nl_model'] == "mead2015" or prec['powspec']['nl_model'] == "mead2016":
                self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'], HMCode_A_baryon = prec['powspec']['HMCode_A_baryon'], HMCode_eta_baryon = prec['powspec']['HMCode_eta_baryon'])
            else:
                self.camb_pars_new.NonLinearModel.set_params(halofit_version=prec['powspec']['nl_model'])
            results = camb.get_results(self.camb_pars_new)
            logging_disabled = logging.root.manager.disable
            logging.disable(logging.WARNING)
            try:
                _,_, aux_mm = results.get_matter_power_spectrum(minkh=self.mass_func.k[0],
                                                                maxkh=self.mass_func.k[-1],
                                                                npoints = len(self.mass_func.k))
            finally:
                logging.disable(logging_disabled)
        t0 = time.time()
        
        if self.csmf:
            aux_stellar_mass_func = np.zeros((self.los_interpolation_sampling, len(self.log10csmf_mass_bins)))
            aux_stellar_mass_func_bias = np.zeros((self.los_interpolation_sampling, len(self.log10csmf_mass_bins)))
            aux_stellar_mass_func_overlap = np.zeros((self.los_interpolation_sampling, len(self.csmf_bin_overlaps)))
            self.csmf_count_matter_bispectrum = np.zeros((len(self.mass_func.k), self.los_interpolation_sampling,len(self.log10csmf_mass_bins)))
            self.csmf_galaxy_bispectra = (self.gg or self.gm) and not self.csmf_auto_only \
                and not self.redshift_dep_bias and not self.unbiased_clustering
            if self.csmf_galaxy_bispectra:
                self.csmf_count_galaxy_galaxy_bispectrum = np.zeros((len(self.mass_func.k), self.los_interpolation_sampling, len(self.log10csmf_mass_bins), self.sample_dim))
                self.csmf_count_galaxy_matter_bispectrum = np.zeros((len(self.mass_func.k), self.los_interpolation_sampling, len(self.log10csmf_mass_bins), self.sample_dim))
            aux_response_mm = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            aux_response_gg = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            aux_response_gm = np.zeros((len(self.los_chi),
                                    len(self.mass_func.k),
                                    self.sample_dim))
            if self.redshift_dep_bias:
                # responses including the tomographic bias, as in covELL_ssc
                aux_response_gg_tomo = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust,
                                        self.n_tomo_clust))
                aux_response_gm_tomo = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust))
            
            save_mm = False
            if not self.mm:
                save_mm = True
                self.mm = True

        for zet in range(self.los_interpolation_sampling):
            self.update_mass_func(self.los_z[zet], bias_dict, hod_dict, prec)
            aux_ngal[zet, :] = self.ngal
            if self.samples_overlap:
                aux_ngal_overlap[zet] = self.nbar_overlap(hod_dict)
            if self.gg or self.gm:
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        if self.unbiased_clustering:
                            aux_gg[zet, :, i_sample, j_sample] = aux_mm[zet,:]
                        else:
                            aux_gg[zet, :, i_sample, j_sample] = self.Pgg[:, i_sample, j_sample]
                    if self.unbiased_clustering:
                        aux_gm[zet, :, i_sample] = aux_mm[zet,:]
                    elif self.gm or (self.gg and self.mm) or mag:
                        aux_gm[zet, :, i_sample] = self.Pgm[:, i_sample]
            if prec['hm']['transfer_model'] != 'CAMB' or self.Pxy_tab['mm'] is not None :
                aux_mm[zet, :] = self.Pmm[:, 0]
            self.power_mm_lin_z[zet, :] = self.mass_func.power[:]
            eta = (time.time()-t0) * \
                (self.los_interpolation_sampling/(zet+1)-1)
            if self.csmf:
                # SMF, Phi~ and count bispectra per dex, averaged over the stellar mass bins
                nodes = 10**self.csmf_quad_nodes
                aux_stellar_mass_func[zet,:] = self.csmf_quad_weights @ (self.galaxy_smf_c(nodes) + self.galaxy_smf_s(nodes))
                aux_stellar_mass_func_bias[zet,:] = self.csmf_quad_weights @ (self.galaxy_smf_bias_c(nodes) + self.galaxy_smf_bias_s(nodes))
                if self.csmf_bin_overlaps:
                    nodes_overlap = 10**self.csmf_overlap_nodes
                    aux_stellar_mass_func_overlap[zet, :] = self.csmf_overlap_weights @ (self.galaxy_smf_c(nodes_overlap) + self.galaxy_smf_s(nodes_overlap))
                if not self.csmf_auto_only:
                    self.csmf_count_matter_bispectrum[:, zet,:] = \
                        self.get_count_matter_bispectrum(bias_dict, hod_dict, prec['hm'], self.csmf_quad_nodes) @ self.csmf_quad_weights.T
                    if self.csmf_galaxy_bispectra:
                        bispec_cgg, bispec_cgm = self.get_count_galaxy_bispectra(bias_dict, hod_dict, prec['hm'], self.csmf_quad_nodes)
                        self.csmf_count_galaxy_galaxy_bispectrum[:, zet] = np.einsum('kns,mn->kms', bispec_cgg, self.csmf_quad_weights)
                        self.csmf_count_galaxy_matter_bispectrum[:, zet] = np.einsum('kns,mn->kms', bispec_cgm, self.csmf_quad_weights)
                    aux_response_gg[zet, :, :], aux_response_gm[zet, :, :], aux_response_mm[zet, :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
                    if self.redshift_dep_bias:
                        bias_at_chi = np.array([self.bias_of_zet[t](self.los_z[zet]) for t in range(self.n_tomo_clust)])
                        aux_response_gm_tomo[zet] = aux_response_mm[zet, :, :, None] - self.Pgm[:, :, None]*bias_at_chi
                        aux_response_gg_tomo[zet] = aux_response_mm[zet, :, :, None, None] \
                            - (bias_at_chi[:, None] + bias_at_chi[None, :])*np.diagonal(self.Pgg, axis1 = -2, axis2 =-1)[:, :, None, None]
            print('\rPreparations for C_ell calculation at '
                    + str(round((zet+1)/self.los_interpolation_sampling*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)
                                    ) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
        print(" ")
        if self.csmf and save_mm:
            self.mm = False
            if self.mm:
                self.spline_responsePmm = RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), np.log(aux_response_mm[:, :, 0]).T,bounds_error= False, fill_value = None)
            if self.gm:
                self.spline_responsePgm = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgm.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gm[:, :, i_sample]).T,bounds_error= False, fill_value = None))
            if self.gg:
                self.spline_responsePgg = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgg.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gg[:, :, i_sample]).T,bounds_error= False, fill_value = None))
        elif self.csmf and not self.csmf_auto_only:
            if self.mm:
                self.spline_responsePmm = RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), np.log(aux_response_mm[:, :, 0]).T,bounds_error= False, fill_value = None)
            if self.gm:
                self.spline_responsePgm = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgm.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gm[:, :, i_sample]).T,bounds_error= False, fill_value = None))
            if self.gg:
                self.spline_responsePgg = []
                for i_sample in range(self.sample_dim):
                    self.spline_responsePgg.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gg[:, :, i_sample]).T,bounds_error= False, fill_value = None))
        if self.csmf and not self.csmf_auto_only and self.redshift_dep_bias:
            self.spline_responsePgm_tomo = []
            self.spline_responsePgg_tomo = []
            for i_sample in range(self.sample_dim):
                for i_tomo in range(self.n_tomo_clust):
                    self.spline_responsePgm_tomo.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gm_tomo[:, :, i_sample, i_tomo]).T,bounds_error= False, fill_value = None))
                    for j_tomo in range(self.n_tomo_clust):
                        self.spline_responsePgg_tomo.append(RegularGridInterpolator((np.log(self.mass_func.k),self.los_chi), (aux_response_gg_tomo[:, :, i_sample, i_tomo, j_tomo]).T,bounds_error= False, fill_value = None))

        self.power_mm_lin_spline = RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),np.log(self.power_mm_lin_z),bounds_error= False, fill_value = None)
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        if self.csmf:
            self.csmf_at_tomo_and_mass = np.zeros((len(self.log10csmf_mass_bins),self.n_tomo_csmf))
            self.phi_tilde_spline = []
            for i_smf_m_bins in range(len(self.log10csmf_mass_bins)):
                aux_csmf_spline = UnivariateSpline(self.los_chi, aux_stellar_mass_func[:, i_smf_m_bins], k=2, s=0, ext=0)
                self.phi_tilde_spline.append(UnivariateSpline(self.los_chi, aux_stellar_mass_func_bias[:, i_smf_m_bins], k=2, s=0, ext=0))
                for i_tomo in range(self.n_tomo_csmf):
                    los_chi_csmf = self._csmf_los_chi(i_smf_m_bins, i_tomo, 0, np.inf, covELLspacesettings)
                    self.csmf_at_tomo_and_mass[i_smf_m_bins, i_tomo] = simpson(
                        self._csmf_window(i_smf_m_bins, i_tomo, los_chi_csmf) * aux_csmf_spline(los_chi_csmf), x=los_chi_csmf)
            self.csmf_overlap_mean = np.zeros((len(self.csmf_bin_overlaps), self.n_tomo_csmf))
            for i_overlap, (i_mass, j_mass) in enumerate(self.csmf_bin_overlaps):
                aux_csmf_spline = UnivariateSpline(self.los_chi, aux_stellar_mass_func_overlap[:, i_overlap], k=2, s=0, ext=0)
                for i_tomo in range(self.n_tomo_csmf):
                    chi_low = min(self.csmf_window_chi_range[i_mass, i_tomo, 0], self.csmf_window_chi_range[j_mass, i_tomo, 0])
                    chi_high = max(self.csmf_window_chi_range[i_mass, i_tomo, 1], self.csmf_window_chi_range[j_mass, i_tomo, 1])
                    los_chi_csmf = self._csmf_los_chi(i_mass, i_tomo, 0, np.inf, covELLspacesettings) if chi_low >= chi_high else \
                        np.geomspace(chi_low, chi_high, covELLspacesettings['integration_steps'])
                    window = 0.5*(self._csmf_window(i_mass, i_tomo, los_chi_csmf) + self._csmf_window(j_mass, i_tomo, los_chi_csmf))
                    self.csmf_overlap_mean[i_overlap, i_tomo] = simpson(window*aux_csmf_spline(los_chi_csmf), x=los_chi_csmf)

        # angular number density (per steradian) of each sample in each tomographic bin from the HoD, N = int dchi chi^2 nbar(chi) S(chi), with the selection S fixed by the normalised n(chi):
        # N p(chi) = chi^2 nbar(chi) S(chi). Assuming S = 1 wherever the sample is selected gives N = 1/int dchi p^2/(chi^2 nbar), exact for
        # a top-hat selection and an upper bound otherwise.
        self.Ngal = np.zeros((self.sample_dim, self.n_tomo_clust))
        for i_sample in range(self.sample_dim):
            spline_nbar = UnivariateSpline(self.los_chi, aux_ngal[:, i_sample], k=2, s=0, ext=0)
            nbar_chi2 = spline_nbar(self.los_integration_chi) * self.los_integration_chi**2
            for tomo_i in range(self.n_tomo_clust):
                prob = self.spline_zclust[tomo_i](self.los_integration_chi)
                selected = prob > 0
                self.Ngal[i_sample,tomo_i] = 1.0/simpson(prob[selected]**2/nbar_chi2[selected], x = self.los_integration_chi[selected])
        self.sample_overlap_fraction = np.eye(self.sample_dim)[:, :, None]*np.ones(self.n_tomo_clust)[None, None, :]
        if self.samples_overlap:
            aux_ngal_overlap = 0.5*(aux_ngal_overlap + np.swapaxes(aux_ngal_overlap, 1, 2))
            nbar_chi = np.array([[UnivariateSpline(self.los_chi, aux_ngal_overlap[:, i_sample, j_sample], k=2, s=0, ext=0)(self.los_integration_chi)
                                  for j_sample in range(self.sample_dim)] for i_sample in range(self.sample_dim)])
            for tomo_i in range(self.n_tomo_clust):
                prob = self.spline_zclust[tomo_i](self.los_integration_chi)
                mean_nbar = simpson(prob[None, None, :]*np.maximum(nbar_chi, 0), x = self.los_integration_chi, axis = -1)
                self.sample_overlap_fraction[:, :, tomo_i] = mean_nbar/np.minimum(np.diag(mean_nbar)[:, None], np.diag(mean_nbar)[None, :])
        if self.gg or self.gm:
            self.spline_Pgg, spline_Pgm = [], []
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    self.spline_Pgg.append(RegularGridInterpolator((self.los_chi, np.log10(self.mass_func.k)),
                                            np.log10(aux_gg[:, :, i_sample, j_sample]),bounds_error= False, fill_value = None))
                if self.gm or self.mm or mag:
                    spline_Pgm.append(RegularGridInterpolator((self.los_chi,np.log10(self.mass_func.k)), np.log10(aux_gm[:, :, i_sample]),bounds_error= False, fill_value = None))
            if self.gm or self.mm or mag:
                self.spline_Pgg = self.spline_Pgg
                self.spline_Pgm = spline_Pgm
    
        if ((self.mm or self.gm) and not self.tab_bools[2]) or mag:
            spline_Pmm = RegularGridInterpolator((self.los_chi,np.log10(self.mass_func.k)), np.log10(aux_mm[:, :]),bounds_error= False, fill_value = None)
            self.spline_Pmm = spline_Pmm

        if not self.do_arbitrary_radial_mm or self.add_to_matter:
            self.__set_lensweight_splines(covELLspacesettings, iA_dict)
        self.Cell_gg_mag, self.Cell_gm_mag = None, None
        if mag:
            self.__set_magweight_splines(covELLspacesettings)
            if self.Cells is not None and (self.tab_bools[0] or self.tab_bools[1]):
                print("ConfigWarning: Tabulated C_ell (gg, gm) are used as given, the magnification " +
                      "bias is NOT added to them.")
        if self.Cells is not None:
            if ((self.gg and self.tab_bools[0]) or not self.gg) and ((self.mm and self.tab_bools[2]) or not self.mm) and ((self.gm and np.all(self.tab_bools)) or not self.gm):
                return self.Cells[0], self.Cells[1], self.Cells[2]

        fil = np.sqrt((self.ellrange + 2)*(self.ellrange + 1)*(self.ellrange)*(self.ellrange -1))
        self.fijl = fil**2*(self.ellrange+0.5)**(-4)
        print("Calculating angular power spectra (C_ell's).")
        if (self.gg or self.gm) and not self.tab_bools[0]:
            Cell_gg = np.zeros((len(self.ellrange), self.sample_dim, self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_clust))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for tomo_i in range(self.n_tomo_clust):
                        for tomo_j in range(tomo_i, self.n_tomo_clust):
                            chi_low = max(
                                self.chi_min_clust[tomo_i], self.chi_min_clust[tomo_j])
                            chi_high = min(
                                self.chi_max_clust[tomo_i], self.chi_max_clust[tomo_j])
                            if chi_low >= chi_high:
                                continue
                            self.__update_los_integration_chi(
                                chi_low, chi_high, covELLspacesettings)
                            chi_grid, ell_grid = np.meshgrid(self.los_integration_chi, self.ellrange, indexing='ij')
                            x_values = np.stack([chi_grid.ravel(), np.log10((ell_grid.ravel() + 0.5) / chi_grid.ravel())])
                            integrand = self.spline_Pgg[i_sample*self.sample_dim + j_sample](x_values.T).reshape((len(self.los_integration_chi),len(self.ellrange)))
                            Cell_gg[:, i_sample, j_sample, tomo_i, tomo_j] = simpson(10**integrand*(self.spline_zclust[tomo_i](self.los_integration_chi)*self.spline_zclust[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x = self.los_integration_chi, axis = 0)
                            Cell_gg[:, i_sample, j_sample, tomo_j, tomo_i] = \
                                Cell_gg[:, i_sample, j_sample,  tomo_i, tomo_j]
                            self.__update_los_integration_chi(
                                self.chimin, self.chimax, covELLspacesettings)
        elif self.tab_bools[0]:
            Cell_gg = self.Cells[0]
        else:
            Cell_gg = 0

        if (self.gm or (self.gg and self.mm and self.cross_terms)) and \
           not self.tab_bools[1]:
            Cell_gm = np.zeros((len(self.ellrange), self.sample_dim,
                                self.n_tomo_clust, self.n_tomo_lens))
            for i_sample in range(self.sample_dim):
                for tomo_i in range(self.n_tomo_clust):
                    for tomo_j in range(self.n_tomo_lens):
                        chi_low = self.chi_min_clust[tomo_i]
                        chi_high = self.chi_max_clust[tomo_i]
                        self.__update_los_integration_chi(
                            chi_low, chi_high, covELLspacesettings)

                        chi_grid, ell_grid = np.meshgrid(self.los_integration_chi, self.ellrange, indexing='ij')
                        x_values = np.stack([chi_grid.ravel(), np.log10((ell_grid.ravel() + 0.5) / chi_grid.ravel())])
                        integrand = spline_Pgm[i_sample](x_values.T).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        Cell_gm[:, i_sample, tomo_i, tomo_j] = np.sqrt(self.fijl)*simpson(10**integrand*(self.spline_zclust[tomo_i](self.los_integration_chi)*self.spline_lensweight[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x = (self.los_integration_chi), axis = 0)
                        self.__update_los_integration_chi(
                            self.chimin, self.chimax, covELLspacesettings)
        elif self.tab_bools[1]:
            Cell_gm = self.Cells[1]
        else:
            Cell_gm = 0
        if (self.mm or self.gm) and not self.tab_bools[2]:
            Cell_mm = np.zeros((len(self.ellrange),
                                self.n_tomo_lens, self.n_tomo_lens))
            chi_grid, ell_grid = np.meshgrid(self.los_integration_chi, self.ellrange, indexing='ij')
            x_values = np.stack([chi_grid.ravel(), np.log10((ell_grid.ravel() + 0.5) / chi_grid.ravel())])
            xi = x_values.T
            for tomo_i in range(self.n_tomo_lens):
                for tomo_j in range(tomo_i, self.n_tomo_lens):
                    integrand = spline_Pmm(xi).reshape((len(self.los_integration_chi),len(self.ellrange)))
                    Cell_mm[:, tomo_i, tomo_j] = self.fijl*simpson(10**integrand*(self.spline_lensweight[tomo_i](self.los_integration_chi)*self.spline_lensweight[tomo_j](self.los_integration_chi)/ self.los_integration_chi**2)[:,None], x =(self.los_integration_chi), axis = 0)
                    Cell_mm[:, tomo_j, tomo_i] = Cell_mm[:, tomo_i, tomo_j]
            # the shear spectrum does not depend on the galaxy sample: one
            # sample axis of length one
            Cell_mm = Cell_mm[:, None, :, :]
        elif self.tab_bools[2]:
            Cell_mm = self.Cells[2]
        else:
            Cell_mm = 0

        if mag:
            print("Adding the magnification bias to the Limber C_ell (gg, gm).")
            dCell_gg, dCell_gm = self.__magnification_Cells(spline_Pgm, spline_Pmm, covELLspacesettings)
            if (self.gg or self.gm) and not self.tab_bools[0]:
                Cell_gg = Cell_gg + dCell_gg
                self.Cell_gg_mag = dCell_gg
            if dCell_gm is not None and not self.tab_bools[1] and \
                    (self.gm or (self.gg and self.mm and self.cross_terms)):
                Cell_gm = Cell_gm + dCell_gm
                self.Cell_gm_mag = dCell_gm

        return Cell_gg, Cell_gm, Cell_mm

    def calc_covELL(self,
                    obs_dict,
                    output_dict,
                    bias_dict,
                    hod_dict,
                    survey_params_dict,
                    prec,
                    read_in_tables):
        r"""
        Calculates the full covariance between the specified
        observables in ell-space config file.

        Parameters
        ----------
        obs_dict : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
            'observables' : dictionary
                Specifies which observables (cosmic shear, galaxy-galaxy
                lensing and/or clustering) should be calculated. Also,
                indicates whether cross-terms are evaluated.
            'ELLspace' : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.
        output_dict : dictionary
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
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
            'survey_area_clust' : dictionary
                Contains information about the survey footprint that is read
                from the fits file of the survey (processed by healpy).
                Possible keys: 'area', 'ell', 'a_lm'
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


        Returns
        -------
        gauss, nongauss, ssc : list of arrays
            each with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        print("Calculating covariance for angular power spectra (C_ell's).")
        if not self.cov_dict['split_gauss']:
            if self.ellrange_photo is not None:
                gaussELLgggg_ssss_new, gaussELLgggg_sssp_new, gaussELLgggg_sspp_new, \
                gaussELLgggg_spsp_new, gaussELLgggg_ppsp_new, gaussELLgggg_pppp_new, \
                gaussELLgggm_sssm_new, gaussELLgggm_sspm_new, gaussELLgggm_spsm_new, \
                gaussELLgggm_sppm_new, gaussELLgggm_ppsm_new, gaussELLgggm_pppm_new, \
                gaussELLggmm_ssmm_new, gaussELLggmm_spmm_new, gaussELLggmm_ppmm_new, \
                gaussELLgmgm_smsm_new, gaussELLgmgm_smpm_new, gaussELLgmgm_pmsm_new, \
                gaussELLgmgm_pmpm_new, gaussELLmmgm_mmsm_new, gaussELLmmgm_mmpm_new, \
                gaussELLmmmm_mmmm_new, gaussELLgggg_ssss_new_sn, gaussELLgggg_sssp_new_sn, \
                gaussELLgggg_sspp_new_sn, gaussELLgggg_spsp_new_sn, gaussELLgggg_ppsp_new_sn, \
                gaussELLgggg_pppp_new_sn, gaussELLgmgm_smsm_new_sn, gaussELLgmgm_smpm_new_sn, \
                gaussELLgmgm_pmsm_new_sn, gaussELLgmgm_pmpm_new_sn, gaussELLmmmm_mmmm_new_sn = \
                self.covELL_gaussian(obs_dict['ELLspace'],
                                     survey_params_dict, True)
                gauss = [gaussELLgggg_ssss_new + gaussELLgggg_ssss_new_sn,
                         gaussELLgggg_sssp_new + gaussELLgggg_sssp_new_sn,
                         gaussELLgggg_sspp_new + gaussELLgggg_sspp_new_sn,
                         gaussELLgggg_spsp_new + gaussELLgggg_spsp_new_sn,
                         gaussELLgggg_ppsp_new + gaussELLgggg_ppsp_new_sn,
                         gaussELLgggg_pppp_new + gaussELLgggg_pppp_new_sn,
                         gaussELLgggm_sssm_new, gaussELLgggm_sspm_new, gaussELLgggm_spsm_new,
                         gaussELLgggm_sppm_new, gaussELLgggm_ppsm_new, gaussELLgggm_pppm_new,
                         gaussELLggmm_ssmm_new, gaussELLggmm_spmm_new, gaussELLggmm_ppmm_new, 
                         gaussELLgmgm_smsm_new + gaussELLgmgm_smsm_new_sn,
                         gaussELLgmgm_smpm_new + gaussELLgmgm_smpm_new_sn,
                         gaussELLgmgm_pmsm_new + gaussELLgmgm_pmsm_new_sn,
                         gaussELLgmgm_pmpm_new + gaussELLgmgm_pmpm_new_sn,
                         gaussELLmmgm_mmsm_new, gaussELLmmgm_mmpm_new,
                         gaussELLmmmm_mmmm_new + gaussELLmmmm_mmmm_new_sn]
            elif self.csmf:
                gaussgggg, gaussgggm, gaussggmm, \
                gaussgmgm, gaussmmgm, gaussmmmm, \
                gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn, \
                csmf_auto, csmf_gg, csmf_gm, csmf_mm = \
                self.covELL_gaussian(obs_dict['ELLspace'],
                                    survey_params_dict, True)
                gauss = [gaussgggg + gaussgggg_sn, gaussgggm,
                        gaussggmm, gaussgmgm + gaussgmgm_sn,
                        gaussmmgm, gaussmmmm + gaussmmmm_sn,
                        csmf_auto, csmf_gg, csmf_gm, csmf_mm]
            else:
                gaussgggg, gaussgggm, gaussggmm, \
                gaussgmgm, gaussmmgm, gaussmmmm, \
                gaussgggg_sn, gaussgmgm_sn, gaussmmmm_sn = \
                self.covELL_gaussian(obs_dict['ELLspace'],
                                    survey_params_dict, True)
                gauss = [gaussgggg + gaussgggg_sn, gaussgggm,
                        gaussggmm, gaussgmgm + gaussgmgm_sn,
                        gaussmmgm, gaussmmmm + gaussmmmm_sn]
        else:
            gauss = self.covELL_gaussian(obs_dict['ELLspace'],
                                         survey_params_dict, True)
        if self.ellrange_photo is None:
            gauss = self.__add_Cell_mult_shear_bias(list(gauss))

        nongauss = self.covELL_non_gaussian(obs_dict['ELLspace'],
                                            output_dict,
                                            bias_dict,
                                            hod_dict,
                                            prec,
                                            read_in_tables['tri'])
        nongauss = list(nongauss)

                    
        if self.cov_dict['nongauss'] and self.ellrange_photo is None:
            nongauss_new = []
            subtract = 0
            if self.gg:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gggg contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_clust'],
                                                              nongauss[0 - subtract],
                                                              True,
                                                              True,
                                                              True))
                    del nongauss[0-subtract]
                    subtract += 1
                else:
                    nongauss[0][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[0][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_clust']/self.deg2torad2)
                    nongauss_new.append(nongauss[0][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gg and self.gm and self.cross_terms:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gggm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[1-subtract],
                                                              True,
                                                              True,
                                                              False))
                    del nongauss[1-subtract]
                    subtract += 1
                else:
                    nongauss[1][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[1][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])/self.deg2torad2)
                    nongauss_new.append(nongauss[1][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gg and self.mm and self.cross_terms:
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian ggmm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_lens'],
                                                              nongauss[2-subtract],
                                                              True,
                                                              True,
                                                              True))
                    del nongauss[2-subtract]
                    subtract += 1
                else:
                    nongauss[2][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[2][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens'])/self.deg2torad2)
                    nongauss_new.append(nongauss[2][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.gm:
                if self.ellrange_clustering_ul is not None:
                    print("Binning non-Gaussian gmgm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_ggl'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[3-subtract],
                                                              True,
                                                              False,
                                                              False))
                    del nongauss[3-subtract]
                    subtract += 1
                else:
                    nongauss[3][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[3][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_ggl']/self.deg2torad2)
                    nongauss_new.append(nongauss[3][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.mm and self.gm and self.cross_terms:           
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian mmgm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              nongauss[4-subtract],
                                                              True,
                                                              True,
                                                              False))
                    del nongauss[4-subtract]
                    subtract += 1
                else:
                    nongauss[4][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[4][:, :, :, :, :, :, :, :])/(max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl'])/self.deg2torad2)
                    nongauss_new.append(nongauss[4][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            if self.mm:
                if self.ellrange_lensing_ul is not None:
                    print("Binning non-Gaussian mmmm contribution")
                    nongauss_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_lens'],
                                                              nongauss[5-subtract],
                                                              True,
                                                              True,
                                                              True))
                    del nongauss[5-subtract]
                    subtract += 1
                else:
                    nongauss[5][:, :, :, :, :, :, :, :] *= np.ones_like(nongauss[5][:, :, :, :, :, :, :, :])/(survey_params_dict['survey_area_lens']/self.deg2torad2)
                    nongauss_new.append(nongauss[5][:, :, :, :, :, :, :, :])
            else:
                nongauss_new.append(0)
            nongauss = nongauss_new
        if self.ellrange_photo is not None:
            nongaussELLgggg_ssss_new = 0
            nongaussELLgggg_sssp_new = 0
            nongaussELLgggg_sspp_new = 0
            nongaussELLgggg_spsp_new = 0
            nongaussELLgggg_ppsp_new = 0
            nongaussELLgggg_pppp_new = 0

            nongaussELLgggm_sssm_new = 0
            nongaussELLgggm_sspm_new = 0
            nongaussELLgggm_spsm_new = 0
            nongaussELLgggm_sppm_new = 0
            nongaussELLgggm_ppsm_new = 0
            nongaussELLgggm_pppm_new = 0

            nongaussELLggmm_ssmm_new = 0
            nongaussELLggmm_spmm_new = 0
            nongaussELLggmm_ppmm_new = 0

            nongaussELLgmgm_smsm_new = 0
            nongaussELLgmgm_smpm_new = 0
            nongaussELLgmgm_pmsm_new = 0
            nongaussELLgmgm_pmpm_new = 0

            nongaussELLmmgm_mmsm_new = 0
            nongaussELLmmgm_mmpm_new = 0

            nongaussELLmmmm_mmmm_new = 0
            if self.cov_dict['nongauss']:
                subtract = 0
                if self.gg:
                    print("Binning non-Gaussian g_s g_s g_s g_s contribution")
                    nongaussELLgggg_ssss_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_s g_s g_p contribution")
                    nongaussELLgggg_sssp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_s g_p g_p contribution")
                    nongaussELLgggg_sspp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_p g_s g_p contribution")
                    nongaussELLgggg_spsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p g_p g_s g_p contribution")
                    nongaussELLgggg_ppsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p g_p g_p g_p contribution")
                    nongaussELLgggg_pppp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[0-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',True
                                                                )
                    #del nongauss[0-subtract]
                    #subtract += 1
                if self.gm and self.gg:
                    print("Binning non-Gaussian g_s g_s g_s m contribution")
                    nongaussELLgggm_sssm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_s g_p m contribution")
                    nongaussELLgggm_sspm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_p g_s m contribution")
                    nongaussELLgggm_spsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_p g_p m contribution")
                    nongaussELLgggm_sppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p g_p g_s m contribution")
                    nongaussELLgggm_ppsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p g_p g_p m contribution")
                    nongaussELLgggm_pppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[1-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                True
                                                                )
                    #del nongauss[1-subtract]
                    #subtract +=1
                if self.mm and self.gg:
                    print("Binning non-Gaussian g_s g_s m m contribution")
                    nongaussELLggmm_ssmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s g_p m m contribution")
                    nongaussELLggmm_spmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p g_p m m contribution")
                    nongaussELLggmm_ppmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[2-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                True
                                                                )
                    #del nongauss[2-subtract]
                    #subtract += 1
                if self.gm:
                    print("Binning non-Gaussian g_s m g_s m contribution")
                    nongaussELLgmgm_smsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_s m g_p m contribution")
                    nongaussELLgmgm_smpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p m g_s m contribution")
                    nongaussELLgmgm_pmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian g_p m g_p m contribution")
                    nongaussELLgmgm_pmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[3-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                True
                                                                )
                    #del nongauss[3-subtract]
                    #subtract += 1
                if self.mm and self.gm:
                    print("Binning non-Gaussian m m g_s m contribution")
                    nongaussELLmmgm_mmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[4-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                True
                                                                )
                    print("Binning non-Gaussian m m g_p m contribution")
                    nongaussELLmmgm_mmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[4-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                True
                                                                )
                    #del nongauss[4-subtract]
                    #subtract += 1
                if self.mm:
                    print("Binning non-Gaussian m m m m contribution")
                    nongaussELLmmmm_mmmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                nongauss[5-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens',
                                                                True
                                                                )
                    #del nongauss[5-subtract]
                    
            nongauss = nongaussELLgggg_ssss_new, nongaussELLgggg_sssp_new, nongaussELLgggg_sspp_new, \
                nongaussELLgggg_spsp_new, nongaussELLgggg_ppsp_new, nongaussELLgggg_pppp_new, \
                nongaussELLgggm_sssm_new, nongaussELLgggm_sspm_new, nongaussELLgggm_spsm_new, \
                nongaussELLgggm_sppm_new, nongaussELLgggm_ppsm_new, nongaussELLgggm_pppm_new, \
                nongaussELLggmm_ssmm_new, nongaussELLggmm_spmm_new, nongaussELLggmm_ppmm_new, \
                nongaussELLgmgm_smsm_new, nongaussELLgmgm_smpm_new, nongaussELLgmgm_pmsm_new, \
                nongaussELLgmgm_pmpm_new, nongaussELLmmgm_mmsm_new, nongaussELLmmgm_mmpm_new, \
                nongaussELLmmmm_mmmm_new
        
        ssc = self.covELL_ssc(bias_dict,
                              hod_dict,
                              prec,
                              survey_params_dict,
                              obs_dict['ELLspace'])
        ssc = list(ssc)
        if self.ellrange_photo is None and self.cov_dict['ssc']:
            subtract = 0
            ssc_new = []    
            if self.gg:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gggg contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_clust'],
                                                              ssc[0-subtract],
                                                              False,
                                                              True,
                                                              True))
                    del ssc[0-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[0])
            else:
                ssc_new.append(0)
            if self.gg and self.gm and self.cross_terms:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gggm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[1-subtract],
                                                              False,
                                                              True,
                                                              False))
                    del ssc[1-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[1])
            else:
                ssc_new.append(0)
            if self.gg and self.mm and self.cross_terms:
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning SSC ggmm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_clust'],
                                                              survey_params_dict['survey_area_lens'],
                                                              ssc[2-subtract],
                                                              False,
                                                              True,
                                                              True))
                    del ssc[2-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[2])
            else:
                ssc_new.append(0)
            if self.gm:
                if self.ellrange_clustering_ul is not None:
                    print("Binning SSC gmgm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_clustering_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_clustering,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_ggl'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[3-subtract],
                                                              False,
                                                              False,
                                                              False))
                    del ssc[3-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[3])
            else:
                ssc_new.append(0)
            if self.mm and self.gm and self.cross_terms:           
                if self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                    print("Binning SSC mmgm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_clustering_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_clustering,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_ggl'],
                                                              ssc[4-subtract],
                                                              False,
                                                              True,
                                                              False))
                    del ssc[4-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[4])
            else:
                ssc_new.append(0)
            if self.mm:
                if self.ellrange_lensing_ul is not None:
                    print("Binning SSC mmmm contribution")
                    ssc_new.append(self.__bin_cov_ell_nongauss(self.ellrange_lensing_ul,
                                                              self.ellrange_lensing_ul,
                                                              self.ellrange_lensing,
                                                              self.ellrange_lensing,
                                                              survey_params_dict['survey_area_lens'],
                                                              survey_params_dict['survey_area_lens'],
                                                              ssc[5-subtract],
                                                              False,
                                                              True,
                                                              True))
                    del ssc[5-subtract]
                    subtract += 1
                else:
                    ssc_new.append(ssc[5])
            else:
                ssc_new.append(0)
            ssc = ssc_new

        if self.ellrange_photo is not None:
            subtract = 0
            sscELLgggg_ssss_new = 0
            sscELLgggg_sssp_new = 0
            sscELLgggg_sspp_new = 0
            sscELLgggg_spsp_new = 0
            sscELLgggg_ppsp_new = 0
            sscELLgggg_pppp_new = 0

            sscELLgggm_sssm_new = 0
            sscELLgggm_sspm_new = 0
            sscELLgggm_spsm_new = 0
            sscELLgggm_sppm_new = 0
            sscELLgggm_ppsm_new = 0
            sscELLgggm_pppm_new = 0

            sscELLggmm_ssmm_new = 0
            sscELLggmm_spmm_new = 0
            sscELLggmm_ppmm_new = 0

            sscELLgmgm_smsm_new = 0
            sscELLgmgm_smpm_new = 0
            sscELLgmgm_pmsm_new = 0
            sscELLgmgm_pmpm_new = 0

            sscELLmmgm_mmsm_new = 0
            sscELLmmgm_mmpm_new = 0

            sscELLmmmm_mmmm_new = 0
            if self.cov_dict['ssc']:
                if self.gg:
                    print("Binning SSC g_s g_s g_s g_s contribution")
                    sscELLgggg_ssss_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                True,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    print("Binning SSC g_s g_s g_s g_p contribution")
                    sscELLgggg_sssp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    print("Binning SSC g_s g_s g_p g_p contribution")
                    sscELLgggg_sspp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    print("Binning SSC g_s g_p g_s g_p contribution")
                    sscELLgggg_spsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    print("Binning SSC g_p g_p g_s g_p contribution")
                    sscELLgggg_ppsp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    print("Binning SSC g_p g_p g_p g_p contribution")                                           
                    sscELLgggg_pppp_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[0-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'clust',
                                                                False
                                                                )
                    del ssc[0-subtract]
                    subtract += 1
                if self.gm and self.gg:
                    print("Binning SSC g_s g_s g_s m contribution")
                    sscELLgggm_sssm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                True,
                                                                True,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_s g_s g_p m contribution")
                    sscELLgggm_sspm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_s g_p g_s m contribution")
                    sscELLgggm_spsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_s g_p g_p m contribution")
                    sscELLgggm_sppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_p g_p g_s m contribution")
                    sscELLgggm_ppsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_p g_p g_p m contribution")
                    sscELLgggm_pppm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[1-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'ggl',
                                                                False
                                                                )
                    del ssc[1-subtract]
                    subtract += 1
                if self.mm and self.gg:
                    print("Binning SSC g_s g_s m m contribution")
                    sscELLggmm_ssmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2-subtract],
                                                                True,
                                                                True,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                    print("Binning SSC g_s g_p m m contribution")
                    sscELLggmm_spmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                    print("Binning SSC g_p g_p m m contribution")
                    sscELLggmm_ppmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[2-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'clust',
                                                                'lens',
                                                                False
                                                                )
                    del ssc[2-subtract]
                    subtract += 1
                if self.gm:
                    print("Binning SSC g_s m g_s m contribution")
                    sscELLgmgm_smsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3-subtract],
                                                                True,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_s m g_p m contribution")
                    sscELLgmgm_smpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3-subtract],
                                                                True,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_p m g_s m contribution")
                    sscELLgmgm_pmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC g_p m g_p m contribution")
                    sscELLgmgm_pmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[3-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'ggl',
                                                                'ggl',
                                                                False
                                                                )
                    del ssc[3-subtract]
                    subtract += 1
                if self.mm and self.gm:
                    print("Binning SSC m m g_s m contribution")
                    sscELLmmgm_mmsm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[4-subtract],
                                                                False,
                                                                False,
                                                                True,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                False
                                                                )
                    print("Binning SSC m m g_p m contribution")
                    sscELLmmgm_mmpm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[4-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'ggl',
                                                                False
                                                                )
                    del ssc[4-subtract]
                    subtract += 1
                if self.mm:
                    print("Binning SSC m m m m contribution")
                    sscELLmmmm_mmmm_new = self.__bin_non_Gaussian(obs_dict['ELLspace'],survey_params_dict,
                                                                ssc[5-subtract],
                                                                False,
                                                                False,
                                                                False,
                                                                False,
                                                                'lens',
                                                                'lens',
                                                                False
                                                                )
                    del ssc[5-subtract]
                    subtract += 1
            ssc = sscELLgggg_ssss_new, sscELLgggg_sssp_new, sscELLgggg_sspp_new, \
                sscELLgggg_spsp_new, sscELLgggg_ppsp_new, sscELLgggg_pppp_new, \
                sscELLgggm_sssm_new, sscELLgggm_sspm_new, sscELLgggm_spsm_new, \
                sscELLgggm_sppm_new, sscELLgggm_ppsm_new, sscELLgggm_pppm_new, \
                sscELLggmm_ssmm_new, sscELLggmm_spmm_new, sscELLggmm_ppmm_new, \
                sscELLgmgm_smsm_new, sscELLgmgm_smpm_new, sscELLgmgm_pmsm_new, \
                sscELLgmgm_pmpm_new, sscELLmmgm_mmsm_new, sscELLmmgm_mmpm_new, \
                sscELLmmmm_mmmm_new
        
        if self.Cell_mmB is not None and self.est_shear == 'C_ell' and self.ellrange_photo is None:
            gauss, nongauss, ssc = self.__add_Cell_Bmode_blocks(list(gauss), list(nongauss), list(ssc))
        return self._split_csmf_terms(list(gauss), list(nongauss), list(ssc))

    def __Cell_mult_shear_bias(self,
                               Cell_12,
                               Cell_34,
                               bias,
                               lensing_12,
                               lensing_34,
                               unique_12,
                               unique_34):
        """
        Covariance due to the uncertainty of the multiplicative shear bias,
        Eq. (60) of arXiv:2410.06962, bias * C_12(ell_1) C_34(ell_2), for
        the C_ell estimator, averaged over the ell bins if the C_ell are
        binned. Unlike the connected terms it does not scale with the
        inverse survey area.

        Parameters
        ----------
        Cell_12, Cell_34 : arrays
            angular power spectra with shape (ell, sample, tomo, tomo)
        bias : array
            products of the multiplicative shear bias uncertainties with
            shape (sample, sample, tomo, tomo, tomo, tomo)
        lensing_12, lensing_34 : bool
            whether the ell binning of the lensing (or else the clustering)
            spectra applies
        unique_12, unique_34 : bool
            whether only the unique tomographic bin combinations are needed

        Returns
        -------
        cov : array
            with shape (ell bins, ell bins, sample, sample, tomo, tomo, tomo,
            tomo)
        """
        cov = bias[None, None] \
            * Cell_12[:, None, :, None, :, :, None, None] \
            * Cell_34[None, :, None, :, None, None, :, :]
        ul_12 = self.ellrange_lensing_ul if lensing_12 else self.ellrange_clustering_ul
        ul_34 = self.ellrange_lensing_ul if lensing_34 else self.ellrange_clustering_ul
        if ul_12 is None or ul_34 is None:
            return cov
        return self.__bin_cov_ell_nongauss(ul_12, ul_34, None, None, 1, 1, cov,
                                           False, unique_12, unique_34)

    def __add_Cell_mult_shear_bias(self,
                                   gauss):
        """
        Adds the covariance due to the multiplicative shear bias uncertainty
        to the Gaussian C_ell covariance (to the sample-variance part if it
        is split), as for the projected estimators.

        Parameters
        ----------
        gauss : list
            Gaussian covariance as returned by covELL_gaussian (6 or, split
            into sva, mix and sn, 18 entries, plus the stellar mass function
            terms)

        Returns
        -------
        gauss : list
        """
        mult = 3 if self.cov_dict['split_gauss'] else 1
        terms = []
        if self.gm and self.est_shear == "C_ell":
            terms.append((3, self.Cell_gm, self.Cell_gm, self.gaussELLgmgm_sva_mult_shear_bias,
                          False, False, False, False))
        if self.gm and self.mm and self.cross_terms and self.est_shear == "C_ell" and self.est_ggl == "C_ell":
            terms.append((4, self.Cell_mm, self.Cell_gm, self.gaussELLmmgm_sva_mult_shear_bias,
                          True, False, True, False))
        if self.mm and self.est_shear == "C_ell":
            terms.append((5, self.Cell_mm, self.Cell_mm, self.gaussELLmmmm_sva_mult_shear_bias,
                          True, True, True, True))
        for probe, Cell_12, Cell_34, bias, lensing_12, lensing_34, unique_12, unique_34 in terms:
            if not isinstance(gauss[mult*probe], np.ndarray) or not np.any(bias):
                continue
            gauss[mult*probe] = gauss[mult*probe] + self.__Cell_mult_shear_bias(
                Cell_12, Cell_34, bias, lensing_12, lensing_34, unique_12, unique_34)
        return gauss

    def __add_Cell_Bmode_blocks(self,
                                gauss,
                                nongauss,
                                ssc):
        """
        Rearranges the C_ell covariance with intrinsic B-modes into the
        E/B layout of the projected estimators, ['gggg', 'gggm', 'ggmmE',
        'ggmmB', 'gmgm', 'mmEgm', 'mmBgm', 'mmEmmE', 'mmEmmB', 'mmBmmB'],
        where E (B) is the E-mode (B-mode) shear spectrum. Only the
        Gaussian term of the B-mode spectrum and the multiplicative shear
        bias contribute to the new blocks (C_EB = C_gB = 0, and there is no
        model for the connected B-mode terms).

        Parameters
        ----------
        gauss, nongauss, ssc : lists
            covariance terms in the C_ell layout ['gggg', 'gggm', 'ggmm',
            'gmgm', 'mmgm', 'mmmm'] (Gaussian split into sva, mix and sn if
            requested, followed by the stellar mass function terms)

        Returns
        -------
        gauss, nongauss, ssc : lists
        """
        mult = 3 if self.cov_dict['split_gauss'] else 1
        n_lss = 6*mult
        blocks = [gauss[mult*probe:mult*(probe + 1)] for probe in range(6)]
        tail = gauss[n_lss:]

        def zeros(term):
            return np.zeros_like(term) if isinstance(term, np.ndarray) else 0

        def total(terms):
            return sum(terms[1:], terms[0])

        BB_sva, BB_mix, BB_sn = self.gaussELLmmmm_BB
        mmmm_ref = total(blocks[5])
        EB_bias, BB_bias = zeros(mmmm_ref), 0
        if np.any(self.gaussELLmmmm_sva_mult_shear_bias):
            EB_bias = self.__Cell_mult_shear_bias(self.Cell_mm, self.Cell_mmB, self.gaussELLmmmm_sva_mult_shear_bias,
                                                  True, True, True, True)
            BB_bias = self.__Cell_mult_shear_bias(self.Cell_mmB, self.Cell_mmB, self.gaussELLmmmm_sva_mult_shear_bias,
                                                  True, True, True, True)
        EB_block = [EB_bias] + [zeros(mmmm_ref)]*(mult - 1)
        BB_block = [BB_sva + BB_bias, BB_mix, BB_sn] if mult == 3 else [BB_sva + BB_mix + BB_sn + BB_bias]
        ggB_block = [zeros(term) for term in blocks[2]]
        gmB_block = [zeros(term) for term in blocks[4]]
        if isinstance(total(blocks[4]), np.ndarray) and self.est_ggl == "C_ell" \
           and np.any(self.gaussELLmmgm_sva_mult_shear_bias):
            gmB_block[0] = gmB_block[0] + self.__Cell_mult_shear_bias(
                self.Cell_mmB, self.Cell_gm, self.gaussELLmmgm_sva_mult_shear_bias,
                True, False, True, False)
        new_gauss = blocks[0] + blocks[1] + blocks[2] + ggB_block + blocks[3] \
            + blocks[4] + gmB_block + blocks[5] + EB_block + BB_block
        new_nongauss = [nongauss[0], nongauss[1], nongauss[2], zeros(nongauss[2]), nongauss[3],
                        nongauss[4], zeros(nongauss[4]), nongauss[5], zeros(nongauss[5]), zeros(nongauss[5])]
        new_ssc = [ssc[0], ssc[1], ssc[2], zeros(ssc[2]), ssc[3],
                   ssc[4], zeros(ssc[4]), ssc[5], zeros(ssc[5]), zeros(ssc[5])]
        return new_gauss + tail, new_nongauss + nongauss[6:], new_ssc + ssc[6:]

    def __print_csmf_settings(self, read_in_tables):
        """
        Writes the settings of the stellar mass function (SMF) to the log.
        """
        print("Stellar mass function (SMF): " + str(len(self.log10csmf_mass_bins)) + " stellar mass bins with " +
              "log10(M*) from " + str(round(float(np.min(self.log10csmf_mass_bins)), 3)) + " to " +
              str(round(float(np.max(self.log10csmf_mass_bins)), 3)) + " (bin centres) in " + str(self.n_tomo_csmf) +
              " tomographic bin(s)" + (", with overlapping bins" if len(self.csmf_bin_overlaps) else "") +
              ", per dex and averaged over the bins (" + str(len(self.csmf_quad_nodes)) + " quadrature nodes).")
        if read_in_tables['zcsmf']['pz_mass'] is not None:
            print("SMF line-of-sight windows: redshift distribution per stellar mass bin (zcsmf_mass_file).")
        else:
            print("SMF line-of-sight windows: V_max top-hats in comoving volume (V_max_file" +
                  (", V_max_zmin" if self.csmf_V_max_zmin is not None else
                   ", lower redshift from the redshift distribution of the tomographic bin") + ").")
        options = [name for name, active in (('csmf_diagonal', self.csmf_diagonal),
                                             ('csmf_diagonal_lenses', self.csmf_diagonal_lenses),
                                             ('csmf_auto_only', self.csmf_auto_only)) if active]
        if options:
            print("SMF options: " + ", ".join(options) + ".")

    def calc_covELL_csmf(self,
                         covELLspacesettings,
                         survey_params_dict):
        r"""
        Calculates the covariance of the conditional stellar mass function and its cross-correlation
        with the tracers defined in the config file.
        
        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        
        Returns
        -------

        """
        terms = ["shot noise", "sample variance (super-sample term of the counts)"]
        if self.csmf_auto_only:
            terms.append("no cross-covariance with the 2-point functions (csmf_auto_only)")
        else:
            if self.cov_dict['nongauss']:
                terms.append("three-point cross term with the 2-point functions (" +
                             ("count-galaxy-galaxy/-matter and count-matter-matter bispectra" if getattr(self, 'csmf_galaxy_bispectra', False)
                              else "count-matter-matter bispectrum") + ")")
            if self.cov_dict['ssc']:
                terms.append("super-sample cross term with the 2-point functions")
        print("Calculating the covariance of the stellar mass function: " + ", ".join(terms) + ".")
        csmf_gg_sva,csmf_gm_sva,csmf_mm_sva = self.covELL_csmf_cross_LSS_sva(covELLspacesettings, survey_params_dict,
                                                                             compute=self.cov_dict['nongauss'])
        csmf_gg_ssc,csmf_gm_ssc,csmf_mm_ssc = self.covELL_csmf_cross_LSS_ssc(covELLspacesettings, survey_params_dict,
                                                                             compute=self.cov_dict['ssc'])
        csmf_sn = self.covELL_csmf_SN()
        # the sample variance of the counts (their super-sample term) is
        # always included: without it the SMF has only the shot noise, which
        # is inconsistent with the three-point cross term
        csmf_ssc = self.covELL_csmf_SSC(covELLspacesettings, survey_params_dict)
        self.csmf_parts = {'auto_sn': csmf_sn, 'auto_ssc': csmf_ssc,
                           'cross_ssc': [csmf_gg_ssc, csmf_gm_ssc, csmf_mm_ssc]}
        return csmf_sn + csmf_ssc, csmf_gg_sva + csmf_gg_ssc, csmf_gm_sva + csmf_gm_ssc, csmf_mm_sva + csmf_mm_ssc
    
    def __bin_cov_ell_csmf(self,
                           ellrange_12_ul,
                           cov,
                           unique_12):
        r"""
        Auxillary function binning the cross-covariance matrix 
        between the SMF and the LSS in ell-bins.

        """
        
        if not isinstance(cov, np.ndarray):
            return 0
        binned_covariance = np.zeros(np.zeros(len(ellrange_12_ul) - 1).shape + cov[0].shape)
        for i_ell in range(len(ellrange_12_ul) - 1):
            area12_ell = np.pi*((ellrange_12_ul[i_ell +1])**2 - (ellrange_12_ul[i_ell])**2)                                            
            integration_ell = np.geomspace(ellrange_12_ul[i_ell],ellrange_12_ul[i_ell +1],100)
            for i_csmf in range(len(cov[0,:,0,0,0,0])):
                for i_sample in range(len(cov[0,0,:,0,0,0])):
                    for csmf_tomo in range(len(cov[0,0,0,:,0,0])):
                        for i_tomo in range(len(cov[0,0,0,0,:,0])):
                            j_tomo_start = 0
                            if unique_12:
                                j_tomo_start = i_tomo
                            for j_tomo in range(j_tomo_start, len(cov[0,0,0,0,0,:])):
                                if len(np.where(cov[:, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo])[0]):
                                    spline = UnivariateSpline(self.ellrange, self.ellrange*cov[:, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo], k=2, s=0, ext=1)   
                                    binned_covariance[i_ell, i_csmf, i_sample, csmf_tomo, i_tomo, j_tomo] = simpson(spline(integration_ell),integration_ell)*2.*np.pi/area12_ell
        return binned_covariance

    
    def __bin_Gaussian(self,
                       covELLspacesettings,
                       survey_params_dict,
                       covariance,
                       field1_spec,
                       field2_spec,
                       field3_spec,
                       field4_spec,
                       probe12,
                       probe34):
        r"""
        Auxillary function binning the Gaussian covariance matrix in ell-bins

        """
        if not isinstance(covariance, np.ndarray):
            return 0
        full_sky_angle = 4*np.pi * self.deg2torad2

        if probe12 == "clust":
            if field1_spec and field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][0]
            if field1_spec and not field2_spec or field2_spec and not field1_spec:
                area_12 = survey_params_dict['survey_area_clust'][1]
            if not field1_spec and not field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][2]
        if probe12 == "ggl":
            area_12 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_12 = survey_params_dict['survey_area_ggl'][0]
        if probe12 == "lens":
            area_12 = survey_params_dict['survey_area_lens']

        if probe34 == "clust":
            if field3_spec and field4_spec:
                area_34 = survey_params_dict['survey_area_clust'][0]
            if field3_spec and not field4_spec or field4_spec and not field3_spec:
                area_34 = survey_params_dict['survey_area_clust'][1]
            if not field3_spec and not field4_spec:
                area_34 = survey_params_dict['survey_area_clust'][2]
        if probe34 == "ggl":
            area_34 = survey_params_dict['survey_area_ggl'][1]
            if field3_spec:
                area_34 = survey_params_dict['survey_area_ggl'][0]
        if probe34 == "lens":
            area_34 = survey_params_dict['survey_area_lens']

        if probe12 == "clust" and probe34 == "clust":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = covELLspacesettings['n_spec']
            up_limit_4 = self.n_tomo_clust
            if field4_spec:
                lo_limit_4 = 0
                up_limit_4 = covELLspacesettings['n_spec']
        
        if probe12 == "clust" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "clust" and probe34 == "lens":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "ggl" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "ggl":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "lens":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
    
        covariance_aux = covariance[:, :, :, :, lo_limit_1 : up_limit_1, lo_limit_2 : up_limit_2, lo_limit_3 : up_limit_3, lo_limit_4 : up_limit_4]
        
        ellrange_12 = self.ellrange_photo
        ellrange_12_ul = self.ellrange_photo_ul
        
        
        if field1_spec and field2_spec:
            ellrange_12 = self.ellrange_spec
            ellrange_12_ul = self.ellrange_spec_ul
        if (field1_spec and not field2_spec) or (field2_spec and not field1_spec):
            ellrange_12 = self.ellrange_spec_photo
            ellrange_12_ul = self.ellrange_spec_photo_ul

        ellrange_34 = self.ellrange_photo
        ellrange_34_ul = self.ellrange_photo_ul
        if field3_spec or field4_spec:
            ellrange_34 = self.ellrange_spec
            ellrange_34_ul = self.ellrange_spec_ul
        if (field3_spec and not field4_spec) or (field4_spec and not field3_spec):
            ellrange_34 = self.ellrange_spec_photo
            ellrange_34_ul = self.ellrange_spec_photo_ul

        
        if probe12 == "lens":
            ellrange_12 = self.ellrange_lensing
            ellrange_12_ul = self.ellrange_lensing_ul

        if probe34 == "lens":
            ellrange_34 = self.ellrange_lensing
            ellrange_34_ul = self.ellrange_lensing_ul

        
        binned_covariance = np.zeros((len(ellrange_12), len(ellrange_34), self.sample_dim, self.sample_dim, up_limit_1 - lo_limit_1, up_limit_2 - lo_limit_2, up_limit_3 - lo_limit_3, up_limit_4 - lo_limit_4))
        for i_ell in range(len(ellrange_12)):
            for j_ell in range(len(ellrange_34)):
                integration_ell_12 = np.arange(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1]).astype(int)
                integration_ell_34 = np.arange(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1]).astype(int)
                overlapping_elements = np.array(list(set(integration_ell_12).intersection(set(integration_ell_34))))
                if len(overlapping_elements) == 0:
                    continue
                else:
                    for i_tomo in range(up_limit_1 - lo_limit_1):
                        for j_tomo in range(up_limit_2 - lo_limit_2):
                            for k_tomo in range(up_limit_3 - lo_limit_3):
                                for l_tomo in range(up_limit_4 - lo_limit_4):
                                    if np.any(np.diagonal(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo])!=0):
                                        spline = UnivariateSpline(self.ellrange,np.diagonal(covariance_aux[:, :, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo]), k=2, s=0, ext=1)
                                        result = full_sky_angle / self._max_area(area_12, area_34)*np.sum(spline(overlapping_elements)/(2.*overlapping_elements + 1))/len(integration_ell_12)/len(integration_ell_34)
                                        binned_covariance[i_ell, j_ell, 0, 0, i_tomo, j_tomo, k_tomo, l_tomo] = result
        return binned_covariance
    

    def __bin_cov_ell_gauss(self,
                            ellrange_12_ul,
                            ellrange_34_ul,
                            area_12,
                            area_34,
                            cov,
                            unique_12,
                            unique_34):
        if not isinstance(cov, np.ndarray):
            return 0
        fsky = self._max_area(area_12, area_34)/(4.0*np.pi * self.deg2torad2)
        binned_covariance = np.zeros((len(ellrange_12_ul) - 1, len(ellrange_34_ul) - 1, len(cov[0,0,:,0,0,0,0,0]), len(cov[0,0,0,:,0,0,0,0]), len(cov[0,0,0,0,:,0,0,0]), len(cov[0,0,0,0,0,:,0,0]), len(cov[0,0,0,0,0,0,:,0]), len(cov[0,0,0,0,0,0,0,:])))
        for i_ell in range(len(ellrange_12_ul) - 1):
            for j_ell in range(len(ellrange_34_ul) - 1):
                integration_ell_12 = np.arange(ellrange_12_ul[i_ell], ellrange_12_ul[i_ell+1]).astype(int)
                integration_ell_34 = np.arange(ellrange_34_ul[j_ell], ellrange_34_ul[j_ell+1]).astype(int)
                overlapping_elements = np.array(list(set(integration_ell_12).intersection(set(integration_ell_34))))
                if len(overlapping_elements) == 0:
                    continue
                else:
                    for i_sample in range(len(cov[0,0,:,0,0,0,0,0])):
                        for j_sample in range(len(cov[0,0,0,:,0,0,0,0])):
                            for i_tomo in range(len(cov[0,0,0,0,:,0,0,0])):
                                j_tomo_start = 0
                                if unique_12:
                                    j_tomo_start = i_tomo
                                for j_tomo in range(j_tomo_start, len(cov[0,0,0,0,0,:,0,0])):
                                    for k_tomo in range(len(cov[0,0,0,0,0,0,:,0])):
                                        l_tomo_start = 0
                                        if unique_34:
                                            l_tomo_start = k_tomo
                                        for l_tomo in range(l_tomo_start, len(cov[0,0,0,0,0,0,0,:])):
                                            if len(np.where(np.diagonal(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]))[0]):
                                                spline = UnivariateSpline(self.ellrange, np.diagonal(cov[:, :, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo]), k=1, s=0, ext=1)
                                                binned_covariance[i_ell, j_ell, i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo] = np.sum(spline(overlapping_elements)/(2.*overlapping_elements + 1))/fsky/len(integration_ell_12)/len(integration_ell_34)
        return binned_covariance            

    def __bin_cov_ell_nongauss(self,
                               ellrange_12_ul,
                               ellrange_34_ul,
                               ellrange_12,
                               ellrange_34,
                               area_12,
                               area_34,
                               cov,
                               connected,
                               unique_12,
                               unique_34):
        if not isinstance(cov, np.ndarray):
            return 0
        binned_covariance = np.zeros((len(ellrange_12_ul) - 1, len(ellrange_34_ul) - 1) + cov.shape[2:])
        n_tomo_1, n_tomo_2, n_tomo_3, n_tomo_4 = cov.shape[4:]
        tomo_12 = np.ones((n_tomo_1, n_tomo_2), dtype=bool)
        tomo_34 = np.ones((n_tomo_3, n_tomo_4), dtype=bool)
        if unique_12:
            tomo_12 = np.triu(tomo_12)
        if unique_34:
            tomo_34 = np.triu(tomo_34)
        selected = np.broadcast_to(tomo_12[:, :, None, None]*tomo_34[None, None, :, :], cov.shape[2:])
        binned_covariance.reshape(binned_covariance.shape[:2] + (-1,))[:] = \
            self.__bin_4pt_ell_columns(ellrange_12_ul, ellrange_34_ul,
                                       cov.reshape(cov.shape[:2] + (-1,)),
                                       selected.reshape(-1), area_12, area_34, connected)
        return binned_covariance

    def __bin_average_matrix(self, ellrange_ul):
        """
        Matrix mapping a function tabulated on self.ellrange to its integral
        times ell (Simpson's rule) over each ell bin of ellrange_ul. The
        integration points of each bin are geometrically spaced (3 to 10 of
        them, depending on the number of ellrange nodes in the bin). Also
        returns the interpolation matrix at the points: the linear
        interpolation of RegularGridInterpolator (with linear extrapolation,
        fill_value = None) on (ellrange, ellrange) is the tensor product of
        it, so the interpolated integrand is B_12 V B_34^T.
        """
        ellrange = np.asarray(self.ellrange, dtype=float)
        points, weights = [], []
        for i_ell in range(len(ellrange_ul) - 1):
            number = int((np.abs(ellrange - ellrange_ul[i_ell+1])).argmin() - (np.abs(ellrange - ellrange_ul[i_ell])).argmin())
            number = min(max(number, 3), 10)
            integration_ell = np.geomspace(ellrange_ul[i_ell], ellrange_ul[i_ell+1], number)
            points.append(integration_ell)
            weights.append(simpson_weights(integration_ell)*integration_ell)
        weight_matrix = np.zeros((len(points), sum(len(x) for x in points)))
        offset = 0
        for i_ell, w in enumerate(weights):
            weight_matrix[i_ell, offset:offset + len(w)] = w
            offset += len(w)
        points = np.concatenate(points)
        # same cell search as RegularGridInterpolator (edge cells extrapolate)
        index = np.clip(np.searchsorted(ellrange, points) - 1, 0, len(ellrange) - 2)
        t = (points - ellrange[index])/(ellrange[index + 1] - ellrange[index])
        interpolation = np.zeros((len(points), len(ellrange)))
        rows = np.arange(len(points))
        interpolation[rows, index] = 1 - t
        interpolation[rows, index + 1] += t
        return weight_matrix, interpolation

    def __bin_4pt_ell_columns(self,
                              ellrange_12_ul,
                              ellrange_34_ul,
                              cov,
                              selected,
                              area_12,
                              area_34,
                              connected):
        """
        Bin averages of the columns cov[:, :, m] (shape (ell, ell, columns))
        in the ell bins ellrange_12_ul x ellrange_34_ul for the selected
        columns with a non-vanishing ell diagonal; the others are zero.
        Columns that are positive everywhere are interpolated in log. All
        columns are done at once as matrix products.
        """
        weights_12, interp_12 = self.__bin_average_matrix(ellrange_12_ul)
        weights_34, interp_34 = self.__bin_average_matrix(ellrange_34_ul)
        area12_ell = np.pi*np.diff(np.asarray(ellrange_12_ul, dtype=float)**2)
        area34_ell = np.pi*np.diff(np.asarray(ellrange_34_ul, dtype=float)**2)
        norm = 4*np.pi**2/(area12_ell[:, None]*area34_ell[None, :])
        if connected:
            norm *= self.deg2torad2/self._max_area(area_12, area_34)
        n_ell = cov.shape[0]
        binned = np.zeros((len(area12_ell), len(area34_ell), cov.shape[2]))
        diagonal = cov[np.arange(n_ell), np.arange(n_ell), :]
        columns = np.flatnonzero(selected & np.any(diagonal != 0, axis=0))
        if len(columns) == 0:
            return binned
        n_points_12, n_points_34 = len(interp_12), len(interp_34)
        size = n_ell*n_ell + n_points_12*(n_ell + n_points_34)
        for block in self._blocks(len(columns), size):
            cols = columns[block]
            # columns first, so that all products are single matrix products
            values = np.ascontiguousarray(np.moveaxis(cov[:, :, cols], 2, 0))
            positive = np.all(values > 0, axis=(1, 2))
            if np.any(positive):
                values[positive] = np.log(values[positive])
            # (m, ell_12, ell_34) -> (m, ell_12, p_34) -> (m, p_34, p_12)
            integrand = (values.reshape(-1, n_ell) @ interp_34.T).reshape(len(cols), n_ell, n_points_34)
            integrand = (integrand.transpose(0, 2, 1).reshape(-1, n_ell) @ interp_12.T).reshape(len(cols), n_points_34, n_points_12)
            if np.any(positive):
                integrand[positive] = np.exp(integrand[positive])
            # (m, p_34, p_12) -> (m, p_34, bin_12) -> (m, bin_12, bin_34)
            result = (integrand.reshape(-1, n_points_12) @ weights_12.T).reshape(len(cols), n_points_34, -1)
            result = (result.transpose(0, 2, 1).reshape(-1, n_points_34) @ weights_34.T).reshape(len(cols), len(area12_ell), len(area34_ell))
            binned[:, :, cols] = np.moveaxis(result, 0, 2)*norm[:, :, None]
        return binned


    def __bin_non_Gaussian(self,
                            covELLspacesettings,
                            survey_params_dict,
                            covariance,
                            field1_spec,
                            field2_spec,
                            field3_spec,
                            field4_spec,
                            probe12,
                            probe34,
                            connected):
        r"""
        Auxillary function binning the Non-Gaussian covariance matrix in ell-bins

        """
        if not isinstance(covariance, np.ndarray):
            return 0
        full_sky_angle = 1 * self.deg2torad2

        if probe12 == "clust":
            if field1_spec and field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][0]
            if field1_spec and not field2_spec or field2_spec and not field1_spec:
                area_12 = survey_params_dict['survey_area_clust'][1]
            if not field1_spec and not field2_spec:
                area_12 = survey_params_dict['survey_area_clust'][2]
        if probe12 == "ggl":
            area_12 = survey_params_dict['survey_area_ggl'][1]
            if field1_spec:
                area_12 = survey_params_dict['survey_area_ggl'][0]
        if probe12 == "lens":
            area_12 = survey_params_dict['survey_area_lens']

        if probe34 == "clust":
            if field3_spec and field4_spec:
                area_34 = survey_params_dict['survey_area_clust'][0]
            if field3_spec and not field4_spec or field4_spec and not field3_spec:
                area_34 = survey_params_dict['survey_area_clust'][1]
            if not field3_spec and not field4_spec:
                area_34 = survey_params_dict['survey_area_clust'][2]
        if probe34 == "ggl":
            area_34 = survey_params_dict['survey_area_ggl'][1]
            if field3_spec:
                area_34 = survey_params_dict['survey_area_ggl'][0]
        if probe34 == "lens":
            area_34 = survey_params_dict['survey_area_lens']

        if probe12 == "clust" and probe34 == "clust":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = covELLspacesettings['n_spec']
            up_limit_4 = self.n_tomo_clust
            if field4_spec:
                lo_limit_4 = 0
                up_limit_4 = covELLspacesettings['n_spec']
        
        if probe12 == "clust" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "clust" and probe34 == "lens":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = covELLspacesettings['n_spec']
            up_limit_2 = self.n_tomo_clust
            if field2_spec:
                lo_limit_2 = 0
                up_limit_2 = covELLspacesettings['n_spec']
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
        
        if probe12 == "ggl" and probe34 == "ggl":
            lo_limit_1 = covELLspacesettings['n_spec']
            up_limit_1 = self.n_tomo_clust
            if field1_spec:
                lo_limit_1 = 0
                up_limit_1 = covELLspacesettings['n_spec']
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "ggl":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = covELLspacesettings['n_spec']
            up_limit_3 = self.n_tomo_clust
            if field3_spec:
                lo_limit_3 = 0
                up_limit_3 = covELLspacesettings['n_spec']
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens

        if probe12 == "lens" and probe34 == "lens":
            lo_limit_1 = 0
            up_limit_1 = self.n_tomo_lens
            lo_limit_2 = 0
            up_limit_2 = self.n_tomo_lens
            lo_limit_3 = 0
            up_limit_3 = self.n_tomo_lens
            lo_limit_4 = 0
            up_limit_4 = self.n_tomo_lens
    
        covariance_aux = covariance[:, :, :, :, lo_limit_1 : up_limit_1, lo_limit_2 : up_limit_2, lo_limit_3 : up_limit_3, lo_limit_4 : up_limit_4]
        ellrange_12 = self.ellrange_photo
        ellrange_12_ul = self.ellrange_photo_ul
        if field1_spec and field2_spec:
            ellrange_12 = self.ellrange_spec
            ellrange_12_ul = self.ellrange_spec_ul
        if (field1_spec and not field2_spec) or (field2_spec and not field1_spec):
            ellrange_12 = self.ellrange_spec_photo
            ellrange_12_ul = self.ellrange_spec_photo_ul

        ellrange_34 = self.ellrange_photo
        ellrange_34_ul = self.ellrange_photo_ul
        if field3_spec or field4_spec:
            ellrange_34 = self.ellrange_spec
            ellrange_34_ul = self.ellrange_spec_ul
        if (field3_spec and not field4_spec) or (field4_spec and not field3_spec):
            ellrange_34 = self.ellrange_spec_photo
            ellrange_34_ul = self.ellrange_spec_photo_ul

        
        if probe12 == "lens":
            ellrange_12 = self.ellrange_lensing
            ellrange_12_ul = self.ellrange_lensing_ul

        if probe34 == "lens":
            ellrange_34 = self.ellrange_lensing
            ellrange_34_ul = self.ellrange_lensing_ul
        
        binned_covariance = np.zeros((len(ellrange_12), len(ellrange_34), self.sample_dim, self.sample_dim, up_limit_1 - lo_limit_1, up_limit_2 - lo_limit_2, up_limit_3 - lo_limit_3, up_limit_4 - lo_limit_4))

        # only the first sample pair is binned
        covariance_00 = covariance_aux[:, :, 0, 0]
        binned_covariance[:, :, 0, 0] = self.__bin_4pt_ell_columns(
            ellrange_12_ul, ellrange_34_ul,
            covariance_00.reshape(covariance_00.shape[:2] + (-1,)),
            np.ones(np.prod(covariance_00.shape[2:], dtype=int), dtype=bool),
            area_12, area_34, connected).reshape(binned_covariance.shape[:2] + covariance_00.shape[2:])
        return binned_covariance
                                    

    def covELL_gaussian(self,
                        covELLspacesettings,
                        survey_params_dict,
                        calc_prefac=True):
        r"""
        Calculates the Gaussian (disconnected) covariance in ell space
        between two observables.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : bool
            default : True
            If True, returns the full Cell-covariance. If False, it
            omits the prefactor 1/(2*ell+1)/f_sky, where f_sky is the
            observed fraction on the sky.

        Returns
        -------
        gaussELLgggg, gaussELLgggm, gaussELLggmm, \
        gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
        gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn : list of
                                                            arrays
            with shape (ell bins, ell bins,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the pure kappa-kappa ('mmmm'),
        tracer-tracer ('gggg'), and kappa-tracer ('gmgm') terms, one
        needs to add gaussELLxxyy + gaussELLxxyy_sn. They are kept
        separate for a later numerical integration.

        """

        print("Calculating gaussian covariance for angular power spectra " +
              "(C_ell's).")

        if not calc_prefac or self.ellrange_spec is not None:
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                self.__covELL_split_gaussian(covELLspacesettings,
                                            survey_params_dict,
                                            False)
        elif calc_prefac and ((self.mm and self.ellrange_lensing_ul is None) or (self.gm and self.ellrange_clustering_ul is None) or (self.gg and self.ellrange_clustering_ul is None)):
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                self.__covELL_split_gaussian(covELLspacesettings,
                                            survey_params_dict,
                                            True)
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = \
                self.__covELL_split_gaussian(covELLspacesettings,
                                            survey_params_dict,
                                            False)
        if self.csmf:
            cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm = self.calc_covELL_csmf(covELLspacesettings, survey_params_dict)

            if self.gg and self.ellrange_clustering_ul is not None and self.est_clust == "C_ell" and not self.do_arbitrary:
                cov_csmf_gg = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cov_csmf_gg,True)
            if self.gm and self.ellrange_clustering_ul is not None and self.est_ggl == "C_ell"  and not self.do_arbitrary:
                cov_csmf_gm = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cov_csmf_gm,False)
            if self.mm and self.ellrange_lensing_ul is not None and self.est_shear == "C_ell" and not self.do_arbitrary:
                cov_csmf_mm = self.__bin_cov_ell_csmf(self.ellrange_lensing_ul,cov_csmf_mm,True)
            # the SSC part of the cross terms, binned in the same way
            cross_ssc = self.csmf_parts['cross_ssc']
            if self.gg and self.ellrange_clustering_ul is not None and self.est_clust == "C_ell" and not self.do_arbitrary:
                cross_ssc[0] = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cross_ssc[0],True)
            if self.gm and self.ellrange_clustering_ul is not None and self.est_ggl == "C_ell"  and not self.do_arbitrary:
                cross_ssc[1] = self.__bin_cov_ell_csmf(self.ellrange_clustering_ul,cross_ssc[1],False)
            if self.mm and self.ellrange_lensing_ul is not None and self.est_shear == "C_ell" and not self.do_arbitrary:
                cross_ssc[2] = self.__bin_cov_ell_csmf(self.ellrange_lensing_ul,cross_ssc[2],True)
        if covELLspacesettings['pixelised_cell']:
            
            gaussELLgggg_sva *= self._pixel_weights(gaussELLgggg_sva) 
            gaussELLgggg_mix *= self._pixel_weights(gaussELLgggg_mix) 
            gaussELLgggg_sn *= self._pixel_weights(gaussELLgggg_sn) 
            gaussELLgggm_sva *= self._pixel_weights(gaussELLgggm_sva) 
            gaussELLgggm_mix *= self._pixel_weights(gaussELLgggm_mix)  
            gaussELLgggm_sn *= self._pixel_weights(gaussELLgggm_sn) 
            gaussELLggmm_sva *= self._pixel_weights(gaussELLggmm_sva) 
            gaussELLggmm_mix *= self._pixel_weights(gaussELLggmm_mix) 
            gaussELLggmm_sn *= self._pixel_weights(gaussELLggmm_sn) 
            gaussELLgmgm_sva *= self._pixel_weights(gaussELLgmgm_sva) 
            gaussELLgmgm_mix *= self._pixel_weights(gaussELLgmgm_mix) 
            gaussELLgmgm_sn *= self._pixel_weights(gaussELLgmgm_sn) 
            gaussELLmmgm_sva *= self._pixel_weights(gaussELLmmgm_sva) 
            gaussELLmmgm_mix *= self._pixel_weights(gaussELLmmgm_mix) 
            gaussELLmmgm_sn *= self._pixel_weights(gaussELLmmgm_sn) 
            gaussELLmmmm_sva *= self._pixel_weights(gaussELLmmmm_sva) 
            gaussELLmmmm_mix *= self._pixel_weights(gaussELLmmmm_mix) 
            gaussELLmmmm_sn *= self._pixel_weights(gaussELLmmmm_sn)
            if self.gaussELLmmmm_BB is not None:
                for term in self.gaussELLmmmm_BB:
                    term *= self._pixel_weights(term)
        if calc_prefac and self.ellrange_spec is None:
            if self.gg and self.ellrange_clustering_ul is not None:
                gaussELLgggg_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_sva,
                                                            True,
                                                            True)
                gaussELLgggg_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_mix,
                                                            True,
                                                            True)
                gaussELLgggg_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_clust'],
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True)
            if self.gg and self.gm and self.ellrange_clustering_ul is not None:
                gaussELLgggm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_sva,
                                                            True,
                                                            False)
                gaussELLgggm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_mix,
                                                            True,
                                                            False)
                gaussELLgggm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgggm_sn,
                                                            True,
                                                            False)
            if self.gg and self.mm and self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:
                gaussELLggmm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_sva,
                                                            True,
                                                            True)
                gaussELLggmm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_mix,
                                                            True,
                                                            True)
                gaussELLggmm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_clust'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLggmm_sn,
                                                            True,
                                                            True)
            if self.gm and self.ellrange_clustering_ul is not None:
                gaussELLgmgm_sva = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_sva,
                                                            False,
                                                            False)
                gaussELLgmgm_mix = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_mix,
                                                            False,
                                                            False)
                gaussELLgmgm_sn = self.__bin_cov_ell_gauss(self.ellrange_clustering_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_ggl'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False)
            if self.gm and self.mm and self.ellrange_clustering_ul is not None and self.ellrange_lensing_ul is not None:                  
                gaussELLmmgm_sva = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLmmgm_sva,
                                                            True,
                                                            False)
                gaussELLmmgm_mix = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLmmgm_mix,
                                                            True,
                                                            False)
                gaussELLmmgm_sn = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_clustering_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_ggl'],
                                                            gaussELLmmgm_sn,
                                                            True,
                                                            False)
            
            if self.mm and self.ellrange_lensing_ul is not None:
            
                gaussELLmmmm_sva = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_sva,
                                                            True,
                                                            True)
                gaussELLmmmm_mix = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_mix,
                                                            True,
                                                            True)
                gaussELLmmmm_sn = self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                            self.ellrange_lensing_ul,
                                                            survey_params_dict['survey_area_lens'],
                                                            survey_params_dict['survey_area_lens'],
                                                            gaussELLmmmm_sn,
                                                            True,
                                                            True)
                if self.gaussELLmmmm_BB is not None:
                    self.gaussELLmmmm_BB = [self.__bin_cov_ell_gauss(self.ellrange_lensing_ul,
                                                                     self.ellrange_lensing_ul,
                                                                     survey_params_dict['survey_area_lens'],
                                                                     survey_params_dict['survey_area_lens'],
                                                                     term,
                                                                     True,
                                                                     True)
                                            for term in self.gaussELLmmmm_BB]

        if not self.cov_dict['split_gauss'] and self.ellrange_spec is not None and calc_prefac:
            gaussELLgggg = gaussELLgggg_sva + gaussELLgggg_mix
            gaussELLgggm = gaussELLgggm_sva + gaussELLgggm_mix
            gaussELLgmgm = gaussELLgmgm_sva + gaussELLgmgm_mix
            gaussELLmmgm = gaussELLmmgm_sva + gaussELLmmgm_mix
            gaussELLmmmm = gaussELLmmmm_sva + gaussELLmmmm_mix
            gaussELLggmm = gaussELLggmm_sva + gaussELLggmm_mix    
            
            gaussELLgggg_ssss_new = 0
            gaussELLgggg_sssp_new = 0
            gaussELLgggg_sspp_new = 0
            gaussELLgggg_spsp_new = 0
            gaussELLgggg_ppsp_new = 0
            gaussELLgggg_pppp_new = 0

            gaussELLgggm_sssm_new = 0
            gaussELLgggm_sspm_new = 0
            gaussELLgggm_spsm_new = 0
            gaussELLgggm_sppm_new = 0
            gaussELLgggm_ppsm_new = 0
            gaussELLgggm_pppm_new = 0

            gaussELLggmm_ssmm_new = 0
            gaussELLggmm_spmm_new = 0
            gaussELLggmm_ppmm_new = 0

            gaussELLgmgm_smsm_new = 0
            gaussELLgmgm_smpm_new = 0
            gaussELLgmgm_pmsm_new = 0
            gaussELLgmgm_pmpm_new = 0

            gaussELLmmgm_mmsm_new = 0
            gaussELLmmgm_mmpm_new = 0

            gaussELLmmmm_mmmm_new = 0


            gaussELLgggg_ssss_new_sn = 0
            gaussELLgggg_sssp_new_sn = 0
            gaussELLgggg_sspp_new_sn = 0
            gaussELLgggg_spsp_new_sn = 0
            gaussELLgggg_ppsp_new_sn = 0
            gaussELLgggg_pppp_new_sn = 0

            gaussELLgmgm_smsm_new_sn = 0
            gaussELLgmgm_smpm_new_sn = 0
            gaussELLgmgm_pmsm_new_sn = 0
            gaussELLgmgm_pmpm_new_sn = 0

            gaussELLmmmm_mmmm_new_sn = 0
            
            if self.gg:
                gaussELLgggg_ssss_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )

                gaussELLgggg_sssp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                del gaussELLgggg
                gaussELLgggg_ssss_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                del gaussELLgggg_sn
            if self.gm and self.gg:
                gaussELLgggm_sssm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sspm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_spsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sppm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_ppsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_pppm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                del gaussELLgggm
            if self.mm and self.gg:
                gaussELLggmm_ssmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_spmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_ppmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                del gaussELLggmm
            if self.gm:
                gaussELLgmgm_smsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                del gaussELLgmgm
                gaussELLgmgm_smsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                del gaussELLgmgm_sn
            if self.mm and self.gm:
                gaussELLmmgm_mmsm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                gaussELLmmgm_mmpm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                del gaussELLmmgm
            if self.mm:
                gaussELLmmmm_mmmm_new = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                del gaussELLmmmm
                gaussELLmmmm_mmmm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                del gaussELLmmmm_sn
            return gaussELLgggg_ssss_new, \
                    gaussELLgggg_sssp_new, \
                    gaussELLgggg_sspp_new, \
                    gaussELLgggg_spsp_new, \
                    gaussELLgggg_ppsp_new, \
                    gaussELLgggg_pppp_new, \
                    gaussELLgggm_sssm_new, \
                    gaussELLgggm_sspm_new, \
                    gaussELLgggm_spsm_new, \
                    gaussELLgggm_sppm_new, \
                    gaussELLgggm_ppsm_new, \
                    gaussELLgggm_pppm_new, \
                    gaussELLggmm_ssmm_new, \
                    gaussELLggmm_spmm_new, \
                    gaussELLggmm_ppmm_new, \
                    gaussELLgmgm_smsm_new, \
                    gaussELLgmgm_smpm_new, \
                    gaussELLgmgm_pmsm_new, \
                    gaussELLgmgm_pmpm_new, \
                    gaussELLmmgm_mmsm_new, \
                    gaussELLmmgm_mmpm_new, \
                    gaussELLmmmm_mmmm_new, \
                    gaussELLgggg_ssss_new_sn, \
                    gaussELLgggg_sssp_new_sn, \
                    gaussELLgggg_sspp_new_sn, \
                    gaussELLgggg_spsp_new_sn, \
                    gaussELLgggg_ppsp_new_sn, \
                    gaussELLgggg_pppp_new_sn, \
                    gaussELLgmgm_smsm_new_sn, \
                    gaussELLgmgm_smpm_new_sn, \
                    gaussELLgmgm_pmsm_new_sn, \
                    gaussELLgmgm_pmpm_new_sn, \
                    gaussELLmmmm_mmmm_new_sn

                                       
        if not self.cov_dict['split_gauss']:
            gaussELLgggg = gaussELLgggg_sva + gaussELLgggg_mix
            gaussELLgggm = gaussELLgggm_sva + gaussELLgggm_mix
            gaussELLgmgm = gaussELLgmgm_sva + gaussELLgmgm_mix
            gaussELLmmgm = gaussELLmmgm_sva + gaussELLmmgm_mix
            gaussELLmmmm = gaussELLmmmm_sva + gaussELLmmmm_mix
            gaussELLggmm = gaussELLggmm_sva + gaussELLggmm_mix
            if self.csmf:
                return gaussELLgggg, gaussELLgggm, gaussELLggmm, \
                    gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
                    gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn, \
                    cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm
            else:
                return gaussELLgggg, gaussELLgggm, gaussELLggmm, \
                    gaussELLgmgm, gaussELLmmgm, gaussELLmmmm, \
                    gaussELLgggg_sn, gaussELLgmgm_sn, gaussELLmmmm_sn
        elif self.ellrange_spec is not None and calc_prefac:
            gaussELLgggg_ssss_new_sva = 0
            gaussELLgggg_sssp_new_sva = 0
            gaussELLgggg_sspp_new_sva = 0
            gaussELLgggg_spsp_new_sva = 0
            gaussELLgggg_ppsp_new_sva = 0
            gaussELLgggg_pppp_new_sva = 0

            gaussELLgggm_sssm_new_sva = 0
            gaussELLgggm_sspm_new_sva = 0
            gaussELLgggm_spsm_new_sva = 0
            gaussELLgggm_sppm_new_sva = 0
            gaussELLgggm_ppsm_new_sva = 0
            gaussELLgggm_pppm_new_sva = 0

            gaussELLggmm_ssmm_new_sva = 0
            gaussELLggmm_spmm_new_sva = 0
            gaussELLggmm_ppmm_new_sva = 0

            gaussELLgmgm_smsm_new_sva = 0
            gaussELLgmgm_smpm_new_sva = 0
            gaussELLgmgm_pmsm_new_sva = 0
            gaussELLgmgm_pmpm_new_sva = 0

            gaussELLmmgm_mmsm_new_sva = 0
            gaussELLmmgm_mmpm_new_sva = 0

            gaussELLmmmm_mmmm_new_sva = 0


            gaussELLgggg_ssss_new_mix = 0
            gaussELLgggg_sssp_new_mix = 0
            gaussELLgggg_sspp_new_mix = 0
            gaussELLgggg_spsp_new_mix = 0
            gaussELLgggg_ppsp_new_mix = 0
            gaussELLgggg_pppp_new_mix = 0

            gaussELLgggm_sssm_new_mix = 0
            gaussELLgggm_sspm_new_mix = 0
            gaussELLgggm_spsm_new_mix = 0
            gaussELLgggm_sppm_new_mix = 0
            gaussELLgggm_ppsm_new_mix = 0
            gaussELLgggm_pppm_new_mix = 0

            gaussELLggmm_ssmm_new_mix = 0
            gaussELLggmm_spmm_new_mix = 0
            gaussELLggmm_ppmm_new_mix = 0

            gaussELLgmgm_smsm_new_mix = 0
            gaussELLgmgm_smpm_new_mix = 0
            gaussELLgmgm_pmsm_new_mix = 0
            gaussELLgmgm_pmpm_new_mix = 0

            gaussELLmmgm_mmsm_new_mix = 0
            gaussELLmmgm_mmpm_new_mix = 0

            gaussELLmmmm_mmmm_new_mix = 0


            gaussELLgggg_ssss_new_sn = 0
            gaussELLgggg_sssp_new_sn = 0
            gaussELLgggg_sspp_new_sn = 0
            gaussELLgggg_spsp_new_sn = 0
            gaussELLgggg_ppsp_new_sn = 0
            gaussELLgggg_pppp_new_sn = 0

            gaussELLgggm_sssm_new_sn = 0
            gaussELLgggm_sspm_new_sn = 0
            gaussELLgggm_spsm_new_sn = 0
            gaussELLgggm_sppm_new_sn = 0
            gaussELLgggm_ppsm_new_sn = 0
            gaussELLgggm_pppm_new_sn = 0

            gaussELLggmm_ssmm_new_sn = 0
            gaussELLggmm_spmm_new_sn = 0
            gaussELLggmm_ppmm_new_sn = 0

            gaussELLmmgm_mmsm_new_sn = 0
            gaussELLmmgm_mmpm_new_sn = 0


            gaussELLgmgm_smsm_new_sn = 0
            gaussELLgmgm_smpm_new_sn = 0
            gaussELLgmgm_pmsm_new_sn = 0
            gaussELLgmgm_pmpm_new_sn = 0

            gaussELLmmmm_mmmm_new_sn = 0

            if self.gg:
                gaussELLgggg_ssss_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                del gaussELLgggg_sva
                gaussELLgggg_ssss_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                del gaussELLgggg_mix
                gaussELLgggg_ssss_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            True,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sssp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_sspp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_spsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_ppsp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                gaussELLgggg_pppp_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggg_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'clust'
                                                            )
                del gaussELLgggg_sn

            if self.gm and self.gg:
                gaussELLgggm_sssm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sspm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_spsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sppm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_ppsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_pppm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                del gaussELLgggm_sva
                gaussELLgggm_sssm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            True,
                                                            True,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sspm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_spsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_sppm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_ppsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                gaussELLgggm_pppm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgggm_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'ggl'
                                                            )
                del gaussELLgggm_mix
            if self.mm and self.gg:
                gaussELLggmm_ssmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_sva,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_spmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_sva,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_ppmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                del gaussELLggmm_sva
                gaussELLggmm_ssmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_mix,
                                                            True,
                                                            True,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_spmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_mix,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                gaussELLggmm_ppmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLggmm_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'clust',
                                                            'lens'
                                                            )
                del gaussELLggmm_mix
            if self.gm:
                gaussELLgmgm_smsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sva,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sva,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sva,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                del gaussELLgmgm_sva
                gaussELLgmgm_smsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_mix,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_mix,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_mix,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                del gaussELLgmgm_mix
                gaussELLgmgm_smsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_smpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            True,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmsm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                gaussELLgmgm_pmpm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLgmgm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'ggl',
                                                            'ggl'
                                                            )
                del gaussELLgmgm_sn
            if self.mm and self.gm:
                gaussELLmmgm_mmsm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm_sva,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                gaussELLmmgm_mmpm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                del gaussELLmmgm_sva
                gaussELLmmgm_mmsm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm_mix,
                                                            False,
                                                            False,
                                                            True,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                gaussELLmmgm_mmpm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmgm_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'ggl'
                                                            )
                del gaussELLmmgm_mix
            if self.mm:
                gaussELLmmmm_mmmm_new_sva = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm_sva,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                del gaussELLmmmm_sva
                gaussELLmmmm_mmmm_new_mix = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm_mix,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                del gaussELLmmmm_mix
                gaussELLmmmm_mmmm_new_sn = self.__bin_Gaussian(covELLspacesettings,survey_params_dict,
                                                            gaussELLmmmm_sn,
                                                            False,
                                                            False,
                                                            False,
                                                            False,
                                                            'lens',
                                                            'lens'
                                                            )
                del gaussELLmmmm_sn
            return gaussELLgggg_ssss_new_sva, gaussELLgggg_ssss_new_mix, gaussELLgggg_ssss_new_sn, \
                   gaussELLgggg_sssp_new_sva, gaussELLgggg_sssp_new_mix, gaussELLgggg_sssp_new_sn, \
                   gaussELLgggg_sspp_new_sva, gaussELLgggg_sspp_new_mix, gaussELLgggg_sspp_new_sn, \
                   gaussELLgggg_spsp_new_sva, gaussELLgggg_spsp_new_mix, gaussELLgggg_spsp_new_sn, \
                   gaussELLgggg_ppsp_new_sva, gaussELLgggg_ppsp_new_mix, gaussELLgggg_ppsp_new_sn, \
                   gaussELLgggg_pppp_new_sva, gaussELLgggg_pppp_new_mix, gaussELLgggg_pppp_new_sn, \
                   gaussELLgggm_sssm_new_sva, gaussELLgggm_sssm_new_mix, gaussELLgggm_sssm_new_sn, \
                   gaussELLgggm_sspm_new_sva, gaussELLgggm_sspm_new_mix, gaussELLgggm_sspm_new_sn, \
                   gaussELLgggm_spsm_new_sva, gaussELLgggm_spsm_new_mix, gaussELLgggm_spsm_new_sn, \
                   gaussELLgggm_sppm_new_sva, gaussELLgggm_sppm_new_mix, gaussELLgggm_sppm_new_sn, \
                   gaussELLgggm_ppsm_new_sva, gaussELLgggm_ppsm_new_mix, gaussELLgggm_ppsm_new_sn, \
                   gaussELLgggm_pppm_new_sva, gaussELLgggm_pppm_new_mix, gaussELLgggm_pppm_new_sn, \
                   gaussELLggmm_ssmm_new_sva, gaussELLggmm_ssmm_new_mix, gaussELLggmm_ssmm_new_sn, \
                   gaussELLggmm_spmm_new_sva, gaussELLggmm_spmm_new_mix, gaussELLggmm_spmm_new_sn, \
                   gaussELLggmm_ppmm_new_sva, gaussELLggmm_ppmm_new_mix, gaussELLggmm_ppmm_new_sn, \
                   gaussELLgmgm_smsm_new_sva, gaussELLgmgm_smsm_new_mix, gaussELLgmgm_smsm_new_sn, \
                   gaussELLgmgm_smpm_new_sva, gaussELLgmgm_smpm_new_mix, gaussELLgmgm_smpm_new_sn, \
                   gaussELLgmgm_pmsm_new_sva, gaussELLgmgm_pmsm_new_mix, gaussELLgmgm_pmsm_new_sn, \
                   gaussELLgmgm_pmpm_new_sva, gaussELLgmgm_pmpm_new_mix, gaussELLgmgm_pmpm_new_sn, \
                   gaussELLmmgm_mmsm_new_sva, gaussELLmmgm_mmsm_new_mix, gaussELLmmgm_mmsm_new_sn,\
                   gaussELLmmgm_mmpm_new_sva, gaussELLmmgm_mmpm_new_mix, gaussELLmmgm_mmpm_new_sn,\
                   gaussELLmmmm_mmmm_new_sva, gaussELLmmmm_mmmm_new_mix, gaussELLmmmm_mmmm_new_sn
        elif self.csmf:
            return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn, \
                cov_csmf_auto, cov_csmf_gg, cov_csmf_gm, cov_csmf_mm
        else:
            return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
                gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
                gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn

    def __covELL_split_gaussian(self,
                                covELLspacesettings,
                                survey_params_dict,
                                calc_prefac):
        r"""
        Calculates the Gaussian (disconnected) covariance in ell space
        for the specified observables and splits it into sample-variance
        (SVA), shot noise (SN) and SNxSVA(mix) terms.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the redshift spacing used for the line-of-sight
            integration.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        calc_prefac : bool
            default : True
            If True, returns the full Cell-covariance. If False, it
            omits the prefactor 1/(2*ell+1)/f_sky, where f_sky is the
            observed fraction on the sky. Alternatively, all prefactors
            are unity.

        Returns
        -------
        gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn,
        gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn,
        gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn,
        gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn,
        gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn,
        gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn: list of
                                                             arrays
            with shape (ell bins, ell bins,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        To get the full covariance contribution, one needs to add
        gaussELLwxyz_sva + gaussELLwxyz_mix + gaussELLwxyz_sn. Some
        terms are zero by definition (e.g., gaussgggm_sn).

        """

        self.gaussELLmmmm_BB = None
        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if self.mm or self.gm:
            if survey_params_dict['n_eff_lens'] is not None:
                noise_kappa = \
                    np.diag(survey_params_dict['ellipticity_dispersion']**2 \
                    / survey_params_dict['n_eff_lens'] / self.arcmin2torad2)
            else:
                raise Exception("WildError: This shouldn't happen...")


        
        if self.gg or self.gm:
            n_eff_clust = self._n_eff_clust_per_sample(survey_params_dict)
            fraction = getattr(self, 'sample_overlap_fraction', np.eye(self.sample_dim)[:, :, None]*np.ones(self.n_tomo_clust)[None, None, :])
            self.sample_overlap = np.eye(self.sample_dim)[:, :, None]*np.ones(self.n_tomo_clust)[None, None, :]
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    if i_sample != j_sample:
                        n_1, n_2 = n_eff_clust[:, i_sample], n_eff_clust[:, j_sample]
                        self.sample_overlap[i_sample, j_sample] = fraction[i_sample, j_sample]*np.minimum(n_1, n_2)/np.sqrt(n_1*n_2)
            noise_g =np.zeros((self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust))
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        if i_sample == j_sample:
                            noise_g[i_sample, i_sample, i_tomo, i_tomo] = 1 / n_eff_clust[i_tomo, i_sample] / self.arcmin2torad2
                        elif self.sample_overlap[i_sample, j_sample, i_tomo] > 0:
                            noise_g[i_sample, j_sample, i_tomo, i_tomo] = self.sample_overlap[i_sample, j_sample, i_tomo] \
                                / np.sqrt(n_eff_clust[i_tomo, i_sample]*n_eff_clust[i_tomo, j_sample]) / self.arcmin2torad2
       
        # diagonal in ell: the full (ell, ell) matrix, or only the diagonal
        # (shape (ell, 1, ...)) for the projections
        ell_diagonal = np.ones((len(self.ellrange), 1)) if self._compact_ell_cov else np.eye(len(self.ellrange))
        reshape_mat = ell_diagonal[
            :, :, None, None, None, None, None, None]*np.ones((self.sample_dim,self.sample_dim))[None, None, :, :, None, None, None, None]
        if self.gg:
            # Cov_sva(gg^ij gg^kl) = Cgg^ik Cgg^jl + Cgg^il Cgg^jk
            gaussELLgggg_sva = self.Cell_gg[:, :, :, :, None, :, None] \
                * self.Cell_gg[:, :, :, None, :, None, :] \
                + self.Cell_gg[:, :, :, :, None, None, :] \
                * self.Cell_gg[:, :, :, None, :, :, None]
            
            # Cov_mix(gg^ij gg^kl) = Cgg^ik noise_g^jl + Cgg^jl noise_g^ik
            #                      + Cgg^il noise_g^jk + Cgg^jk noise_g^il
            gaussELLgggg_mix = self.Cell_gg[:, :, :, :, None, :, None] \
                * noise_g[None, :, :, None, :, None, :] \
                + self.Cell_gg[:, :, :, None, :, None, :] \
                * noise_g[None, :, :, :, None, :, None] \
                + self.Cell_gg[:, :, :, :, None, None, :]  \
                * noise_g[None, :, :, None, :, :, None] \
                + self.Cell_gg[:, :, :, None, :, :, None]\
                * noise_g[None, :, :, :, None, None, :]
            # Cov_sn(gg^ij gg^kl) = noise_g^ik noise_g^jl
            #                     + noise_g^il noise_g^jk
            gaussELLgggg_sn = \
                noise_g[None, :, :, :, None, :, None] \
                * noise_g[None, :, :, None, :, None, :]  \
                + noise_g[None, :, :, :, None, None, :] \
                * noise_g[None, :, :, None, :, :, None]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_clust, self.n_tomo_clust]
                prefac_gggg = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'])
                gaussELLgggg_sva = prefac_gggg * gaussELLgggg_sva
                gaussELLgggg_mix = prefac_gggg * gaussELLgggg_mix
                gaussELLgggg_sn = prefac_gggg * \
                    gaussELLgggg_sn
                
            gaussELLgggg_sva = gaussELLgggg_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgggg_mix = gaussELLgggg_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
            gaussELLgggg_sn = gaussELLgggg_sn[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn = 0, 0, 0

        if self.gg and self.gm and self.cross_terms:
            # Cov_sva(gg^ij gm^kl) = Cgg^ik Cgm^jl + Cgg^jk Cgm^il
            gaussELLgggm_sva = self.Cell_gg[:, :, :, :, None, :, None] \
                * self.Cell_gm[:, :, None, None, :, None, :] \
                + self.Cell_gg[:, :, :, None, :, :, None] \
                * self.Cell_gm[:, :, None, :, None, None, :]
            # Cov_mix(gg^ij gm^kl) = Cgm^jl noise_g^ik + Cgm^il noise_g^jk
            gaussELLgggm_mix = self.Cell_gm[:, :, None, None, :, None, :] \
                * noise_g[None,:, :, :, None, :, None] \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * noise_g[None, :, :, None, :, :, None]
            gaussELLgggm_sn = 0

            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim, 
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_gggm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'],
                    survey_params_dict['survey_area_ggl'])
                gaussELLgggm_sva = prefac_gggm * gaussELLgggm_sva
                gaussELLgggm_mix = prefac_gggm * gaussELLgggm_mix

            gaussELLgggm_sva = gaussELLgggm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgggm_mix = gaussELLgggm_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
        else:
            gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn = 0, 0, 0

        if self.gg and self.mm and self.cross_terms:
            # Cov_sva(gg^ij mm^kl) = Cgm^ik Cgm^jl + Cgm^il Cgm^jk
            gaussELLggmm_sva = self.Cell_gm[:, :, None ,:, None, :, None] \
                * self.Cell_gm[:, :, None, None, :, None, :] \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * self.Cell_gm[:, :, None, None, :, :, None]
            gaussELLggmm_mix = 0
            gaussELLggmm_sn = 0
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust,
                              self.n_tomo_lens, self.n_tomo_lens]
                prefac_ggmm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_clust'],
                    survey_params_dict['survey_area_lens'])
                gaussELLggmm_sva = prefac_ggmm * gaussELLggmm_sva
            gaussELLggmm_sva = gaussELLggmm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLggmm_sva = gaussELLggmm_sva[:, :, :, :1, :, :, :, :]
        else:
            gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn = 0, 0, 0

        if self.gm:
            # Cov_sva(gm^ij gm^kl) = Cgg^ik Cmm^jl + Cgm^il Cgm^kj
            gaussELLgmgm_sva = (self.Cell_gg[:, :, :, :, None, :, None]
                                * self.Cell_mm[:, :, None, None, :, None, :]) \
                + self.Cell_gm[:, :, None, :, None, None, :] \
                * self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None]
            # Cov_mix(gm^ij gm^kl) = Cgg^ik noise_m^jl + Cmm^jl noise_g^ik
            gaussELLgmgm_mix = self.Cell_gg[:, :, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :] \
                + self.Cell_mm[:, :, None, None, :, None, :] \
                * noise_g[None, :, :, :, None, :, None]
            # Cov_sn(gm^ij gm^kl) = noise_g^ik noise_m^jl
            gaussELLgmgm_sn = noise_g[None, :, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_lens,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_gmgm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_ggl'])
                gaussELLgmgm_sva = prefac_gmgm * gaussELLgmgm_sva
                gaussELLgmgm_mix = prefac_gmgm * gaussELLgmgm_mix
                gaussELLgmgm_sn = prefac_gmgm * \
                    gaussELLgmgm_sn
            gaussELLgmgm_sva = gaussELLgmgm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
                
            gaussELLgmgm_mix = gaussELLgmgm_mix[:, None, :, :, :, :, :, :]\
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLgmgm_sn = gaussELLgmgm_sn[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            
        else:
            gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn = 0, 0, 0

        if self.mm and self.gm and self.cross_terms:
            # Cov_sva(mm^ij gm^kl) = Cmm^lj Cgm^ki + Cmm^li Cgm^kj
            gaussELLmmgm_sva = self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, :, None, :, None] \
                * self.Cell_mm[:, :, None, None, :, None, :] \
                + self.Cell_mm[:, :, None, :, None, None, :] \
                * self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None]
            # Cov_mix(mm^ij gm^kl) = Cgm^ki noise_m^lj + Cgm^kj noise_m^li
            gaussELLmmgm_mix = self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, :, None, :, None] \
                * noise_kappa[None, None, None, None, :, None, :] \
                + self.Cell_gm.transpose(0, 1, 3, 2)[:, None, :, None, :, :, None] \
                * noise_kappa[None, None, None, :, None, None, :]
            gaussELLmmgm_sn = 0

            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens,
                              self.n_tomo_clust, self.n_tomo_lens]
                prefac_mmgm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_lens'],
                    survey_params_dict['survey_area_ggl'])
                gaussELLmmgm_sva = prefac_mmgm * gaussELLmmgm_sva
                gaussELLmmgm_mix = prefac_mmgm * gaussELLmmgm_mix

            gaussELLmmgm_sva = gaussELLmmgm_sva[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLmmgm_mix = gaussELLmmgm_mix[:, None, :, :, :, :, :, :] \
                * reshape_mat  # [:, :, :, None, None, None, None]
            gaussELLmmgm_sva = gaussELLmmgm_sva[:, :, :1, :, :, :, :, :]
            gaussELLmmgm_mix = gaussELLmmgm_mix[:, :, :1, :, :, :, :, :]    
        else:
            gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn = 0, 0, 0

        def mmmm_terms(Cmm):
            # Gaussian mmmm terms for the shear spectrum Cmm (the E- or
            # the B-mode one); the shape noise is the same for both
            # Cov_sva(mm^ij mm^kl) = Cmm^ik Cmm^jl + Cmm^il Cmm^jk
            gaussELLmmmm_sva = Cmm[:, :, None, :, None, :, None] \
                * Cmm[:, None, :, None, :, None, :] \
                + Cmm[:, :, None, :, None, None, :] \
                * Cmm[:, None, :, None, :, :, None]
            # Cov_mix(mm^ij mm^kl) = Cmm^ik noise_m^jl + Cmm^jl noise_m^ik
            #                      + Cmm^il noise_m^jk + Cmm^jk noise_m^il
            gaussELLmmmm_mix = Cmm[:, :, :, None, :, None] \
                * noise_kappa[None, None, None, :, None, :] \
                + Cmm[:, :, None, :, None, :] \
                * noise_kappa[None, None, :, None, :, None] \
                + Cmm[:, :, :, None, None, :] \
                * noise_kappa[None, None, None, :, :, None] \
                + Cmm[:, :, None, :, :, None] \
                * noise_kappa[None, None :, None, None, :]
            # Cov_sn(mm^ij mm^kl) = noise_m^ik noise_m^jl
            #                     + noise_m^il noise_m^jk
            gaussELLmmmm_sn = \
                noise_kappa[:, None, :, None] \
                * noise_kappa[None, :, None, :] \
                + noise_kappa[:, None, None, :] \
                * noise_kappa[None, :, :, None]
            if calc_prefac:
                tomo_shape = [self.sample_dim, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens,
                              self.n_tomo_lens, self.n_tomo_lens]
                prefac_mmmm = self.__calc_prefac_covELL(
                    covELLspacesettings,
                    tomo_shape,
                    survey_params_dict['survey_area_lens'])
                gaussELLmmmm_sva = prefac_mmmm * gaussELLmmmm_sva
                gaussELLmmmm_mix = prefac_mmmm * gaussELLmmmm_mix[:, None, :, :, :, :, :]
                gaussELLmmmm_sn = prefac_mmmm * \
                    gaussELLmmmm_sn[None, None, None, :, :, :, :]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, None, :, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, :, :1, :1, :, :, :, :]
            else:
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, None, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, None, None, :, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sn = gaussELLmmmm_sn[None, None, None, None, :, :, :, :] \
                    * reshape_mat  # [:, :, :, None, None, None, None]
                gaussELLmmmm_sva = gaussELLmmmm_sva[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_mix = gaussELLmmmm_mix[:, :, :1, :1, :, :, :, :]
                gaussELLmmmm_sn = gaussELLmmmm_sn[:, :, :1, :1, :, :, :, :]
            return gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn

        # intrinsic B-modes: the same terms with C_BB (C_EB = 0, so there
        # is no Gaussian EE-BB cross term)
        if self.mm:
            gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = mmmm_terms(self.Cell_mm)
            if self.Cell_mmB is not None:
                self.gaussELLmmmm_BB = list(mmmm_terms(self.Cell_mmB))
        else:
            gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn = 0, 0, 0

        if self.mm or self.gm:
            if len(covELLspacesettings['mult_shear_bias']) < self.n_tomo_lens:
                covELLspacesettings['mult_shear_bias'] = np.zeros(self.n_tomo_lens)
                print("Multiplicative shear bias needs to be given for every tomographic bin. Is set to zero")
            if self.gm:
                self.gaussELLgmgm_sva_mult_shear_bias = np.zeros_like(gaussELLgmgm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_clust):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLgmgm_sva_mult_shear_bias[i_sample, j_sample, i_tomo,j_tomo,k_tomo,l_tomo] = covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
            if self.gm and self.mm and self.cross_terms:
                self.gaussELLmmgm_sva_mult_shear_bias = np.zeros_like(gaussELLmmgm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(1):
                    for j_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_clust):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLmmgm_sva_mult_shear_bias[i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = (covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo])
            if self.mm:
                self.gaussELLmmmm_sva_mult_shear_bias = np.zeros_like(gaussELLmmmm_sva[0,0, :, :, :, :, :, :])
                for i_sample in range(1):
                    for j_sample in range(1):
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(self.n_tomo_lens):
                                for k_tomo in range(self.n_tomo_lens):
                                    for l_tomo in range(self.n_tomo_lens):
                                        self.gaussELLmmmm_sva_mult_shear_bias[i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] =  (covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][k_tomo] 
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][i_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][l_tomo]
                                                                                                                                        + covELLspacesettings['mult_shear_bias'][j_tomo]*covELLspacesettings['mult_shear_bias'][k_tomo])

        if self.est_shear != "C_ell" and self.mm:
            for i_sample in range(1):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLmmmm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLmmmm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
            if self.gaussELLmmmm_BB is not None:
                unique = np.triu(np.ones((self.n_tomo_lens, self.n_tomo_lens)))
                for term in self.gaussELLmmmm_BB[:2]:
                    term *= unique[:, :, None, None]*unique[None, None, :, :]


        if self.est_shear != "C_ell" and self.gg:
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_clust):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLgggg_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLgggg_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and self.gg and self.mm:
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo or l_tomo < k_tomo:
                                        gaussELLggmm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and (self.mm and self.gm and self.cross_terms):
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo:
                                        gaussELLmmgm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLmmgm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.est_shear != "C_ell" and (self.gg and self.gm and self.cross_terms):
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        for j_tomo in range(self.n_tomo_clust):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    if j_tomo < i_tomo:
                                        gaussELLgggm_sva[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
                                        gaussELLgggm_mix[:,:,i_sample, j_sample,i_tomo,j_tomo,k_tomo,l_tomo] = 0.0
        if self.csmf_diagonal_lenses:
            for probes, terms in (('gggg', (gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn)),
                                  ('gggm', (gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn)),
                                  ('ggmm', (gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn)),
                                  ('gmgm', (gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn)),
                                  ('mmgm', (gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn))):
                for term in terms:
                    if isinstance(term, np.ndarray) and np.ndim(term) == 8:
                        term *= self._diagonal_lens_mask(probes)[None, None]

        return gaussELLgggg_sva, gaussELLgggg_mix, gaussELLgggg_sn, \
            gaussELLgggm_sva, gaussELLgggm_mix, gaussELLgggm_sn, \
            gaussELLggmm_sva, gaussELLggmm_mix, gaussELLggmm_sn, \
            gaussELLgmgm_sva, gaussELLgmgm_mix, gaussELLgmgm_sn, \
            gaussELLmmgm_sva, gaussELLmmgm_mix, gaussELLmmgm_sn, \
            gaussELLmmmm_sva, gaussELLmmmm_mix, gaussELLmmmm_sn

    def __calc_prefac_covELL(self,
                             covELLspacesettings,
                             tomo_shape,
                             survey_area,
                             survey_area2=None):
        r"""
        Calculates the prefactor,  1/(2*ell+1)/f_sky, which is used
        if the pure ell-space covariance should be used for inference.


        """
        if covELLspacesettings['ell_type'] == 'lin':
            delta_ellrange = self.ellrange[1]-self.ellrange[0]
        elif covELLspacesettings['ell_type'] == 'log':
            logdelta = np.log10(self.ellrange[1]/self.ellrange[0])/2
            ell_ul_range = \
                np.append(np.log10(self.ellrange) - logdelta,
                          np.log10(self.ellrange[-1]) + logdelta)
            delta_ellrange = 10**ell_ul_range[1:] - 10**ell_ul_range[:-1]
        full_sky_angle = 4*np.pi * self.deg2torad2
        prefac_noarea = full_sky_angle / (2*self.ellrange + 1)/ delta_ellrange
        prefac = None
        if len(survey_area) == 1:
            if survey_area2 is None:
                prefac = prefac_noarea / survey_area[0]
            elif len(survey_area2) == 1:
                prefac = prefac_noarea / max(survey_area[0], survey_area2[0])
        if prefac is not None:
            prefac = prefac[:, None, None, None, None, None, None] \
                * np.ones([len(self.ellrange)] + tomo_shape)
        else:
            prefac = self.__calc_prefac6x2pt_covELL(prefac_noarea,
                                                    tomo_shape,
                                                    survey_area,
                                                    survey_area2)
        return prefac

    def __calc_prefac6x2pt_covELL(self,
                                  prefac_noarea,
                                  tomo_shape,
                                  survey_area,
                                  survey_area2):
        prefac = np.zeros([len(self.ellrange)] + tomo_shape)
        prefac_noarea = prefac_noarea[:, None, None, None, None]
        t_sz = self.tomos_6x2pt_clust
        if survey_area2 is None:
            # gggg combs from ssss to pppp (nothing fixed)
            if tomo_shape[0] == tomo_shape[1]:
                # ssss
                prefac_ssss = prefac_noarea / survey_area[0]
                prefac[:, :t_sz, :t_sz, :t_sz, :t_sz] = prefac_ssss
                # pppp
                prefac_pppp = prefac_noarea / survey_area[1]
                prefac[:, t_sz:, t_sz:, t_sz:, t_sz:] = prefac_pppp
                # spsp + spps + pssp + psps
                prefac_spsp = prefac_noarea / survey_area[2]
                prefac[:, :t_sz, t_sz:, :t_sz, t_sz:] = prefac_spsp
                prefac[:, :t_sz, t_sz:, t_sz:, :t_sz] = prefac_spsp
                prefac[:, t_sz:, :t_sz, :t_sz, t_sz:] = prefac_spsp
                prefac[:, t_sz:, :t_sz, t_sz:, :t_sz] = prefac_spsp
                # sspp + ppss
                prefac_sspp = \
                    prefac_noarea / max(survey_area[0], survey_area[1])
                prefac[:, :t_sz, :t_sz, t_sz:, t_sz:] = prefac_sspp
                prefac[:, t_sz:, t_sz:, :t_sz, :t_sz] = prefac_sspp
                # sssp + ssps + spss + psss
                prefac_sssp = \
                    prefac_noarea / max(survey_area[0], survey_area[2])
                prefac[:, :t_sz, :t_sz, :t_sz, t_sz:] = prefac_sssp
                prefac[:, :t_sz, :t_sz, t_sz:, :t_sz] = prefac_sssp
                prefac[:, :t_sz, t_sz:, :t_sz, :t_sz] = prefac_sssp
                prefac[:, t_sz:, :t_sz, :t_sz, :t_sz] = prefac_sssp
                # sppp + pspp + ppsp + ppps
                prefac_ppps = \
                    prefac_noarea / max(survey_area[1], survey_area[2])
                prefac[:, :t_sz, t_sz:, t_sz:, t_sz:] = prefac_ppps
                prefac[:, t_sz:, :t_sz, t_sz:, t_sz:] = prefac_ppps
                prefac[:, t_sz:, t_sz:, :t_sz, t_sz:] = prefac_ppps
                prefac[:, t_sz:, t_sz:, t_sz:, :t_sz] = prefac_ppps
            # gmgm combs from sPsP to pPpP (P fixed)
            elif tomo_shape[0] != tomo_shape[1]:
                # sPsP
                prefac_sPsP = prefac_noarea / survey_area[0]
                prefac[:, :t_sz, :, :t_sz, :] = prefac_sPsP
                # pPpP
                prefac_pPpP = prefac_noarea / survey_area[1]
                prefac[:, t_sz:, :, t_sz:, :] = prefac_pPpP
                # sPpP + pPsP
                prefac_sPpP = \
                    prefac_noarea / max(survey_area[0], survey_area[1])
                prefac[:, :t_sz, :, t_sz:, :] = prefac_sPpP
                prefac[:, t_sz:, :, :t_sz, :] = prefac_sPpP
            else:
                raise Exception("too stupid for 6x2pt, diagonal case")
        # gggm combs from sssP to pppP (P fixed)
        elif tomo_shape[0] == tomo_shape[1] and \
           tomo_shape[0] == tomo_shape[2]:
            # sssP
            prefac_sssP = \
                prefac_noarea / max(survey_area[0], survey_area2[0])
            prefac[:, :t_sz, :t_sz, :t_sz, :] = prefac_sssP
            # sspP
            prefac_sspP = \
                prefac_noarea / max(survey_area[0], survey_area2[1])
            prefac[:, :t_sz, :t_sz, t_sz:, :] = prefac_sspP
            # ppsP
            prefac_ppsP = \
                prefac_noarea / max(survey_area[1], survey_area2[0])
            prefac[:, t_sz:, t_sz:, :t_sz, :] = prefac_ppsP
            # pppP
            prefac_pppP = \
                prefac_noarea / max(survey_area[1], survey_area2[1])
            prefac[:, t_sz:, t_sz:, t_sz:, :] = prefac_pppP
            # spsP + spsP
            prefac_spsP = \
                prefac_noarea / max(survey_area[2], survey_area2[0])
            prefac[:, :t_sz, t_sz:, :t_sz, :] = prefac_spsP
            prefac[:, t_sz:, :t_sz, :t_sz, :] = prefac_spsP
            # sppP + pspP
            prefac_sppP = \
                prefac_noarea / max(survey_area[2], survey_area2[1])
            prefac[:, :t_sz, t_sz:, t_sz:, :] = prefac_sppP
            prefac[:, t_sz:, :t_sz, t_sz:, :] = prefac_sppP
        # ggmm combs from ssPP to ppPP (P fixed)
        elif tomo_shape[0] == tomo_shape[1] and \
                tomo_shape[2] == tomo_shape[3]:
            # ssPP + PPss
            prefac_ssPP = \
                prefac_noarea / max(survey_area[0], survey_area2[0])
            prefac[:, :t_sz, :t_sz, :, :] = prefac_ssPP
            # ppPP + PPpp
            prefac_ppPP = \
                prefac_noarea / max(survey_area[1], survey_area2[0])
            prefac[:, t_sz:, t_sz:, :, :] = prefac_ppPP
            # spPP + psPP
            prefac_spPP = \
                prefac_noarea / max(survey_area[2], survey_area2[0])
            prefac[:, :t_sz, t_sz:, :, :] = prefac_spPP
            prefac[:, t_sz:, :t_sz, :, :] = prefac_spPP
        # mmgm combs from PPsP to PPpP (P fixed)
        elif tomo_shape[0] == tomo_shape[1] and \
                tomo_shape[0] == tomo_shape[3]:
            # PPsP
            prefac_sPPP = \
                prefac_noarea / max(survey_area[0], survey_area2[0])
            prefac[:, :, :, :t_sz, :] = prefac_sPPP
            # PPpP
            prefac_pPPP = \
                prefac_noarea / max(survey_area[0], survey_area2[1])
            prefac[:, :, :, t_sz:, :] = prefac_pPPP
        else:
            raise Exception("too stupid for 6x2pt, diagonal case")

        return prefac

    def covELL_non_gaussian(self,
                            covELLspacesettings,
                            output_dict,
                            bias_dict,
                            hod_dict,
                            prec,
                            tri_tab):
        r"""
        Calculates the non-Gaussian part of the covariance using
        Limber's approximation for all specified observables in
        ell-space config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.
        output_dict : dictionary
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
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
            'trispec' : dictionary
                Contains precision information about the trispectra, this
                includes k-range and spacing and the desired precision
                limits.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'


        Returns
        -------
        nongauss : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """
        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating non-Gaussian covariance in ell space")
        if getattr(self, 'magnification_active', False):
            print("ConfigWarning: The non-Gaussian covariance does NOT contain the magnification bias " +
                  "(only the C_ell and the Gaussian covariance do).")
        # line-of-sight integrals over the range of all tracers
        self.los_integration_chi = np.copy(self.los_integration_chi_full)
        splines_gggg = [[]for _ in range(self.sample_dim)]
        splines_gggm = [[]for _ in range(self.sample_dim)]
        splines_ggmm = [[]for _ in range(self.sample_dim)]
        splines_gmgm = [[]for _ in range(self.sample_dim)]
        splines_mmgm = [[]for _ in range(self.sample_dim)]
        splines_mmmm = [[]for _ in range(self.sample_dim)]
        
        '''splines_gggm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_ggmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]'''

        zet = self.zet_min
        # We want the trispectrum at least at two zet
        if (self.zet_min + covELLspacesettings['tri_delta_z']) >= self.zet_max:
            covELLspacesettings['tri_delta_z'] = self.zet_max - self.zet_min
        chi_list, idx_z =  [], 0
        
        gggg_z = []
        gggm_z = []
        ggmm_z = []
        gmgm_z = []
        mmgm_z = []
        mmmm_z = []

        zet_list = []
        self.num_cores = min(self.num_cores, 16)
        while zet < self.zet_max:
            zet = self.zet_min + \
                covELLspacesettings['tri_delta_z']*idx_z
            zet = min(zet, self.zet_max)
            if (idx_z == 0):
                chi_list.append(self.los_integration_chi[0])
            else:
                chi_list.append(self.cosmology.comoving_distance(
                    zet).value * self.cosmology.h)
            idx_z += 1
            self.update_mass_func(
                zet, bias_dict, hod_dict, prec)
            aux_tri = self.trispectra(
                bias_dict, hod_dict, prec['hm'], tri_tab)
            gggg_z.append(aux_tri[0])
            gggm_z.append(aux_tri[1])
            ggmm_z.append(aux_tri[2])
            gmgm_z.append(aux_tri[3])
            mmgm_z.append(aux_tri[4])
            mmmm_z.append(aux_tri[5])
            zet_list.append(zet)
        zet += covELLspacesettings['tri_delta_z']
        chi_list.append(self.cosmology.comoving_distance(
            zet).value * self.cosmology.h)
        zet_list.append(zet)

        self.update_mass_func(
            zet, bias_dict, hod_dict, prec)
        aux_tri = self.trispectra(
            bias_dict, hod_dict, prec['hm'], tri_tab)
        gggg_z.append(aux_tri[0])
        gggm_z.append(aux_tri[1])
        ggmm_z.append(aux_tri[2])
        gmgm_z.append(aux_tri[3])
        mmgm_z.append(aux_tri[4])
        mmmm_z.append(aux_tri[5])
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        print('Producing splines for non-Gaussian computation')
        gggg_z = np.array(gggg_z)
        gggm_z = np.array(gggm_z)
        ggmm_z = np.array(ggmm_z)
        gmgm_z = np.array(gmgm_z)
        mmgm_z = np.array(mmgm_z)
        mmmm_z = np.array(mmmm_z)
        log_k = np.log((self.ellrange[None, :] + 0.5)/self.los_integration_chi[:, None])
        log_k_table = np.log(self.mass_func.k)

        def log_tri_spline(trispec_z):
            log_k_nodes, trispec_array = self._pad_log_trispectrum(
                log_k_table, np.log(trispec_z), np.min(log_k), np.max(log_k))
            return RegularGridInterpolator(
                (chi_list, log_k_nodes, log_k_nodes), trispec_array, bounds_error=False, fill_value=None)
        for i_sample in range(self.sample_dim):
            for j_sample in range(self.sample_dim):
                if self.gg:
                    splines_gggg[i_sample].append(log_tri_spline(gggg_z[:,:,:,i_sample,j_sample]))
                if self.gg and self.gm and self.cross_terms:
                    splines_gggm[i_sample].append(log_tri_spline(gggm_z[:,:,:,i_sample,j_sample]))
                if self.gg and self.mm and self.cross_terms:
                    splines_ggmm[i_sample].append(log_tri_spline(ggmm_z[:,:,:,i_sample,j_sample]))
                if self.gm:
                    splines_gmgm[i_sample].append(log_tri_spline(gmgm_z[:,:,:,i_sample,j_sample]))
                if self.gm and self.mm and self.cross_terms:
                    splines_mmgm[i_sample].append(log_tri_spline(mmgm_z[:,:,:,i_sample,j_sample]))
                if self.mm:
                    splines_mmmm[i_sample].append(log_tri_spline(mmmm_z[:,:,:,i_sample,j_sample]))
        mesh_shape = (len(self.los_integration_chi), len(self.ellrange), len(self.ellrange))
        integrand_cache = {}

        def trispec_integrand(splines, i_sample, j_sample):
            key = (id(splines), i_sample, j_sample)
            if key not in integrand_cache:
                integrand_cache.clear()
                values = np.empty(mesh_shape)
                for rows in self._blocks(mesh_shape[0], mesh_shape[1]*mesh_shape[2]):
                    values[rows] = np.exp(self._trilinear_on_mesh(splines[i_sample][j_sample],
                                                                  self.los_integration_chi[rows], log_k[rows]))
                integrand_cache[key] = values
            return integrand_cache[key]
        self.num_cores = self.num_cores_save
        nongaussELLgggg = None
        nongaussELLgggm = None
        nongaussELLggmm = None
        nongaussELLgmgm = None
        nongaussELLmmgm = None
        nongaussELLmmmm = None
        print('Line-of-sight integration for non-Gaussian components')
        zclust_chi = [spline(self.los_integration_chi) for spline in getattr(self, 'spline_zclust', [])]
        lensweight_chi = [spline(self.los_integration_chi) for spline in getattr(self, 'spline_lensweight', [])]
        if self.gg:
            nongaussELLgggg = self._ell_4pt_array('gggg',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust),
                self.ellrange_photo is None, self.ellrange_photo is None, compact=covELLspacesettings['nglimber'])
            los_gggg = _LineOfSightIntegrals(nongaussELLgggg, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_clust):
                                l_tomo_start = k_tomo
                                if self.ellrange_photo is not None:
                                    l_tomo_start = 0
                                for l_tomo in range(l_tomo_start, self.n_tomo_clust):
                                    if self.csmf_diagonal_lenses and (j_sample != k_tomo or l_tomo != k_tomo):
                                        continue
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo], self.chi_min_clust[l_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo], self.chi_max_clust[l_tomo])
                                    if chi_low >= chi_high or (self.clustering_z and (i_tomo >= 2 or k_tomo >= 2)):
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*zclust_chi[i_tomo] * zclust_chi[j_tomo]*zclust_chi[k_tomo]*zclust_chi[l_tomo]
                                    los_gggg.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_gggg, i_sample, j_sample))        
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gggg at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            
            los_gggg.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgggg *= self._pixel_weights(nongaussELLgggg)
            print("")
        else:
            nongaussELLgggg = 0

        if self.gg and self.gm and self.cross_terms:
            nongaussELLgggm = self._ell_4pt_array('gggm',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens),
                self.ellrange_photo is None, False)
            los_gggm = _LineOfSightIntegrals(nongaussELLgggm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                    continue
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*zclust_chi[i_tomo] * zclust_chi[j_tomo]*zclust_chi[k_tomo]*lensweight_chi[l_tomo]
                                    los_gggm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_gggm, i_sample, j_sample))                                    
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gggm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
                
            los_gggm.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgggm *= self._pixel_weights(nongaussELLgggm)
            print("")
        else:
            nongaussELLgggm = 0

        if self.gg and self.mm and self.cross_terms:
            nongaussELLggmm = self._ell_4pt_array('ggmm',
                (self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens),
                self.ellrange_photo is None, True)
            
            los_ggmm = _LineOfSightIntegrals(nongaussELLggmm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*zclust_chi[i_tomo] * zclust_chi[j_tomo]*lensweight_chi[k_tomo]*lensweight_chi[l_tomo]
                                    los_ggmm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_ggmm, i_sample, j_sample))
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance ggmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_ggmm.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLggmm *= self._pixel_weights(nongaussELLggmm)
            print("")
        else:
            nongaussELLggmm = 0

        if self.gm:
            nongaussELLgmgm = self._ell_4pt_array('gmgm',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens),
                False, False)
            los_gmgm = _LineOfSightIntegrals(nongaussELLgmgm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_clust*(self.n_tomo_lens)*self.sample_dim**2
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_lens*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2*self.n_tomo_lens
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_clust):
                        if self.csmf_diagonal_lenses and (i_sample != i_tomo):
                            continue
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                    continue
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1/self.los_integration_chi**6.0*zclust_chi[i_tomo] * lensweight_chi[j_tomo]*zclust_chi[k_tomo]*lensweight_chi[l_tomo]
                                    los_gmgm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_gmgm, i_sample, j_sample))
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance gmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_gmgm.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLgmgm *= self._pixel_weights(nongaussELLgmgm)
            if self.est_shear == "C_ell" and self.ellrange_photo is not None:
                # otherwise added to the Gaussian term in calc_covELL
                adding = self.gaussELLgmgm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_gm[:, None, :, None, :,:, None, None]*self.Cell_gm[None, :, None, :, None, None, :, :])
                nongaussELLgmgm[:, :, :, :, :, :, :, :] = nongaussELLgmgm[:, :, :, :, :, :, :, :] + adding[:, :, :, :, :, :, :, :]
            print("")
        else:
            nongaussELLgmgm = 0

        if self.gm and self.mm and self.cross_terms:
            nongaussELLmmgm = self._ell_4pt_array('mmgm',
                (1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens),
                True, False)
            los_mmgm = _LineOfSightIntegrals(nongaussELLmmgm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_lens*(self.n_tomo_lens + 1)/2*self.sample_dim
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                    continue
                                for l_tomo in range(self.n_tomo_lens):
                                    weight = 1/self.los_integration_chi**6.0*lensweight_chi[i_tomo] * lensweight_chi[j_tomo]*zclust_chi[k_tomo]*lensweight_chi[l_tomo]
                                    los_mmgm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_mmgm, i_sample, j_sample))
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance mmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_mmgm.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLmmgm *= self._pixel_weights(nongaussELLmmgm)
            if self.est_shear == "C_ell" and self.est_ggl == "C_ell" and self.ellrange_photo is not None:
                # otherwise added to the Gaussian term in calc_covELL
                adding = self.gaussELLmmgm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_mm[:, None, :, None, :,:, None, None]*self.Cell_gm[None, :, None, :, None, None, :, :])
                nongaussELLmmgm[:, :, 0, :, :, :, :, :] = nongaussELLmmgm[:, :, 0, :, :, :, :, :] + adding[:, :, 0, :, :, :, :, :]
            print("")
        else:
            nongaussELLmmgm = 0

        if self.mm:
            nongaussELLmmmm = self._ell_4pt_array('mmmm',
                (1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens),
                True, True)
            los_mmmm = _LineOfSightIntegrals(nongaussELLmmmm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_lens*(self.n_tomo_lens + 1)/2*1
            for i_sample in range(1):
                for j_sample in range(1):
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    weight = 1./self.los_integration_chi**6.0*lensweight_chi[i_tomo] * lensweight_chi[j_tomo]*lensweight_chi[k_tomo]*lensweight_chi[l_tomo]
                                    los_mmmm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, trispec_integrand(splines_mmmm, i_sample, j_sample))
                                    
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for nonGaussian term for the '
                                'ell-space covariance mmmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_mmmm.flush()
            if covELLspacesettings['pixelised_cell']:
                nongaussELLmmmm *= self._pixel_weights(nongaussELLmmmm)
            print("")
            if self.est_shear == "C_ell" and self.ellrange_photo is not None:
                # otherwise added to the Gaussian term in calc_covELL
                adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, :, :, :, :, :]*(self.Cell_mm[:, None, :, None, :,:, None, None]*self.Cell_mm[None, :, None, :, None, None, :, :])
                nongaussELLmmmm[:, :, 0, 0, :, :, :, :] = nongaussELLmmmm[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
        else:
            nongaussELLmmmm = 0
        if not covELLspacesettings['nglimber']:
            nongaussELLgggg = self.covELL_non_gaussian_non_Limber(
                covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab, nongaussELLgggg)
        return nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm

    def __get_flat_index_spec(self, i_tomo, j_tomo, case):
        if case == "gg":
            if i_tomo >= self.n_spec and j_tomo >= self.n_spec:
                return 2
            if i_tomo < self.n_spec and j_tomo >= self.n_spec or i_tomo >= self.n_spec and j_tomo < self.n_spec:
                return 1
            else:
                return 0
        if case == "ggl":
            if i_tomo >= self.n_spec:
                return 1
            else:
                return 0


    def covELL_ssc(self,
                   bias_dict,
                   hod_dict,
                   prec,
                   survey_params_dict,
                   covELLspacesettings):
        r"""
        Calculates the super-sample part of the covariance using
        Limber's approximation for all specified observables in
        ell-space config file.

        Parameters
        ----------
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
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
            'trispec' : dictionary
                Contains precision information about the trispectra, this
                includes k-range and spacing and the desired precision
                limits.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.


        Returns
        -------
        ssc : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating the SSC terms in ell space.")
        if getattr(self, 'magnification_active', False):
            print("ConfigWarning: The SSC covariance does NOT contain the magnification bias " +
                  "(only the C_ell and the Gaussian covariance do).")
        # line-of-sight integrals over the range of all tracers
        self.los_integration_chi = np.copy(self.los_integration_chi_full)
        SSCELLgggg = None
        SSCELLgggm = None
        SSCELLggmm = None
        SSCELLgmgm = None
        SSCELLmmgm = None
        SSCELLmmmm = None
        spline_responsePgg = []
        spline_responsePgm = []
        spline_responsePmm = []

        survey_variance_gggg = np.zeros((self._n_areas(survey_params_dict['survey_area_clust']), self._n_areas(survey_params_dict['survey_area_clust']), len(self.los_integration_chi)))
        survey_variance_mmmm = np.zeros((self._n_areas(survey_params_dict['survey_area_lens']), self._n_areas(survey_params_dict['survey_area_lens']), len(self.los_integration_chi)))
        survey_variance_gmgm = np.zeros((self._n_areas(survey_params_dict['survey_area_ggl']), self._n_areas(survey_params_dict['survey_area_ggl']), len(self.los_integration_chi)))
        survey_variance_gggm = np.zeros((self._n_areas(survey_params_dict['survey_area_clust']), self._n_areas(survey_params_dict['survey_area_ggl']), len(self.los_integration_chi)))
        survey_variance_mmgm = np.zeros((self._n_areas(survey_params_dict['survey_area_lens']), self._n_areas(survey_params_dict['survey_area_ggl']), len(self.los_integration_chi)))
        survey_variance_ggmm = np.zeros((self._n_areas(survey_params_dict['survey_area_clust']), self._n_areas(survey_params_dict['survey_area_lens']), len(self.los_integration_chi)))

        
        if self.gg:
            print("Getting survey modes for gggg")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'gg', survey_params_dict)
            ell = ell[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            flat_area_idx = 0
            for i_area in range(len(survey_params_dict['survey_area_clust'])):
                for j_area in range(len(survey_params_dict['survey_area_clust'])):
                    # cross modes of the two footprints normalised by the product of their areas
                    survey_variance_gggg[i_area, j_area, :] = np.sum(power * sum_m_a_lm[flat_area_idx],axis = 1)/(survey_params_dict['survey_area_clust'][i_area]*survey_params_dict['survey_area_clust'][j_area]/self.deg2torad2**2)
                    flat_area_idx += 1

        if self.mm:
            print("Getting survey modes for mmmm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('mm', 'mm', survey_params_dict)
            ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            survey_variance_mmmm[0,0,:] = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_lens']**2/self.deg2torad2**2)

        if self.gm:
            print("Getting survey modes for gmgm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gm', 'gm', survey_params_dict)
            ell = ell[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            flat_area_idx = 0
            for i_area in range(len(survey_params_dict['survey_area_ggl'])):
                for j_area in range(len(survey_params_dict['survey_area_ggl'])):
                    # cross modes of the two footprints normalised by the product of their areas
                    survey_variance_gmgm[i_area, j_area, :] = np.sum(power * sum_m_a_lm[flat_area_idx],axis = 1)/(survey_params_dict['survey_area_ggl'][i_area]*survey_params_dict['survey_area_ggl'][j_area]/self.deg2torad2**2)
                    flat_area_idx += 1

        if self.gg and self.gm and self.cross_terms:
            print("Getting survey modes for gggm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'gm', survey_params_dict)
            ell = ell[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            flat_area_idx = 0
            for i_area in range(len(survey_params_dict['survey_area_clust'])):
                for j_area in range(len(survey_params_dict['survey_area_ggl'])):
                    # cross modes of the two footprints normalised by the product of their areas
                    survey_variance_gggm[i_area, j_area, :] = np.sum(power * sum_m_a_lm[flat_area_idx],axis = 1)/(survey_params_dict['survey_area_clust'][i_area]*survey_params_dict['survey_area_ggl'][j_area]/self.deg2torad2**2)
                    flat_area_idx += 1
            
        if self.gg and self.mm and self.cross_terms:
            print("Getting survey modes for ggmm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('gg', 'mm', survey_params_dict)
            ell = ell[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            flat_area_idx = 0
            for i_area in range(len(survey_params_dict['survey_area_clust'])):
                # cross modes of the two footprints normalised by the product of their areas
                survey_variance_ggmm[i_area, 0, :] = np.sum(power * sum_m_a_lm[flat_area_idx],axis = 1)/(survey_params_dict['survey_area_clust'][i_area]*survey_params_dict['survey_area_lens'][0]/self.deg2torad2**2)
                flat_area_idx += 1
        
        if self.gm and self.mm and self.cross_terms:
            print("Getting survey modes for mmgm")
            ell, sum_m_a_lm = \
                self.calc_a_lm('mm', 'gm', survey_params_dict)
            ell = ell[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
            flat_area_idx = 0
            for j_area in range(len(survey_params_dict['survey_area_ggl'])):
                # cross modes of the two footprints normalised by the product of their areas
                survey_variance_mmgm[0, j_area, :] = np.sum(power * sum_m_a_lm[flat_area_idx],axis = 1)/(survey_params_dict['survey_area_lens'][0]*survey_params_dict['survey_area_ggl'][j_area]/self.deg2torad2**2)
                flat_area_idx += 1


        x_values = np.array(self._chi_ell_points(self.los_integration_chi, self.ellrange))
        if not self.redshift_dep_bias:  
            self.aux_response_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            self.aux_response_gm = np.zeros_like(self.aux_response_gg)
            self.aux_response_mm = np.zeros_like(self.aux_response_gg)
            t0 = time.time()
            for i_chi in range(self.los_interpolation_sampling):
                self.update_mass_func(self.los_z[i_chi], bias_dict, hod_dict, prec)
                self.aux_response_gg[i_chi, :, :], self.aux_response_gm[i_chi, :, :], self.aux_response_mm[i_chi,
                                                                                            :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
                eta = (time.time()-t0) * \
                    (len(self.los_z)/(i_chi+1)-1)
                print('\rPreparations for SSC calculation at '
                    + str(round((i_chi+1)/len(self.los_z)*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
            
            for i_sample in range(self.sample_dim):
                spline_responsePgg.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gg[:, :, i_sample]),bounds_error= False, fill_value = None))
                spline_responsePgm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gm[:, :, i_sample]),bounds_error= False, fill_value = None))
                spline_responsePmm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_mm[:, :, i_sample]),bounds_error= False, fill_value = None))
        else:
            save_mm = self.mm
            self.mm = True
            aux_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            aux_gm = np.zeros_like(aux_gg)
            self.aux_response_gg = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust,
                                        self.n_tomo_clust))
            self.aux_response_gm = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim,
                                        self.n_tomo_clust))
            self.aux_response_mm = np.zeros((len(self.los_chi),
                                        len(self.mass_func.k),
                                        self.sample_dim))
            t0 = time.time()
            for i_chi in range(self.los_interpolation_sampling):
                self.update_mass_func(self.los_z[i_chi], bias_dict, hod_dict, prec)
                
                aux_gg[i_chi, :, :], aux_gm[i_chi, :, :], self.aux_response_mm[i_chi, :, :] = self.powspec_responses(bias_dict, hod_dict, prec['hm'])
                if self.gm or self.gg:
                    bias_at_chi = [self.bias_of_zet[t](self.los_z[i_chi]) for t in range(self.n_tomo_clust)]
                if self.gm:
                    for i_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            self.aux_response_gm[i_chi, :, i_sample, i_tomo] = self.aux_response_mm[i_chi, :, i_sample]
                            self.aux_response_gm[i_chi, :, i_sample, i_tomo] -= self.Pgm[:, i_sample]*bias_at_chi[i_tomo]
                if self.gg:
                    for i_sample in range(self.sample_dim):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(self.n_tomo_clust):
                                self.aux_response_gg[i_chi, :, i_sample, i_tomo, j_tomo] = self.aux_response_mm[i_chi, :, i_sample]
                                self.aux_response_gg[i_chi, :, i_sample, i_tomo, j_tomo] -= (bias_at_chi[i_tomo] + bias_at_chi[j_tomo])*np.diagonal(self.Pgg, axis1 = -2, axis2 =-1)[:,i_sample]
                eta = (time.time()-t0) * \
                    (len(self.los_z)/(i_chi+1)-1)
                print('\rPreparations for SSC calculation at '
                    + str(round((i_chi+1)/len(self.los_z)*100, 1))
                    + '% in ' + str(round((time.time()-t0), 1)) + 'sek  ETA in '
                    + str(round(eta, 1)) + 'sek', end="")
            

            
            responsePmm = np.zeros((self.sample_dim,len(self.los_integration_chi),len(self.ellrange)))
            responsePgm = np.zeros((self.sample_dim*self.n_tomo_clust, len(self.los_integration_chi),len(self.ellrange)))
            responsePgg = np.zeros((self.sample_dim*self.n_tomo_clust*self.n_tomo_clust, len(self.los_integration_chi),len(self.ellrange)))
            for i_sample in range(self.sample_dim):
                spline_responsePmm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_mm[:, :, i_sample]),bounds_error= False, fill_value = None))
                responsePmm[i_sample, :, :] = (spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange))))
                for i_tomo in range(self.n_tomo_clust):
                    spline_responsePgm.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                (self.aux_response_gm[:, :, i_sample, i_tomo]),bounds_error= False, fill_value = None))
                    responsePgm[i_sample*self.n_tomo_clust + i_tomo, :, :] =(spline_responsePgm[i_sample*self.n_tomo_clust + i_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange))))
                    for j_tomo in range(self.n_tomo_clust):        
                        spline_responsePgg.append(RegularGridInterpolator((self.los_chi, np.log(self.mass_func.k)),
                                                        (self.aux_response_gg[:, :, i_sample, i_tomo, j_tomo]),bounds_error= False, fill_value = None))
                        responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo, :, :] = (spline_responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange))))
            self.spline_responsePgg = spline_responsePgg    
            self.spline_responsePgm = spline_responsePgm   
            self.spline_responsePmm = spline_responsePmm

            self.mm = save_mm

        print("")
        print("Calculating SSC contribution in ell space")
        # n(z) and lensing weights on the line-of-sight grid, once per bin
        zclust_chi = [spline(self.los_integration_chi) for spline in getattr(self, 'spline_zclust', [])]
        lensweight_chi = [spline(self.los_integration_chi) for spline in getattr(self, 'spline_lensweight', [])]
        self.update_mass_func(0, bias_dict, hod_dict, prec)
        if self.gg:
            print("At gggg contribution")
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2
            SSCELLgggg = self._ell_4pt_array('gggg',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust),
                self.ellrange_photo is None, self.ellrange_photo is None, compact=covELLspacesettings['nglimber'])
            los_gggg = _FactorisedLineOfSightIntegrals(SSCELLgggg, self._los_weights)
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    if not self.redshift_dep_bias:
                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        P2_response = spline_responsePgg[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))                            
                        if len(survey_variance_gggg[:, 0,0]) == 1:
                            variance = survey_variance_gggg[0, 0, :]
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_clust):
                                l_tomo_start = k_tomo
                                if self.ellrange_photo is not None:
                                    l_tomo_start = 0
                                for l_tomo in range(l_tomo_start, self.n_tomo_clust):
                                    if self.csmf_diagonal_lenses and (j_sample != k_tomo or k_tomo != l_tomo):
                                        continue
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo], self.chi_min_clust[l_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo], self.chi_max_clust[l_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        zclust_chi[i_tomo] * \
                                        zclust_chi[j_tomo] * \
                                        zclust_chi[k_tomo] * \
                                        zclust_chi[l_tomo]
                                    if self.redshift_dep_bias:
                                        P1_response = responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]
                                        P2_response = responsePgg[j_sample*self.n_tomo_clust**2 + k_tomo*self.n_tomo_clust + l_tomo]
                                        if len(survey_variance_gggg[:, 0,0]) == 1:
                                            variance = survey_variance_gggg[0, 0, :]
                                    if len(survey_variance_gggg[:, 0,0]) != 1:
                                        variance = survey_variance_gggg[self.__get_flat_index_spec(i_tomo, j_tomo, "gg"), self.__get_flat_index_spec(k_tomo, l_tomo, "gg"), :]
                                    los_gggg.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, P1_response, P2_response, variance)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance gggg at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_gggg.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLgggg *= self._pixel_weights(SSCELLgggg)
            print("")
        else:
            SSCELLgggg = 0

        if self.gg and self.gm and self.cross_terms:
            print("At gggm contribution")
            SSCELLgggm = self._ell_4pt_array('gggm',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens),
                self.ellrange_photo is None, False)
            los_gggm = _FactorisedLineOfSightIntegrals(SSCELLgggm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim**2)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    if not self.redshift_dep_bias:
                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        if len(survey_variance_gggm[:, 0,0]) == 1 and len(survey_variance_gggm[0, :,0]) == 1:
                            variance = survey_variance_gggm[0, 0, :]
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                    continue
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        zclust_chi[i_tomo] * \
                                        zclust_chi[j_tomo] * \
                                        zclust_chi[k_tomo] * \
                                        lensweight_chi[l_tomo]
                                    if self.redshift_dep_bias:
                                        P1_response = responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]
                                        P2_response = responsePgm[j_sample*self.n_tomo_clust + k_tomo]
                                        if len(survey_variance_gggm[:, 0,0]) == 1 and len(survey_variance_gggm[0, :,0]) == 1:
                                            variance = survey_variance_gggm[0, 0, :]
                                    if len(survey_variance_gggm[:, 0,0]) != 1:
                                        variance = survey_variance_gggm[self.__get_flat_index_spec(i_tomo, j_tomo, "gg"), self.__get_flat_index_spec(k_tomo, l_tomo, "ggl"), :]
                                    los_gggm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, P1_response, P2_response, variance)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance gggm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_gggm.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLgggm *= self._pixel_weights(SSCELLgggm)
            print("")
        else:
            SSCELLgggm = 0

        if self.gg and self.mm and self.cross_terms:
            print("At ggmm contribution")
            SSCELLggmm = self._ell_4pt_array('ggmm',
                (self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens),
                self.ellrange_photo is None, True)
            los_ggmm = _FactorisedLineOfSightIntegrals(SSCELLggmm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = int(self.n_tomo_clust*(self.n_tomo_clust + 1)/2*self.sample_dim)
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_clust*self.sample_dim
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim
            for i_sample in range(self.sample_dim):
                for j_sample in range(1):
                    if not self.redshift_dep_bias:
                        P1_response = spline_responsePgg[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        P2_response = spline_responsePmm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        if len(survey_variance_ggmm[:, 0,0]) == 1:
                            variance = survey_variance_ggmm[0, 0, :]
                    for i_tomo in range(self.n_tomo_clust):
                        j_tomo_start = i_tomo
                        if self.ellrange_photo is not None:
                            j_tomo_start = 0
                        for j_tomo in range(j_tomo_start, self.n_tomo_clust):
                            if self.csmf_diagonal_lenses and (i_sample != i_tomo or i_tomo != j_tomo):
                                continue
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        zclust_chi[i_tomo] * \
                                        zclust_chi[j_tomo] * \
                                        lensweight_chi[k_tomo] * \
                                        lensweight_chi[l_tomo]
                                    if self.redshift_dep_bias:
                                        P1_response = responsePgg[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]
                                        P2_response = responsePmm[j_sample]
                                        if len(survey_variance_ggmm[:, 0,0]) == 1:
                                            variance = survey_variance_ggmm[0, 0, :]
                                    if len(survey_variance_ggmm[:, 0,0]) != 1:
                                        variance = survey_variance_ggmm[self.__get_flat_index_spec(i_tomo, j_tomo, "gg"), 0, :]
                                    los_ggmm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, P1_response, P2_response, variance)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance ggmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_ggmm.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLggmm *= self._pixel_weights(SSCELLggmm)
            print("")
        else:
            SSCELLggmm = 0

        if self.gm:
            print("At gmgm contribution")
            SSCELLgmgm = self._ell_4pt_array('gmgm',
                (self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens),
                False, False)
            los_gmgm = _FactorisedLineOfSightIntegrals(SSCELLgmgm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_clust*(self.n_tomo_lens)*self.sample_dim**2
            if self.ellrange_photo is not None:
                tomos_comb = self.n_tomo_clust*self.n_tomo_lens*self.sample_dim**2
            if self.csmf_diagonal_lenses:
                # only sample i in clustering bin i
                tomos_comb = self.sample_dim**2*self.n_tomo_lens
            for i_sample in range(self.sample_dim):
                for j_sample in range(self.sample_dim):
                    if not self.redshift_dep_bias:
                        P1_response = spline_responsePgm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        if len(survey_variance_gmgm[:, 0,0]) == 1:
                            variance = survey_variance_gmgm[0, 0, :]
                    for i_tomo in range(self.n_tomo_clust):
                        if self.csmf_diagonal_lenses and (i_sample != i_tomo):
                            continue
                        for j_tomo in range(self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                for l_tomo in range(self.n_tomo_lens):
                                    if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                        continue
                                    chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[k_tomo])
                                    chi_high = min(
                                        self.chi_max_clust[i_tomo], self.chi_max_clust[k_tomo])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        zclust_chi[i_tomo] * \
                                        zclust_chi[k_tomo] * \
                                        lensweight_chi[j_tomo] * \
                                        lensweight_chi[l_tomo]
                                    if self.redshift_dep_bias:
                                        P1_response = responsePgm[i_sample*self.n_tomo_clust + i_tomo]
                                        P2_response = responsePgm[j_sample*self.n_tomo_clust + k_tomo]
                                        if len(survey_variance_gmgm[:, 0,0]) == 1:
                                            variance = survey_variance_gmgm[0, 0, :]
                                    if len(survey_variance_gmgm[:, 0,0]) != 1:
                                        variance = survey_variance_gmgm[self.__get_flat_index_spec(i_tomo, j_tomo, "ggl"), self.__get_flat_index_spec(k_tomo, l_tomo, "ggl"), :]
                                    los_gmgm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, P1_response, P2_response, variance)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance gmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_gmgm.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLgmgm *= self._pixel_weights(SSCELLgmgm)
            print("")
        else:
            SSCELLgmgm = 0

        if self.gm and self.mm and self.cross_terms:
            print("At mmgm contribution")
            SSCELLmmgm = self._ell_4pt_array('mmgm',
                (1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens),
                True, False)
            los_mmgm = _FactorisedLineOfSightIntegrals(SSCELLmmgm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = self.n_tomo_lens*(self.n_tomo_lens + 1)/2*self.sample_dim
            
            for i_sample in range(1):
                for j_sample in range(self.sample_dim):
                    if not self.redshift_dep_bias:
                        P1_response = spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        P2_response = spline_responsePgm[j_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                        if len(survey_variance_mmgm[:, 0,0]) == 1:
                            variance = survey_variance_mmgm[0, 0, :]
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and (j_sample != k_tomo):
                                    continue       
                                for l_tomo in range(self.n_tomo_lens):
                                    chi_low = self.chi_min_clust[k_tomo]
                                    chi_high = self.chi_max_clust[k_tomo]
                                    chi_low = max(chi_low, self.los_integration_chi[0])
                                    chi_high = min(chi_high, self.los_integration_chi[-1])
                                    if chi_low >= chi_high:
                                        continue
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        lensweight_chi[i_tomo] * \
                                        zclust_chi[k_tomo] * \
                                        lensweight_chi[j_tomo] * \
                                        lensweight_chi[l_tomo]
                                    if self.redshift_dep_bias:
                                        P1_response = responsePmm[i_sample]
                                        P2_response = responsePgm[j_sample*self.n_tomo_clust + k_tomo]
                                        if len(survey_variance_mmgm[:, 0,0]) == 1:
                                            variance = survey_variance_mmgm[0, 0, :]
                                    if len(survey_variance_mmgm[:, 0,0]) != 1:
                                        variance = survey_variance_mmgm[0, self.__get_flat_index_spec(k_tomo, l_tomo, "gl"), :]
                                    los_mmgm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, P1_response, P2_response, variance)
                                    self.__update_los_integration_chi(
                                        self.chimin, self.chimax, covELLspacesettings)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance mmgm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_mmgm.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLmmgm *= self._pixel_weights(SSCELLmmgm)
            print("")
        else:
            SSCELLmmgm = 0

        if self.mm:
            print("At mmmm contribution")
            SSCELLmmmm = self._ell_4pt_array('mmmm',
                (1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens),
                True, True)
            los_mmmm = _FactorisedLineOfSightIntegrals(SSCELLmmmm, self._los_weights)
            t0, tomos = time.time(), 0
            tomos_comb = (self.n_tomo_lens*(self.n_tomo_lens+1)/2)
            for i_sample in range(1):
                for j_sample in range(1):
                    # the response and the integrand only depend on chi and ell, not on the tomographic bins
                    x_values = np.array(self._chi_ell_points(self.los_integration_chi, self.ellrange))
                    Pmm_response = spline_responsePmm[i_sample]((x_values[0,:],x_values[1,:])).reshape((len(self.los_integration_chi),len(self.ellrange)))
                    variance = survey_variance_mmmm[0, 0, :]
                    for i_tomo in range(self.n_tomo_lens):
                        for j_tomo in range(i_tomo, self.n_tomo_lens):
                            for k_tomo in range(self.n_tomo_lens):
                                for l_tomo in range(k_tomo, self.n_tomo_lens):
                                    weight = 1.0/self.los_integration_chi**6.0 * \
                                        lensweight_chi[i_tomo] * \
                                        lensweight_chi[k_tomo] * \
                                        lensweight_chi[j_tomo] * \
                                        lensweight_chi[l_tomo]                                   
                                    los_mmmm.add((i_sample, j_sample, i_tomo, j_tomo, k_tomo, l_tomo), weight, Pmm_response, Pmm_response, variance)
                            tomos += 1
                            eta = (time.time()-t0)/60 * (tomos_comb/tomos-1)
                            print('\rProjection for SSC term for the '
                                'ell-space covariance mmmm at ' +
                                str(round(tomos/tomos_comb*100, 1)) + '% in ' +
                                str(round((time.time()-t0)/60, 1)) + 'min  ETA in ' +
                                str(round(eta, 1)) + 'min', end="")
            los_mmmm.flush()
            if covELLspacesettings['pixelised_cell']:
                SSCELLmmmm *= self._pixel_weights(SSCELLmmmm)
            print("")       
        else:
            SSCELLmmmm = 0

        return SSCELLgggg, SSCELLgggm, SSCELLggmm, SSCELLgmgm, SSCELLmmgm, SSCELLmmmm

    def covELL_csmf_SN(self):
        r"""
        Calculates the shot noise component of the stellar mass function covariance matrix,
        Cov[Phi_mu^i, Phi_nu^i] = N_mu,nu^i/(Delta log10 M*_mu Delta log10 M*_nu V_mu V_nu)
        for the SMF per dex, Phi_mu^i = N_mu^i/(Delta log10 M*_mu V_mu), with N_mu,nu^i the
        number of galaxies the bins share (N_mu^i for mu = nu, galaxies in the overlap of
        overlapping bins, with the geometric mean of the two volumes).

        Returns
        -------
        smf_sn : array
            with shape (csmf_mass_bins, csmf_mass_bins, csmf_tomo_bins, csmf_tomo_bins)
        """
        amplitude = self.csmf_at_tomo_and_mass/(self.deltalog10M_csmf[:, None]*self.Vmax)
        result = np.eye(len(self.log10csmf_mass_bins))[:,:, None, None]*np.eye(self.n_tomo_csmf)[None, None, :, :]*amplitude[:, None, :, None]
        for i_overlap, (i_mass, j_mass) in enumerate(self.csmf_bin_overlaps):
            for i_tomo in range(self.n_tomo_csmf):
                result[i_mass, j_mass, i_tomo, i_tomo] = self.csmf_overlap_mean[i_overlap, i_tomo] \
                    / (self.deltalog10M_csmf[i_mass]*self.deltalog10M_csmf[j_mass]*np.sqrt(self.Vmax[i_mass, i_tomo]*self.Vmax[j_mass, i_tomo]))
                result[j_mass, i_mass, i_tomo, i_tomo] = result[i_mass, j_mass, i_tomo, i_tomo]
        return result

    def covELL_csmf_SSC(self,
                        covELLspacesettings,
                        survey_params_dict):
        r"""
        Calculates the SSC component of the stellar mass function covariance matrix
        Cov[Phi_mu^i, Phi_nu^j] = int dchi p_mu^i(chi) p_nu^j(chi) Phi~_mu(chi) Phi~_nu(chi) sigma_b^2(chi)/chi^2,
        with p_mu^i the line-of-sight distribution of the galaxies in stellar mass bin mu and
        tomographic bin i (see __set_csmf_windows), sigma_b^2 the survey variance and Phi~_mu
        the bias-weighted SMF per dex averaged over the bin (see _csmf_bin_quadrature).

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            here the number of line-of-sight integration steps.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            To be passed from the read_input method of the Input class.

        Returns
        -------
        smf_ssc : array
            with shape (csmf_mass_bins, csmf_mass_bins, csmf_tomo_bins, csmf_tomo_bins)
        """
        ell, sum_m_a_lm = \
            self.calc_a_lm('gg', 'gg', survey_params_dict)
        if ell is not None:
            ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
            x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
            power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))

        survey_variance_mmmm = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust'][0]**2/self.deg2torad2**2)
        survey_variance_mmmm_spline = UnivariateSpline(
                self.los_integration_chi, survey_variance_mmmm, k=1, s=0, ext=0)
        n_mass = len(self.log10csmf_mass_bins)
        result = np.zeros((n_mass, n_mass, self.n_tomo_csmf, self.n_tomo_csmf))
        for i_tomo in range(self.n_tomo_csmf):
            for j_tomo in range(i_tomo, self.n_tomo_csmf):
                for i_mass in range(n_mass):
                    for j_mass in range(n_mass):
                        if j_tomo == i_tomo and j_mass < i_mass:
                            continue
                        los_chi_csmf = self._csmf_los_chi(i_mass, i_tomo,
                                                          self.csmf_window_chi_range[j_mass, j_tomo, 0],
                                                          self.csmf_window_chi_range[j_mass, j_tomo, 1],
                                                          covELLspacesettings)
                        if los_chi_csmf is None:
                            continue
                        integrand = self._csmf_window(i_mass, i_tomo, los_chi_csmf)*self._csmf_window(j_mass, j_tomo, los_chi_csmf) \
                            * self.phi_tilde_spline[i_mass](los_chi_csmf)*self.phi_tilde_spline[j_mass](los_chi_csmf) \
                            * survey_variance_mmmm_spline(los_chi_csmf)/los_chi_csmf**2
                        result[i_mass, j_mass, i_tomo, j_tomo] = simpson(integrand, x = los_chi_csmf)
                        result[j_mass, i_mass, j_tomo, i_tomo] = result[i_mass, j_mass, i_tomo, j_tomo]
        return result


    def _csmf_bispectrum_splines(self,
                                 probe,
                                 i_mass):
        """
        Splines in (log k, chi) of the logarithm of the count bispectrum
        for the cross-covariance of stellar mass bin i_mass of the SMF with
        the probe ('gg', 'gm' or 'mm'): one per galaxy sample for the
        count-galaxy-galaxy and count-galaxy-matter bispectra, a single one
        for the count-matter bispectrum (cosmic shear, or galaxies without
        an HOD).
        """
        if probe == 'mm' or not getattr(self, 'csmf_galaxy_bispectra', False):
            tables = [self.csmf_count_matter_bispectrum[:, :, i_mass]]
        elif probe == 'gg':
            tables = [self.csmf_count_galaxy_galaxy_bispectrum[:, :, i_mass, i_sample] for i_sample in range(self.sample_dim)]
        else:
            tables = [self.csmf_count_galaxy_matter_bispectrum[:, :, i_mass, i_sample] for i_sample in range(self.sample_dim)]
        return [RegularGridInterpolator((np.log(self.mass_func.k), self.los_chi), np.log(np.maximum(table, np.finfo(float).tiny)),
                                        bounds_error= False, fill_value = None) for table in tables]

    def covELL_csmf_cross_LSS_sva(self,
                                  covELLspacesettings,
                                  survey_params_dict,
                                  compute=True):
        r"""
        Calculates the sample variance term for the stellar mass function cross LSS
        covariance, Cov[Phi_mu^i, C_ab(ell)] = int dchi p_mu^i(chi) q_a(chi) q_b(chi)
        nB_mu(ell/chi)/(A chi^4), with p_mu^i the line-of-sight distribution of the
        galaxies in stellar mass bin mu and tomographic bin i (see __set_csmf_windows),
        q_a,b the clustering or lensing kernels and nB_mu the count matter bispectrum.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant value is the survey area of the clustering
            measurement. To be passed from the read_input method of the
            Input class.

        compute : bool
            if False, the arrays are returned with zeros (the term is not
            requested)

        Returns
        -------
        covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm : list of arrays
            with shapes (number of ell bins, number of smf bins, sample dims, n_tomo_smf, n_tomo_clust/lens, n_tomo_clust/lens)
        """
        area_smf = survey_params_dict['survey_area_clust'][0]/self.deg2torad2
        area_factor = {}
        for probe, key in (('gg', 'survey_area_clust'), ('gm', 'survey_area_ggl'), ('mm', 'survey_area_lens')):
            area_probe = survey_params_dict[key][0]/self.deg2torad2 if survey_params_dict[key] is not None else area_smf
            area_factor[probe] = min(area_smf, area_probe)/(area_smf*area_probe)
        if self.gg:
            covELL_smf_cross_gg = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
            if compute and not self.csmf_auto_only:
                for i_mass in range(len(self.log10csmf_mass_bins)):
                    splines = self._csmf_bispectrum_splines('gg', i_mass)
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(i_tomo, self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and i_tomo != j_tomo:
                                    continue
                                chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                chi_high = min(
                                    self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, chi_low, chi_high, covELLspacesettings)
                                if los_integration_chi_update is None:
                                    continue
                                weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_zclust[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)/los_integration_chi_update**4
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                for i_spline, spline in enumerate(splines):
                                    samples = slice(None) if len(splines) == 1 else i_spline
                                    if self.csmf_diagonal_lenses and len(splines) > 1 and i_spline != i_tomo:
                                        continue
                                    spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                    value = area_factor['gg']*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
                                    covELL_smf_cross_gg[: ,i_mass, samples, i_smf_tomo, i_tomo, j_tomo] = value[:, None] if len(splines) == 1 else value
                                    covELL_smf_cross_gg[: ,i_mass, samples, i_smf_tomo, j_tomo, i_tomo] = covELL_smf_cross_gg[: ,i_mass, samples, i_smf_tomo, i_tomo, j_tomo]
        else:
            covELL_smf_cross_gg = 0

        if self.gm:
            covELL_smf_cross_gm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
            if compute and not self.csmf_auto_only:
                for i_mass in range(len(self.log10csmf_mass_bins)):
                    splines = self._csmf_bispectrum_splines('gm', i_mass)
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(self.n_tomo_lens):
                                los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, self.chi_min_clust[i_tomo], self.chi_max_clust[i_tomo], covELLspacesettings)
                                if los_integration_chi_update is None:
                                    continue
                                weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)/los_integration_chi_update**4
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                for i_spline, spline in enumerate(splines):
                                    samples = slice(None) if len(splines) == 1 else i_spline
                                    if self.csmf_diagonal_lenses and len(splines) > 1 and i_spline != i_tomo:
                                        continue
                                    spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                    value = area_factor['gm']*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
                                    covELL_smf_cross_gm[: ,i_mass, samples, i_smf_tomo, i_tomo, j_tomo] = value[:, None] if len(splines) == 1 else value
        else:
            covELL_smf_cross_gm = 0
        if self.mm:
            covELL_smf_cross_mm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
            if compute and not self.csmf_auto_only:
                for i_mass in range(len(self.log10csmf_mass_bins)):
                    spline = self._csmf_bispectrum_splines('mm', i_mass)[0]
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, 0, np.inf, covELLspacesettings)
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(i_tomo, self.n_tomo_lens):
                                weight = self.spline_lensweight[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)/los_integration_chi_update**4
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                spline_eval = np.exp(spline((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                covELL_smf_cross_mm[: ,i_mass, 0, i_smf_tomo, i_tomo, j_tomo] = area_factor['mm']*simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
                                covELL_smf_cross_mm[: ,i_mass, 0, i_smf_tomo, j_tomo, i_tomo] = covELL_smf_cross_mm[: ,i_mass, 0, i_smf_tomo, i_tomo, j_tomo]
        else:
            covELL_smf_cross_mm = 0
        return covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm


    def covELL_csmf_cross_LSS_ssc(self,
                                  covELLspacesettings,
                                  survey_params_dict,
                                  compute=True):
        r"""
        Calculates the super sample variance term for the stellar mass function cross LSS
        covariance, Cov[Phi_mu^i, C_ab(ell)] = int dchi p_mu^i(chi) q_a(chi) q_b(chi)
        Phi~_mu(chi) dP_ab(ell/chi)/ddelta_b sigma_b^2(chi)/chi^4, with p_mu^i the
        line-of-sight distribution of the galaxies in stellar mass bin mu and tomographic
        bin i (see __set_csmf_windows) and sigma_b^2 the survey variance.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space,
            e.g., ell_min/max and the number of ell-modes to be
            calculated.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.

        compute : bool
            if False, the arrays are returned with zeros (the term is not
            requested)

        Returns
        -------
        covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm : list of arrays
            with shapes (number of ell bins, number of smf bins, sample dims, n_tomo_smf, n_tomo_clust/lens, n_tomo_clust/lens)
        """
        if self.gg:
            covELL_smf_cross_gg = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
            if compute and not self.csmf_auto_only:
                ell, sum_m_a_lm = \
                    self.calc_a_lm('gg', 'gg', survey_params_dict)
                if ell is not None:
                    ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                    x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
                    power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust'][0]**2/self.deg2torad2**2)
                survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)
                for i_mass in range(len(self.log10csmf_mass_bins)):
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(i_tomo, self.n_tomo_clust):
                                if self.csmf_diagonal_lenses and i_tomo != j_tomo:
                                    continue
                                chi_low = max(
                                        self.chi_min_clust[i_tomo], self.chi_min_clust[j_tomo])
                                chi_high = min(
                                    self.chi_max_clust[i_tomo], self.chi_max_clust[j_tomo])
                                los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, chi_low, chi_high, covELLspacesettings)
                                if los_integration_chi_update is None:
                                    continue
                                weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_zclust[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)
                                weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)/los_integration_chi_update**4
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                for i_sample in range(self.sample_dim):
                                    if self.csmf_diagonal_lenses and i_sample != i_tomo:
                                        continue
                                    if self.redshift_dep_bias:
                                        spline_response = self.spline_responsePgg_tomo[i_sample*self.n_tomo_clust**2 + i_tomo*self.n_tomo_clust + j_tomo]
                                    else:
                                        spline_response = self.spline_responsePgg[i_sample]
                                    spline_eval = spline_response((x_values[0,:],x_values[1,:])).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                    covELL_smf_cross_gg[: ,i_mass, i_sample, i_smf_tomo, i_tomo, j_tomo] = simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
                                    covELL_smf_cross_gg[: ,i_mass, i_sample, i_smf_tomo, j_tomo, i_tomo] = covELL_smf_cross_gg[: ,i_mass, i_sample, i_smf_tomo, i_tomo, j_tomo]
        else:
            covELL_smf_cross_gg = 0

        if self.gm:
            covELL_smf_cross_gm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
            if compute and not self.csmf_auto_only:
                # SMF (clustering footprint) crossed with the ggl footprint
                ell, sum_m_a_lm = \
                    self.calc_a_lm('gg', 'gm', survey_params_dict)
                if ell is not None:
                    ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                    x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
                    power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust'][0]*survey_params_dict['survey_area_ggl'][0]/self.deg2torad2**2)
                survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)

                for i_mass in range(len(self.log10csmf_mass_bins)):
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        for i_tomo in range(self.n_tomo_clust):
                            for j_tomo in range(self.n_tomo_lens):
                                los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, self.chi_min_clust[i_tomo], self.chi_max_clust[i_tomo], covELLspacesettings)
                                if los_integration_chi_update is None:
                                    continue
                                weight = self.spline_zclust[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)/los_integration_chi_update**4
                                weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                for i_sample in range(self.sample_dim):
                                    if self.csmf_diagonal_lenses and i_sample != i_tomo:
                                        continue
                                    if self.redshift_dep_bias:
                                        spline_response = self.spline_responsePgm_tomo[i_sample*self.n_tomo_clust + i_tomo]
                                    else:
                                        spline_response = self.spline_responsePgm[i_sample]
                                    spline_eval = spline_response((x_values[0,:],x_values[1,:])).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                    covELL_smf_cross_gm[: ,i_mass, i_sample, i_smf_tomo, i_tomo, j_tomo] = simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)
        else:
            covELL_smf_cross_gm = 0

        if self.mm:
            covELL_smf_cross_mm = np.zeros((len(self.ellrange), len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
            if compute and not self.csmf_auto_only:
                # SMF (clustering footprint) crossed with the lensing footprint
                ell, sum_m_a_lm = \
                    self.calc_a_lm('gg', 'mm', survey_params_dict)
                if ell is not None:
                    ell, sum_m_a_lm = ell[0], sum_m_a_lm[0]
                    x_values = np.array(self._chi_ell_points(self.los_integration_chi, ell, log_k=False))
                    power = np.exp(self.power_mm_lin_spline((x_values[0,:],np.log(x_values[1,:])))).reshape((len(self.los_integration_chi),len(ell)))
                survey_variance = np.sum(power * sum_m_a_lm,axis = 1)/(survey_params_dict['survey_area_clust'][0]*survey_params_dict['survey_area_lens'][0]/self.deg2torad2**2)
                survey_variance_spline = UnivariateSpline(self.los_integration_chi, survey_variance, k=1, s=0, ext=0)

                for i_mass in range(len(self.log10csmf_mass_bins)):
                    for i_smf_tomo in range(self.n_tomo_csmf):
                        los_integration_chi_update = self._csmf_los_chi(i_mass, i_smf_tomo, self.chimin, self.chimax, covELLspacesettings)
                        if los_integration_chi_update is None:
                            continue
                        for i_tomo in range(self.n_tomo_lens):
                            for j_tomo in range(i_tomo, self.n_tomo_lens):
                                weight = self.spline_lensweight[i_tomo](los_integration_chi_update)*self.spline_lensweight[j_tomo](los_integration_chi_update)*self._csmf_window(i_mass, i_smf_tomo, los_integration_chi_update)/los_integration_chi_update**4
                                weight *= survey_variance_spline(los_integration_chi_update)*self.phi_tilde_spline[i_mass](los_integration_chi_update)
                                x_values = np.array(self._chi_ell_points(los_integration_chi_update, self.ellrange)[::-1])
                                spline_eval = np.exp(self.spline_responsePmm((x_values[0,:],x_values[1,:]))).reshape((len(los_integration_chi_update),len(self.ellrange)))
                                covELL_smf_cross_mm[: ,i_mass, :, i_smf_tomo, i_tomo, j_tomo] = simpson(spline_eval*weight[:, None],x = los_integration_chi_update, axis = 0)[:, None]
                                covELL_smf_cross_mm[: ,i_mass, :, i_smf_tomo, j_tomo, i_tomo] = covELL_smf_cross_mm[: ,i_mass, :, i_smf_tomo, i_tomo, j_tomo]
        else:
            covELL_smf_cross_mm = 0
        return covELL_smf_cross_gg, covELL_smf_cross_gm, covELL_smf_cross_mm



    def covELL_non_gaussian_non_Limber(self,
                                       covELLspacesettings,
                                       output_dict,
                                       bias_dict,
                                       hod_dict,
                                       prec,
                                       tri_tab,
                                       nongaussELLgggg):
        r"""
        Calculates the non-Gaussian part of the covariance using the
        full expression. This will slow down the code significantly. (still work in progress)

        Parameters
        ----------
        covELLspacesettings : dictionary
                Specifies the exact details of the projection to ell space,
                e.g., ell_min/max and the number of ell-modes to be
                calculated.
        output_dict : dictionary
            Specifies the full path and name, and the style(s) for the
            output file(s). To be passed from the read_input method of
            the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
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
            'trispec' : dictionary
                Contains precision information about the trispectra, this
                includes k-range and spacing and the desired precision
                limits.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'


        Returns
        -------
        nongauss : list of arrays
            with 6 entries for the observables
                ['gggg', 'gggm', 'ggmm', 'gmgm', 'mmgm', 'mmmm']
            each entry with shape (if given in ini file)
                                  (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        if not self.cov_dict['nongauss'] or covELLspacesettings['nglimber']:
            return 0, 0, 0, 0, 0, 0
        print("Calculating non-Gaussian covariance in ell space using the full projection")
        splines_gggg = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gggm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_ggmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_gmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmgm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]
        splines_mmmm = [[[[] for _ in range(self.sample_dim)] for _ in range(
            len(self.mass_func.k))] for _ in range(len(self.mass_func.k))]

        zet = self.zet_min
        # We want the trispectrum at least at two zet
        if (self.zet_min + covELLspacesettings['tri_delta_z']) >= self.zet_max:
            covELLspacesettings['tri_delta_z'] = self.zet_max - self.zet_min
        trispec_at_z, chi_list, idx_z = [], [], 0
        while zet < self.zet_max:
            zet = self.zet_min + \
                covELLspacesettings['tri_delta_z']*idx_z
            zet = min(zet, self.zet_max)
            if (idx_z == 0):
                chi_list.append(self.los_integration_chi[0])
            else:
                chi_list.append(self.cosmology.comoving_distance(
                    zet).value * self.cosmology.h)
            idx_z += 1
            self.update_mass_func(
                zet, bias_dict, hod_dict, prec)
            trispec_at_z.append(self.trispectra(
                bias_dict, hod_dict, prec['hm'], tri_tab))

        gggg_z = np.zeros(len(chi_list))
        gggm_z = np.zeros(len(chi_list))
        ggmm_z = np.zeros(len(chi_list))
        gmgm_z = np.zeros(len(chi_list))
        mmgm_z = np.zeros(len(chi_list))
        mmmm_z = np.zeros(len(chi_list))
        k_spline = 2
        if len(chi_list) < 4:
            k_spline = 1
        print('Producing splines for non-Gaussian computation')
        for i_k in range(len(self.mass_func.k)):
            for j_k in range(len(self.mass_func.k)):
                for i_sample in range(self.sample_dim):
                    for j_sample in range(self.sample_dim):
                        for i_chi in range(len(chi_list)):
                            if self.gg:
                                gggg_z[i_chi] = trispec_at_z[i_chi][0][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gg and self.gm and self.cross_terms:
                                gggm_z[i_chi] = trispec_at_z[i_chi][1][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gg and self.mm and self.cross_terms:
                                ggmm_z[i_chi] = trispec_at_z[i_chi][2][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gm:
                                gmgm_z[i_chi] = trispec_at_z[i_chi][3][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.gm and self.mm and self.cross_terms:
                                mmgm_z[i_chi] = trispec_at_z[i_chi][4][i_k,
                                                                       j_k, i_sample, j_sample]
                            if self.mm:
                                mmmm_z[i_chi] = trispec_at_z[i_chi][5][i_k,
                                                                       j_k, i_sample, j_sample]
                        splines_gggg[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gggg_z, k=k_spline, s=0, ext=1))
                        splines_gggm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gggm_z, k=k_spline, s=0, ext=1))
                        splines_ggmm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, ggmm_z, k=k_spline, s=0, ext=1))
                        splines_gmgm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, gmgm_z, k=k_spline, s=0, ext=1))
                        splines_mmgm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, mmgm_z, k=k_spline, s=0, ext=1))
                        splines_mmmm[i_k][j_k][i_sample].append(
                            UnivariateSpline(
                                chi_list, mmmm_z, k=k_spline, s=0, ext=1))

        ell_max_ng_nonlimber = 20
        n_ell_ng_nonlimber = 2
        ell_non_tri_non_limber = np.unique(np.geomspace(2,ell_max_ng_nonlimber, n_ell_ng_nonlimber).astype(int)).astype(float)
        nonlimbernongauss = np.zeros((len(ell_non_tri_non_limber),len(ell_non_tri_non_limber),self.sample_dim,self.sample_dim,self.n_tomo_clust,self.n_tomo_clust,self.n_tomo_clust,self.n_tomo_clust))
        
        integral_over_k1 = np.zeros((len(self.ellrange),len(self.mass_func.k),len(self.los_integration_chi), len(self.los_integration_chi)))

        lev = levin.Levin(3, 8, 32, 1e-6, self.integration_intervals)
        t0 = time.time()
        print('Carrying out first integral over two Bessel functions')
        
        for i_ell in range(len(self.ellrange)):
            for i_chi_1 in range(len(self.los_integration_chi)):
                for i_chi_x in range(len(self.los_integration_chi)):
                    for i_k1 in range(len(self.mass_func.k)):
                        integrand = np.zeros_like(self.mass_func.k)
                        for i_k3 in range(len(self.mass_func.k)):
                            integrand[i_k3] = np.sqrt(splines_gggg[i_k1][i_k3][0](self.los_integration_chi[i_chi_1]))*self.mass_func.k[i_k3]**2
                        lev.init_integral(
                            self.mass_func.k, integrand[:, None], True, True)
                        integral_over_k1[i_ell, i_chi_1, i_chi_x, i_k1] = float(lev.double_bessel(self.los_integration_chi[i_chi_1], self.los_integration_chi[i_chi_x], int(ell_non_tri_non_limber[i_ell]), int(ell_non_tri_non_limber[i_ell]), self.mass_func.k[0], self.mass_func.k[-1])[0])

        print('Carrying out second integral over two Bessel functions')
        for i_ell in range(len(self.ellrange)):
            for j_ell in range(i_ell, len(self.ellrange)):
                for i_chi_1 in range(len(self.los_integration_chi)):
                    for i_chi_3 in range(len(self.los_integration_chi)):
                        for i_chi_x in range(len(self.los_integration_chi)):
                            integrand = integral_over_k1[i_ell, i_chi_1, i_chi_x,:]
                            lev.init_integral(
                            self.mass_func.k, integrand[:, None], True, True)
                        

        
        return nongaussELLgggg
 

        

