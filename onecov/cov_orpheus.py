import hashlib
import os
import tempfile

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


def _thinned(GGCovariance):
    """
    Subclass of the orpheus GGCovariance which measures the triplets and
    quadruplets on a random subset of the catalogue, keeping each galaxy
    with probability f. As orpheus excludes multiplets with repeated
    galaxies, a triplet survives with probability f^3 and a quadruplet
    with f^4, so the counts are rescaled by thin_scale^3 and thin_scale^4
    with thin_scale = 1/f. The pairs, which set the shape noise, are
    measured on the full catalogue.
    """
    class ThinnedGGCovariance(GGCovariance):
        def __init__(self, *args, catalog_thin=None, thin_scale=1., **kwargs):
            self._catalog_thin = catalog_thin
            self._thin_scale = thin_scale
            self._instances_thin = None
            super().__init__(*args, **kwargs)

        def computeDiscrete(self, contribution):
            if contribution == 'pairs':
                return super().computeDiscrete(contribution)
            # orpheus measures and caches the multiplets of the catalogue it
            # holds, so the thinned one is swapped in with its own
            # correlators and hash before the cache file is looked up
            full = (self.catalog, self.orpheus_instances, self._catalog_sha1,
                    self._unit_weights)
            self.catalog = self._catalog_thin
            if self._instances_thin is None:
                self.orpheus_instances = {}
                self.set_orpheus_instances([])
                self._instances_thin = (self.orpheus_instances, self._catalog_sha1,
                                        self._unit_weights)
            self.orpheus_instances, self._catalog_sha1, self._unit_weights = \
                self._instances_thin
            try:
                counts = super().computeDiscrete(contribution)
            finally:
                self.catalog, self.orpheus_instances, self._catalog_sha1, \
                    self._unit_weights = full
            # out of place, as orpheus keeps the arrays it read from its cache
            return counts*self._thin_scale**{'triplets': 3, 'quadruplets': 4}[contribution]

    return ThinnedGGCovariance


class OrpheusCovTHETASpace():
    """
    This class calculates the Gaussian covariance of the shear-shear
    correlation functions xi_+/- from the discrete pairs, triplets and
    quadruplets of galaxies of a survey (Schneider et al. 2002), using
    the orpheus package. All parameters that orpheus shares with this code
    (theta bins, effective number densities, ellipticity dispersion,
    survey area, xi_+/- and the number of cores) are passed on from
    here.

    Parameters
    ----------
    discrete_dict : dictionary
        Specifies the discrete covariance, i.e. the method for each
        multiplet, the catalogue, the measurement and integration
        accuracy. To be passed from the read_input method of the Input
        class as obs_dict['THETAspace_discrete'].
    theta_ul_bins : array
        Edges of the logarithmic theta bins of the lensing correlation
        functions in arcmin.
    n_eff : array
        Effective number density of the tomographic lensing bins in
        arcmin^-2.
    sigma_eps : array
        Ellipticity dispersion of a single component for each
        tomographic lensing bin.
    survey_area : float
        Survey area in deg^2.
    num_cores : int
        Number of threads used by orpheus.
    cache_dir : string
        Directory in which the measured multiplets are cached (in the
        subdirectory orpheus/) unless 'path_multiplets' is set. Can be
        None, in which case they only live in a temporary directory.

    Example :
    ---------
    disccov = OrpheusCovTHETASpace(obs_dict['THETAspace_discrete'],
                                   theta_ul_bins, n_eff, sigma_eps,
                                   survey_area, num_cores, cache_dir)
    xi_theta = disccov.get_xi_theta()
    # xip, xim tabulated at xi_theta with shape (len(xi_theta), n, n)
    sva, mix, sn = disccov.calc_covTHETA_discrete(xi_theta, xip, xim)

    """

    def __init__(self,
                 discrete_dict,
                 theta_ul_bins,
                 n_eff,
                 sigma_eps,
                 survey_area,
                 num_cores,
                 cache_dir):
        try:
            import orpheus
        except ImportError:
            raise Exception("ConfigError: [covTHETAspace discrete]: " +
                            "'do_discrete = True' requires the orpheus package " +
                            "(https://github.com/lporth93/orpheus) in the python " +
                            "environment.")
        if not hasattr(getattr(orpheus, 'GGCovariance', None), 'set_quadruplets_from_multipoles'):
            raise Exception("ConfigError: [covTHETAspace discrete]: the " +
                            "orpheus package at " + orpheus.__file__ + " does " +
                            "not provide the GGCovariance with multiplets in Fourier " +
                            "modes, please update it.")
        self.orpheus = orpheus
        self.settings = discrete_dict
        self.theta_ul_bins = np.asarray(theta_ul_bins, dtype=float)
        dlnt = np.diff(np.log(self.theta_ul_bins))
        if not np.allclose(dlnt, dlnt.mean(), rtol=1e-6, atol=0.):
            raise Exception("ConfigError: [covTHETAspace discrete]: orpheus needs " +
                            "theta bins of equal width in ln(theta). With " +
                            "'theta_to_obs = True' these are the fine bins, set " +
                            "[covTHETAspace settings]: 'theta_to_obs_type_lensing = log' " +
                            "(the fine bins then have no other edges than the ends of the filters).")
        self.n_eff = np.atleast_1d(np.asarray(n_eff, dtype=float)).ravel()
        # orpheus expects the total ellipticity dispersion
        self.sigma_eps_total = np.sqrt(2.)*np.atleast_1d(
            np.asarray(sigma_eps, dtype=float)).ravel()
        self.n_tomo = len(self.n_eff)
        self.area = float(survey_area)*60.**2
        self.num_cores = int(num_cores)
        self.methods = (discrete_dict['method_pairs'],
                        discrete_dict['method_triplets'],
                        discrete_dict['method_quadruplets'])
        self.needs_catalog = 'discrete' in self.methods or \
            (self.methods[2] == 'frompairs' and
             discrete_dict['frompairs_suppression_file'] is None)
        self.store_multiplets = discrete_dict['store_multiplets']
        if discrete_dict['path_multiplets'] is not None:
            self.path_multiplets = discrete_dict['path_multiplets']
        elif cache_dir is not None:
            self.path_multiplets = os.path.join(cache_dir, 'orpheus')
        else:
            self.path_multiplets = tempfile.mkdtemp(prefix='orpheus_multiplets_')
            self.store_multiplets = False
        # The catalogue is read here, as its extent sets the default
        # largest separation of the two pairs and with it the xi_pm table
        self.catalog = self.__read_catalog() if self.needs_catalog else None
        phidist_max = discrete_dict['phidist_max']
        if phidist_max is None:
            if self.catalog is None:
                phidist_max = 1e4
            else:
                ra, dec, weight = self.catalog
                phidist_max = 60.*self.orpheus.ScalarTracerCatalog(
                    pos1=ra, pos2=dec, tracer=np.ones_like(ra), units_pos1='deg',
                    units_pos2='deg', geometry='spherical').max_separation()
                print("The largest separation in the catalogue is " +
                      str(np.round(phidist_max, 1)) + " arcmin, the integral of " +
                      "the sample variance ends there ('phidist_max').")
        self.phi_grid = np.geomspace(discrete_dict['phidist_min'], phidist_max,
                                     discrete_dict['phidist_bins'])
        self.varphi_grid = np.linspace(0., 2.*np.pi,
                                       discrete_dict['varphi_bins'],
                                       endpoint=False)

    def get_xi_theta(self):
        """
        Returns the separations in arcmin at which xi_+/- has to be
        tabulated: the mixed term evaluates it on triangles whose third
        side reaches below the theta bins, for which 0.01 arcmin suffices,
        and the sample variance up to phidist_max plus the largest theta.

        Returns
        -------
        xi_theta : array
            with shape ('xi_theta_bins',)
        """
        theta_lo = min(.01, self.theta_ul_bins[0], self.phi_grid[0])
        theta_hi = self.phi_grid[-1] + self.theta_ul_bins[-1]
        return np.geomspace(theta_lo, theta_hi, self.settings['xi_theta_bins'])

    def __xi_spline(self, xi_theta, xi):
        """
        Wraps a tabulated correlation function into a callable which
        interpolates linearly in log theta and vanishes beyond the table.
        """
        spl = interp1d(np.log(xi_theta), xi, bounds_error=False, fill_value=0.)
        def xi_func(theta):
            return spl(np.log(np.asarray(theta, dtype=float)))
        xi_func.x = xi_theta
        return xi_func

    def __triplet_oversample(self, theta_edges):
        """Number of radial sub-bins per theta bin for the triplet counts."""
        if self.settings['triplet_oversample'] is not None:
            oversample = self.settings['triplet_oversample']
        # Not given --> choose .05 in logtheta which usually works ok
        else:
            oversample = int(np.ceil(np.log(theta_edges[1]/theta_edges[0])/.05))
        return oversample

    def __subgrid(self, theta_edges):
        """
        Radial nodes per theta bin of the sample-variance integral, as
        offsets from the bin centres in units of the bin width, spread from
        edge to edge. None for the bin centres alone.
        """
        nodes = self.settings['subgrid_nodes']
        # Not given --> an odd number of nodes at most .1 in logtheta apart
        if nodes is None:
            nodes = 2*int(np.ceil((np.log(theta_edges[1]/theta_edges[0])/.1 - 1.)/2.)) + 1
        if nodes <= 1:
            return None
        center = np.sqrt(theta_edges[0]*theta_edges[1])
        width = theta_edges[1] - theta_edges[0]
        return np.linspace((theta_edges[0] - center)/width, (theta_edges[1] - center)/width, nodes)

    def __read_catalog(self):
        """
        Reads positions (in degree) and weights from the catalogue. The
        weights are normalised to their mean so that the multiplet counts
        are numbers of galaxies.

        Returns
        -------
        ra, dec, weight : arrays
        """
        try:
            data = Table.read(self.settings['catalog_file'])
        except Exception:
            data = Table.read(self.settings['catalog_file'], format='ascii')
        ra = np.asarray(data[self.settings['col_name_pos1']], dtype=float)
        dec = np.asarray(data[self.settings['col_name_pos2']], dtype=float)
        if self.settings['col_name_weight'] is None:
            weight = np.ones_like(ra)
        else:
            weight = np.asarray(data[self.settings['col_name_weight']], dtype=float)
        weight = weight/np.mean(weight)
        print("Read " + str(len(ra)) + " galaxies from " + self.settings['catalog_file'] +
              ", their density is " + str(np.round(np.sum(weight)/self.area, 4)) +
              " arcmin^-2.")
        return ra, dec, weight

    def __frompairs_suppression(self, ra, dec, weight):
        """
        Pair-count suppression s(theta) = N_p / N_p^ideal of the footprint
        over the full range of the quadruplet integral, which sets the
        quadruplet density of the method 'frompairs'. It is read from
        'frompairs_suppression_file' if given and otherwise measured with
        orpheus on the sphere on the (thinned) catalogue (cached in the
        multiplet directory).

        Returns
        -------
        suppression : callable
            s(theta) with theta in arcmin. Its attribute x holds the
            separations it was tabulated on.
        """
        if self.settings['frompairs_suppression_file'] is not None:
            table = np.loadtxt(self.settings['frompairs_suppression_file'])
            # Keep s(theta_min) below and zero above the tabulated range
            spl = interp1d(np.log(table[:, 0]), table[:, 1], bounds_error=False,
                           fill_value=(table[0, 1], 0.))
            def suppression(theta):
                return spl(np.log(np.asarray(theta, dtype=float)))
            suppression.x = table[:, 0]
            return suppression

        theta_min = self.settings['quad_theta_min']
        if theta_min is None:
            theta_min = self.theta_ul_bins[0]
        theta_max = self.phi_grid[-1] + self.theta_ul_bins[-1]
        binsize = .05
        nbins = int(np.ceil(np.log(theta_max/theta_min)/binsize))
        edges = np.geomspace(theta_min, theta_max, nbins + 1)
        centers = np.sqrt(edges[1:]*edges[:-1])
        # the tree of the pair counts, coarsened further up to theta_max
        rmin_pixsize, tree_resos = self.__tree('pairs')
        tree_resos = list(tree_resos)
        while rmin_pixsize*tree_resos[-1] < theta_max:
            tree_resos.append(2.*tree_resos[-1])
        hashed = hashlib.sha1()
        for arr in (ra, dec, weight):
            hashed.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        hashed.update(repr(('orpheus', edges[0], edges[-1], nbins, rmin_pixsize, tree_resos)).encode())
        fname = os.path.join(self.path_multiplets,
                             "widepairs_%s.npz" % hashed.hexdigest()[:16])
        if os.path.exists(fname):
            npairs = np.load(fname)['pairs']
        else:
            print("Measuring the pair counts between " + str(np.round(edges[0], 4)) + " and " +
                  str(np.round(edges[-1], 1)) + " arcmin for the quadruplet density 'frompairs'.")
            cat = self.orpheus.ScalarTracerCatalog(pos1=ra, pos2=dec, tracer=np.ones_like(ra),
                                                   weight=weight, units_pos1='deg', units_pos2='deg',
                                                   geometry='spherical')
            # random cell centres, as the centres of mass put the separations of the cells on the
            # lattice of the pixels, which nothing normalises here
            nn = self.orpheus.NNCorrelation(min_sep=edges[0], max_sep=edges[-1], nbinsr=nbins,
                                            sep_units='arcmin', process_spherical=True,
                                            tree_resos=tree_resos, rmin_pixsize=rmin_pixsize,
                                            shuffle_pix=1, nthreads=self.num_cores)
            nn.process(cat, dotomo=False)
            # both orders of every pair, as GGCovariance counts them
            npairs = nn.npair[0]
            if self.store_multiplets:
                os.makedirs(self.path_multiplets, exist_ok=True)
                np.savez(fname, pairs=npairs, edges=edges)
        return self.orpheus.GGCovariance.suppression_spline_from_pairs(
            npairs, centers, edges, self.area, np.sum(weight)/self.area, weights_mean=1.)

    def __tree(self, multiplet):
        """
        rmin_pixsize and tree_resos of the measurement of one multiplet,
        those of [covTHETAspace discrete] unless set for it alone.
        """
        rmin_pixsize = self.settings['rmin_pixsize_' + multiplet]
        tree_resos = self.settings.get('tree_resos_' + multiplet)
        return (self.settings['rmin_pixsize'] if rmin_pixsize is None else rmin_pixsize,
                self.settings['tree_resos'] if tree_resos is None else tree_resos)

    def __to_theta_space_shape(self, cov, matrix):
        """
        Reorders a (n_bp*n_r, n_bp*n_r) matrix of orpheus, ordered by the
        tomographic bin pairs i <= j, into the shape used for the real
        space covariance, filling both orders of each bin pair.

        Returns
        -------
        array
            with shape (theta bins, theta bins, 1, 1, n_tomo_lens,
                        n_tomo_lens, n_tomo_lens, n_tomo_lens)
        """
        n_r = cov.n_r
        out = np.zeros((n_r, n_r, 1, 1,
                        self.n_tomo, self.n_tomo, self.n_tomo, self.n_tomo))
        for idx1, (i_tomo, j_tomo) in enumerate(cov.tomopairs_list):
            for idx2, (k_tomo, l_tomo) in enumerate(cov.tomopairs_list):
                block = matrix[idx1*n_r:(idx1 + 1)*n_r, idx2*n_r:(idx2 + 1)*n_r]
                for i, j in {(i_tomo, j_tomo), (j_tomo, i_tomo)}:
                    for k, l in {(k_tomo, l_tomo), (l_tomo, k_tomo)}:
                        out[:, :, 0, 0, i, j, k, l] = block
        return out

    def calc_covTHETA_discrete(self,
                               xi_theta,
                               xip,
                               xim):
        """
        Calculates the Gaussian covariance of xi_+/- from the multiplet
        counts specified in [covTHETAspace discrete] on the theta bins of
        the lensing correlation functions.

        Parameters
        ----------
        xi_theta : array
            Separations in arcmin at which xi_+/- is tabulated, see
            get_xi_theta.
        xip, xim : arrays
            xi_+ and xi_- of all lensing bin pairs with shape
            (len(xi_theta), n_tomo_lens, n_tomo_lens).

        Returns
        -------
        sva, mix, sn : dictionaries
            with keys 'xipxip', 'xipxim', 'ximxim', each of shape
            (theta bins, theta bins, 1, 1, n_tomo_lens, n_tomo_lens,
             n_tomo_lens, n_tomo_lens). The shot noise of xipxim is zero.
        """
        settings = self.settings
        method_pairs, method_triplets, method_quadruplets = self.methods
        GGCovariance = self.orpheus.GGCovariance
        bp_list = [(i, j) for i in range(self.n_tomo) for j in range(i, self.n_tomo)]
        xip_list = [(i, j, self.__xi_spline(xi_theta, xip[:, i, j])) for i, j in bp_list]
        xim_list = [(i, j, self.__xi_spline(xi_theta, xim[:, i, j])) for i, j in bp_list]
        theta_edges = self.theta_ul_bins
        subgrid = self.__subgrid(theta_edges)
        common_kw = dict(theta_centers=np.sqrt(theta_edges[1:]*theta_edges[:-1]),
                         theta_edges=theta_edges,
                         triplet_oversample=self.__triplet_oversample(theta_edges),
                         method_pairs=method_pairs, method_triplets=method_triplets,
                         method_quadruplets=method_quadruplets,
                         spline_xip=xip_list, spline_xim=xim_list,
                         sigma_eps=self.sigma_eps_total, nbar=self.n_eff, area=self.area,
                         weights_mean=1., weights_std=0., phi_grid=self.phi_grid,
                         varphi1_grid=self.varphi_grid, varphi2_grid=self.varphi_grid,
                         phi1_subgrid_frac=subgrid, phi2_subgrid_frac=subgrid,
                         nthreads=self.num_cores)

        # Catalogue and reference density of the multiplets: the
        # catalogue density if measured, otherwise the first bin
        nbar_ref = self.n_eff[0]
        catalog = None
        if self.needs_catalog:
            ra, dec, weight = self.catalog
            nbar_ref = np.sum(weight)/self.area
            os.makedirs(self.path_multiplets, exist_ok=True)
            catalog = dict(ra=ra, dec=dec, weight=weight)
            # the settings orpheus takes as None are left at its defaults
            rmin_pixsize, tree_resos = self.__tree('pairs')
            t_rmin_pixsize, t_tree_resos = self.__tree('triplets')
            q_rmin_pixsize, q_tree_resos = self.__tree('quadruplets')
            measure_kw = dict(catalog=catalog, path_save=self.path_multiplets,
                              store_multiplets=self.store_multiplets,
                              nmaxs=settings['nmaxs'], t_nmaxs=settings['nmaxs_triplets'],
                              q_nmaxs=settings['nmaxs_quadruplets'],
                              rmin_pixsize=rmin_pixsize, tree_resos=list(tree_resos),
                              t_rmin_pixsize=t_rmin_pixsize, t_tree_resos=list(t_tree_resos),
                              q_rmin_pixsize=q_rmin_pixsize,
                              q_tree_resos=None if settings['tree_resos_quadruplets'] is None
                              else list(settings['tree_resos_quadruplets']),
                              memory_bound=settings['memory_bound'],
                              q_count_min=settings['q_count_min'])
            common_kw.update({key: value for key, value in measure_kw.items() if value is not None})
        common_kw.update(nbar_ref=nbar_ref)

        # One random subset of the catalogue for the triplets, the
        # quadruplets and the pair-count suppression of 'frompairs'
        thinned = catalog is not None and settings['thinning_factor'] > 1.
        if thinned:
            # fixed seed, so that the cached multiplets are reused
            keep = np.random.default_rng(1234).random(len(ra)) < 1./settings['thinning_factor']
            ra, dec, weight = ra[keep], dec[keep], weight[keep]
            GGCovariance = _thinned(GGCovariance)
            common_kw.update(catalog_thin=dict(ra=ra, dec=dec, weight=weight),
                             thin_scale=np.sum(catalog['weight'])/np.sum(weight))
            print("Measuring the triplets and quadruplets on " + str(np.sum(keep)) +
                  " galaxies (thinning_factor = " + str(settings['thinning_factor']) +
                  "), the pairs on all of them.")
        if method_quadruplets == 'frompairs':
            # Without a catalogue the suppression comes from its file
            common_kw.update(spl_suppression_ww=self.__frompairs_suppression(
                *((ra, dec, weight) if self.needs_catalog else (None, None, None))))
        if method_quadruplets == 'discrete':
            common_kw.update(q_binsize=settings['quad_binsize'])
            if settings['quad_richardson']:
                common_kw.update(q_richardson=True)
            # orpheus bins the density from the first theta edge to the
            # largest separation in the catalogue unless one of them is set
            if settings['quad_theta_min'] is not None or settings['quad_theta_max'] is not None:
                q_lo = theta_edges[0] if settings['quad_theta_min'] is None else settings['quad_theta_min']
                q_hi = self.phi_grid[-1] if settings['quad_theta_max'] is None else settings['quad_theta_max']
                n_qr = int(round(np.log(q_hi/q_lo)/settings['quad_binsize']))
                # an even number, as quad_richardson merges pairs of radial bins
                n_qr += n_qr % 2
                q_theta_edges = np.geomspace(q_lo, q_hi, n_qr + 1)
                common_kw.update(q_theta_centers=np.sqrt(q_theta_edges[1:]*q_theta_edges[:-1]),
                                 q_theta_edges=q_theta_edges)

        print("Computing the discrete covariance with orpheus (multiplets: " + method_pairs +
              ", " + method_triplets + ", " + method_quadruplets + ")")
        cov = GGCovariance(**common_kw)

        # Assemble covariance blocks and match to onecov shape
        terms = cov.get_cov_terms(verbose=True)
        sva, mix, sn = dict(), dict(), dict()
        for name, block in [('xipxip', 'pp'), ('xipxim', 'pm'), ('ximxim', 'mm')]:
            sva[name] = self.__to_theta_space_shape(cov, terms[block]['sva'])
            mix[name] = self.__to_theta_space_shape(cov, terms[block]['mix'])
            sn[name] = self.__to_theta_space_shape(cov, terms[block]['D'])
        return sva, mix, sn
