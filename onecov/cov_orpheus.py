import hashlib
import os
import tempfile

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d


class OrpheusCovTHETASpace():
    """
    This class calculates the Gaussian covariance of the shear-shear
    correlation functions xi_+/- from the discrete pairs, triplets and
    quadruplets of galaxies of a survey (Schneider et al. 2002, using 
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
        if not hasattr(getattr(orpheus, 'GGCovariance', None), 'get_cov_terms_tomo'):
            raise Exception("ConfigError: [covTHETAspace discrete]: the " +
                            "orpheus package at " + orpheus.__file__ + " does " +
                            "not provide GGCovariance.get_cov_terms_tomo, please " +
                            "update it.")
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
        self.continuous = self.methods[2] == 'continuous'
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
        self.phi_grid = np.geomspace(discrete_dict['phidist_min'],
                                     discrete_dict['phidist_max'],
                                     discrete_dict['phidist_bins'])
        self.varphi_grid = np.linspace(0., 2.*np.pi,
                                       discrete_dict['varphi_bins'],
                                       endpoint=False)

    def get_xi_theta(self):
        """
        Returns the separations in arcmin at which xi_+/- has to be
        tabulated: orpheus evaluates it between the smallest theta bin
        edge (or phidist_min) and phidist_max plus the largest theta.

        Returns
        -------
        xi_theta : array
            with shape ('xi_theta_bins',)
        """
        theta_lo = min(self.theta_ul_bins[0], self.phi_grid[0])
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
        return ra, dec, weight

    def __frompairs_suppression(self, ra, dec, weight):
        """
        Pair-count suppression s(theta) = N_p / N_p^ideal of the footprint
        over the full range of the quadruplet integral, which sets the
        quadruplet density of the method 'frompairs'. It is read from
        'frompairs_suppression_file' if given and otherwise measured with
        orpheus on the sphere on a random subset of the catalogue (cached
        in the multiplet directory).

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

        theta_max = self.phi_grid[-1] + self.theta_ul_bins[-1]
        binsize = .05
        nbins = int(np.ceil(np.log(theta_max/self.settings['quad_theta_min'])/binsize))
        edges = np.geomspace(self.settings['quad_theta_min'], theta_max, nbins + 1)
        centers = np.sqrt(edges[1:]*edges[:-1])
        thin = self.settings['frompairs_thin']
        # the tree of the multiplet measurement, coarsened further up to theta_max
        rmin_pixsize = self.settings['rmin_pixsize']
        tree_resos = list(self.settings['tree_resos'])
        while rmin_pixsize*tree_resos[-1] < theta_max:
            tree_resos.append(2.*tree_resos[-1])
        hashed = hashlib.sha1()
        for arr in (ra, dec, weight):
            hashed.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        hashed.update(repr(('orpheus', edges[0], edges[-1], nbins, thin, rmin_pixsize, tree_resos)).encode())
        fname = os.path.join(self.path_multiplets,
                             "widepairs_%s.npz" % hashed.hexdigest()[:16])
        if os.path.exists(fname):
            data = np.load(fname)
            npairs, nbar_thin = data['pairs'], float(data['nbar_thin'])
        else:
            print("Measuring the pair counts between " + str(edges[0]) + " and " +
                  str(edges[-1]) + " arcmin for the quadruplet density 'frompairs'.")
            sel = np.random.RandomState(12345).rand(len(ra)) < 1./thin
            cat = self.orpheus.ScalarTracerCatalog(pos1=ra[sel], pos2=dec[sel], tracer=np.ones(np.sum(sel)),
                                                   weight=weight[sel], units_pos1='deg', units_pos2='deg',
                                                   geometry='spherical')
            nn = self.orpheus.NNCorrelation(min_sep=edges[0], max_sep=edges[-1], nbinsr=nbins,
                                            sep_units='arcmin', process_spherical=True,
                                            tree_resos=tree_resos, rmin_pixsize=rmin_pixsize,
                                            nthreads=self.num_cores)
            nn.process(cat, dotomo=False)
            # both orders of every pair, as GGCovariance counts them
            npairs = nn.npair[0]
            nbar_thin = np.sum(weight[sel])/self.area
            if self.store_multiplets:
                np.savez(fname, pairs=npairs, edges=edges, nbar_thin=nbar_thin)
        return self.orpheus.GGCovariance.suppression_spline_from_pairs(
            npairs, centers, edges, self.area, nbar_thin, weights_mean=1.)

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
        for idx1, (i_tomo, j_tomo) in enumerate(cov.bp_list):
            for idx2, (k_tomo, l_tomo) in enumerate(cov.bp_list):
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

        # Catalogue and reference density of the multiplets: the
        # catalogue density if measured, otherwise the first bin
        catalog = None
        nbar_ref = self.n_eff[0]
        if self.needs_catalog:
            ra, dec, weight = self.__read_catalog()
            nbar_ref = np.sum(weight)/self.area
            os.makedirs(self.path_multiplets, exist_ok=True)
            catalog = dict(ra=ra, dec=dec, weight=weight,
                           path_save=self.path_multiplets,
                           rmin_pixsize=settings['rmin_pixsize'],
                           tree_resos=list(settings['tree_resos']),
                           npatches=settings['npatches'], nmaxs=settings['nmaxs'],
                           method=settings['tree_method'],
                           shuffle_pix=settings['shuffle_pix'],
                           memory_bound=settings['memory_bound'], verbosity=1)
            print("Read " + str(len(ra)) + " galaxies from " + settings['catalog_file'] +
                  ", their density is " + str(np.round(nbar_ref, 4)) + " arcmin^-2.")
        suppression = None
        if method_quadruplets == 'frompairs':
            # Without a catalogue the suppression comes from its file
            suppression = self.__frompairs_suppression(
                *((ra, dec, weight) if self.needs_catalog else (None, None, None)))

        phi_edges = np.linspace(0., 2.*np.pi, settings['nbins_phi_triplets'] + 1)
        common_kw = dict(phi_centers=.5*(phi_edges[1:] + phi_edges[:-1]), phi_edges=phi_edges,
                         spline_xip=xip_list, spline_xim=xim_list,
                         sigma_eps=self.sigma_eps_total, nbar=self.n_eff,
                         nbar_ref=nbar_ref, area=self.area,
                         weights_mean=1., weights_std=0.,
                         spl_suppression_ww=suppression, phi_grid=self.phi_grid,
                         varphi1_grid=self.varphi_grid, varphi2_grid=self.varphi_grid,
                         store_multiplets=self.store_multiplets,
                         nthreads=self.num_cores)
        if method_quadruplets == 'discrete':
            n_qr = int(round(np.log(settings['quad_theta_max']/settings['quad_theta_min'])
                             /settings['quad_binsize']))
            q_theta_edges = np.geomspace(settings['quad_theta_min'],
                                         settings['quad_theta_max'], n_qr + 1)
            q_phi_edges = np.linspace(0., 2.*np.pi, settings['quad_nbins_phi'] + 1)
            common_kw.update(q_theta_centers=np.sqrt(q_theta_edges[1:]*q_theta_edges[:-1]),
                             q_theta_edges=q_theta_edges,
                             q_phi_centers=.5*(q_phi_edges[1:] + q_phi_edges[:-1]),
                             q_phi_edges=q_phi_edges)
        methods_kw = dict(method_pairs=method_pairs, method_triplets=method_triplets,
                          method_quadruplets=method_quadruplets, catalog=catalog)

        # The covariance is computed on n_coarse bins and refined to the
        # theta bins with a recomputed band around the diagonal
        theta_edges = self.theta_ul_bins
        hybrid = settings['n_coarse'] is not None and not self.continuous
        if hybrid:
            theta_edges = np.geomspace(self.theta_ul_bins[0], self.theta_ul_bins[-1],
                                       settings['n_coarse'] + 1)
        print("Allocating the orpheus GGCovariance (multiplets: " + method_pairs + ", " +
              method_triplets + ", " + method_quadruplets + ")")
        cov = GGCovariance(theta_centers=np.sqrt(theta_edges[1:]*theta_edges[:-1]),
                           theta_edges=theta_edges,
                           triplet_oversample=self.__triplet_oversample(theta_edges),
                           **methods_kw, **common_kw)
        if self.continuous:
            print("Computing the continuous covariance with orpheus")
            cov.get_continuous_tomo_c(verbose=True)
        elif hybrid:
            fine_edges = self.theta_ul_bins
            fine_centers = np.sqrt(fine_edges[1:]*fine_edges[:-1])
            # orpheus builds the fine instance itself if all multiplets are
            # measured and then reuses the coarse quadruplet density
            cov_fine = None
            if self.methods != ('discrete', 'discrete', 'discrete'):
                fine_kw = dict(common_kw)
                if method_quadruplets == 'discrete':
                    fine_kw.update(quadruplets=cov.quadruplets,
                                   q_theta_centers=cov.q_theta_centers)
                cov_fine = GGCovariance(theta_centers=fine_centers, theta_edges=fine_edges,
                                        triplet_oversample=self.__triplet_oversample(fine_edges),
                                        **methods_kw, **fine_kw)
            print("Computing the discrete covariance with orpheus on " +
                  str(settings['n_coarse']) + " bins and refining it to " +
                  str(len(fine_centers)) + " bins")
            slab_width = dict(zip(('pp', 'mm', 'pm'), settings['slab_width']))
            cov = cov.subbin_tomo(fine_centers, fine_edges,
                                  new_pairs=None if cov_fine is None else cov_fine.pairs,
                                  slab_instance=cov_fine, recompute_slab=True,
                                  slab_width=slab_width, recompute_full=True,
                                  verbose=True)
        else:
            print("Computing the discrete covariance with orpheus")

        # Assemble covariance blocks and match to onecov shape
        terms = cov.get_cov_terms_tomo(verbose=True)
        sva, mix, sn = dict(), dict(), dict()
        for name, block in [('xipxip', 'pp'), ('xipxim', 'pm'), ('ximxim', 'mm')]:
            sva[name] = self.__to_theta_space_shape(cov, terms[block]['sva'])
            mix[name] = self.__to_theta_space_shape(cov, terms[block]['mix'])
            sn[name] = self.__to_theta_space_shape(cov, terms[block]['D'])
        return sva, mix, sn
