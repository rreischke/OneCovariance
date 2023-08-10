import sys
import numpy as np
from scipy.signal import correlate

from .cov_discrete_utils import *

class TwoDimGrid:
    
    def __init__(self, shape, pos1start, pos2start, dpix):
        self.pos1start = pos1start
        self.pos2start = pos2start
        self.dpix = dpix
        self.outer_shape = shape[:-2]
        self.grid_shape = shape[-2:]
        self.npix2, self.npix1 = self.grid_shape
        self.len1 = self.dpix*self.npix1
        self.len2 = self.dpix*self.npix2
        dpix2 = self.len2/self.npix2
        assert((max(self.dpix,dpix2)/min(self.dpix,dpix2)-1)<1e-10)
        
        Y, X = np.ogrid[0:self.npix2, 0:self.npix1]
        self.dist = np.hypot(Y - self.npix2/2., X - self.npix1/2.)*self.dpix
        self.azimuthalAngle = np.arctan2((Y-self.npix2/2.),(X-self.npix1/2.))
        self.azimuthalAngle[self.azimuthalAngle<0] += 2*np.pi
        
    def assign_ngp(self, pos1, pos2, weights):
        """
        Maps datapoints to pixelgrid.
        
        Parameters:
        -----------
        pos1 : array
            The x-positions of the objects that are mapped to the grid
        pos2 : array
            The y-positions of the objects that are mapped to the grid
        weights : array
             The weights of the objects that are mapped to the grid
             
        Returns:
        --------
        
        datagrid : array
            with shape (self.npix2, self.npix1)
        """
        
        bins_x = np.linspace(self.pos1start, self.pos1start+self.len1, self.npix1+1)
        bins_y = np.linspace(self.pos2start, self.pos2start+self.len2, self.npix2+1)
        datagrid, _, _ = np.histogram2d(pos1, pos2, bins=[bins_x, bins_y], weights=weights)
        return datagrid.transpose()
    
    def get_gn(self, n, rmin, rmax):  
        """ 
        Only returns region that has nonzeros in it.
        """
        assert(not self.npix2%2)
        assert(not self.npix1%2)
        npix_rmax = int(np.ceil(rmax/self.dpix))
        lox = (self.npix1//2-npix_rmax)
        hix = lox+2*npix_rmax
        loy = (self.npix2//2-npix_rmax)
        hiy = loy+2*npix_rmax
        tmpdist = self.dist[loy:hiy,lox:hix]
        gn = np.zeros((2*npix_rmax,2*npix_rmax),dtype=np.complex)
        sel = np.logical_and(tmpdist<rmax,tmpdist>=rmin)
        gn[sel] = 1.
        gn = gn * np.exp(1.0j*n*self.azimuthalAngle[loy:hiy,lox:hix])
        return gn
        
    def get_gn_big(self, n, rmin, rmax):  
        """
        Returns full region
        """
        gn = np.zeros((self.npix2,self.npix1),dtype=np.complex)
        sel = np.logical_and(self.dist<rmax,self.dist>=rmin)
        gn[sel] = 1.
        gn = gn * np.exp(1.0j*n*self.azimuthalAngle)
        return gn
        
class DiscreteData:
    
    def __init__(self, path_to_data, colname_weight, colname_pos1, 
                 colname_pos2, colname_zbin, isspherical, sigma2_eps, target_patchsize,
                 do_overlap):
        self.path_to_data = path_to_data
        self.colname_weight = colname_weight
        self.colname_pos1 = colname_pos1
        self.colname_pos2 = colname_pos2
        self.colname_zbin = colname_zbin
        self.isspherical = isspherical
        self.sigma2_eps = sigma2_eps
        self.target_patchsize = target_patchsize
        self.do_overlap = do_overlap
        self.patchcats = None
        self.npatches = None
        
        tmpdata = Table.read(self.path_to_data)
        self.weight = tmpdata[self.colname_weight]
        self.pos1 = tmpdata[self.colname_pos1]
        self.pos2 = tmpdata[self.colname_pos2]
        self.zbin = tmpdata[self.colname_zbin]
        self.nbinsz = 1+int(np.max(self.zbin)-np.min(self.zbin))
        
    
    def gen_patches(self, func=cygnus_patches, func_args=None):
        """ 
        Dummy for creating overlapping flat-sky patches from a full-sky catalog
        [pos1_patch1, pos1_patch2, ..., pos1_patchn]
        """
        
        funcs_avail = [cygnus_patches]
        if func is not None and func in funcs_avail:
            self.patchcats = func(**func_args)
        self.npatches = len(self.patchcats.keys())
                
        
    def gen_datagrids(self, elpatch, dpix, rmax=150, forcedivide=2):
        """
        Creates datagrids that are needed for the discrete covariance computation.
        
        Parameters:
        -----------
        elpatch : int
            Number of the patch for which the datagrids are produced
        dpix : float
            Sidelength of the pixels used in the grid in arcmin
        rmax : float (defaults to 150)
            The largest 2pcf scale considered. This is relevant to set in order
            to make the grid large enough s.t. all multipoles can be allocaeted
            without errors
        forcedivide: int (defaults to 2)
            Smallest integer divisor of the number of pixels in any grid dimension
            
        Returns:
        --------
        gridinst : TwoDimGrid
            Instance of a TwoDimGrid class with the underlying grid specifications
        weightfields: list of arrays
            The gridded weightfields. Each element in the list corresponds to a 
            tomographic redshift bin
        weightsqfields: list of arrays
            The gridded fields of the squared weights. Each element in the list 
            corresponds to a tomographic redshift bin
        isinnerfields: list of arrays
            The gridded fields of the boolean value that siginfies whether a galaxy
            is in the interior region of overlapping patches
        """
        weightfields = []
        weightsqfields = [] 
        isinnerfields = []
        forcedivide = 2
        _minx = np.min(self.patchcats[elpatch]["x"])
        _maxx = np.max(self.patchcats[elpatch]["x"])
        _miny = np.min(self.patchcats[elpatch]["y"])
        _maxy = np.max(self.patchcats[elpatch]["y"])
        
        extent = [_minx,_maxx, _miny,_maxy]
        if extent[1] - extent[0] < 2.1*rmax:
            extent[1] = extent[0] + 2.1*rmax
        if extent[3] - extent[2] < 2.1*rmax:
            extent[3] = extent[2] + 2.1*rmax
        n1 = int((extent[1] - extent[0])/dpix)
        n2 = int((extent[3] - extent[2])/dpix)
        extent[1] += dpix*(forcedivide-n1%forcedivide) - ((extent[1] - extent[0])%dpix)
        extent[3] += dpix*(forcedivide-n2%forcedivide) - ((extent[3] - extent[2])%dpix)
        n1 = int((extent[1] - extent[0])/dpix)
        n2 = int((extent[3] - extent[2])/dpix)
        gridinst = TwoDimGrid((n2,n1), extent[0], extent[2], dpix)
        
        for z in range(self.nbinsz):
            selz = self.patchcats[elpatch]["zbin"]==z

            weightfields.append(gridinst.assign_ngp(pos1=self.patchcats[elpatch]["x"][selz],
                                                    pos2=self.patchcats[elpatch]["y"][selz], 
                                                    weights=self.patchcats[elpatch]["weight"][selz]))
            weightsqfields.append(gridinst.assign_ngp(pos1=self.patchcats[elpatch]["x"][selz],
                                                      pos2=self.patchcats[elpatch]["y"][selz], 
                                                      weights=self.patchcats[elpatch]["weight"][selz]**2))
            isinnerfields.append(gridinst.assign_ngp(pos1=self.patchcats[elpatch]["x"][selz],
                                                      pos2=self.patchcats[elpatch]["y"][selz], 
                                                      weights=self.patchcats[elpatch]["incenter"][selz].astype(float)))
        return gridinst, weightfields, weightsqfields, isinnerfields
    
                
class DiscreteCovTHETASpace:
    
    def __init__(self, discrete, xi_spl, bin_edges, nmax, nbinsphi, do_ec, 
                 savepath_triplets=None, loadpath_triplets=None, 
                 dpix_min_force=0.25, terms=["xipxip"]):
        
        self.discrete = discrete
        self.xi_spl = xi_spl
        self.bin_edges = bin_edges
        self.nmax = nmax
        self.nbinsphi = nbinsphi
        self.do_ec = do_ec
        self.savepath_triplets = savepath_triplets
        self.loadpath_triplets = loadpath_triplets
        self.allowed_terms = ["xipxip"]
        for term in terms:
            assert(term in self.allowed_terms)
        self.terms = terms
        self.dpix_min_force = dpix_min_force
        self.nbinsr = len(self.bin_edges)-1
        self.dpix = max(self.dpix_min_force, self.bin_edges[0]/20.)
        self.flatz = gen_flatz(self.discrete.nbinsz)
        
        self.bin_centers = None
        self.tripletcounts = None
        self.paircounts = None
        self.paircounts_sq = None
        
    def mixed_covariance(self):
        """
        Computes the mixed covariance terms for xip. Note that this assumes that the
        relevant pair/triplet counts have already been allocated. 
        
        Returns:
        --------
        
        """
        
        assert(self.bin_centers is not None)
        assert(self.paircounts is not None)
        assert(self.paircounts_sq is not None)
        assert(self.tripletcounts is not None)
        
        sig_eps = np.sqrt(self.discrete.sigma2_eps)
        nbinsz = self.discrete.nbinsz
        nbinsr = len(self.bin_edges)-1
        nzcombi = int((nbinsz*(nbinsz+1))/2)
        allmixedmeas = np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
        allshape= np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
        flatz = self.flatz
        for elz1 in range(nbinsz):
            for elz2 in range(elz1,nbinsz):
                for elz3 in range(nbinsz):
                    for elz4 in range(elz3,nbinsz):
                        start1 = flatz[elz1,elz2]*nbinsr
                        start2 = flatz[elz3,elz4]*nbinsr
                        if elz2==elz4:
                            tmpxispl = lambda r: np.e**self.xi_spl["xip"][flatz[elz1,elz3]](np.log(r))
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz1], 
                                                                   tmpxispl, sig_eps[elz2],
                                                                   self.nbinsphi,
                                                                   self.tripletcounts[elz2,elz1,elz3], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz2,elz4])
                            allmixedmeas[start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed
                            if elz1==elz3:
                                allshape[start1:start1+nbinsr,start2:start2+nbinsr] += nexts

                        if elz2==elz3:
                            tmpxispl = lambda r: np.e**self.xi_spl["xip"][flatz[elz1,elz4]](np.log(r))
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz1], 
                                                                   tmpxispl, 
                                                                   sig_eps[elz2],
                                                                   self.nbinsphi,
                                                                   self.tripletcounts[elz2,elz1,elz4], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz2,elz3])
                            allmixedmeas[start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed
                            if elz1==elz3:
                                allshape[start1:start1+nbinsr,start2:start2+nbinsr] += nexts

                        if elz1==elz4:
                            tmpxispl = lambda r: np.e**self.xi_spl["xip"][flatz[elz2,elz3]](np.log(r))
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz2],
                                                                   tmpxispl, 
                                                                   sig_eps[elz1], 
                                                                   self.nbinsphi,
                                                                   self.tripletcounts[elz1,elz2,elz3], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz1,elz4])
                            allmixedmeas[start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed
                            if elz1==elz3:
                                allshape[start1:start1+nbinsr,start2:start2+nbinsr] += nexts

                        if elz1==elz3:
                            tmpxispl = lambda r: np.e**self.xi_spl["xip"][flatz[elz2,elz4]](np.log(r))
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz2],
                                                                   tmpxispl,
                                                                   sig_eps[elz1],
                                                                   self.nbinsphi,
                                                                   self.tripletcounts[elz1,elz2,elz4], 
                                                                   self.paircounts[elz1,elz2],
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz1,elz3])
                            allmixedmeas[start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed
                            if elz1==elz3:
                                allshape[start1:start1+nbinsr,start2:start2+nbinsr] += nexts

        return allmixedmeas, allshape

    def compute_triplets(self):
        """
        Does either compute and save the relevant pair/triplet counts 
        or loads them in from a file. 
        
        """
        
        # Load triplets from file
        # Does not check for correct format etc!
        if self.loadpath_triplets is not None:
            tripletdata = Table.read(self.loadpath_triplets)
            self.bin_centers = tripletdata["bin_centers"]
            self.paircounts = tripletdata["paircounts"] 
            self.paircounts_sq = tripletdata["paircounts_sq"]
            self.tripletcounts = tripletdata["tripletcounts"]
        
        # Compute triplets and store
        else:
            if self.discrete.npatches is None:
                patchcats = self.gen_patches()
                
            for elp, patchcat in enumerate(self.discrete.patchcats):
                # Build datagrid
                grid, wf, w2f, innerf = self.discrete.gen_datagrids(elpatch=elp, 
                                                            dpix=self.dpix_min_force,
                                                            rmax=self.bin_edges[-1],
                                                            forcedivide=2)
                # Allocate paircounts and multipoles of triplet counts
                nextmultipoles = _compute_w2ww_patch(grid, 
                                                     wf, 
                                                     w2f, 
                                                     self.bin_edges,
                                                     nmax=self.nmax,
                                                     sel_inner=innerf)
                bincenters, paircounts_ret, paircountssq_ret, mixedcounts  = nextmultipoles
                # Optionally perform an edge-correction for the multipoles
                if self.do_ec:
                    mixedcounts = edge_correctiontomo(mixedcounts,
                                                      mixedcounts,
                                                      ret_matrices=False)
                # Transform the multipoles to triplet counts
                tripletcounts = tomomultipoles2triplets(mixedcounts, nphis=self.nbinsphi)
                # Update the pair/triplet counts
                if not elp:
                    allpaircounts_ret = np.zeros_like(paircounts_ret)
                    allpaircountssq_ret = np.zeros_like(paircountssq_ret)
                    alltripletcounts = np.zeros_like(tripletcounts)

                allpaircounts_ret += paircounts_ret
                allpaircountssq_ret += paircountssq_ret
                alltripletcounts += tripletcounts
                
            # Add complete pair/triplet counts to class instance variables
            self.bin_centers = bincenters
            self.paircounts = allpaircounts_ret
            self.paircounts_sq = allpaircountssq_ret
            self.tripletcounts = alltripletcounts

            if self.savepath_triplets is not None:
                tosave = Table()
                tosave["bin_centers"] = bincenters
                tosave["paircounts"] = allpaircounts_ret
                tosave["paircounts_sq"] = allpaircountssq_ret
                tosave["tripletcounts"] = alltripletcounts
                Table.write(tosave, self.savepath_triplets, overwrite=True)
                
        
                        
def _compute_w2ww_patch(inst, weightfields, weightsqfields, binedges, nmax=10, sel_inner=None):
    """
    Computes the pair counts w*w, w^2*w^2, as well as the triplet counts w^2*w*w for some patch
    
    Parameters:
    -----------
    inst : TwoDimGrid
        Instance of TwoDimGrid for which the datagrids were allocated
    weightfields : list
        List of 2d-histogram of the w-field. Each element in the list
        corresponds a fixed photometric redshift bin
    weightsqfields : list
        List of 2d-histogram of the w^2-field. Each element in the list
        corresponds a fixed photometric redshift bin
    binedges : array
        Radial bins on which the doublet counts, as well as triplet multipole
        components are allocated
    nmax : int (defaults to 10)
        The largest multipole considered for the triplet counts
    
    Returns:
    --------
    bincenters : array
    
    paircounts_ret : array
    
    paircountssq_ret : array
    
    mixedcounts : array
    
    """
    nbinsz = len(weightfields)
    nbins = len(binedges)-1

    weights_f = {}
    weightssq_f = {}
    n_nonempty = {}
    sel_nonempty = {}
    for elz in range(nbinsz):
        sel_nonempty[elz] = np.logical_and(weightfields[elz] != 0, sel_inner[elz] > 0)
        n_nonempty[elz] = np.sum(sel_nonempty[elz])    
        weights_f[elz] = weightfields[elz][sel_nonempty[elz]].flatten()
        weightssq_f[elz] = weightsqfields[elz][sel_nonempty[elz]].flatten()
        bincenters = np.zeros((nbinsz,nbins))
        paircounts_ret = np.zeros((nbinsz,nbinsz,nbins))
        paircountssq_ret = np.zeros((nbinsz,nbinsz,nbins))
        mixedcounts = np.zeros((nmax+1, nbinsz, nbinsz, nbinsz, nbins,nbins))
        sys.stdout.write("We have %i/%i (inner) galaxies in patch\n"%(np.sum(weightfields[elz] != 0),np.sum(sel_inner[elz] > 0)))
    for n in range(nmax+1):
        paircounts_f = {} # paircounts_f[zbin_computed][zbin_thinned]
        for elz in range(nbinsz):
            paircounts_f[elz] = {} 
            for elbin in range(nbins):
                sys.stdout.write("\r Computing radial shells (%i/%i) at zbin %i/%i"%(elbin,nbins,elz,nbinsz))
                thisfilter = inst.get_gn(n, binedges[elbin],  binedges[elbin+1])
                nextpaircounts = correlate(weightfields[elz],np.conj(thisfilter),'same','fft')
                for elz2 in range(nbinsz):
                    if elbin==0:    
                        paircounts_f[elz][elz2] = np.zeros(( nbins, n_nonempty[elz2]))
                    _ = nextpaircounts[sel_nonempty[elz2]].flatten()
                    paircounts_f[elz][elz2][elbin] = nextpaircounts[sel_nonempty[elz2]].flatten()
                if n==0:
                    binweight = inst.get_gn_big(0,binedges[elbin],  binedges[elbin+1])
                    bincenters[elz,elbin] = (np.sum(binweight*inst.dist)/np.sum(binweight)).real
                    #paircountssq = correlate(weightsqfields[elz], np.conj(thisfilter),'same','fft')
                    for elz2 in range(nbinsz):
                        paircounts_ret[elz,elz2,elbin] = np.sum(weights_f[elz2]*paircounts_f[elz][elz2][elbin])
                        paircountssq_ret[elz,elz2,elbin] = np.sum(weightssq_f[elz2]*paircounts_f[elz][elz2][elbin])
        sys.stdout.write("\n")

        for elz1 in range(nbinsz):
            for elz2 in range(nbinsz):
                for elz3 in range(nbinsz):
                    sys.stdout.write("\r Collecting triplet counts (%i %i %i / %i)"%(elz1,elz2,elz3,nbinsz))
                    for elb1 in range(nbins):
                        for elb2 in range(nbins):
                            nextmixedcount = np.sum(weightssq_f[elz1]*paircounts_f[elz2][elz1][elb1]*paircounts_f[elz3][elz1][elb2].conj())
                            mixedcounts[n,elz1,elz2,elz3,elb1,elb2] = nextmixedcount

    return bincenters, paircounts_ret, paircountssq_ret, mixedcounts

def tomomultipoles2triplets(multipoles, nphis=100):
    """
    Transforms multipoles w2ww_n to triplet counts
    """
    nmax, nbinsz, _, _, nbinsr, _ = multipoles.shape
    phis = np.linspace(-np.pi,np.pi,nphis)
    triplets = np.zeros((nbinsz, nbinsz, nbinsz, nbinsr, nbinsr, nphis), dtype=np.complex)
    for n in range(nmax):
        for elz1 in range(nbinsz):
            for elz2 in range(nbinsz):
                for elz3 in range(nbinsz):
                    for elb1 in range(nbinsr):
                        for elb2 in range(nbinsr):
                            if n==0:
                                triplets[elz1,elz2,elz3,elb1,elb2] += multipoles[n,elz1,elz2,elz3,elb1,elb2].real*np.ones(nphis)
                            else:
                                triplets[elz1,elz2,elz3,elb1,elb2] += multipoles[n,elz1,elz2,elz3,elb1,elb2]*np.e**(1J*n*phis)
                                triplets[elz1,elz3,elz2,elb2,elb1] += (multipoles[n,elz1,elz2,elz3,elb1,elb2]*np.e**(1J*n*phis)).conj()

    return triplets/nphis

def cov_estimate_single(bin_edges, bin_centers, xi_spline, sigma_eps1, nphibins,
                 multipole_tripletcounts, paircounts_ret12, paircounts_ret34, paircountssq_ret,
                 sigma_eps2=None, N2=None):
    if sigma_eps2 is None:
        sigma_eps2 = sigma_eps1
    sigma2_eps = sigma_eps1**2 + sigma_eps2**2
    _,_,nphibins = multipole_tripletcounts.shape
    dphis = np.linspace(-np.pi,np.pi,nphibins)
    cov_third_multipole = np.zeros((len(bin_centers),len(bin_centers)))
    cov_shape = sigma2_eps**2 * np.diag(paircountssq_ret/(paircounts_ret12*paircounts_ret34))
    for elb1, b1 in enumerate(bin_centers):
        for elb2, b2 in enumerate(bin_centers):
            #print(b1)
            norm = sigma2_eps/(2*paircounts_ret12[elb1]*paircounts_ret34[elb2])
            rconnect = b1**2+b2**2-2*b1*b2*np.cos(dphis)
            cov_third_multipole[elb1,elb2] = norm*np.nansum(multipole_tripletcounts[elb1,elb2]*xi_spline(np.sqrt(rconnect)))
    return cov_shape, cov_third_multipole

def edge_correctiontomo(threepcf_n,threepcf_n_norm,ret_matrices=False):
    """
    Perform edge-corrections on a triplet correlator
    
    Parameters:
    -----------
    threepcf_n : array
        The bare triplet correlator
    threepcf_n_norm : array
        Multipole expansion of the normalization of the triplet correlator
    ret_matrices : bool (defaults to False)
        Flag whether to return the mode-coupling matrices
        
    Returns:
    --------
    threepcf_n_corr : array
        The edge-corrected triplet correlator
        
    """
    
    threepcf_n_corr = np.zeros(threepcf_n.shape, dtype=np.complex)
    nvals, nz, _, _, ntheta, _ = threepcf_n_norm.shape
    nmax = nvals-1
    
    #(11, 6, 6, 6, 20, 20)
    # Required to perform edge correction over full -nmax,..+nmax range
    # --> Need to explicitly recover the negative entries
    mirroredthreepcf_n = np.zeros((2*nmax+1, *(threepcf_n.shape)[1:]), dtype=np.complex)
    mirroredthreepcf_n_norm = np.zeros((2*nmax+1, *(threepcf_n.shape)[1:]), dtype=np.complex)
    mirroredthreepcf_n[nmax:] = threepcf_n
    mirroredthreepcf_n_norm[nmax:] = threepcf_n_norm
    mirroredthreepcf_n[:nmax] = np.transpose(threepcf_n[1:][::-1], axes=(0,1,3,2,5,4))
    mirroredthreepcf_n_norm[:nmax] = np.transpose(threepcf_n_norm[1:][::-1], axes=(0,1,3,2,5,4))
    for thet1 in range(ntheta):
        for thet2 in range(ntheta):
            for elz1 in range(nbinsz):
                for elz2 in range(nbinsz):
                    for elz3 in range(nbinsz):
                        sys.stdout.write("\r%i %i %i %i %i"%(thet1,thet2,elz1,elz2,elz3))
                        thnis3pcfn = threepcf_n[:,elz1,elz2,elz3,thet1,thet2]
                        nextM = _gen_M_matrix(thet1,thet2,mirroredthreepcf_n_norm[:,elz1,elz2,elz3])
                        threepcf_n_corr[:,elz1,elz2,elz3,thet1,thet2] = (np.linalg.inv(nextM)@mirroredthreepcf_n[:,elz1,elz2,elz3,thet1,thet2])[nmax:]
    return threepcf_n_corr

def _gen_M_matrix(thet1,thet2,threepcf_n_norm):
    """
    Allocates mode coupling matrix of the edge corrections
    """
    nvals, ntheta, _ = threepcf_n_norm.shape
    nmax = (nvals-1)//2
    narr = np.arange(-nmax,nmax+1, dtype=np.int)
    nextM = np.zeros((nvals,nvals))
    for ind, ell in enumerate(narr):
        lminusn = ell-narr
        sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
        nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real/threepcf_n_norm[nmax,thet1,thet2].real
    return nextM