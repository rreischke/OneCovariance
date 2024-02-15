import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.signal import correlate
import ctypes as ct
from numpy.ctypeslib import ndpointer


try:
    from onecov.cov_discrete_utils import *
except:
    from cov_discrete_utils import *

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
        gn = np.zeros((2*npix_rmax,2*npix_rmax),dtype=complex)
        sel = np.logical_and(tmpdist<rmax,tmpdist>=rmin)
        gn[sel] = 1.
        gn = gn * np.exp(1.0j*n*self.azimuthalAngle[loy:hiy,lox:hix])
        return gn
        
    def get_gn_big(self, n, rmin, rmax):  
        """
        Returns full region
        """
        gn = np.zeros((self.npix2,self.npix1),dtype=complex)
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
        hdul = fits.open(self.path_to_data)
        self.weight = None
        self.pos1 = None
        self.pos2 = None
        self.zbin = None
        self.nbinsz = None
        self.mixed_fail = False
        for i in range(len(hdul) - 1):
            try:
                tmpdata = Table.read(self.path_to_data, hdu = i+1)
                self.weight = tmpdata[self.colname_weight]
                self.pos1 = tmpdata[self.colname_pos1]
                self.pos2 = tmpdata[self.colname_pos2]
                self.zbin = tmpdata[self.colname_zbin]
                self.nbinsz = len(sigma2_eps)
                break
            except:
                continue
        if self.weight is None:
            self.mixed_fail = True
            print("WARNING: Couldnt find",self.colname_weight,"in",self.path_to_data,"proceeding with standard mixed term")
        if self.pos1 is None:
            self.mixed_fail = True
            print("WARNING: Couldnt find",self.colname_pos1,"in",self.path_to_data,"proceeding with standard mixed term")
        if self.pos2 is None:
            self.mixed_fail = True
            print("WARNING: Couldnt find",self.colname_pos2,"in",self.path_to_data,"proceeding with standard mixed term")
        if self.zbin is None:
            self.mixed_fail = True
            print("WARNING: Couldnt find",self.colname_zbin,"in",self.path_to_data,"proceeding with standard mixed term")
        #self.nbinsz = 1+int(np.max(self.zbin)-np.min(self.zbin))
        
    
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
    
    def __init__(self, discrete, xi_spl, bin_edges, nmax, nbinsphi, nsubbins, do_ec, 
                 nthreads=16, savepath_triplets=None, loadpath_triplets=None, 
                 terms=["xipxip","xipxim","ximxim"]):
        
        self.discrete = discrete
        self.xi_spl = xi_spl
        self.bin_edges = bin_edges
        self.nmax = nmax
        self.nbinsphi = nbinsphi
        self.nsubbins = nsubbins
        self.do_ec = do_ec
        self.nthreads=nthreads
        self.savepath_triplets = savepath_triplets
        self.loadpath_triplets = loadpath_triplets
        self.allowed_terms = ["xipxip","xipxim","ximxim"]
        for term in terms:
            assert(term in self.allowed_terms)
        self.terms = terms
        self.nbinsr = len(self.bin_edges)-1
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
        allmixedmeas = {}
        allmixed_no_mat = ()
        for term in self.allowed_terms:
            allmixedmeas[term] = np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
            allmixed_no_mat += (np.zeros((nbinsr,nbinsr,1,1,nbinsz,nbinsz,nbinsz,nbinsz)),) 
        for elz1 in range(nbinsz):
            for elz2 in range(elz1,nbinsz):
                for elz3 in range(nbinsz):
                    for elz4 in range(elz3,nbinsz):
                        start1 = self.flatz[elz1,elz2]*nbinsr
                        start2 = self.flatz[elz3,elz4]*nbinsr
                        if elz2==elz4:
                            tmpxipspl = lambda r: self.xi_spl["xip"][self.flatz[elz1,elz3]](r)
                            tmpximspl = lambda r: self.xi_spl["xim"][self.flatz[elz1,elz3]](r)
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz1], 
                                                                   tmpxipspl, tmpximspl, 
                                                                   sig_eps[elz2],
                                                                   self.nbinsphi,
                                                                   self.nsubbins,
                                                                   self.tripletcounts[elz2,elz1,elz3], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz2,elz4])
                            for elt, term in enumerate(self.allowed_terms):
                                allmixedmeas[term][start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed[term]
                                for ir1 in range(nbinsr):
                                    for ir2 in range(nbinsr):
                                        allmixed_no_mat[elt][ir1,ir2,0,0, elz1, elz2, elz3, elz4] += nextmixed[term][ir1, ir2] 

                        if elz2==elz3:
                            tmpxipspl = lambda r: self.xi_spl["xip"][self.flatz[elz1,elz4]](r)
                            tmpximspl = lambda r: self.xi_spl["xim"][self.flatz[elz1,elz4]](r)
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz1], 
                                                                   tmpxipspl, tmpximspl,  
                                                                   sig_eps[elz2],
                                                                   self.nbinsphi,
                                                                   self.nsubbins,
                                                                   self.tripletcounts[elz2,elz1,elz4], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz2,elz3])
                            for elt, term in enumerate(self.allowed_terms):
                                allmixedmeas[term][start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed[term]
                                for ir1 in range(nbinsr):
                                    for ir2 in range(nbinsr):
                                        allmixed_no_mat[elt][ir1,ir2,0,0, elz1, elz2, elz3, elz4] += nextmixed[term][ir1, ir2] 
                                        
                        if elz1==elz4:
                            tmpxipspl = lambda r: self.xi_spl["xip"][self.flatz[elz2,elz3]](r)
                            tmpximspl = lambda r: self.xi_spl["xim"][self.flatz[elz2,elz3]](r)
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz2],
                                                                   tmpxipspl, tmpximspl,  
                                                                   sig_eps[elz1], 
                                                                   self.nbinsphi,
                                                                   self.nsubbins,
                                                                   self.tripletcounts[elz1,elz2,elz3], 
                                                                   self.paircounts[elz1,elz2], 
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz1,elz4])
                            for elt, term in enumerate(self.allowed_terms):
                                allmixedmeas[term][start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed[term]
                                for ir1 in range(nbinsr):
                                    for ir2 in range(nbinsr):
                                        allmixed_no_mat[elt][ir1,ir2,0,0, elz1, elz2, elz3, elz4] += nextmixed[term][ir1, ir2] 

                        if elz1==elz3:
                            tmpxipspl = lambda r: self.xi_spl["xip"][self.flatz[elz2,elz4]](r)
                            tmpximspl = lambda r: self.xi_spl["xim"][self.flatz[elz2,elz4]](r)
                            nexts, nextmixed = cov_estimate_single(self.bin_edges, 
                                                                   self.bin_centers[elz2],
                                                                   tmpxipspl, tmpximspl,  
                                                                   sig_eps[elz1],
                                                                   self.nbinsphi,
                                                                   self.nsubbins,
                                                                   self.tripletcounts[elz1,elz2,elz4], 
                                                                   self.paircounts[elz1,elz2],
                                                                   self.paircounts[elz3,elz4], 
                                                                   self.paircounts_sq[elz1,elz3])
                            for elt, term in enumerate(self.allowed_terms):
                                allmixedmeas[term][start1:start1+nbinsr,start2:start2+nbinsr] += nextmixed[term]
                                for ir1 in range(nbinsr):
                                    for ir2 in range(nbinsr):
                                        allmixed_no_mat[elt][ir1,ir2,0,0, elz1, elz2, elz3, elz4] += nextmixed[term][ir1, ir2]

        return allmixed_no_mat

    def compute_triplets(self, fthin=10):
        """
        Does either compute and save the relevant pair/triplet counts 
        or loads them in from a file. 
        
        """
        
        # Load triplets from file
        # Does not check for correct format etc!
        triplet_right_format = False
        if self.loadpath_triplets is not None:
            try:
                tripletdata = Table.read(self.loadpath_triplets)
                self.bin_centers = tripletdata["bin_centers"]
                self.paircounts = tripletdata["paircounts"] 
                self.paircounts_sq = tripletdata["paircounts_sq"]
                self.tripletcounts = tripletdata["tripletcounts"]
                if len(self.tripletcounts[:,0,0,0,0,0]) == self.discrete.nbinsz and  len(self.tripletcounts[0,0,0,:,0,0]) == (int(len(self.bin_edges)-1)*self.nsubbins):
                    triplet_right_format = True
                else:
                    print("\rWarning, triplet count file", self.loadpath_triplets, "has not the right format will compute triplets")
            except:
                print("\rWarning, triplet count file", self.loadpath_triplets, "not found, will compute triplets")
                trplet_right_format = False
        # Compute triplets and store
        if not triplet_right_format:
            if self.discrete.npatches is None:
                patchcats = self.discrete.gen_patches()
                
            for elp, patchcat in self.discrete.patchcats.items():
                print("\rComputing triplets from patch %i/%i\n"%(elp+1,len(self.discrete.patchcats.keys())),end="")
                # Allocate instances of ScalarTracerCatalog and XipmMixedCovariance
                scat = ScalarTracerCatalog(pos1=patchcat["x"][::fthin],
                                           pos2=patchcat["y"][::fthin],
                                           weight=patchcat["weight"][::fthin],
                                           zbins=patchcat["zbin"][::fthin],
                                           isinner=patchcat["incenter"][::fthin],
                                          tracer=patchcat["weight"][::fthin])
                xipmmixed = XipmMixedCovariance(min_sep_xi=self.bin_edges[0], 
                                                max_sep_xi=self.bin_edges[-1], 
                                                nbins_xi=len(self.bin_edges)-1, 
                                                nsubbins=self.nsubbins, 
                                                nmax=self.nmax, 
                                                shuffle_pix=True, 
                                                multicountcorr=True, 
                                                method="Tree", 
                                                tree_resos=[0,0.25,0.5,1.,2.,4.,8.])
                # Compute the tripletcounts
                xipmmixed.process(scat, nthreads=self.nthreads, dotomo=True)
                
                # Optionally perform an edge-correction for the multipoles
                if self.do_ec:
                    mixedcounts = edge_correctiontomo(xipmmixed.npcf_multipoles,
                                                      xipmmixed.npcf_multipoles_norm,
                                                      ret_matrices=False)
                # Transform the multipoles to triplet counts
                tripletcounts = tomomultipoles2triplets(xipmmixed.npcf_multipoles, nphis=self.nbinsphi)
                # Update the pair/triplet counts
                if not elp:
                    allbincenters_ret = np.zeros_like(xipmmixed.bin_centers)
                    allpaircounts_ret = np.zeros_like(xipmmixed.wwcounts)
                    allpaircountssq_ret = np.zeros_like(xipmmixed.w2wcounts)
                    alltripletcounts = np.zeros_like(tripletcounts)
                    ngaltot = 0
                
                ngaltot += np.sum(patchcat["incenter"])
                allbincenters_ret += np.sum(patchcat["incenter"])*xipmmixed.bin_centers
                allpaircounts_ret += xipmmixed.wwcounts
                allpaircountssq_ret += xipmmixed.w2wcounts
                alltripletcounts += tripletcounts
                
            # Add complete pair/triplet counts to class instance variables
            # Note that we divide tripletcounts by fthin to get the ratio
            # sum(i,j,k)w_i^2w_jw_k/((sum(i,j)w_iw_j)*(sum(k,l)w_kw_l) right
            # at the the stage when computing the mixed cov contribution.
            self.bin_centers = allbincenters_ret/ngaltot
            self.paircounts = allpaircounts_ret
            self.paircounts_sq = allpaircountssq_ret
            self.tripletcounts = alltripletcounts/fthin

            if self.savepath_triplets is not None:
                tosave = Table()
                tosave["bin_centers"] = self.bin_centers
                tosave["paircounts"] = self.paircounts
                tosave["paircounts_sq"] = self.paircounts_sq
                tosave["tripletcounts"] = self.tripletcounts
                tosave.write(self.savepath_triplets, overwrite=True,format='fits')
                
#####################################
### CLASSES IMPORTED FROM ORPHEUS ###
#####################################
class Catalog:
    
    def __init__(self, pos1, pos2, weight=None, zbins=None, isinner=None):
        self.pos1 = pos1.astype(np.float64)
        self.pos2 = pos2.astype(np.float64)
        self.weight = weight
        self.zbins = zbins
        self.ngal = len(self.pos1)
        # Normalize weight s.t. <weight> = 1
        if self.weight is None:
            self.weight = np.ones(self.ngal)
        self.weight = self.weight.astype(np.float64)
        #self.weight /= np.mean(self.weight)
        # Require zbins to only contain elements in {0, 1, ..., nbinsz-1}
        if self.zbins is None:
            self.zbins = np.zeros(self.ngal)        
        self.zbins = self.zbins.astype(int)
        self.nbinsz = len(np.unique(self.zbins))
        assert(np.max(self.zbins)-np.min(self.zbins)==self.nbinsz-1)
        self.zbins -= (np.min( self.zbins))
        if isinner is None:
            isinner = np.ones(self.ngal, dtype=np.float64)
        self.isinner = np.asarray(isinner, dtype=np.float64)
        assert(np.min(self.isinner) >= 0.)
        assert(np.max(self.isinner) <= 1.)
        assert(len(self.isinner)==self.ngal)
        assert(len(self.pos2)==self.ngal)
        assert(len(self.weight)==self.ngal)
        assert(len(self.zbins)==self.ngal)
        
        self.min1 = np.min(self.pos1)
        self.min2 = np.min(self.pos2)
        self.max1 = np.max(self.pos1)
        self.max2 = np.max(self.pos2)
        self.len1 = self.max1-self.min1
        self.len2 = self.max2-self.min2
        
        self.spatialhash = None
        self.hasspatialhash = False
        self.index_matcher = None
        self.pixs_galind_bounds = None
        self.pix_gals = None
        self.pix1_start = None
        self.pix1_d = None
        self.pix1_n = None
        self.pix2_start = None
        self.pix2_d = None
        self.pix2_n = None
        
        self.assign_methods = {"NGP":0, "CIC":1, "TSC":2}
        
        ## Link compiled libraries ##
        self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"discretecov"))
        p_c128 = ndpointer(np.complex128, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        
        # Generate pixel --> galaxy mapping
        # Safely called within other wrapped functions
        self.clib.build_spatialhash.restype = ct.c_void_p
        self.clib.build_spatialhash.argtypes = [
            p_f64, p_f64, ct.c_int32, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
            ct.c_int32, ct.c_int32,
            np.ctypeslib.ndpointer(dtype=np.int32)]
        
        self.clib.reducecat.restype = ct.c_void_p
        self.clib.reducecat.argtypes = [
            p_f64, p_f64, p_f64, p_f64, ct.c_int32, ct.c_int32, 
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int32, ct.c_int32, ct.c_int32,
            p_f64_nof, p_f64_nof, p_f64_nof, p_f64_nof,ct.c_int32]        
        
    # Reduces catalog to smaller catalog where positions & quantities are
    # averaged over regular grid
    def _reduce(self, fields, dpix, tomo=False, normed=False, shuffle=False,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        
        # Initialize grid
        start1, start2, n1, n2 = self._gengridprops(dpix, forcedivide, extent)
        
        # Prepare arguments
        if not tomo:
            zbinarr = np.zeros(self.ngal).astype(np.int32)
        else:
            zbinarr = self.zbins.astype(np.int32)
        nbinsz = len(np.unique(zbinarr))
        ncompfields = []
        scalarquants = []
        nfields = 0
        for field in fields:
            if type(field[0].item()) is float:
                scalarquants.append(field)
                nfields += 1
                ncompfields.append(1)
            if type(field[0].item()) is complex:
                scalarquants.append(field.real)
                scalarquants.append(field.imag)
                nfields += 2
                ncompfields.append(2)
        scalarquants = np.asarray(scalarquants)
        
        # Compute reduction (individually for each zbin)
        w_red = np.zeros(self.ngal, dtype=float)
        pos1_red = np.zeros(self.ngal, dtype=float)
        pos2_red = np.zeros(self.ngal, dtype=float)
        zbins_red = np.zeros(self.ngal, dtype=int)
        scalarquants_red = np.zeros((nfields, self.ngal), dtype=float)
        ind_start = 0
        for elz in range(nbinsz):
            sel_z = zbinarr==elz
            ngal_z = np.sum(sel_z)
            ngal_red_z = 0
            red_shape = (len(fields), ngal_z)
            w_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos1_red_z = np.zeros(ngal_z, dtype=np.float64)
            pos2_red_z = np.zeros(ngal_z, dtype=np.float64)
            scalarquants_red_z = np.zeros(nfields*ngal_z, dtype=np.float64)
            self.clib.reducecat(self.weight[sel_z].astype(np.float64), 
                                self.pos1[sel_z].astype(np.float64), 
                                self.pos2[sel_z].astype(np.float64),
                                scalarquants[:,sel_z].flatten().astype(np.float64),
                                ngal_z, nfields,
                                dpix, dpix, start1, start2, n1, n2, np.int32(shuffle),
                                w_red_z, pos1_red_z, pos2_red_z, scalarquants_red_z, ngal_red_z)
            w_red[ind_start:ind_start+ngal_z] = w_red_z
            pos1_red[ind_start:ind_start+ngal_z] = pos1_red_z
            pos2_red[ind_start:ind_start+ngal_z] = pos2_red_z
            zbins_red[ind_start:ind_start+ngal_z] = elz*np.ones(ngal_z, dtype=int)
            scalarquants_red[:,ind_start:ind_start+ngal_z] = scalarquants_red_z.reshape((nfields, ngal_z))
            ind_start += ngal_z
            
        # Accumulate reduced atalog
        sel_nonzero = w_red>0
        w_red = w_red[sel_nonzero]
        pos1_red = pos1_red[sel_nonzero]
        pos2_red = pos2_red[sel_nonzero]
        zbins_red = zbins_red[sel_nonzero]
        scalarquants_red = scalarquants_red[:,sel_nonzero]
        fields_red = []
        tmpcomp = 0
        for elf in range(len(fields)):
            if ncompfields[elf]==1:
                fields_red.append(scalarquants_red[tmpcomp])
            if ncompfields[elf]==2:
                fields_red.append(scalarquants_red[tmpcomp]+1J*scalarquants_red[tmpcomp+1])
            tmpcomp += ncompfields[elf]
            
        if ret_inst:
            return Catalog(pos1=pos1_red, pos2=pos2_red, weight=w_red, zbins=zbins_red), fields_red
            
        return w_red, pos1_red, pos2_red, zbins_red, fields_red
    
    def _multihash(self, dpixs, fields, dpix_hash=None, tomo=False, normed=False, shuffle=False,
                  extent=[None,None,None,None], forcedivide=1):
        """ Builds spatialhash for a base catalog and its reductions. """
        
        dpixs = sorted(dpixs)
        if dpix_hash is None:
            dpix_hash = dpixs[-1]
        if extent[0] is None:
            extent = [self.min1-dpix_hash, self.max1+dpix_hash, self.min2-dpix_hash, self.max2+dpix_hash]
            #extent = [self.min1, self.max1+dpix_hash-self.len1%dpix_hash, 
            #          self.min2, self.max2+dpix_hash-self.len2%dpix_hash]
        
        # Initialize spatial hash for discrete catalog
        self.build_spatialhash(dpix=dpix_hash, extent=extent)
        ngals = [self.ngal]
        pos1s = [self.pos1]
        pos2s = [self.pos2]
        weights = [self.weight]
        zbins = [self.zbins*tomo]
        allfields = [fields]
        index_matchers = [self.index_matcher]
        pixs_galind_bounds = [self.pixs_galind_bounds]
        pix_gals = [self.pix_gals]
        # Build spatial hashes for reduced catalogs 
        for dpix in dpixs:
            #print(dpix, len(self.pos1))
            nextcat, fields_red = self._reduce(fields=fields,
                                               dpix=dpix, 
                                               tomo=tomo, 
                                               normed=normed, 
                                               shuffle=shuffle,
                                               extent=extent, 
                                               forcedivide=forcedivide, 
                                               ret_inst=True)
            nextcat.build_spatialhash(dpix=dpix_hash, extent=extent)
            ngals.append(nextcat.ngal)
            pos1s.append(nextcat.pos1)
            pos2s.append(nextcat.pos2)
            weights.append(nextcat.weight)
            zbins.append(nextcat.zbins)
            allfields.append(fields_red)
            index_matchers.append(nextcat.index_matcher)
            pixs_galind_bounds.append(nextcat.pixs_galind_bounds)
            pix_gals.append(nextcat.pix_gals)
            
        return ngals, pos1s, pos2s, weights, zbins, allfields, index_matchers, pixs_galind_bounds, pix_gals
                                
    
    def build_spatialhash(self, dpix=1., extent=[None, None, None, None]):
        
        # Build extent
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)
            
        # Collect arguments
        # Note that the C function assumes the mask to start at zero, that's why we shift
        # the galaxy positions
        self.pix1_start = thismin1 - dpix/1.
        self.pix2_start = thismin2 - dpix/1.
        stop1 = thismax1 + dpix/1.
        stop2 = thismax2 + dpix/1.
        self.pix1_n = int(np.round((stop1-self.pix1_start)/dpix))
        self.pix2_n = int(np.round((stop2-self.pix2_start)/dpix))
        npix = self.pix1_n * self.pix2_n
        self.pix1_d = (stop1-self.pix1_start)/(self.pix1_n)
        self.pix2_d = (stop2-self.pix2_start)/(self.pix2_n)

        # Compute hashtable
        #print(np.min(self.pos1), np.min(self.pos2), np.max(self.pos1), np.max(self.pos2))
        #print(self.pix1_start, self.pix2_start, 
        #      self.pix1_start+self.pix1_n*self.pix1_d,
        #     self.pix2_start+self.pix2_n*self.pix2_d)
        #print(self.pix1_n,self.pix2_n,self.pix1_d,self.pix2_d)
        result = np.zeros(2 * npix + 3 * self.ngal + 1).astype(np.int32)
        self.clib.build_spatialhash(self.pos1, self.pos2, self.ngal,
                                  self.pix1_d, self.pix2_d, 
                                  self.pix1_start, self.pix2_start, 
                                  self.pix1_n, self.pix2_n,
                                  result)

        # Allocate result
        start_isoutside = 0
        start_index_matcher = self.ngal
        start_pixs_galind_bounds = self.ngal + npix
        start_pixs_gals = self.ngal + npix + self.ngal + 1
        start_ngalinpix = self.ngal + npix + self.ngal + 1 + self.ngal
        self.index_matcher = result[start_index_matcher:start_pixs_galind_bounds]
        self.pixs_galind_bounds = result[start_pixs_galind_bounds:start_pixs_gals]
        self.pix_gals = result[start_pixs_gals:start_ngalinpix]
        self.hasspatialhash = True
        

    def _gengridprops(self, dpix, forcedivide, extent=[None,None,None,None]):
        
        # Define inner extent of the grid
        fixedsize = False
        if extent[0] is not None:
            fixedsize = True
        if extent[0] is None:
            thismin1 = self.min1
        else:
            thismin1 = extent[0]
            assert(thismin1 <= self.min1)
        if extent[1] is None:
            thismax1 = self.max1
        else:
            thismax1 = extent[1]
            assert(thismax1 >= self.max1)
        if extent[2] is None:
            thismin2 = self.min2
        else:
            thismin2 = extent[2]
            assert(thismin2 <= self.min2)
        if extent[3] is None:
            thismax2 = self.max2
        else:
            thismax2 = extent[3]
            assert(thismax2 >= self.max2)
            
        # Add buffer to grid and get associated pixelization
        if not fixedsize:
            start1 = thismin1 - 4*dpix
            start2 = thismin2 - 4*dpix
            n1 = int(np.ceil((thismax1+4*dpix - start1)/dpix))
            n2 = int(np.ceil((thismax2+4*dpix - start2)/dpix))
            n1 += (forcedivide - n1%forcedivide)%forcedivide
            n2 += (forcedivide - n2%forcedivide)%forcedivide
        else:
            start1=extent[0]
            start2=extent[2]
            n1 = int((thismax1-thismin1)/dpix)
            n2 = int((thismax2-thismin2)/dpix)
            assert(not n1%forcedivide)
            assert(not n2%forcedivide)
            
        return start1, start2, n1, n2
    
class ScalarTracerCatalog(Catalog):
    
    def __init__(self, pos1, pos2, tracer, **kwargs):
        super().__init__(pos1=pos1, pos2=pos2, **kwargs)
        self.tracer = tracer
        self.spin = 0
        
    def reduce(self, dpix, tomo=False, normed=False, shuffle=False,
               extent=[None,None,None,None], forcedivide=1, 
               ret_inst=False):
        res = super()._reduce(dpix=dpix, 
                              fields=[self.tracer], 
                              tomo=tomo, 
                              normed=normed, 
                              shuffle=shuffle,
                              extent=extent,
                              forcedivide=forcedivide,
                              ret_inst=False)
        (w_red, pos1_red, pos2_red, zbins_red, fields_red) = res
        if ret_inst:
            return ScalarTracerCatalog(self.spin, pos1_red, pos2_red, 
                                       fields_red[0], 
                                       weight=w_red, zbins=zbins_red)
        return res
    
    def multireduce(self, dpixs, tomo=False, normed=False, 
                    extent=[None,None,None,None], forcedivide=1):
        pass
    
    def multihash(self, dpixs, dpix_hash=None, tomo=False, normed=False, shuffle=False,
                  extent=[None,None,None,None], forcedivide=1):
        res = super()._multihash(dpixs=dpixs, 
                                fields=[self.tracer], 
                                dpix_hash=dpix_hash,
                                tomo=tomo, 
                                normed=normed, 
                                shuffle=shuffle,
                                extent=extent,
                                forcedivide=forcedivide)
        return res
    
class BinnedNPCF:
    
    def __init__(self, order, spins, n_cfs, min_sep, max_sep, nbinsr=None, binsize=None, nbinsphi=100, 
                 nmaxs=30, method="Tree", multicountcorr=True, shuffle_pix=True,
                 tree_resos=[0,0.25,0.5,1.,2.], tree_redges=None, rmin_pixsize=20):
        
        self.order = np.int32(order)
        self.n_cfs = np.int32(n_cfs)
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbinsphi = nbinsphi
        self.nmaxs = nmaxs
        self.method = method
        self.multicountcorr = int(multicountcorr)
        self.shuffle_pix = shuffle_pix
        self.methods_avail = ["Discrete", "Tree", "DoubleTree"]
        self.tree_resos = np.asarray(tree_resos, dtype=float)
        self.tree_nresos = np.int32(len(self.tree_resos))
        self.tree_redges = tree_redges
        self.rmin_pixsize = rmin_pixsize
        self.tree_resosatr = None
        self.bin_centers = None
        self.phis = [None]*self.order
        self.npcf = None
        self.npcf_norm = None
        self.npcf_multipoles = None
        self.npcf_multipoles_norm = None
        
        # Check types or arguments
        if isinstance(self.nbinsphi, int):
            self.nbinsphi = self.nbinsphi*np.ones(order-2).astype(np.int32)
        if isinstance(self.nmaxs, int):
            self.nmaxs = self.nmaxs*np.ones(order-2).astype(np.int32)
        if isinstance(spins, int):
            spins = spins*np.ones(order).astype(np.int32)
        self.spins = np.asarray(spins, dtype=np.int32)
        #print(self.spins)
        assert(isinstance(self.order, np.int32))
        assert(isinstance(self.spins, np.ndarray))
        assert(isinstance(self.spins[0], np.int32))
        assert(len(spins)==self.order)
        assert(isinstance(self.n_cfs, np.int32))
        assert(isinstance(self.min_sep, float))
        assert(isinstance(self.max_sep, float))
        assert(isinstance(self.nbinsphi, np.ndarray))
        assert(isinstance(self.nbinsphi[0], np.int32))
        assert(len(self.nbinsphi)==self.order-2)
        assert(isinstance(self.nmaxs, np.ndarray))
        assert(isinstance(self.nmaxs[0], np.int32))
        assert(len(self.nmaxs)==self.order-2)
        assert(self.method in self.methods_avail)
        assert(isinstance(self.tree_resos, np.ndarray))
        assert(isinstance(self.tree_resos[0], float))
        
        # Setup radial bins
        # Note that we always have self.binsize <= binsize
        assert((binsize!=None) or (nbinsr!=None))
        if nbinsr != None:
            self.nbinsr = np.int32(nbinsr)
        if binsize != None:
            assert(isinstance(binsize, float))
            self.nbinsr = np.int32(np.ceil(np.log(self.max_sep/self.min_sep)/binsize))
        assert(isinstance(self.nbinsr, np.int32))
        self.bin_edges = np.geomspace(self.min_sep, self.max_sep, self.nbinsr+1)
        self.binsize = np.log(self.bin_edges[1]/self.bin_edges[0])
        # Setup variable for tree estimator
        if self.tree_redges != None:
            assert(isinstance(self.tree_redges, np.ndarray))
            assert(isinstance(self.tree_redges[0], float))
            assert(len(self.tree_redges)==self.tree_resos+1)
            self.tree_redges = np.sort(self.tree_redges)
            assert(self.tree_redges[0]==self.min_sep)
            assert(self.tree_redges[-1]==self.max_sep)
        else:
            self.tree_redges = np.zeros(len(self.tree_resos)+1)
            self.tree_redges[-1] = self.max_sep
            for elreso, reso in enumerate(self.tree_resos):
                self.tree_redges[elreso] = (reso==0.)*self.min_sep + (reso!=0.)*self.rmin_pixsize*reso
        _tmpreso = 0
        self.tree_resosatr = np.zeros(self.nbinsr, dtype=np.int32)
        for elbin, rbin in enumerate(self.bin_edges[:-1]):
            if rbin > self.tree_redges[_tmpreso+1]:
                _tmpreso += 1
            self.tree_resosatr[elbin] = _tmpreso
            
        # Setup phi bins
        for elp in range(self.order-2):
            _ = np.linspace(0,2*np.pi,self.nbinsphi[elp]+1)
            self.phis[elp] = .5*(_[1:] + _[:-1])      
          
        #############################
        ## Link compiled libraries ##
        #############################
        self.clib = ct.CDLL(search_file_in_site_package(get_site_packages_dir(),"discretecov"))
        
        p_c128 = ndpointer(np.complex128, flags="C_CONTIGUOUS")
        p_f64 = ndpointer(np.float64, flags="C_CONTIGUOUS")
        p_f32 = ndpointer(np.float32, flags="C_CONTIGUOUS")
        p_i32 = ndpointer(np.int32, flags="C_CONTIGUOUS")
        p_f64_nof = ndpointer(np.float64)
        
        if self.order==3:
            self.clib.alloc_triplets_tree_xipxipcov.restype = ct.c_void_p
            self.clib.alloc_triplets_tree_xipxipcov.argtypes = [
                p_i32, p_f64, p_f64, p_f64, p_i32, ct.c_int32, ct.c_int32, 
                ct.c_int32, p_f64, p_i32,
                p_f64, p_f64, p_f64 , p_i32,
                p_i32, p_i32, p_i32, ct.c_double, ct.c_double, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32,
                ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, p_f64, ct.c_int32, ct.c_int32, ct.c_int32, 
                np.ctypeslib.ndpointer(dtype=np.double), 
                np.ctypeslib.ndpointer(dtype=np.double), 
                np.ctypeslib.ndpointer(dtype=np.double), 
                np.ctypeslib.ndpointer(dtype=complex),
                np.ctypeslib.ndpointer(dtype=complex)] 

        
    ############################################################
    ## Functions that deal with different projections of NPCF ##
    ############################################################
    def _initprojections(self, child):
        assert(child.projection in child.projections_avail)
        child.project = {}
        for proj in child.projections_avail:
            child.project[proj] = {}
            for proj2 in child.projections_avail:
                if proj==proj2:
                    child.project[proj][proj2] = lambda: child.npcf
                else:
                    child.project[proj][proj2] = None
                    
    def _projectnpcf(self, child, projection):
        """
        Projects npcf to a new basis.
        """
        assert(child.npcf is not None)
        if projection not in child.projections_avail:
            print("Projection %s is not yet supported."%(projection))
            self.gen_npcfprojections_avail(child)

        if child.project[child.projection][projection] is not None:
            child.npcf = child.project[child.projection][projection]()
            child.projection = projection
        else:
            print("Projection from %s to %s is not yet implemented."%(child.projection,projection))
            self._gen_npcfprojections_avail(child)
                    
    def _gen_npcfprojections_avail(self, child):
        print("The following projections are available in the class %s:"%child.__class__.__name__)
        for proj in child.projections_avail:
            for proj2 in child.projections_avail:
                if child.project[proj][proj2] is not None:
                    print("  %s --> %s"%(proj,proj2))

    
    ####################
    ## MISC FUNCTIONS ##
    ####################
    def _checkcats(self, cats, spins):
        if isinstance(cats, list):
            assert(len(cats)==self.order)
        for els, s in enumerate(self.spins):
            if not isinstance(cats, list):
                thiscat = cats
            else:
                thiscat = cats[els]
            assert(thiscat.spin == s)
        
            
class XipmMixedCovariance(BinnedNPCF):
    
    def __init__(self, min_sep_xi, max_sep_xi, nbins_xi, nsubbins, nmax=10, **kwargs):
        self.nsubbins = int(nsubbins)
        nbinsr = nsubbins*nbins_xi
        super().__init__(order=3, spins=[0,0,0], n_cfs=1, min_sep=min_sep_xi, max_sep=max_sep_xi, nbinsr=nbinsr, 
                         nmaxs=nmax, **kwargs)
        self.nmax = self.nmaxs[0]
        self.phi = self.phis[0]
        self.projection = None
        self.projections_avail = [None]
        self.nbinsz = None
        
        # (Add here any newly implemented projections)
        self._initprojections(self)
        
    def process(self, cat, nthreads=16, dotomo=True):
        #self._checkcats(cat, self.spins)
        # Within the c-code we parallelize over stripes across the pos1 direction. If we have
        # multiple patches with a nonzero overlap, only the galaxies within the inner region
        # can form the base of a triplet and therefore the threads within the stripes of the
        # overlap will not run anything. Therefore, we increase the number of threads by a factor
        # s.t. the actual number of threads actively running ~matches the nthreads argument
        _leninner = np.max(cat.pos1[cat.isinner.astype(bool)])-np.min(cat.pos1[cat.isinner.astype(bool)])
        nthreads_ext = int(cat.len1/_leninner * nthreads)
        if not dotomo:
            self.nbinsz = 1
            zbins = np.zeros(cat.ngal, dtype=np.int32)
        else:
            self.nbinsz = cat.nbinsz
            zbins = cat.zbins
        sc = (self.nmax+1,self.nbinsz,self.nbinsz,self.nbinsz,self.nbinsr,self.nbinsr)
        sn = (self.nmax+1,self.nbinsz,self.nbinsz,self.nbinsz,self.nbinsr,self.nbinsr)
        szr = (self.nbinsz, self.nbinsr)
        szzr = (self.nbinsz, self.nbinsz, self.nbinsr)
        bin_centers = np.zeros(self.nbinsz*self.nbinsr).astype(np.float64)
        w2wwtriplets = np.zeros((self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(np.complex128)
        wwwtriplets = np.zeros((self.nmax+1)*self.nbinsz*self.nbinsz*self.nbinsz*self.nbinsr*self.nbinsr).astype(np.complex128)
        wwcounts = np.zeros(self.nbinsz*self.nbinsz*self.nbinsr).astype(np.float64)
        w2wcounts = np.zeros(self.nbinsz*self.nbinsz*self.nbinsr).astype(np.float64)
        args_basecat = (cat.isinner.astype(np.int32), cat.weight.astype(np.float64), cat.pos1.astype(np.float64), cat.pos2.astype(np.float64), 
                        zbins.astype(np.int32), np.int32(self.nbinsz), np.int32(cat.ngal), )
        args_basesetup = (np.int32(0), np.int32(self.nmax), np.float64(self.min_sep), np.float64(self.max_sep), np.array([-1.]).astype(np.float64), 
                          np.int32(self.nbinsr), np.int32(self.multicountcorr), )
        if self.method=="Discrete":
            raise NotImplementedError
        elif self.method=="Tree":
            cutfirst = int(self.tree_resos[0]==0.)
            mhash = cat.multihash(dpixs=self.tree_resos[cutfirst:],tomo=dotomo,shuffle=self.shuffle_pix)
            ngal_resos, pos1s, pos2s, weights, zbins, _, index_matchers, pixs_galind_bounds, pix_gals = mhash
            weight_resos = np.concatenate(weights).astype(np.float64)
            pos1_resos = np.concatenate(pos1s).astype(np.float64)
            pos2_resos = np.concatenate(pos2s).astype(np.float64)
            zbin_resos = np.concatenate(zbins).astype(np.int32)
            index_matcher = np.concatenate(index_matchers).astype(np.int32)
            pixs_galind_bounds = np.concatenate(pixs_galind_bounds).astype(np.int32)
            pix_gals = np.concatenate(pix_gals).astype(np.int32)
            args_pixgrid = (np.float64(cat.pix1_start), np.float64(cat.pix1_d), np.int32(cat.pix1_n), 
                            np.float64(cat.pix2_start), np.float64(cat.pix2_d), np.int32(cat.pix2_n), )

            args = (*args_basecat,
                    self.tree_nresos,
                    self.tree_redges,
                    np.array(ngal_resos, dtype=np.int32),
                    weight_resos,
                    pos1_resos,
                    pos2_resos,
                    zbin_resos,
                    index_matcher,
                    pixs_galind_bounds,
                    pix_gals,
                    *args_pixgrid,
                    *args_basesetup,
                    np.int32(nthreads_ext),
                    bin_centers,
                    wwcounts,
                    w2wcounts,
                    w2wwtriplets,
                    wwwtriplets)
            func = self.clib.alloc_triplets_tree_xipxipcov
        elif self.method=="DoubleTree":
            raise NotImplementedError 
        func(*args)
        
        self.bin_centers = bin_centers.reshape(szr)
        self.wwcounts = wwcounts.reshape(szzr)
        self.w2wcounts = w2wcounts.reshape(szzr)
        self.npcf_multipoles = w2wwtriplets.reshape(sc)
        self.npcf_multipoles_norm = wwwtriplets.reshape(sn)
        self.projection = None 
       
    
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
    dphi = phis[1]-phis[0]
    triplets = np.zeros((nbinsz, nbinsz, nbinsz, nbinsr, nbinsr, nphis), dtype=complex)
    for n in range(nmax):
        einphis = 1./(2*np.pi)*np.e**(1J*n*phis)
        einphis_c = 1./(2*np.pi)*np.e**(-1J*n*phis)
        ones = 1./(2*np.pi)*np.ones(nphis)
        for elz1 in range(nbinsz):
            for elz2 in range(nbinsz):
                for elz3 in range(nbinsz):
                    for elb1 in range(nbinsr):
                        for elb2 in range(nbinsr):
                            if n==0:
                                triplets[elz1,elz2,elz3,elb1,elb2] += multipoles[n,elz1,elz2,elz3,elb1,elb2].real*ones
                            else:
                                triplets[elz1,elz2,elz3,elb1,elb2] += multipoles[n,elz1,elz2,elz3,elb1,elb2]*einphis
                                triplets[elz1,elz2,elz3,elb1,elb2] += multipoles[n,elz1,elz3,elz2,elb2,elb1]*einphis_c
    return triplets*dphi

def cov_estimate_single(bin_edges, bin_centers, xip_spline, xim_spline, sigma_eps1, nphibins, nbins_subr,
                 multipole_tripletcounts, paircounts_ret12, paircounts_ret34, paircountssq_ret,
                 sigma_eps2=None, N2=None):
    if sigma_eps2 is None:
        sigma_eps2 = sigma_eps1
    sigma2_eps = sigma_eps1**2 + sigma_eps2**2
    _,_,nphibins = multipole_tripletcounts.shape
    dphis = np.linspace(-np.pi,np.pi,nphibins)
    ddphis = dphis[1] - dphis[0]
    terms = ["xipxip", "xipxim", "ximxim"]
    cov_third_multipole = {}
    for k in terms:
        cov_third_multipole[k] = np.zeros((int(len(bin_centers)//nbins_subr),int(len(bin_centers)//nbins_subr)))
    cov_shape = sigma2_eps**2/2 * np.diag(paircountssq_ret/(paircounts_ret12*paircounts_ret34))
    for elb1, b1 in enumerate(bin_centers):
        elb1_cov = int(elb1//nbins_subr)
        for elb2, b2 in enumerate(bin_centers):
            elb2_cov = int(elb2//nbins_subr)
            b3 = np.sqrt(b1**2+b2**2-2*b1*b2*np.cos(dphis))
            # For Cov(xip,xim) have cos(4*(phi_jk-phi_ij))
            # * As the difference of those angles is invariant under rotations 
            #   we can set phi_ij==0, such that phi_ik==phi
            # * Using the exterior angle formula we have 
            #   phi_ik - phi_jk = phi - (pi - phi - beta)
            #                   = -pi + beta
            #   where beta is the inner angle between b2 and b3
            # * Using cosine properties, this leaves
            #   cos(4*(phi_jk-phi_ij)) = cos(4*beta)
            #   and beta can be inferred from the three triangle sides
            beta = np.arccos((b3**2+b2**2-b1**2)/(2*b2*b3))
            norm = sigma2_eps/(2*paircounts_ret12[elb1]*paircounts_ret34[elb2])
            cov_third_multipole["xipxip"][elb1_cov,elb2_cov] = norm * np.nansum(multipole_tripletcounts[elb1,elb2].real*
                                                                        xip_spline(b3))*ddphis
            cov_third_multipole["ximxim"][elb1_cov,elb2_cov] = norm * np.nansum(multipole_tripletcounts[elb1,elb2].real*
                                                                        np.cos(4*dphis)*xip_spline(b3))*ddphis
            cov_third_multipole["xipxim"][elb1_cov,elb2_cov] = norm * np.nansum(multipole_tripletcounts[elb1,elb2].real*
                                                                        np.cos(4*beta)*xim_spline(b3))*ddphis
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
    
    threepcf_n_corr = np.zeros(threepcf_n.shape, dtype=complex)
    nvals, nbinsz, _, _, ntheta, _ = threepcf_n_norm.shape
    nmax = nvals-1
    
    #(11, 6, 6, 6, 20, 20)
    # Required to perform edge correction over full -nmax,..+nmax range
    # --> Need to explicitly recover the negative entries
    mirroredthreepcf_n = np.zeros((2*nmax+1, *(threepcf_n.shape)[1:]), dtype=complex)
    mirroredthreepcf_n_norm = np.zeros((2*nmax+1, *(threepcf_n.shape)[1:]), dtype=complex)
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
    narr = np.arange(-nmax,nmax+1, dtype=int)
    nextM = np.zeros((nvals,nvals))
    for ind, ell in enumerate(narr):
        lminusn = ell-narr
        sel = np.logical_and(lminusn+nmax>=0, lminusn+nmax<nvals)
        nextM[ind,sel] = threepcf_n_norm[(lminusn+nmax)[sel],thet1,thet2].real/threepcf_n_norm[nmax,thet1,thet2].real
    return nextM