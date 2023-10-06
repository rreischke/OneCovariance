import numpy as np
from astropy.table import Table
import healpy as hp


def toorigin(ras, decs, gammas1, gammas2, rotangle=None, inv=False, radec_units="deg"):
    """
    Rotates survey patch such that its center of mass lies in the origin. 
    
    Parameters:
    -----------
    ras : array
        RA-coordinates of the galaxies
    decs : array
        DEC-coordinates of the galaxies
    gammas1 : array
        First component of the galaxies' shape
    gammas2 : array
        Second component of the galaxies' shape
    rotangle : list or None (defaults to None)
        Angle by which the positions and shapes will be rotated
    inv : bool (defaults to False)
        Whether to apply the inverse rotation
    radec_units : str (defaults to "deg")
        Units in which ra and dec parameters are given
        
    Returns:
    --------
    ras : array
        RA-coordinates of the rotated galaxies
    decs : array
        DEC-coordinates of the rotated galaxies
    gammas1 : array
        First component of the rotated galaxies' shape
    gammas2 : array
        Second component of the rotated galaxies' shape
    """
    
    assert(radec_units in ["rad", "deg"])
    
    # Map (ra, dec) --> (theta, phi)
    if radec_units=="deg":
        decs_rad = decs*np.pi/180.
        ras_rad = ras*np.pi/180.
    thetas = np.pi/2. + decs_rad
    phis = ras_rad
    # Sloppy way of catching bimodal distributions...does not affect statistics!
    try:
        if (np.max(phis)-np.min(phis))/np.std(phis)<3.:
            phis[phis>np.pi] -= 2*np.pi
    except:
        pass
    
    # Compute rotation angle
    if rotangle is None:
        rotangle = [np.mean(phis),np.pi/2.-np.mean(thetas)]
    thisrot = hp.Rotator(rot=rotangle, deg=False, inv=inv)
    rotatedthetas, rotatedphis = thisrot(thetas,phis,inv=False)
    gamma_rot = (gammas1+1J*gammas2) * np.exp(1J * 2 * thisrot.angle_ref(rotatedthetas, rotatedphis,inv=True))
    
    # Transform back to (ra,dec)
    ra_rot = rotatedphis
    dec_rot = rotatedthetas - np.pi/2.
    if radec_units=="deg":
        dec_rot *= 180./np.pi
        ra_rot *= 180./np.pi
    
    return ra_rot, dec_rot, gamma_rot.real, gamma_rot.imag


def gen_flatz(nbinsz):
    # Construct zmatcher
    flatz = np.zeros((nbinsz,nbinsz),dtype=int)
    _ind = 0
    for i in range(nbinsz):
        for j in range(i,nbinsz):
            flatz[i,j] = _ind
            flatz[j,i] = _ind
            _ind += 1
    return flatz

############################
## UTILS FOR CYGNUS MOCKS ##
############################
def download_cygnus(los, base_save):
    import ssl
    import wget
    ssl._create_default_https_context = ssl._create_unverified_context
    base = "https://www.star.ucl.ac.uk/maxivonw/cygnus/"
    filelist = []
    outlist = []
    for i in range(16):
        fname = "galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(los,i)
        filelist.append(base + fname)
        outlist.append(base_save+fname)
        wget.download(base+fname, out=base_save)
    return outlist

def load_cygnus_mock(run, basepath):
    ['ALPHA_J2000', 'DELTA_J2000', 'z_spec', 'g1', 'g2', 'e1', 'e2', 'weight']
    ra = np.array([])
    dec = np.array([])
    zspec = np.array([])
    zbin = np.array([])
    e1 = np.array([])
    e2 = np.array([])
    g1 = np.array([])
    g2 = np.array([])
    weight = np.array([])
    
    for zind in range(6):
        #print(zind)
        nextbit = Table.read(basepath+"galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(run,4+2*zind))
        ra = np.append(ra, nextbit["ALPHA_J2000"])
        dec = np.append(dec, nextbit["DELTA_J2000"])
        zspec = np.append(zspec, nextbit["z_spec"])
        g1 = np.append(g1, nextbit["g1"])
        g2 = np.append(g2, nextbit["g2"])
        e1 = np.append(e1, nextbit["e1"])
        e2 = np.append(e2, nextbit["e2"])
        weight = np.append(weight, nextbit["weight"])
        zbin = np.append(zbin, zind*np.ones(len(nextbit["e1"])).astype(np.int))
        nextbit = Table.read(basepath+"galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(run,5+2*zind))
        ra = np.append(ra, nextbit["ALPHA_J2000"])
        dec = np.append(dec, nextbit["DELTA_J2000"])
        zspec = np.append(zspec, nextbit["z_spec"])
        g1 = np.append(g1, nextbit["g1"])
        g2 = np.append(g2, nextbit["g2"])
        e1 = np.append(e1, nextbit["e1"])
        e2 = np.append(e2, nextbit["e2"])
        weight = np.append(weight, nextbit["weight"])
        zbin = np.append(zbin, zind*np.ones(len(nextbit["e1"])).astype(np.int))
    
    return weight, ra, dec, g1, g2, e1, e2, zbin, zspec

def cyg2disc(run, basepath="/cosma6/data/dp004/dc-port3/KiDS/cygnus_mocks/", fpath_save="tmpcygnus.fits", tomo=True):
    
    #['ALPHA_J2000', 'DELTA_J2000', 'z_spec', 'g1', 'g2', 'e1', 'e2', 'weight']
    ra = np.array([])
    dec = np.array([])
    zspec = np.array([])
    zbin = np.array([])
    e1 = np.array([])
    e2 = np.array([])
    g1 = np.array([])
    g2 = np.array([])
    weight = np.array([])
    ngal = np.zeros(6)
    
    for zind in range(6):
        nextbit = Table.read(basepath+"galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(run,4+2*zind))
        ra = np.append(ra, nextbit["ALPHA_J2000"])
        dec = np.append(dec, nextbit["DELTA_J2000"])
        zspec = np.append(zspec, nextbit["z_spec"])
        g1 = np.append(g1, nextbit["g1"])
        g2 = np.append(g2, nextbit["g2"])
        e1 = np.append(e1, nextbit["e1"])
        e2 = np.append(e2, nextbit["e2"])
        weight = np.append(weight, nextbit["weight"])
        zbin = np.append(zbin, tomo*zind*np.ones(len(nextbit["e1"])).astype(np.int))
        ngal[zind] += len(nextbit["g1"])
        nextbit = Table.read(basepath+"galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(run,5+2*zind))
        ra = np.append(ra, nextbit["ALPHA_J2000"])
        dec = np.append(dec, nextbit["DELTA_J2000"])
        zspec = np.append(zspec, nextbit["z_spec"])
        g1 = np.append(g1, nextbit["g1"])
        g2 = np.append(g2, nextbit["g2"])
        e1 = np.append(e1, nextbit["e1"])
        e2 = np.append(e2, nextbit["e2"])
        weight = np.append(weight, nextbit["weight"])
        zbin = np.append(zbin, tomo*zind*np.ones(len(nextbit["e1"])).astype(np.int))
        ngal[zind] += len(nextbit["g1"])

        
    totcyg = Table()
    totcyg["weight"] = weight
    totcyg["ALPHA_J2000"] = ra
    totcyg["DELTA_J2000"] = dec
    totcyg["e1"] = e1
    totcyg["e2"] = e2
    totcyg["zbin"] = zbin
    
    Table.write(totcyg, fpath_save, overwrite=True)
    
    return ngal

def cygnus_patches(ra, dec, g1, g2, e1, e2, zbin, weight, overlap_arcmin=0.):
    dec_base = dec
    ra_base = (ra+40)%360
    issouth = dec_base < -10
    
    patchcats = {}
    
    # Patches of kids south
    npatches_south = 7
    overlap_arcmin = overlap_arcmin / np.sin(np.pi/2.-dec_base*np.pi/180.)
    south_min = np.min(ra_base[issouth])
    south_max = np.max(ra_base[issouth])
    dpatch = (south_max - south_min)/npatches_south
    patchcats = {}
    indpatches = (np.floor((ra_base-south_min)/dpatch)).astype(np.int)
    for i in range(npatches_south):
        centersel = np.argwhere(np.logical_and(issouth, indpatches==i))
        inoverlap_left = np.argwhere(np.logical_and(indpatches==i-1,south_min+i*dpatch-ra_base<=overlap_arcmin/60.))
        inoverlap_right = np.argwhere(np.logical_and(indpatches==i+1,ra_base-(south_min+(i+1)*dpatch)<=overlap_arcmin/60.))
        thissel = np.append(np.append(centersel, inoverlap_left), inoverlap_right)
        ra_rot, dec_rot, rot_g1, rot_g2 = toorigin(ra_base[thissel], dec_base[thissel], 
                                                  g1[thissel], g2[thissel],
                                                  rotangle=None, inv=False)
        ra_rot, dec_rot, rot_e1, rot_e2 = toorigin(ra_base[thissel], dec_base[thissel], 
                                                  e1[thissel], e2[thissel],
                                                  rotangle=None, inv=False)
        patchcats[i]={}
        patchcats[i]["name"] = "KIDS_South_Patch_%i"%i
        patchcats[i]["weight"] = weight[thissel]
        patchcats[i]["zbin"] = zbin[thissel]
        patchcats[i]["ra"] = ra_rot
        patchcats[i]["dec"] = dec_rot
        patchcats[i]["g1"] = rot_g1
        patchcats[i]["g2"] = rot_g2
        patchcats[i]["e1"] = rot_e1
        patchcats[i]["e2"] = rot_e2
        patchcats[i]["x"] = (ra_rot - np.min(ra_rot))*60 * np.sin(np.pi/2-dec_rot*np.pi/180)
        patchcats[i]["y"] = (dec_rot - np.min(dec_rot))*60
        patchcats[i]["g1_flat"] = rot_g1
        patchcats[i]["g2_flat"] = rot_g2
        patchcats[i]["e1_flat"] = rot_e1
        patchcats[i]["e2_flat"] = rot_e2
        patchcats[i]["incenter"] = np.ones(len(centersel)).astype(np.bool)
        patchcats[i]["incenter"] = np.append(patchcats[i]["incenter"], np.zeros(len(inoverlap_left)).astype(np.bool))
        patchcats[i]["incenter"] = np.append(patchcats[i]["incenter"], np.zeros(len(inoverlap_right)).astype(np.bool))
    
    # G15 patch
    insingle = np.logical_and(~issouth, ra_base<190)
    ra_rot, dec_rot, rot_g1, rot_g2 = toorigin(ra_base[insingle], dec_base[insingle], 
                                               g1[insingle], g2[insingle],
                                               rotangle=None, inv=False)
    ra_rot, dec_rot, rot_e1, rot_e2 = toorigin(ra_base[insingle], dec_base[insingle], 
                                               e1[insingle], e2[insingle],
                                               rotangle=None, inv=False)
    patchcats[npatches_south]={}
    patchcats[npatches_south]["name"] = "KIDS_North_G15"
    patchcats[npatches_south]["weight"] = weight[insingle]
    patchcats[npatches_south]["zbin"] = zbin[insingle]
    patchcats[npatches_south]["ra"] = ra_rot
    patchcats[npatches_south]["dec"] = dec_rot
    patchcats[npatches_south]["g1"] = rot_g1
    patchcats[npatches_south]["g2"] = rot_g2
    patchcats[npatches_south]["e1"] = rot_e1
    patchcats[npatches_south]["e2"] = rot_e2
    patchcats[npatches_south]["x"] = (ra_rot - np.min(ra_rot))*60 * np.sin(np.pi/2-dec_rot*np.pi/180)
    patchcats[npatches_south]["y"] = (dec_rot - np.min(dec_rot))*60
    patchcats[npatches_south]["g1_flat"] = rot_g1
    patchcats[npatches_south]["g2_flat"] = rot_g2
    patchcats[npatches_south]["e1_flat"] = rot_e1
    patchcats[npatches_south]["e2_flat"] = rot_e2
    patchcats[npatches_south]["incenter"] = np.ones(len(ra_rot)).astype(np.bool)
    
    npatches_north = 2
    isnorth = np.logical_and(~issouth, ra_base>=190)
    north_min = np.min(ra_base[isnorth])
    north_max = np.max(ra_base[isnorth])
    dpatch = (north_max - north_min)/npatches_north
    indpatches = (np.floor((ra_base-north_min)/dpatch)).astype(np.int)
    for i in range(npatches_north):
        centersel = np.argwhere(np.logical_and(isnorth, indpatches==i))
        inoverlap_left = np.argwhere(np.logical_and(indpatches==i-1,north_min+i*dpatch-ra_base<=overlap_arcmin/60.))
        inoverlap_right = np.argwhere(np.logical_and(indpatches==i+1,ra_base-(north_min+(i+1)*dpatch)<=overlap_arcmin/60.))
        thissel = np.append(np.append(centersel, inoverlap_left), inoverlap_right)
        ra_rot, dec_rot, rot_g1, rot_g2 = toorigin(ra_base[thissel], dec_base[thissel], 
                                                  g1[thissel], g2[thissel],
                                                  rotangle=None, inv=False)
        ra_rot, dec_rot, rot_e1, rot_e2 = toorigin(ra_base[thissel], dec_base[thissel], 
                                                  e1[thissel], e2[thissel],
                                                  rotangle=None, inv=False)
        patchcats[npatches_south+1+i]={}
        patchcats[npatches_south+1+i]["name"] = "KIDS_North_Patch_%i"%i
        patchcats[npatches_south+1+i]["weight"] = weight[thissel]
        patchcats[npatches_south+1+i]["zbin"] = zbin[thissel]
        patchcats[npatches_south+1+i]["ra"] = ra_rot
        patchcats[npatches_south+1+i]["dec"] = dec_rot
        patchcats[npatches_south+1+i]["g1"] = rot_g1
        patchcats[npatches_south+1+i]["g2"] = rot_g2
        patchcats[npatches_south+1+i]["e1"] = rot_e1
        patchcats[npatches_south+1+i]["e2"] = rot_e2
        patchcats[npatches_south+1+i]["x"] = (ra_rot - np.min(ra_rot))*60 * np.sin(np.pi/2-dec_rot*np.pi/180)
        patchcats[npatches_south+1+i]["y"] = (dec_rot - np.min(dec_rot))*60
        patchcats[npatches_south+1+i]["g1_flat"] = rot_g1
        patchcats[npatches_south+1+i]["g2_flat"] = rot_g2
        patchcats[npatches_south+1+i]["e1_flat"] = rot_e1
        patchcats[npatches_south+1+i]["e2_flat"] = rot_e2
        patchcats[npatches_south+1+i]["incenter"] = np.ones(len(centersel)).astype(np.bool)
        patchcats[npatches_south+1+i]["incenter"] = np.append(patchcats[npatches_south+1+i]["incenter"], np.zeros(len(inoverlap_left)).astype(np.bool))
        patchcats[npatches_south+1+i]["incenter"] = np.append(patchcats[npatches_south+1+i]["incenter"], np.zeros(len(inoverlap_right)).astype(np.bool))

    return patchcats


###############################################
############# UTILS FOR XI-SPLINES ############
##(not needed after integration with onecov) ##
###############################################
import pickle
from scipy.interpolate import UnivariateSpline
import copy
def pickle_load(fname):
    with open(fname, 'rb') as handle:
        res = pickle.load(handle)
    return res

def pickle_save(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def oldprepare_xispl():
    los_start = 2500
    los_batch = 20
    nbatches = 9

    nlos = []
    fbase = "/cosma6/data/dp004/dc-port3/KiDS/cygnus_mocks_meas/tomoxi_cov/"
    for elf in range(nbatches):
        _start = los_start+elf*los_batch
        fname = "treecor_south_noiseless_rbins_0_400_100_los_%i_%i.pickle"%(_start, _start+los_batch)
        nextbatch = pickle_load(fbase+fname)
        nlos.append(len(nextbatch.keys()))
        ntomoz = len(nextbatch[list(nextbatch.keys())[0]].keys())
    nlos_tot = sum(nlos)
    alllos = []

    fulltctomofine = {}
    fulltctomofine["North"] = {}
    fulltctomofine["South"] = {}
    fulltctomofine["Combined"] = {}
    fulltctomofine["Pars"] = {"min_sep":1., "max_sep":200., "nbinsr":100, "los":alllos}
    for patch in ["North", "South", "Combined"]:
        fulltctomofine[patch]["meanr"] = np.zeros((nlos_tot, ntomoz*100), dtype=float)
        fulltctomofine[patch]["weight"] = np.zeros((nlos_tot, ntomoz*100), dtype=float)
        fulltctomofine[patch]["npairs"] = np.zeros((nlos_tot, ntomoz*100), dtype=float)
        fulltctomofine[patch]["xip"] = np.zeros((nlos_tot, ntomoz*100), dtype=complex)
        fulltctomofine[patch]["xim"] = np.zeros((nlos_tot, ntomoz*100), dtype=complex)

    for elf in range(nbatches):
        _start = los_start+elf*los_batch
        fname_north = "treecor_north_noiseless_rbins_0_400_100_los_%i_%i.pickle"%(_start, _start+los_batch)
        fname_south = "treecor_south_noiseless_rbins_0_400_100_los_%i_%i.pickle"%(_start, _start+los_batch)
        nextbatch_north = pickle_load(fbase+fname_north)
        nextbatch_south = pickle_load(fbase+fname_south)
        batchlos = sorted(nextbatch_north.keys())
        thistomoz = sorted(nextbatch_north[batchlos[0]].keys())
        for ellos, nextlos in enumerate(batchlos):
            for k in nextbatch_north[batchlos[0]][thistomoz[0]].keys():
                tmplos = sum(nlos[:elf]) + ellos
                for eltomoz, nexttomoz in enumerate(thistomoz):
                    wnorth = nextbatch_north[nextlos][nexttomoz]["weight"]
                    wsouth = nextbatch_south[nextlos][nexttomoz]["weight"]
                    datnorth = nextbatch_north[nextlos][nexttomoz][k]
                    datsouth = nextbatch_south[nextlos][nexttomoz][k]
                    fulltctomofine["Pars"]["los"].append(tmplos)
                    nextcombined = (wnorth*datnorth + wsouth*datsouth)/(wnorth+wsouth)
                    fulltctomofine["North"][k][tmplos,eltomoz*100:(eltomoz+1)*100] = datnorth
                    fulltctomofine["South"][k][tmplos,eltomoz*100:(eltomoz+1)*100] = datsouth 
                    fulltctomofine["Combined"][k][tmplos,eltomoz*100:(eltomoz+1)*100] = nextcombined

    xispl = {}
    xispl["xip"] = [None]*21
    xispl["xim"] = [None]*21
    nbinsr=100
    tmpzbin = 0
    for zbin1 in range(6):
        for zbin2 in range(zbin1, 6):
            thisr = np.mean(fulltctomofine['Combined']['meanr'][:,tmpzbin*nbinsr:(tmpzbin+1)*nbinsr],axis=0)
            thisxip = np.mean(fulltctomofine['Combined']['xip'][:,tmpzbin*nbinsr:(tmpzbin+1)*nbinsr],axis=0)
            thisxim = np.mean(fulltctomofine['Combined']['xim'][:,tmpzbin*nbinsr:(tmpzbin+1)*nbinsr],axis=0)
            _xipspl = UnivariateSpline(np.log(thisr),np.log(thisxip), s=0)
            _ximspl = UnivariateSpline(np.log(thisr),np.log(thisxim), s=0)
            xispl["xip"][tmpzbin] = copy.deepcopy(_xipspl)
            xispl["xim"][tmpzbin] = copy.deepcopy(_ximspl)
            tmpzbin += 1
    return xispl

def prepare_xispl():
    meanxis = pickle_load("../data/meanxis.pickle")
    xispl = {}
    xispl["xip"] = [None]*21
    xispl["xim"] = [None]*21
    nbinsr=100
    tmpzbin = 0
    for zbin1 in range(6):
        for zbin2 in range(zbin1, 6):
            thisr = meanxis["r"][tmpzbin]
            thisxip = meanxis["xip"][tmpzbin]
            thisxim = meanxis["xim"][tmpzbin]
            _xipspl = UnivariateSpline(np.log(thisr),np.log(thisxip), s=0)
            _ximspl = UnivariateSpline(np.log(thisr),np.log(thisxim), s=0)
            xispl["xip"][tmpzbin] = copy.deepcopy(_xipspl)
            xispl["xim"][tmpzbin] = copy.deepcopy(_ximspl)
            tmpzbin += 1
    return xispl


###############################################
############### UTILS FOR THEORY ##############
##(not needed after integration with onecov) ##
###############################################
def compute_theory(xispl, bin_edges, sigma_eps1, ngals):
    
    nbinsr = len(bin_edges)-1
    nbinsz = len(sigma_eps1)
    nzcombi = int((nbinsz*(nbinsz+1))/2)
    allmixed = np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
    allmixedcounts = np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
    allshape = np.zeros((nzcombi*nbinsr,nzcombi*nbinsr))
    flatz = gen_flatz(nbinsz)
    for elz1 in range(nbinsz):
        for elz2 in range(elz1,nbinsz):
            for elz3 in range(nbinsz):
                for elz4 in range(elz3,nbinsz):
                    #bin_centers = np.mean(fulltctomo["North"]["meanr"][:,elz1*10:(elz1+1)*10],axis=0)
                    bin_centers = bin_edges[:-1] + 2/3*np.diff(bin_edges)
                    start1 = flatz[elz1,elz2]*nbinsr
                    start2 = flatz[elz3,elz4]*nbinsr
                    if elz2==elz4:
                        tmpxispl = lambda r: np.e**xispl["xip"][flatz[elz1,elz3]](np.log(r))
                        nextD, nextq = cov_theory(bin_edges, bin_centers, tmpxispl, sigma_eps1[elz2], ngals[elz2])
                        allmixed[start1:start1+nbinsr,start2:start2+nbinsr] += nextq
                        allmixedcounts[start1:start1+nbinsr,start2:start2+nbinsr] += 1
                        if elz1==elz3:
                            allshape[start1:start1+nbinsr,start2:start2+nbinsr] += np.diag(nextD)
                    if elz2==elz3:
                        tmpxispl = lambda r: np.e**xispl["xip"][flatz[elz1,elz4]](np.log(r))
                        nextD, nextq = cov_theory(bin_edges, bin_centers, tmpxispl, sigma_eps1[elz2], ngals[elz2])
                        allmixed[start1:start1+nbinsr,start2:start2+nbinsr] += nextq
                        allmixedcounts[start1:start1+nbinsr,start2:start2+nbinsr] += 1
                        if elz1==elz4:
                            allshape[start1:start1+nbinsr,start2:start2+nbinsr] += np.diag(nextD)
                    if elz1==elz4:
                        tmpxispl = lambda r: np.e**xispl["xip"][flatz[elz2,elz3]](np.log(r))
                        nextD, nextq = cov_theory(bin_edges, bin_centers, tmpxispl, sigma_eps1[elz1], ngals[elz1])
                        allmixed[start1:start1+nbinsr,start2:start2+nbinsr] += nextq
                        allmixedcounts[start1:start1+nbinsr,start2:start2+nbinsr] += 1
                    if elz1==elz3:
                        tmpxispl = lambda r: np.e**xispl["xip"][flatz[elz2,elz4]](np.log(r))
                        nextD, nextq = cov_theory(bin_edges, bin_centers, tmpxispl, sigma_eps1[elz1], ngals[elz1])
                        allmixed[start1:start1+nbinsr,start2:start2+nbinsr] += nextq
                        allmixedcounts[start1:start1+nbinsr,start2:start2+nbinsr] += 1
    return allmixed, allmixedcounts, allshape
    
def cov_theory(bin_edges, bin_centers, xi_spline, sigma_eps1, N1, sigma_eps2=None, N2=None):
    NBINS_PHI = 100
    if sigma_eps2 is None:
        sigma_eps2 = sigma_eps1
    if N2 is None:
        N2 = N1
    Ngal = np.sqrt(N1*N2)
    sigma2_eps = sigma_eps1**2 + sigma_eps2**2
    dbins = bin_edges[1:] - bin_edges[:-1]
    dphis = np.linspace(-np.pi,np.pi,NBINS_PHI)
    #D = 3.979e-9 * (sigma2_eps/.09)**2 (area*nbar**2/(3600*30**2))**-1 * (bin_centers/1.)**-2 * (dbins/(.1*bin_centers))**-1
    D  = 3.979e-9 * (sigma2_eps/.09)**2 * 3600*900/Ngal * (bin_centers/1.)**-2 * (dbins/(.1*bin_centers))**-1
    qprefac = 2*sigma2_eps/(np.pi*Ngal) 
    q_mm = np.zeros((len(bin_centers),len(bin_centers)))
    for elb1, b1 in enumerate(bin_centers):
        for elb2, b2 in enumerate(bin_centers):
            _arsqrt = b1**2+b2**2-2*b1*b2*np.cos(dphis)
            q_mm[elb1,elb2] = np.nansum(xi_spline(np.sqrt(_arsqrt))) * (dphis[1] - dphis[0])
    q_mm *= qprefac
    return D, q_mm