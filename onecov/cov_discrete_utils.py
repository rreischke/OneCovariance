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
        zbin = np.append(zbin, zind*np.ones(len(nextbit["e1"])).astype(int))
        nextbit = Table.read(basepath+"galCat_run_%i_1_kids1000like_sample0_type%i.fits"%(run,5+2*zind))
        ra = np.append(ra, nextbit["ALPHA_J2000"])
        dec = np.append(dec, nextbit["DELTA_J2000"])
        zspec = np.append(zspec, nextbit["z_spec"])
        g1 = np.append(g1, nextbit["g1"])
        g2 = np.append(g2, nextbit["g2"])
        e1 = np.append(e1, nextbit["e1"])
        e2 = np.append(e2, nextbit["e2"])
        weight = np.append(weight, nextbit["weight"])
        zbin = np.append(zbin, zind*np.ones(len(nextbit["e1"])).astype(int))
    
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
        zbin = np.append(zbin, tomo*zind*np.ones(len(nextbit["e1"])).astype(int))
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
        zbin = np.append(zbin, tomo*zind*np.ones(len(nextbit["e1"])).astype(int))
        ngal[zind] += len(nextbit["g1"])

        
    totcyg = Table()
    totcyg["weight"] = weight
    totcyg["ALPHA_J2000"] = ra
    totcyg["DELTA_J2000"] = dec
    totcyg["e1"] = e1
    totcyg["e2"] = e2
    totcyg["zbin"] = zbin
    
    totcyg.write(fpath_save, format="fits", overwrite=True)
    
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
    indpatches = (np.floor((ra_base-south_min)/dpatch)).astype(int)
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
        patchcats[i]["incenter"] = np.ones(len(centersel)).astype(bool)
        patchcats[i]["incenter"] = np.append(patchcats[i]["incenter"], np.zeros(len(inoverlap_left)).astype(bool))
        patchcats[i]["incenter"] = np.append(patchcats[i]["incenter"], np.zeros(len(inoverlap_right)).astype(bool))
    
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
    patchcats[npatches_south]["incenter"] = np.ones(len(ra_rot)).astype(bool)
    
    npatches_north = 2
    isnorth = np.logical_and(~issouth, ra_base>=190)
    north_min = np.min(ra_base[isnorth])
    north_max = np.max(ra_base[isnorth])
    dpatch = (north_max - north_min)/npatches_north
    indpatches = (np.floor((ra_base-north_min)/dpatch)).astype(int)
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
        patchcats[npatches_south+1+i]["incenter"] = np.ones(len(centersel)).astype(bool)
        patchcats[npatches_south+1+i]["incenter"] = np.append(patchcats[npatches_south+1+i]["incenter"], np.zeros(len(inoverlap_left)).astype(bool))
        patchcats[npatches_south+1+i]["incenter"] = np.append(patchcats[npatches_south+1+i]["incenter"], np.zeros(len(inoverlap_right)).astype(bool))

    return patchcats

