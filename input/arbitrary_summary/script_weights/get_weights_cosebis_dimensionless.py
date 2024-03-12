import numpy as np
import levin
import matplotlib.pyplot as plt
from mpmath import mp
import mpmath
from scipy.interpolate import interp1d
import multiprocessing as mpi
from scipy.signal import argrelextrema
from scipy.special import p_roots
from numpy.polynomial.legendre import legcompanion, legval, legder
import numpy.linalg as la
from scipy import integrate
from scipy.special import eval_legendre


ell_min = 1 # Minimum multipole
ell_max = 1e5 # Maximum multipole
N_ell = int(1e5) # Number of Fourier modes
N_theta = int(1e4)
get_W_ell_as_well = True # If true the Well are calculated
num_cores = 100 #number of cores used


mp.dps = 160
arcmintorad = 1./60./180.*np.pi
#define constants
Nmax_mm = 20 # maximum COSEBI mode
tmin_mm = 0.5 #theta_min in arcmin
tmax_mm = 300.0 #theta_max in armin

tmin_mm *= arcmintorad
tmax_mm *= arcmintorad
theta_mm = np.geomspace(tmin_mm,tmax_mm, N_theta)



#####################
zmax = mp.log(tmax_mm/tmin_mm)
tbar = (tmax_mm + tmin_mm)/2
BB = (tmax_mm - tmin_mm)/(tmax_mm + tmin_mm)
ell = np.geomspace(ell_min, ell_max, N_ell)


if Nmax_mm > 0:
    ### CASE FOR LN(THETA) EVENLY SPACED

    ##We here compute the c_nj
    #compute the Js
    def J(k,j,zmax):
        # using the lower gamma function returns an error
        #J = mp.gammainc(j+1,0,-k*zmax)
        # so instead we use the full gamma - gamma upper incomplete
        J = mp.gamma(j+1) - mp.gammainc(j+1,-k*zmax)
        k_power = mp.power(-k,j+1.)
        J = mp.fdiv(J,k_power)
        return J


    # matrix of all the coefficient cs 
    coeff_j = mp.matrix(Nmax_mm+1,Nmax_mm+2)
    # add the constraint that c_n(n+1) is = 1
    for i in range(Nmax_mm+1):
        coeff_j[i,i+1] = mp.mpf(1.)


    # determining c_10 and c_11
    nn = 1
    aa = [J(2,0,zmax),J(2,1,zmax)],[J(4,0,zmax),J(4,1,zmax)]
    bb = [-J(2,nn+1,zmax),-J(4,nn+1,zmax)]
    coeff_j_ini = mp.lu_solve(aa,bb)

    coeff_j[1,0] = coeff_j_ini[0]
    coeff_j[1,1] = coeff_j_ini[1]

    # generalised for all N
    # iteration over j up to Nmax_mm solving a system of N+1 equation at each step
    # to compute the next coefficient
    # using the N-1 orthogonal equations Eq34 and the 2 equations 33

    #we start from nn = 2 because we compute the coefficients for nn = 1 above
    for nn in np.arange(2,Nmax_mm+1):
        aa = mp.matrix(int(nn+1))
        bb = mp.matrix(int(nn+1),1)
        #orthogonality conditions: equations (34) 
        for m in np.arange(1,nn): 
            #doing a matrix multiplication (seems the easiest this way in mpmath)
            for j in range(0,nn+1):           
                for i in range(0,m+2): 
                    aa[m-1,j] += J(2,i+j,zmax)*coeff_j[m,i] 
                
            for i in range(0,m+2): 
                bb[int(m-1)] -= J(2,i+nn+1,zmax)*coeff_j[m,i]

        #adding equations (33)
        for j in range(nn+1):
            aa[nn-1,j] = J(2,j,zmax) 
            aa[nn,j]   = J(4,j,zmax) 
            bb[int(nn-1)] = -J(2,nn+1,zmax)
            bb[int(nn)]   = -J(4,nn+1,zmax)

        temp_coeff = mp.lu_solve(aa,bb)
        coeff_j[nn,:len(temp_coeff)] = temp_coeff[:,0].T

    #remove the n = 0 line - so now he n^th row is the n-1 mode.
    coeff_j = coeff_j[1:,:]

    ##We here compute the normalization N_nm, solving equation (35)
    Nn = []
    for nn in np.arange(1,Nmax_mm+1):
        temp_sum = mp.mpf(0)
        for i in range(nn+2):
            for j in range(nn+2):
                temp_sum += coeff_j[nn-1,i]*coeff_j[nn-1,j]*J(2,i+j,zmax)

        temp_Nn = 1./(temp_sum)
        #N_n chosen to be > 0.  
        temp_Nn = mp.sqrt(mp.fabs(temp_Nn))*np.sqrt(tbar**2*BB/tmin_mm**2)
        Nn.append(temp_Nn)


    ##We now want the root of the filter t_+n^log 
    #the filter is: 
    rn = []
    for nn in range(1,Nmax_mm+1):
        rn.append(mpmath.polyroots(coeff_j[nn-1,:nn+2][::-1],maxsteps=500,extraprec=100))
        #[::-1])



def tplus(tmin,tmax,n,norm,root,ntheta=N_theta):
    theta=np.logspace(np.log10(tmin),np.log10(tmax),ntheta)
    tplus=np.zeros((ntheta,2))
    tplus[:,0]=theta
    z=np.log(theta/tmin)
    result=1.
    for r in range(n+1):
        result*=(z-root[r])
    result*=norm
    tplus[:,1]=result
    return tplus

def tminus(tmin,tmax,n,norm,root,tp,ntheta=N_theta):
    tplus_func=interp1d(np.log(tp[:,0]/tmin),tp[:,1])
    rtminus = np.zeros_like(tp)
    rtminus[:,0] = tp[:,0]
    z=np.log(theta/tmin)
    rtminus[:,1]= tplus_func(z)
    lev = levin.Levin(0, 16, 32, 1e-8, 200, num_cores)
    wide_theta = np.linspace(theta[0]*0.999, theta[-1]*1.001,int(N_theta))
    lev.init_w_ell(np.log(wide_theta/tmin_mm), np.ones_like(wide_theta)[:,None])
    z=np.log(theta/tmin)
    for i_z, val_z in enumerate(z[1:]):
        y = np.linspace(z[0],val_z, 1000)
        integrand = tplus_func(y)*(np.exp(2*(y-val_z)) - 3*np.exp(4*(y-val_z)))
        limits_at_mode = np.array(y[argrelextrema(integrand, np.less)[0][:]])
        limits_at_mode_append = np.zeros(len(limits_at_mode) + 2)
        if len(limits_at_mode) != 0:
            limits_at_mode_append[1:-1] = limits_at_mode
        limits_at_mode_append[0] = y[0]
        limits_at_mode_append[-1] = y[-1]
        
        lev.init_integral(y,integrand[:,None],False,False)
        rtminus[i_z, 1] +=  4*lev.cquad_integrate_single_well(limits_at_mode_append,0)[0]
    return rtminus

def get_W_ell(theta, tp):
    lev_w = levin.Levin(0, 16, 32, 1e-8, 200, num_cores)
    lev_w.init_integral(theta,(theta*tp)[:,None]*np.ones(num_cores)[None,:],True,True)
    return lev_w.single_bessel_many_args(ell,0,theta[0],theta[-1])



for nn in range(1,Nmax_mm+1):
    n = nn-1
    tpn = tplus(tmin_mm,tmax_mm,nn,Nn[n],rn[n])
    theta = tpn[:,0]
    tmn = tminus(tmin_mm,tmax_mm,1,Nn[n],rn[n], tpn)
    result_Well = get_W_ell(theta,tpn[:,1])
    tpn[:,1] /= 2
    tmn[:,1] /= 2
    tpn[:,0] /= arcmintorad
    tmn[:,0] /= arcmintorad
    if nn < 10:
        file_tpn = "./../dimensionless_Tp_" +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_0"+str(nn)  + ".table"
        file_tmn = "./../dimensionless_Tm_" +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_0"+str(nn)  + ".table"
        file_Wn = "./../dimensionless_Wn_"  +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_0"+str(nn)  + ".table"
    else:
        file_tpn = "./../dimensionless_Tp_" +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_"+str(nn)  + ".table"
        file_tmn = "./../dimensionless_Tm_" +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_"+str(nn)  + ".table"
        file_Wn = "./../dimensionless_Wn_"  +str(tmin_mm/arcmintorad) + "_to_" + str(tmax_mm/arcmintorad) + "_"+str(nn)  + ".table"
    np.savetxt(file_tpn,tpn)
    np.savetxt(file_tmn,tmn)    
    if get_W_ell_as_well:
        np.savetxt(file_Wn, np.array([ell,result_Well]).T)
  