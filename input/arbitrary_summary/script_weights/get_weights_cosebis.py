import numpy as np
import levin
import matplotlib.pyplot as plt
from mpmath import mp
import mpmath
from scipy.interpolate import interp1d
import multiprocessing as mpi
mpi.set_start_method("fork")

mp.dps = 160
num_cores = 8 #number of cores used
#define constants
Nmax = 5 # maximum mode
tmin = 0.5 #theta_min in arcmin
tmax = 300.0 #theta_max in armin
zmax = mp.log(tmax/tmin)
ell_min = 1
ell_max = 1e5
N_ell = int(1e4)
ell = np.geomspace(ell_min, ell_max, N_ell)



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
coeff_j = mp.matrix(Nmax+1,Nmax+2)
# add the constraint that c_n(n+1) is = 1
for i in range(Nmax+1):
    coeff_j[i,i+1] = mp.mpf(1.)


# determining c_10 and c_11
nn = 1
aa = [J(2,0,zmax),J(2,1,zmax)],[J(4,0,zmax),J(4,1,zmax)]
bb = [-J(2,nn+1,zmax),-J(4,nn+1,zmax)]
coeff_j_ini = mp.lu_solve(aa,bb)

coeff_j[1,0] = coeff_j_ini[0]
coeff_j[1,1] = coeff_j_ini[1]

# generalised for all N
# iteration over j up to Nmax solving a system of N+1 equation at each step
# to compute the next coefficient
# using the N-1 orthogonal equations Eq34 and the 2 equations 33

#we start from nn = 2 because we compute the coefficients for nn = 1 above
for nn in np.arange(2,Nmax+1):
    aa = mp.matrix(int(nn+1))
    bb = mp.matrix(int(nn+1),1)
    #orthogonality conditions: equations (34) 
    for m in np.arange(1,nn): 
        #doing a matrix multiplication (seems the easiest this way in mpmath)
        for j in range(0,nn+1):           
            for i in range(0,m+2): 
                aa[m-1,j] += J(1,i+j,zmax)*coeff_j[m,i] 
            
        for i in range(0,m+2): 
            bb[int(m-1)] -= J(1,i+nn+1,zmax)*coeff_j[m,i]

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
for nn in np.arange(1,Nmax+1):
    temp_sum = mp.mpf(0)
    for i in range(nn+2):
        for j in range(nn+2):
            temp_sum += coeff_j[nn-1,i]*coeff_j[nn-1,j]*J(1,i+j,zmax)

    temp_Nn = (mp.expm1(zmax))/(temp_sum)
    #N_n chosen to be > 0.  
    temp_Nn = mp.sqrt(mp.fabs(temp_Nn))
    Nn.append(temp_Nn)


##We now want the root of the filter t_+n^log 
#the filter is: 
rn = []
for nn in range(1,Nmax+1):
    rn.append(mpmath.polyroots(coeff_j[nn-1,:nn+2][::-1],maxsteps=500,extraprec=100))
    #[::-1])



def tplus(tmin,tmax,n,norm,root,ntheta=10000):
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

def tminus(tmin,tmax,n,norm,root,tp,ntheta=10000):
    tplus_func=interp1d(np.log(tp[:,0]/tmin),tp[:,1])
    tminus = np.zeros_like(tp)
    tminus[:,0] = theta
    tminus[:,1]=tplus_func(z)
    for i_z, val_z in enumerate(z[1:]):
        y = np.linspace(z[0],val_z, 1000)
        integrand = tplus_func(y)*(np.exp(2*(y-val_z)) - 3*np.exp(4*(y-val_z)))
        lev.init_integral(y,integrand[:,None],True,True)
        tminus[i_z, 1] +=  4*lev.cquad_integrate_single_well([y[0],y[-1]],0)[0]
    return tminus


for nn in range(1,Nmax+1):
    print("At mode",nn)
    n = nn-1
    tpn = tplus(tmin,tmax,nn,Nn[n],rn[n])
    lev = levin.Levin(0, 16, 32, 1e-8, 200)
    theta = tpn[:,0]
    z=np.log(theta/tmin)
    lev.init_w_ell(z, np.ones_like(theta)[:,None])
    tmn = tminus(tmin,tmax,1,Nn[n],rn[n], tpn)[:,1]
    lev_w = levin.Levin(0, 16, 32, 1e-8, 200)
    arcmintorad = np.pi/180/60
    lev_w.init_integral(theta*arcmintorad,(theta*arcmintorad*tpn[:,1])[:,None],True,True)

    global getWell

    def getWell(ells):
        result = lev_w.single_bessel(ells,0,theta[0]*arcmintorad,theta[-1]*arcmintorad)[0]
        return result
    
    
    pool = mpi.Pool(num_cores)
    result_Well = np.array(pool.map(
                    getWell, ell))
    pool.close()
    pool.terminate()
    if nn < 10:
        file_tpn = "./../Tplus_"+str(nn) + "_0" +str(tmin) + "_to_" + str(tmax) + ".table"
        file_tmn = "./../Tminus_"+str(nn) + "_0" +str(tmin) + "_to_" + str(tmax) + ".table"
        file_Wn = "./../WnLog_"+str(nn) + "_0" +str(tmin) + "_to_" + str(tmax) + ".table"
    else:
        file_tpn = "./../Tplus_"+str(nn) + "_" +str(tmin) + "_to_" + str(tmax) + ".table"
        file_tmn = "./../Tminus_"+str(nn) + "_" +str(tmin) + "_to_" + str(tmax) + ".table"
        file_Wn = "./../WnLog_"+str(nn) + "_" +str(tmin) + "_to_" + str(tmax) + ".table"
    np.savetxt(file_tpn,2*tpn)
    np.savetxt(file_tmn,2*tmn)    
    np.savetxt(file_Wn, np.array([ell,result_Well]).T)

    



