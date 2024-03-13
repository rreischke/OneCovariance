import numpy as np
import levin
import matplotlib.pyplot as plt
from mpmath import mp
import mpmath
from scipy.interpolate import interp1d
import multiprocessing as mpi
from scipy.signal import argrelextrema
from scipy import pi,sqrt,exp
from scipy.special import p_roots
from numpy.polynomial.legendre import legcompanion, legval, legder
import numpy.linalg as la
from scipy import integrate
from scipy.special import eval_legendre
import sys
from argparse import ArgumentParser


parser = ArgumentParser(description='Map COSEBIs to pure E/B mode correlation functions')
parser.add_argument('--data', dest='data', type=str, required=True, help="File containing E and B COSEBIS measurements")
parser.add_argument('--covariance', dest='covariance', type=str, required=True, help="File containing E and B COSEBIS covariance")
parser.add_argument('--ncores', dest='ncores', type=int, required=True, help="Number of cores")
parser.add_argument('--ntomo', dest='ntomo', type=int, required=True, help="Number of tomographic bins")
parser.add_argument('--thetamin', dest='thetamin', type=int, required=True, help="Minimum theta in arcmin")
parser.add_argument('--thetamax', dest='thetamax', type=int, required=True, help="Maximum theta in arcmin")
parser.add_argument('--ntheta', dest='ntheta', type=int, required=True, help="Number of theta bins")
parser.add_argument('--binning', dest='binning', type=str, required=True, help="Type of theta binning. Must be either lin or log.", choices = ['lin', 'log'])
parser.add_argument('--output_data', dest='output_data', type=str, required=True, help="Output directory for the data vector")
parser.add_argument('--output_cov', dest='output_cov', type=str, required=True, help="Output directory for the covariance matrix")
parser.add_argument('--filename_data', dest='filename_data', type=str, required=True, help="Output filename of the combined xi_EB data vector")
parser.add_argument('--filename_cov', dest='filename_cov', type=str, required=True, help="Output filename of the combined xi_EB covariance matrix")



args = parser.parse_args()

num_cores = args.ncores
tmin_mm = args.thetamin #theta_min in arcmin
tmax_mm = args.thetamax #theta_max in armin
ntheta_bins_mm = args.ntheta #number of theta bins for lensing
theta_type_mm = args.binning # type of theta binning for lensing
output_data = args.output_data
output_cov = args.output_cov
filename_data = args.filename_data
filename_cov = args.filename_cov




theta_min_mm = args.thetamin 
theta_max_mm = args.thetamax
N_theta = 1000
arcmintorad = 1./60./180.*np.pi

# if len(sys.argv) == 4:
#     signalfile = str(sys.argv[1])
#     covfile = str(sys.argv[2])
#     n_tomo = int(sys.argv[3])
# else:
#     print(r"Please pass first the signal file, then the covariance matrix file of the COSEBIs and last the number of tomographic bins")
#     print(r"I.e.: python mapping_cosebis_to_pureEBmode_cf.py signal_file.txt covariance_file.txt 6")

signalfile = args.data
covfile = args.covariance
n_tomo = args.ntomo

covariance_cosebi = np.array(np.loadtxt(covfile))
signal_cosebi = np.array(np.loadtxt(signalfile))



n_data = int(n_tomo*(n_tomo + 1)/2)
Nmax_mm = int(len(covariance_cosebi)/2/n_data)

tmin_mm *= arcmintorad
tmax_mm *= arcmintorad
theta_mm = np.geomspace(tmin_mm,tmax_mm, N_theta)

B = (tmax_mm - tmin_mm)/(tmax_mm + tmin_mm)
bar_theta = (tmin_mm + tmax_mm)/2
zmax = mp.log(tmax_mm/tmin_mm)


def get_theta_bins(theta_type, theta_min, theta_max, ntheta_bins, theta_list_boundary = None):
    '''
    This function returns the theta bins and the corresponding bin boundaries

    Parameters:
    -----------
    theta_type : string
        Do you want lin-/log-spaced theta's or a list? Can be 'lin', 'log' or 'list'
    theta_min : float
        Minimum angle (lowest bin boundary NOT the central value) in arcmin
    theta_max : float
        Maximum angle (higest bin boundary NOT the central value) in arcmin
    ntheta_bins : integer
        How many theta bins should there be?
    theta_list_boundary : array
        Array of all bin boundaries (arbitrarily spaced)
    '''
    if theta_type == 'lin':
        theta_ul_bins = np.linspace(theta_min, theta_max, ntheta_bins + 1)
        theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])
    if theta_type == 'log':
        theta_ul_bins = np.geomspace(theta_min, theta_max, ntheta_bins + 1)
        theta_bins = np.exp(.5 * (np.log(theta_ul_bins[1:])
                                + np.log(theta_ul_bins[:-1])))
    if theta_type == 'list' and theta_list_boundary is not None:
        theta_ul_bins = theta_list_boundary
        theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])

    return theta_bins, theta_ul_bins

theta_bins, theta_ul_bins = get_theta_bins(theta_type=theta_type_mm,theta_min = theta_min_mm, theta_max = theta_max_mm, ntheta_bins=ntheta_bins_mm)


### get Tpm and theta from file here
Tplus = np.zeros((Nmax_mm,N_theta))
Tminus = np.zeros((Nmax_mm,N_theta))




En_to_E_plus = np.zeros((len(theta_bins), Nmax_mm))
En_to_E_minus = np.zeros((len(theta_bins), Nmax_mm))
Bn_to_B_plus = np.zeros((len(theta_bins), Nmax_mm))
Bn_to_B_minus = np.zeros((len(theta_bins), Nmax_mm))

for i_theta in range(len(theta_bins)):
    for n in range(Nmax_mm):
        En_to_E_plus[i_theta, n] = 2*bar_theta**2/B*np.interp(theta_bins[i_theta],theta/arcmintorad,Tplus[n,:])
        En_to_E_minus[i_theta, n] = 2*bar_theta**2/B*np.interp(theta_bins[i_theta],theta/arcmintorad,Tminus[n,:])
        Bn_to_B_plus[i_theta, n] = 2*bar_theta**2/B*np.interp(theta_bins[i_theta],theta/arcmintorad,Tplus[n,:])
        Bn_to_B_minus[i_theta, n] = 2*bar_theta**2/B*np.interp(theta_bins[i_theta],theta/arcmintorad,Tminus[n,:])

covariance_xiE_p = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))
covariance_xiE_m = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))
covariance_xiE_pm = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))

covariance_xiB_p = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))
covariance_xiB_m = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))
covariance_xiB_pm = np.zeros((n_data*len(theta_bins), n_data*len(theta_bins)))

signal_xiE_p = np.zeros(n_data*len(theta_bins))
signal_xiE_m = np.zeros(n_data*len(theta_bins))
signal_xiB_p = np.zeros(n_data*len(theta_bins))
signal_xiB_m = np.zeros(n_data*len(theta_bins))

for i in range(n_data):
    for m in range(Nmax_mm):
        signal_xiE_p[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_plus[:, None, m]*signal_cosebi[i*Nmax_mm + m]
        signal_xiE_m[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_minus[:, None, m]*signal_cosebi[i*Nmax_mm + m]
        signal_xiB_p[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_plus[:, None, m]*signal_cosebi[i*Nmax_mm + m + n_data*Nmax_mm]
        signal_xiB_m[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_minus[:, None, m]*signal_cosebi[i*Nmax_mm + m + n_data*Nmax_mm]


for i in range(n_data):
    for j in range(n_data):
        for m in range(Nmax_mm):
            for n in range(Nmax_mm):
                covariance_xiE_p[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += En_to_E_plus[:, None, m]*En_to_E_plus[None, :,n]*(covariance_cosebi[i*Nmax_mm + m, j*Nmax_mm  + n])
                covariance_xiE_m[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += En_to_E_minus[:, None, m]*En_to_E_minus[None, :,n]*covariance_cosebi[i*Nmax_mm + m, j*Nmax_mm  + n]
                covariance_xiE_pm[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += En_to_E_plus[:, None, m]*En_to_E_minus[None, :,n]*covariance_cosebi[i*Nmax_mm + m, j*Nmax_mm  + n]

for i in range(n_data):
    for j in range(n_data):
        for m in range(Nmax_mm):
            for n in range(Nmax_mm):
                covariance_xiB_p[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += Bn_to_B_plus[:, None, m]*Bn_to_B_plus[None, :,n]*covariance_cosebi[i*Nmax_mm + m + n_data*Nmax_mm, j*Nmax_mm  + n + n_data*Nmax_mm]
                covariance_xiB_m[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += Bn_to_B_minus[:, None, m]*Bn_to_B_minus[None, :,n]*covariance_cosebi[i*Nmax_mm + m + n_data*Nmax_mm, j*Nmax_mm  + n + n_data*Nmax_mm]
                covariance_xiB_pm[i*len(theta_bins) : (i+1)*len(theta_bins), j*len(theta_bins) : (j+1)*len(theta_bins)] += Bn_to_B_plus[:, None, m]*Bn_to_B_minus[None, :,n]*covariance_cosebi[i*Nmax_mm + m + n_data*Nmax_mm, j*Nmax_mm  + n + n_data*Nmax_mm]


signal_XiE_pm = np.block([signal_xiE_p,signal_xiE_m]).T/arcmintorad**2
signal_XiB_pm = np.block([signal_xiB_p,signal_xiB_m]).T/arcmintorad**2
covariance_XiE_pm = np.block([[covariance_xiE_p, covariance_xiE_pm],[covariance_xiE_pm.T,covariance_xiE_m]])/arcmintorad**4
covariance_XiB_pm = np.block([[covariance_xiB_p, covariance_xiB_pm],[covariance_xiB_pm.T,covariance_xiB_m]])/arcmintorad**4

signal_XiEB_pm = np.concatenate((signal_XiE_pm,signal_XiB_pm))
covariance_XiEB_pm = np.block([[covariance_XiE_pm, np.zeros(covariance_XiE_pm.shape)],[np.zeros(covariance_XiE_pm.shape),covariance_XiB_pm]])

np.savetxt(output_data+'/signal_XiE_pm.dat',signal_XiE_pm)
np.savetxt(output_data+'/signal_XiB_pm.dat',signal_XiB_pm)
np.savetxt(output_data+'/'+filename_data,signal_XiEB_pm)

np.savetxt(output_cov+'/covariance_XiE_pm.mat',covariance_XiE_pm)
np.savetxt(output_cov+'/covariance_XiB_pm.mat',covariance_XiB_pm)
np.savetxt(output_cov+'/'+filename_cov,covariance_XiEB_pm)

# np.savetxt("signal_XiE_pm.dat", np.block([signal_xiE_p,signal_xiE_m]).T/arcmintorad**2)
# np.savetxt("signal_XiB_pm.dat", np.block([signal_xiB_p,signal_xiB_m]).T/arcmintorad**2)

# np.savetxt("covariance_XiE_pm.mat", np.block([[covariance_xiE_p, covariance_xiE_pm],
#                         [covariance_xiE_pm.T,covariance_xiE_m]])/arcmintorad**4)
# np.savetxt("covariance_XiB_pm.mat", np.block([[covariance_xiB_p, covariance_xiB_pm],
#                         [covariance_xiB_pm.T,covariance_xiB_m]])/arcmintorad**4)