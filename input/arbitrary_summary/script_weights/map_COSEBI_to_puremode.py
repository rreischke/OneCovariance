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
import os


parser = ArgumentParser(description='Map COSEBIs to pure E/B mode correlation functions')
parser.add_argument('--data', dest='data', type=str, required=True, nargs=2, help="File containing E and B COSEBIS measurements")
parser.add_argument('--covariance', dest='covariance', type=str, required=True, help="File containing E and B COSEBIS covariance")
parser.add_argument('--ncores', dest='ncores', type=int, required=True, help="Number of cores")
parser.add_argument('--ntomo', dest='ntomo', type=int, required=True, help="Number of tomographic bins")
parser.add_argument('--thetamin', dest='thetamin', type=float, required=True, help="Minimum theta in arcmin")
parser.add_argument('--thetamax', dest='thetamax', type=float, required=True, help="Maximum theta in arcmin")
parser.add_argument('--ntheta', dest='ntheta', type=int, required=True, help="Number of theta bins")
parser.add_argument('--binning', dest='binning', type=str, required=True, help="Type of theta binning. Must be either lin or log.", choices = ['lin', 'log'])
parser.add_argument('--output_data_E', dest='output_data_E', type=str, required=True, help="Output directory for the E-mode data vector")
parser.add_argument('--output_data_B', dest='output_data_B', type=str, required=True, help="Output directory for the B_mode data vector")
parser.add_argument('--output_cov_E', dest='output_cov_E', type=str, required=True, help="Output directory for the E-mode covariance matrix")
parser.add_argument('--output_cov_B', dest='output_cov_B', type=str, required=True, help="Output directory for the B-mode covariance matrix")
parser.add_argument('--filename_data', dest='filename_data', type=str, required=True, help="Output filename of the combined xi_EB data vector")
parser.add_argument('--filename_cov', dest='filename_cov', type=str, required=True, help="Output filename of the combined xi_EB covariance matrix")
parser.add_argument('--tfoldername', dest="tfoldername", default="Tplus_minus", required=True,
    help='name and full address of the folder for Tplus Tminus files for COSEBIs, will not make it if it does not exist')
parser.add_argument('--tplusfile', dest="tplusfile", default="Tplus", required=False,
    help='name of Tplus file for COSEBIs, will look for it before running the code')
parser.add_argument('--tminusfile', dest="tminusfile", default="Tminus", required=False,
    help='name of Tplus file for COSEBIs, will look for it before running the code')


args = parser.parse_args()

num_cores = args.ncores
tmin_mm = args.thetamin #theta_min in arcmin
tmax_mm = args.thetamax #theta_max in armin
ntheta_bins_mm = args.ntheta #number of theta bins for lensing
theta_type_mm = args.binning # type of theta binning for lensing
output_data_E = args.output_data_E
output_data_B = args.output_data_B
output_cov_E = args.output_cov_E
output_cov_B = args.output_cov_B
filename_data = args.filename_data
filename_cov = args.filename_cov
tfoldername = args.tfoldername
tplusfile=args.tplusfile
tminusfile=args.tminusfile

theta_min_mm = args.thetamin 
theta_max_mm = args.thetamax
arcmintorad = 1./60./180.*np.pi
thetaRange=str(args.thetamin)+'-'+str(args.thetamax)

signalfile = args.data[1]
covfile = args.covariance
n_tomo = args.ntomo

covariance_cosebi = np.array(np.loadtxt(covfile))
signal_cosebi = np.array(np.loadtxt(signalfile))



n_data = int(n_tomo*(n_tomo + 1)/2)
Nmax_mm = int(len(covariance_cosebi)/2/n_data)

tmin_mm *= arcmintorad
tmax_mm *= arcmintorad

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
theta = np.loadtxt(tfoldername+'/'+tplusfile+'_n'+str(1)+'_'+thetaRange+'.table',comments='#').T[0]
N_theta = len(theta)
Tplus = np.zeros((Nmax_mm,N_theta))
Tminus = np.zeros((Nmax_mm,N_theta))

for n in range(1,Nmax_mm+1):
    TplusFileName= tfoldername+'/'+tplusfile+'_n'+str(n)+'_'+thetaRange+'.table'
    TminusFileName= tfoldername+'/'+tminusfile+'_n'+str(n)+'_'+thetaRange+'.table'
    if(os.path.isfile(TplusFileName)):
        file = open(TplusFileName)
        Tplus[n-1]=np.loadtxt(file,comments='#').T[1]
    if(os.path.isfile(TminusFileName)):
        file = open(TminusFileName)
        Tminus[n-1]=np.loadtxt(file,comments='#').T[1]

En_to_E_plus = np.zeros((len(theta_bins), Nmax_mm))
En_to_E_minus = np.zeros((len(theta_bins), Nmax_mm))
Bn_to_B_plus = np.zeros((len(theta_bins), Nmax_mm))
Bn_to_B_minus = np.zeros((len(theta_bins), Nmax_mm))

for n in range(Nmax_mm):
    for i_theta in range(len(theta_bins)):
        theta_int = np.geomspace(theta_ul_bins[i_theta] , theta_ul_bins[i_theta + 1], 100)
        Tplus_inter = np.interp(theta_int,theta,Tplus[n,:])
        Tminus_inter = np.interp(theta_int,theta,Tminus[n,:])
        Tplus_binned = 1/(theta_ul_bins[i_theta + 1] - theta_ul_bins[i_theta])*np.trapz(Tplus_inter,theta_int)
        Tminus_binned = 1/(theta_ul_bins[i_theta + 1] - theta_ul_bins[i_theta])*np.trapz(Tminus_inter,theta_int)
        En_to_E_plus[i_theta, n] = 1/bar_theta**2/B*Tplus_binned
        En_to_E_minus[i_theta, n] = 1/bar_theta**2/B*Tminus_binned
        Bn_to_B_plus[i_theta, n] = 1/bar_theta**2/B*Tplus_binned
        Bn_to_B_minus[i_theta, n] = 1/bar_theta**2/B*Tminus_binned

# for i_theta in range(len(theta_bins)):
#     for n in range(Nmax_mm):
#         En_to_E_plus[i_theta, n] = bar_theta**2/B*np.interp(theta_bins[i_theta],theta,Tplus[n,:])
#         En_to_E_minus[i_theta, n] = bar_theta**2/B*np.interp(theta_bins[i_theta],theta,Tminus[n,:])
#         Bn_to_B_plus[i_theta, n] = bar_theta**2/B*np.interp(theta_bins[i_theta],theta,Tplus[n,:])
#         Bn_to_B_minus[i_theta, n] = bar_theta**2/B*np.interp(theta_bins[i_theta],theta,Tminus[n,:])

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
        signal_xiE_p[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_plus[:, m]*signal_cosebi[i*Nmax_mm + m]
        signal_xiE_m[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_minus[:, m]*signal_cosebi[i*Nmax_mm + m]
        signal_xiB_p[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_plus[:, m]*signal_cosebi[i*Nmax_mm + m + n_data*Nmax_mm]
        signal_xiB_m[i*len(theta_bins) : (i+1)*len(theta_bins)] +=  En_to_E_minus[:, m]*signal_cosebi[i*Nmax_mm + m + n_data*Nmax_mm]


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

signal_XiE_pm = np.block([signal_xiE_p,signal_xiE_m]).T
signal_XiB_pm = np.block([signal_xiB_p,signal_xiB_m]).T
covariance_XiE_pm = np.block([[covariance_xiE_p, covariance_xiE_pm],[covariance_xiE_pm.T,covariance_xiE_m]])
covariance_XiB_pm = np.block([[covariance_xiB_p, covariance_xiB_pm],[covariance_xiB_pm.T,covariance_xiB_m]])
# signal_XiE_pm = np.block([signal_xiE_p,signal_xiE_m]).T/arcmintorad**2
# signal_XiB_pm = np.block([signal_xiB_p,signal_xiB_m]).T/arcmintorad**2
# covariance_XiE_pm = np.block([[covariance_xiE_p, covariance_xiE_pm],[covariance_xiE_pm.T,covariance_xiE_m]])/arcmintorad**4
# covariance_XiB_pm = np.block([[covariance_xiB_p, covariance_xiB_pm],[covariance_xiB_pm.T,covariance_xiB_m]])/arcmintorad**4

signal_XiEB_pm = np.concatenate((signal_XiE_pm,signal_XiB_pm))
covariance_XiEB_pm = np.block([[covariance_XiE_pm, np.zeros(covariance_XiE_pm.shape)],[np.zeros(covariance_XiE_pm.shape),covariance_XiB_pm]])

np.savetxt(output_data_E+'/'+filename_data,signal_XiE_pm)
np.savetxt(output_data_E+'/combined_vector_no_m_bias.txt',signal_XiE_pm)
np.savetxt(output_data_B+'/'+filename_data,signal_XiB_pm)
np.savetxt(output_data_B+'/combined_vector_no_m_bias.txt',signal_XiB_pm)

np.savetxt(output_cov_E+'/'+filename_cov,covariance_XiE_pm)
np.savetxt(output_cov_B+'/'+filename_cov,covariance_XiB_pm)

# np.savetxt("signal_XiE_pm.dat", np.block([signal_xiE_p,signal_xiE_m]).T/arcmintorad**2)
# np.savetxt("signal_XiB_pm.dat", np.block([signal_xiB_p,signal_xiB_m]).T/arcmintorad**2)

# np.savetxt("covariance_XiE_pm.mat", np.block([[covariance_xiE_p, covariance_xiE_pm],
#                         [covariance_xiE_pm.T,covariance_xiE_m]])/arcmintorad**4)
# np.savetxt("covariance_XiB_pm.mat", np.block([[covariance_xiB_p, covariance_xiB_pm],
#                         [covariance_xiB_pm.T,covariance_xiB_m]])/arcmintorad**4)