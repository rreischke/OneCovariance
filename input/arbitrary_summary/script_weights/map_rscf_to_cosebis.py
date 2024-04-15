import numpy as np
from mpmath import mp
import mpmath
from scipy.interpolate import interp1d
import multiprocessing as mpi
from scipy import pi,sqrt,exp
from scipy.special import p_roots
from numpy.polynomial.legendre import legcompanion, legval, legder
import numpy.linalg as la
from scipy import integrate
from scipy.special import eval_legendre
import sys
from argparse import ArgumentParser
import os


parser = ArgumentParser(description='Map XIpm to COSEBIs')
parser.add_argument('--covariance', dest='covariance', type=str, required=True, help="File containing Xipm covariance")
parser.add_argument('--ncores', dest='ncores', type=int, required=True, help="Number of cores")
parser.add_argument('--ntomo', dest='ntomo', type=int, required=True, help="Number of tomographic bins")
parser.add_argument('--nmodes', dest='nmodes', type=int, required=True, help="Number of COSEBIs modes")
parser.add_argument('--thetamin', dest='thetamin', type=float, required=True, help="Minimum theta in arcmin")
parser.add_argument('--thetamax', dest='thetamax', type=float, required=True, help="Maximum theta in arcmin")
parser.add_argument('--ntheta', dest='ntheta', type=int, required=True, help="Number of theta bins")
parser.add_argument('--binning', dest='binning', type=str, required=True, help="Type of theta binning. Must be either lin or log.", choices = ['lin', 'log'])
parser.add_argument('--output_cov_EB', dest='output_cov_EB', type=str, required=True, help="Output directory for the EB-mode covariance matrix")
parser.add_argument('--filename_cov', dest='filename_cov', type=str, required=True, help="Output filename of the combined EB covariance matrix")
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
Nmax_mm = pars.nmodes

theta_min_mm = args.thetamin 
theta_max_mm = args.thetamax
arcmintorad = 1./60./180.*np.pi
thetaRange=str(args.thetamin)+'-'+str(args.thetamax)

covfile = args.covariance
n_tomo = args.ntomo

covariance_xipm = np.array(np.loadtxt(covfile))


n_data = int(n_tomo*(n_tomo + 1)/2)
tmin_mm *= arcmintorad
tmax_mm *= arcmintorad



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


Xipm_to_E_n = np.zeros((Nmax_mm, 2*len(theta_bins)))
Xipm_to_B_n = np.zeros((Nmax_mm, 2*len(theta_bins)))




for n in range(Nmax_mm):
    theta_int = theta_bins
    dtheta = np.zeros_like(theta_int)
    dtheta[:-1] = theta_int[1:] - theta_int[:-1]
    dtheta[-1] = dtheta[-2]/2
    Tplus_inter = np.interp(theta_int,theta,Tplus[n,:])
    Tminus_inter = np.interp(theta_int,theta,Tminus[n,:])
    Xipm_to_E_n[n, :len(theta_bins)] = Tplus_inter*theta_bins*dtheta/2
    Xipm_to_E_n[n, len(theta_bins):] = Tminus_inter*theta_bins*dtheta/2
    Xipm_to_B_n[n, :len(theta_bins)] = Tplus_inter*theta_bins*dtheta/2
    Xipm_to_B_n[n, len(theta_bins):] = -Tminus_inter*theta_bins*dtheta/2




cov_new = np.zeros((2*n_data*Nmax_mm,2*n_data*Nmax_mm))
for i in range(n_data):
    bin_idx = i
    b1i = (bin_idx)*len(theta_bins)
    bp1i = (bin_idx + 1)*len(theta_bins)   
    adding = n_data*len(theta_bins) 
    for j in range(n_data):
        bin_idx = j
        b1j = (bin_idx)*len(theta_bins)
        bp1j = (bin_idx + 1)*len(theta_bins)
        covv = np.block([[covariance_xipm[b1i:bp1i, b1j:bp1j] , covariance_xipm[b1i:bp1i, b1j+ adding:bp1j + adding]] , [covariance_xipm[b1i+ adding:bp1i + adding, b1j:bp1j] , covariance_xipm[b1i+ adding:bp1i +adding, b1j+ adding:bp1j + adding]]])
        cov_new[i*Nmax_mm:(i+1)*Nmax_mm, j*Nmax_mm:(j+1)*Nmax_mm] = Xipm_to_E_n@covv@Xipm_to_E_n.T
        cov_new[i*Nmax_mm:(i+1)*Nmax_mm, j*Nmax_mm + n_data*Nmax_mm:(j+1)*Nmax_mm + n_data*Nmax_mm] = Xipm_to_E_n@covv@Xipm_to_B_n.T
        cov_new[i*Nmax_mm + n_data*Nmax_mm:(i+1)*Nmax_mm + n_data*Nmax_mm, j*Nmax_mm + n_data*Nmax_mm:(j+1)*Nmax_mm + n_data*Nmax_mm] = Xipm_to_B_n@covv@Xipm_to_B_n.T
np.savetxt(output_cov_E+'/'+filename_cov,covariance_cosebis)
