from onecov.cov_input import Input, FileInput
from onecov.cov_output import Output
from onecov.cov_theta_space import CovTHETASpace
from astropy.cosmology import FlatLambdaCDM, Planck15

import os
import numpy as np
import sys
import matplotlib.pyplot as plt


if len(sys.argv) < 1:
    which_survey = "VIPERS"
else:
    which_survey = str(sys.argv[1])

config = "./config_files/config_cz.ini"
r_low = 0.1 # Scales considered 
r_hig = 1.0
limber = True #should the Cells be calculated using Limber projection?
diagonal_only = False # should only be autocorrelations be considered, that is autocorrelations in the spectroscopic sample
nonGaussian = True
npair_spec = True
inp = Input()
covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input(
    config)
fileinp = FileInput(bias)
read_in_tables = fileinp.read_input(inp.config_name)
out = Output(output)


if which_survey == 'mice2_mock':
    path_to_reference_sample = str("./../clustering-z_covariance/data/mice2_mock/nz_reference.dat") # path to the spectroscopic refence sample
    path_to_nz_true = str("./../clustering-z_covariance/data/mice2_mock/nz_true/nz_true/nz_true_") # path to sample to be calibrated must integrate to the actual number of galaxies
    n_tomo_source = 5
    save_path_cz_covariance = str("./../clustering-z_covariance/data_onecov/mice2_mock/cz_covariance_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    save_path_spec_covariance = str("./../clustering-z_covariance/data_onecov/mice2_mock/spec_covariance_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    save_path_spec_Cell = str("./../clustering-z_covariance/data_onecov/mice2_mock/Cell_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    survey_params['survey_area_clust'][0] = float(np.loadtxt(str("./../clustering-z_covariance/data/") +which_survey + str("/area")))
else:
    path_to_reference_sample = str("./../clustering-z_covariance/data/") +which_survey + str("/true/nz_reference.dat")   # path to the spectroscopic refence sample
    path_to_nz_true = str("./../clustering-z_covariance/data/") + which_survey + str("/true/nz_true_") # path to sample to be calibrated must integrate to the actual number of galaxies
    path_to_reference_pair = str("./../clustering-z_covariance/data/") + which_survey + str("/estimate/auto_reference.count") # path to sample to be calibrated must integrate to the actual number of galaxies
    path_to_nz_pair = str("./../clustering-z_covariance/data/") + which_survey + str("/estimate/cross_") # path to sample to be calibrated must integrate to the actual number of galaxies
    
    n_tomo_source = 6
    save_path_cz_covariance = str("./../clustering-z_covariance/data_onecov/") + which_survey+  str("/cz_covariance_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    save_path_spec_covariance = str("./../clustering-z_covariance/data_onecov/") + which_survey+  str("/spec_covariance_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    save_path_spec_Cell = str("./../clustering-z_covariance/data_onecov/") + which_survey+  str("/Cell_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
    survey_params['survey_area_clust'][0] = float(np.loadtxt(str("./../clustering-z_covariance/data/") +which_survey + str("/area")))



# Setting up the OneCovariance code

# How fine should the binning bes
nz_binning = 0.001


# Number counts for the reference sample
galcount = np.array(np.loadtxt(path_to_reference_sample)[:, 2])
zbound = np.array(np.loadtxt(path_to_reference_sample)[:, 1])
zbound = zbound[np.where(galcount > 0)[0]]

measure_index = np.where(galcount > 0)[0]

zbound = np.insert(zbound,0,float(np.loadtxt(path_to_reference_sample)[np.where(galcount > 0)[0][0],0]),0)
Npair = np.array(np.loadtxt(path_to_reference_pair))[np.where(galcount > 0)[0]]
galcount = galcount[np.where(galcount > 0)[0]]

# Survey area of the CZ measurements
survey_area = survey_params['survey_area_clust'][0]*3600.  # in arcmin^2


# Redshift boundaries
zmean = (zbound[1:]+zbound[:-1])/2.
deltaz = (zbound[1:]-zbound[:-1])
# Corresponding co-moving distances
fkchi = Planck15.comoving_transverse_distance(zmean).value 
# Angular diameter distance
d_ang = fkchi/(1.+zmean)
# Corresponding angular range for measurements
theta_low = np.arctan(r_low/d_ang)/np.pi*180.*60. 
theta_hig = np.arctan(r_hig/d_ang)/np.pi*180.*60.

# Number density of the reference sample
if npair_spec:
    ndens_spec = np.sqrt(Npair/((theta_hig**2 - theta_low**2)*np.pi*survey_area))
else:
    ndens_spec = galcount / survey_area


zbins = np.arange(zbound[0],
                  zbound[-1], nz_binning)

# Redshift distributions to be calibrated (including some interpolations)
#n_tomo_source = np.shape(read_in_tables['zclust']['nz'])[
#    0]
nz_interp = np.zeros(n_tomo_source*len(zbins)
                     ).reshape(n_tomo_source, len(zbins))



neff_phot = np.zeros(n_tomo_source)
for j in range(n_tomo_source):
    ndens_phot_file = path_to_nz_true+ str(j+1) +".dat"
    ndens_phot = np.array(np.loadtxt(ndens_phot_file)[:, 2])
    Npair_spec_phot =np.array(np.loadtxt(path_to_nz_pair + str(j+1) + ".count"))[measure_index]
    
    #neff_phot[j,:] = Npair_spec_phot/ndens_spec/((theta_hig**2 - theta_low**2)*np.pi*survey_area)
    #print(neff_phot[j,:]/10)
    #neff_phot[j,:] /= survey_area
    neff_phot[j] = np.sum(ndens_phot)/(survey_area)/15
    #print(neff_phot[j,:])
    nz_interp[j] = np.interp(zbins, np.array(np.loadtxt(ndens_phot_file)[:, 0]), ndens_phot)

    
read_in_tables['zclust']['z'] = zbins

filename_old1 = os.path.splitext(os.path.basename(output['file'][0]))
filename_old2 = os.path.splitext(os.path.basename(output['file'][1]))


# Defining the covariance
clustering_z_covariance_sva = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
clustering_z_covariance_sn = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
clustering_z_covariance_mix = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
clustering_z_covariance_gauss = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
clustering_z_covariance_total = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
clustering_z_spec_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
Cells = []
thetas = []
spec_reference_covariance = np.zeros(((len(zbound)-1), ((len(zbound)-1))))
spec_signal = np.zeros(len(zbound)-1)


def get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_total , signal_w):
    clustering_z_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
    if i_z  + 1 == j_z:
        for s_i in range(n_s):
            for s_j in range(n_s):
                for p_alpha in range (0, n_tomo_source):
                    for p_beta in range(0, n_tomo_source):
                        s_i_theta = 0
                        if s_i != 0:
                            s_i_theta = 2
                        s_j_theta = 0
                        if s_j != 0:
                            s_j_theta = 2
                        covariance = np.array(cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s])
                        flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
                        flat_idx_j = s_j + j_z  - 1 + p_beta*(len(zbound)-1)
                        covariance -= .5*signal_w[s_i_theta, 0, 0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                        covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                        covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                        covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                        clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
                            
    else:
        for p_alpha in range (0, n_tomo_source):
            for p_beta in range(0, n_tomo_source):
                s_i_theta = 0
                s_j_theta = 2
                s_i = 0
                s_j = 1
                covariance = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s]
                flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
                flat_idx_j = s_j + j_z  - 1 + p_beta*(len(zbound)-1)
                covariance -= .5*signal_w[s_i_theta, 0,0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
    return clustering_z_covariance

def get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_total , signal_w):
    clustering_z_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
    clustering_spec_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
    for p_alpha in range (0, n_tomo_source):
        for p_beta in range(0, n_tomo_source):
            s_i_theta = 0
            s_j_theta = 0
            s_i = 0
            s_j = 0
            covariance = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s]
            flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
            flat_idx_j = s_j + j_z + p_beta*(len(zbound)-1)                
            covariance -= .5*signal_w[s_i_theta, 0,0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
            covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
            covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
            covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
            clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
    return clustering_z_covariance

def get_spec_covariance(i_z, j_z, n_tomo_source, n_s, cov_total , signal_w):
    clustering_spec_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
    for p_alpha in range (0, n_tomo_source):
        for p_beta in range(0, n_tomo_source):
            s_i_theta = 0
            s_j_theta = 0
            s_i = 0
            s_j = 0
            flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
            flat_idx_j = s_j + j_z + p_beta*(len(zbound)-1)                
            clustering_spec_covariance[flat_idx_i, flat_idx_j] = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s]
    return clustering_spec_covariance

survey_params['n_eff_clust'] = np.zeros(n_tomo_source)

if not diagonal_only:
    n_s = 2
    read_in_tables['zclust']['nz'] = np.zeros((n_s + n_tomo_source, len(zbins)))
    survey_params['n_eff_clust'] = np.insert(survey_params['n_eff_clust'],0,0)
    survey_params['n_eff_clust'] = np.insert(survey_params['n_eff_clust'],0,0)
else:
    n_s = 1
    read_in_tables['zclust']['nz'] = np.zeros((n_s + n_tomo_source, len(zbins)))
    survey_params['n_eff_clust'] = np.insert(survey_params['n_eff_clust'],0,0)
subtract = 1
for i_z in range(0, len(zbound)- subtract, 1):  # loop over each spec-z bin
    if not diagonal_only:
        for j_z in range(0, len(zbound)- subtract, 1):  # loop over each spec-z bin
            if i_z == j_z:
                continue
            print(i_z, j_z, zmean[i_z], zmean[j_z])

            if i_z  + 1 != j_z:
                covterms["nongauss"] = False
                observables["ELLspace"]['limber'] = True
            else:
                covterms["nongauss"] = nonGaussian
                observables["ELLspace"]['limber'] = limber
            
            read_in_tables['zclust']['z'] = zbins
            zbin_values_i_z=np.zeros(len(zbins))
            zbin_values_i_z=np.where((zbound[i_z]<=zbins),1.,0.)
            zbin_values_i_z=np.where((zbound[i_z+1]<zbins),0.,zbin_values_i_z)
            read_in_tables['zclust']['nz'][0,:] = zbin_values_i_z

            zbin_values_j_z=np.zeros(len(zbins))
            zbin_values_j_z=np.where((zbound[j_z]<=zbins),1.,0.)
            zbin_values_j_z=np.where((zbound[j_z+1]<zbins),0.,zbin_values_j_z)
            read_in_tables['zclust']['nz'][1,:] = zbin_values_j_z
            read_in_tables['zclust']['nz'][n_s:,:] = nz_interp
            survey_params['n_eff_clust'][0] = ndens_spec[i_z]
            survey_params['n_eff_clust'][1] = ndens_spec[j_z]
            survey_params['n_eff_clust'][n_s:] = neff_phot
            if (ndens_spec[i_z] == 0.0 or ndens_spec[j_z] == 0.0):
                print("Iteration ", i_z, j_z, " skipped -- no galaxies")
                continue  # skip this calculation if no galaxies in bin
            # modify ini input
            #observables['THETAspace']['theta_min'] = theta_low[i]
            #observables['THETAspace']['theta_max'] = theta_hig[i]
            observables['THETAspace']['theta_type'] = 'list'
            observables['THETAspace']['theta_list'] = np.array([theta_low[i_z], theta_hig[i_z], theta_low[j_z], theta_hig[j_z]])
            # covariance calculation  

            cov_theta = CovTHETASpace(covterms, observables, output,
                                    cosmo, bias, iA,  hod, survey_params, prec, read_in_tables)
            
            cov_w = cov_theta.calc_covTHETA(
                observables, output, bias, hod, survey_params, prec, read_in_tables)
            #
            # has shape len(theta), len(samples), n_tomo, n_tomo
            
            #out.write_cov(covterms, observables, cov_theta.n_tomo_clust,
            #              cov_theta.n_tomo_lens, cov_theta.thetabins, cov_w[0], cov_w[1], cov_w[2])

            '''
            #SVA
            cov_sva = np.copy(cov_w[0][0][0][0][0][0])
            
            #MIX
            cov_mix = np.copy(cov_w[0][1][0][0][0][0])
            
            #SN
            cov_sn = np.copy(cov_w[0][2][0][0][0][0])
            
            #NG
            cov_NG = np.copy(cov_w[1][0][0][0][0])
            
            #SSC
            cov_SSC = np.copy(cov_w[2][0][0][0][0])
            '''
            cov_total_ = cov_w[0][0] + cov_w[0][1]  + cov_w[0][2]
            clustering_z_covariance_sva += get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_w[0][0],cov_theta.w_gg)
            clustering_z_covariance_sn += get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_w[0][2],cov_theta.w_gg)
            clustering_z_covariance_mix += get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_w[0][1],cov_theta.w_gg)
            clustering_z_covariance_gauss += get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_total_,cov_theta.w_gg)
            if covterms['nongauss']:
                cov_total_ += cov_w[1][0] 
            if covterms['ssc']:
                cov_total_ += cov_w[2][0]
            clustering_z_covariance_total += get_clustering_z_covariance(i_z, j_z, n_tomo_source, n_s, cov_total_,cov_theta.w_gg)
            
            '''if i_z  + 1 == j_z:
                for s_i in range(n_s):
                    for s_j in range(n_s):
                        for p_alpha in range (0, n_tomo_source):
                            for p_beta in range(0, n_tomo_source):
                                s_i_theta = 0
                                if s_i != 0:
                                    s_i_theta = 2
                                s_j_theta = 0
                                if s_j != 0:
                                    s_j_theta = 2
                                covariance = np.array(cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s])
                                flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
                                flat_idx_j = s_j + j_z  - 1 + p_beta*(len(zbound)-1)
                                cross_correlation_covariance[flat_idx_i, flat_idx_j] = covariance
                                covariance -= .5*signal_w[s_i_theta, 0, 0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                                covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                                covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                                covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                                clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
                                print(covariance)
            else:
                for p_alpha in range (0, n_tomo_source):
                    for p_beta in range(0, n_tomo_source):
                        s_i_theta = 0
                        s_j_theta = 2
                        s_i = 0
                        s_j = 1
                        covariance = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s]
                        flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
                        flat_idx_j = s_j + j_z  - 1 + p_beta*(len(zbound)-1)
                        cross_correlation_covariance[flat_idx_i, flat_idx_j] = covariance
                        covariance -= .5*signal_w[s_i_theta, 0,0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                        covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                        covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                        covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                        clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance'''

    else:
        covterms["nongauss"] = nonGaussian
        observables["ELLspace"]['limber'] = limber
        j_z = i_z
        read_in_tables['zclust']['z'] = zbins
        zbin_values_i_z=np.zeros(len(zbins))
        zbin_values_i_z=np.where((zbound[i_z]<=zbins),1.,0.)
        zbin_values_i_z=np.where((zbound[i_z+1]<zbins),0.,zbin_values_i_z)
        read_in_tables['zclust']['nz'][0,:] = zbin_values_i_z

        read_in_tables['zclust']['nz'][n_s:,:] = nz_interp
        survey_params['n_eff_clust'][0] = ndens_spec[i_z]
        survey_params['n_eff_clust'][n_s:] = neff_phot
        print(i_z, zmean[i_z])
        
        
        if (ndens_spec[i_z] == 0.0):
            print("Iteration ", i_z, j_z, " skipped -- no galaxies")
            continue  # skip this calculation if no galaxies in bin
        observables['THETAspace']['theta_type'] = 'list'
        observables['THETAspace']['theta_list'] = np.array([theta_low[i_z], theta_hig[i_z]])
        # covariance calculation  

        cov_theta = CovTHETASpace(covterms, observables, output,
                                cosmo, bias, iA,  hod, survey_params, prec, read_in_tables)
        
        cov_w = cov_theta.calc_covTHETA(
            observables, output, bias, hod, survey_params, prec, read_in_tables)
        #
        # has shape len(theta), len(samples), n_tomo, n_tomo
        
        
        '''
        #SVA
        cov_sva = np.copy(cov_w[0][0][0][0][0][0])
        
        #MIX
        cov_mix = np.copy(cov_w[0][1][0][0][0][0])
        
        #SN
        cov_sn = np.copy(cov_w[0][2][0][0][0][0])
        
        #NG
        cov_NG = np.copy(cov_w[1][0][0][0][0])
        
        #SSC
        cov_SSC = np.copy(cov_w[2][0][0][0][0])
        '''
        cov_total_ = cov_w[0][0] + cov_w[0][1] + cov_w[0][2]
        clustering_z_covariance_sva += get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_w[0][0],cov_theta.w_gg)
        clustering_z_covariance_sn += get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_w[0][2],cov_theta.w_gg)
        clustering_z_covariance_mix += get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_w[0][1],cov_theta.w_gg)
        clustering_z_covariance_gauss += get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_total_,cov_theta.w_gg)
        if covterms['nongauss']:
            cov_total_ += cov_w[1][0] 
        if covterms['ssc']:
            cov_total_ += cov_w[2][0]
        clustering_z_covariance_total += get_clustering_z_covariance_diagonal(i_z, j_z, n_tomo_source, n_s, cov_total_,cov_theta.w_gg)
        clustering_z_spec_covariance += get_spec_covariance(i_z, j_z, n_tomo_source, n_s, cov_total_,cov_theta.w_gg)
        Cells.append(np.array([cov_theta.Cell_gg[:,0,0,0,0], cov_theta.ellrange]).T)
        thetas.append([theta_low[i_z], theta_hig[i_z]])
        '''for p_alpha in range (0, n_tomo_source):
            for p_beta in range(0, n_tomo_source):
                s_i_theta = 0
                s_j_theta = 0
                s_i = 0
                s_j = 0
                covariance = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j, p_beta+ n_s]
                flat_idx_i = s_i + i_z + p_alpha*(len(zbound)-1)
                flat_idx_j = s_j + j_z + p_beta*(len(zbound)-1)                
                cross_correlation_covariance[flat_idx_i, flat_idx_j] = covariance
                covariance -= .5*signal_w[s_i_theta, 0,0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
                spec_reference_covariance[i_z,j_z] = cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j] 
                spec_signal[i_z] = signal_w[s_i_theta,0,0,s_i,s_i]'''

for i_cov in range(len(clustering_z_covariance_sva[:,0])):
    for j_cov in range(i_cov,len(clustering_z_covariance_sva[:,0])):
        clustering_z_covariance_sva[j_cov, i_cov] = clustering_z_covariance_sva[i_cov,j_cov]
        clustering_z_covariance_gauss[j_cov, i_cov] = clustering_z_covariance_gauss[i_cov,j_cov]
        clustering_z_covariance_sn[j_cov, i_cov] = clustering_z_covariance_sn[i_cov,j_cov]
        clustering_z_covariance_total[j_cov, i_cov] = clustering_z_covariance_total[i_cov,j_cov]
        
if diagonal_only:
    np.savetxt(save_path_cz_covariance + "_gauss_diagonal.mat",clustering_z_covariance_gauss)
    np.savetxt(save_path_cz_covariance + "_sva_diagonal.mat",clustering_z_covariance_sva)
    np.savetxt(save_path_cz_covariance + "_mix_diagonal.mat",clustering_z_covariance_mix)
    np.savetxt(save_path_cz_covariance + "_sn_diagonal.mat",clustering_z_covariance_sn)
    np.savetxt(save_path_cz_covariance + "_total_diagonal.mat",clustering_z_covariance_total)
    np.save(save_path_spec_Cell,np.array(Cells))
    np.save("./../clustering-z_covariance/data_onecov/theta_bins_of_z", np.array(thetas))
else:
    np.savetxt(save_path_cz_covariance + "_gauss.mat",clustering_z_covariance_gauss)
    np.savetxt(save_path_cz_covariance + "_sva.mat",clustering_z_covariance_sva)
    np.savetxt(save_path_cz_covariance + "_sn.mat",clustering_z_covariance_sn)
    np.savetxt(save_path_cz_covariance + "_total.mat",clustering_z_covariance_total)
    np.savetxt(save_path_cz_covariance + "_mix.mat",clustering_z_covariance_mix)