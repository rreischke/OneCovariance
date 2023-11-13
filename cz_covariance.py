from onecov.cov_input import Input, FileInput
from onecov.cov_output import Output
from onecov.cov_theta_space import CovTHETASpace
from astropy.cosmology import FlatLambdaCDM, Planck15

import os
import numpy as np
import sys



config = "./config_files/config_cz.ini"
r_low = 0.1 # Scales considered 
r_hig = 1.0
path_to_reference_sample = str("./../clustering-z_covariance/data_paper/nz_reference.dat") # path to the refence sample
path_to_nz_true = str("./../clustering-z_covariance/data_paper/nz_true/nz_true_") # path to the true redshift
diagonal_only = False # should only be autocorrelations be considered

save_path_cz_covariance = str("./../clustering-z_covariance/data_onecov/cz_covariance_r_" +str(r_low) + "_"+str(r_hig)) # Where should the cz covariance be stored?
save_path_cross_covariance = str("./../clustering-z_covariance/data_onecov/cross_covariance_r_" +str(r_low) + "_"+str(r_hig))
save_path_at_i = str("./../clustering-z_covariance/data_onecov/cz_covariance_r_" +str(r_low) + "_"+str(r_hig)+ "at_i")




# Setting up the OneCovariance code
inp = Input()
covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input(
    config)
fileinp = FileInput(bias)
read_in_tables = fileinp.read_input(inp.config_name)
out = Output(output)

# How fine should the binning bes
nz_binning = 0.001


# Number counts for the reference sample
galcount = np.array(np.loadtxt(path_to_reference_sample)[:, 2])
# Survey area of the CZ measurements
survey_area = survey_params['survey_area_clust'][0]*3600.  # in arcmin^2
# Number density of the reference sample
ndens_spec = galcount / survey_area
# Redshift boundaries
zbound = np.array(np.loadtxt(path_to_reference_sample)[:, 1])
zbound = np.insert(zbound,0,float(np.loadtxt(path_to_reference_sample)[0, 0]),0)
zmean = (zbound[1:]+zbound[:-1])/2.
deltaz = (zbound[1:]-zbound[:-1])
# Corresponding co-moving distances
fkchi = Planck15.comoving_transverse_distance(zmean).value 
# Angular diameter distance
d_ang = fkchi/(1.+zmean)
# Corresponding angular range for measurements
theta_low = np.arctan(r_low/d_ang)/np.pi*180.*60. 
theta_hig = np.arctan(r_hig/d_ang)/np.pi*180.*60.

zbins = np.arange(zbound[0],
                  zbound[-1], nz_binning)

# Redshift distributions to be calibrated (including some interpolations)
n_tomo_source = np.shape(read_in_tables['zclust']['nz'])[
    0]
nz_interp = np.zeros(n_tomo_source*len(zbins)
                     ).reshape(n_tomo_source, len(zbins))



neff_phot = np.zeros(n_tomo_source)
for j in range(n_tomo_source):
    ndens_phot_file = path_to_nz_true+ str(j+2) +".dat"
    ndens_phot = np.array(np.loadtxt(ndens_phot_file)[:, 2])
    neff_phot[j] = np.sum(ndens_phot)
    neff_phot[j] /= survey_area
    nz_interp[j] = np.interp(zbins, np.array(np.loadtxt(ndens_phot_file)[:, 0]), ndens_phot)
    
    
read_in_tables['zclust']['z'] = zbins

filename_old1 = os.path.splitext(os.path.basename(output['file'][0]))
filename_old2 = os.path.splitext(os.path.basename(output['file'][1]))


# Defining the covariance
clustering_z_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
cross_correlation_covariance = np.zeros(((len(zbound)-1)*n_tomo_source, ((len(zbound)-1) * n_tomo_source)))
spec_reference_covariance = np.zeros(((len(zbound)-1), ((len(zbound)-1))))
spec_signal = np.zeros(len(zbound)-1)


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
            print(i_z, j_z)
            if i_z  + 1 != j_z:
                covterms["nongauss"] = False
                #observables["ELLspace"]['limber'] = True
            else:
                covterms["nongauss"] = False
                #observables["ELLspace"]['limber'] = False
            
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
            signal_w = cov_theta.w_gg
            
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
            cov_total = cov_w[0][0] + cov_w[0][1]  + cov_w[0][2]
            if covterms['nongauss']:
                cov_total += cov_w[1][0] 
            if covterms['ssc']:
                cov_total += cov_w[2][0]
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
                                cross_correlation_covariance[flat_idx_i, flat_idx_j] = covariance
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
                        cross_correlation_covariance[flat_idx_i, flat_idx_j] = covariance
                        covariance -= .5*signal_w[s_i_theta, 0,0, s_i, p_alpha + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j, p_beta + n_s]
                        covariance -= .5*signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, p_alpha + n_s,s_j,s_j]
                        covariance += .25*signal_w[s_i_theta,0,0,s_i, p_alpha + n_s] * signal_w[s_j_theta,0,0,s_j, p_beta + n_s]/signal_w[s_i_theta,0,0,s_i,s_i]/signal_w[s_j_theta,0,0,s_j,s_j]*cov_total[s_i_theta, s_j_theta, 0, 0,s_i, s_i,s_j,s_j]   
                        covariance /= np.sqrt(signal_w[s_i_theta,0,0,s_i,s_i]*signal_w[s_j_theta,0,0,s_j,s_j])
                        clustering_z_covariance[flat_idx_i, flat_idx_j] = covariance
        file_name_cov_partial = save_path_at_i + str("clustering_z_cov_partial_atz_" + str(i_z) +".dat")
        file_name_cross_partial = save_path_at_i + str("clustering_z_cross_partial_atz_" + str(i_z) +".dat")
        np.savetxt(file_name_cov_partial,clustering_z_covariance)
        np.savetxt(file_name_cross_partial,cross_correlation_covariance)    
    else:
        print(i_z)
        j_z = i_z
        read_in_tables['zclust']['z'] = zbins
        zbin_values_i_z=np.zeros(len(zbins))
        zbin_values_i_z=np.where((zbound[i_z]<=zbins),1.,0.)
        zbin_values_i_z=np.where((zbound[i_z+1]<zbins),0.,zbin_values_i_z)
        read_in_tables['zclust']['nz'][0,:] = zbin_values_i_z

        read_in_tables['zclust']['nz'][n_s:,:] = nz_interp
        survey_params['n_eff_clust'][0] = ndens_spec[i_z]
        survey_params['n_eff_clust'][n_s:] = neff_phot
        
        
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
        signal_w = cov_theta.w_gg
        
        out.write_cov(covterms, observables, cov_theta.n_tomo_clust,
                      cov_theta.n_tomo_lens, cov_theta.thetabins, cov_w[0], cov_w[1], cov_w[2])

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
        cov_total = cov_w[0][0] + cov_w[0][1]  + cov_w[0][2]
        if covterms['nongauss']:
            cov_total += cov_w[1][0] 
        if covterms['ssc']:
            cov_total += cov_w[2][0]
        for p_alpha in range (0, n_tomo_source):
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
                spec_signal[i_z] = signal_w[s_i_theta,0,0,s_i,s_i]

for i_cov in range(len(clustering_z_covariance[:,0])):
    for j_cov in range(i_cov,len(clustering_z_covariance[:,0])):
        clustering_z_covariance[j_cov, i_cov] = clustering_z_covariance[i_cov,j_cov]
        cross_correlation_covariance[j_cov, i_cov] = cross_correlation_covariance[i_cov,j_cov]                  
if diagonal_only:
    np.savetxt("clustering_z_cov_diagonal_biased",clustering_z_covariance)
    np.savetxt("cross_correlation_cov_diagonal_biased",cross_correlation_covariance)
    np.savetxt("spec_z_covariance_biased",spec_reference_covariance)
    np.savetxt("spec_z_signal_biased",spec_signal)
else:
    np.savetxt(save_path_cz_covariance,clustering_z_covariance)
    np.savetxt(save_path_cross_covariance,cross_correlation_covariance)
