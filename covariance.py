from onecov.cov_input import Input, FileInput
from onecov.cov_ell_space import CovELLSpace
from onecov.cov_theta_space import CovTHETASpace
from onecov.cov_output import Output
from onecov.cov_cosebis import CovCOSEBI
from onecov.cov_bandpowers import CovBandPowers
from onecov.cov_arbitrary_summary import CovARBsummary
import sys
import os
import platform
if len(platform.mac_ver()[0]) > 0 and (platform.processor() == 'arm' or int(platform.mac_ver()[0][:(platform.mac_ver()[0]).find(".")]) > 13):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
print("READING INPUT")
print("#############")

inp = Input()



if len(sys.argv) > 1:
    config = str(sys.argv[1])
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input(config)
    fileinp = FileInput(bias)
    read_in_tables = fileinp.read_input(config)
else:
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input()
    fileinp = FileInput(bias)
    read_in_tables = fileinp.read_input()


safe_gauss = covterms['gauss']
covterms['gauss'] = True

if not observables['arbitrary_summary']['do_arbitrary_summary']:
    if ((observables['observables']['est_shear'] == 'C_ell' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'C_ell' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'C_ell' and observables['observables']['clustering']):
        print("CALCULATING COVARIANCE FOR ANGULAR POWER SPECTRA")
        print("################################################")    
        covell = CovELLSpace(covterms, observables, output, cosmo, bias, iA,
                            hod, survey_params, prec, read_in_tables)
        
        
        covariance_in_ell_space = covell.calc_covELL(
            observables, output, bias,  hod, survey_params, prec, read_in_tables)
        observables['observables']['is_cell'] = True
        out = Output(output, covell.ellrange_clustering, covell.ellrange_lensing)
        covterms['gauss'] = safe_gauss
        out.write_cov(covterms, observables, covell.n_tomo_clust,
                    covell.n_tomo_lens, covell.ellrange,
                    covariance_in_ell_space[0],
                    covariance_in_ell_space[1],
                    covariance_in_ell_space[2])


    if ((observables['observables']['est_shear'] == 'xi_pm' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'gamma_t' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'w' and observables['observables']['clustering']):
        print("CALCULATING COVARIANCE FOR REAL SPACE CORRELATION FUNCTIONS")
        print("###########################################################")    
        covtheta = CovTHETASpace(covterms, observables, output,
                                cosmo, bias, iA,  hod, survey_params, prec, read_in_tables)
        covariance_in_theta_space = covtheta.calc_covTHETA(
            observables, output, bias,  hod, survey_params, prec, read_in_tables)
        out = Output(output, covtheta.theta_bins_clustering, covtheta.theta_bins_lensing)
        covterms['gauss'] = safe_gauss
        out.write_cov(covterms, observables, covtheta.n_tomo_clust,
                    covtheta.n_tomo_lens, covtheta.thetabins,
                    covariance_in_theta_space[0],
                    covariance_in_theta_space[1],
                    covariance_in_theta_space[2])
    if (observables['observables']['est_shear'] == 'cosebi' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'cosebi' and observables['observables']['ggl']) or (observables['observables']['est_clust'] == 'cosebi' and observables['observables']['clustering']):
        print("CALCULATING COVARIANCE FOR COSEBIS")
        print("##################################")
        covcosebis = CovCOSEBI(covterms, observables, output,
                            cosmo, bias, iA, hod, survey_params, prec, read_in_tables)
        covariance_COSEBIS = covcosebis.calc_covCOSEBI(observables, output, bias,  hod, survey_params, prec, read_in_tables)
        out = Output(output , covcosebis.array_En_g_modes, covcosebis.array_En_modes)
        covterms['gauss'] = safe_gauss
        out.write_cov(covterms, observables, covcosebis.n_tomo_clust,
                    covcosebis.n_tomo_lens, covcosebis.ellrange,
                    covariance_COSEBIS[0],
                    covariance_COSEBIS[1],
                    covariance_COSEBIS[2])

    if ((observables['observables']['est_shear'] == 'bandpowers' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'bandpowers' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'bandpowers' and observables['observables']['clustering']):
        print("CALCULATING COVARIANCE FOR BANDPOWERS")
        print("#####################################")
        covbp = CovBandPowers(covterms, observables, output,
                            cosmo, bias, iA, hod, survey_params, prec, read_in_tables)
        covariance_bp = covbp.calc_covbandpowers(observables, output, bias,  hod, survey_params, prec, read_in_tables)
        out = Output(output, covbp.ell_bins_clustering, covbp.ell_bins_lensing)
        covterms['gauss'] = safe_gauss
        out.write_cov(covterms, observables, covbp.n_tomo_clust,
                    covbp.n_tomo_lens, covbp.ellrange,
                    covariance_bp[0],
                    covariance_bp[1],
                    covariance_bp[2])
else:
    print("CALCULATING COVARIANCE FOR ARBITRARY SUMMARY STATISTICS")
    print("#######################################################")
    covARB = CovARBsummary(covterms, observables, output,
                            cosmo, bias, iA, hod, survey_params, prec, read_in_tables)
    covariance_arb = covARB.calc_covarbsummary(observables, output, bias,  hod, survey_params, prec, read_in_tables)
    out = Output(output)
    covterms['gauss'] = safe_gauss
    out.write_arbitrary_cov(covterms, observables, covARB.n_tomo_clust,
                covARB.n_tomo_lens, read_in_tables,
                covariance_arb[0],
                covariance_arb[1],
                covariance_arb[2])
    
