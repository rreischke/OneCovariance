from onecov.cov_input import Input, FileInput
from onecov.cov_ell_space import CovELLSpace
from onecov.cov_theta_space import CovTHETASpace
from onecov.cov_output import Output
from onecov.cov_cosebis import CovCOSEBI
from onecov.cov_bandpowers import CovBandPowers
import sys

from scipy.interpolate import RegularGridInterpolator
import numpy as np


x = np.linspace(1,5,100)
y = np.linspace(1,5,50)
f = (x**2)[:,None]*y[None,:]

interp = RegularGridInterpolator((x,y),f)
X,Y  = np.meshgrid(x,x)
print(X)
print(Y)
resu = interp(((X,Y)))
print(resu.shape)



inp = Input()

if len(sys.argv) > 1:
    config = str(sys.argv[1])
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input(config)
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(config)

else:
    covterms, observables, output, cosmo, bias, iA, hod, survey_params, prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input()

if ((observables['observables']['est_shear'] == 'C_ell' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'C_ell' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'C_ell' and observables['observables']['clustering']):
    covell = CovELLSpace(covterms, observables, output, cosmo, bias, iA,
                         hod, survey_params, prec, read_in_tables)
    covariance_in_ell_space = covell.calc_covELL(
        observables, output, bias,  hod, survey_params, prec, read_in_tables)
    out = Output(output)
    out.write_cov(covterms, observables, covell.n_tomo_clust,
                  covell.n_tomo_lens, covell.ellrange,
                  covariance_in_ell_space[0],
                  covariance_in_ell_space[1],
                  covariance_in_ell_space[2])


if ((observables['observables']['est_shear'] == 'xi_pm' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'gamma_t' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'w' and observables['observables']['clustering']):
    covtheta = CovTHETASpace(covterms, observables, output,
                             cosmo, bias, iA,  hod, survey_params, prec, read_in_tables)
    covariance_in_theta_space = covtheta.calc_covTHETA(
        observables, output, bias,  hod, survey_params, prec, read_in_tables)
    out = Output(output)
    out.write_cov(covterms, observables, covtheta.n_tomo_clust,
                  covtheta.n_tomo_lens, covtheta.thetabins,
                  covariance_in_theta_space[0],
                  covariance_in_theta_space[1],
                  covariance_in_theta_space[2])

if (observables['observables']['est_shear'] == 'cosebi' and observables['observables']['cosmic_shear'] or observables['observables']['ggl'] == 'cosebi' or observables['observables']['clustering'] == 'cosebi'):
    covcosebis = CovCOSEBI(covterms, observables, output,
                           cosmo, bias, iA, hod, survey_params, prec, read_in_tables)
    covariance_COSEBIS = covcosebis.calc_covCOSEBI(observables, output, bias,  hod, survey_params, prec, read_in_tables)
    out = Output(output)
    out.write_cov(covterms, observables, covcosebis.n_tomo_clust,
                  covcosebis.n_tomo_lens, covcosebis.array_En_modes,
                  covariance_COSEBIS[0],
                  covariance_COSEBIS[1],
                  covariance_COSEBIS[2])

if ((observables['observables']['est_shear'] == 'bandpowers' and observables['observables']['cosmic_shear']) or (observables['observables']['est_ggl'] == 'bandpowers' and observables['observables']['ggl']) or observables['observables']['est_clust'] == 'bandpowers' and observables['observables']['clustering']):
    covbp = CovBandPowers(covterms, observables, output,
                           cosmo, bias, iA, hod, survey_params, prec, read_in_tables)
    covariance_bp = covbp.calc_covbandpowers(observables, output, bias,  hod, survey_params, prec, read_in_tables)
    out = Output(output)
    out.write_cov(covterms, observables, covbp.n_tomo_clust,
                  covbp.n_tomo_lens, covbp.ell_bins,
                  covariance_bp[0],
                  covariance_bp[1],
                  covariance_bp[2])
