"""
Writes the line-of-sight kernels of the line-of-sight shear of strong lenses
(Fleury et al. 2026, arXiv:2603.03441, eqs. 12, 84 and 87),

    gamma_LOS = gamma_od + gamma_os - gamma_ds,
    W(chi) = 3/2 Omega_m (H0/c)^2 chi/a(chi) <K_LOS(chi; chi_d, chi_s)>,

averaged over the lenses (weighted mean over p(z_d, z_s)), in the convention
of the tabulated radial kernels of OneCovariance ([arbitrary_kernel],
arb_radial_filter_mm_file): two columns, z and W(chi(z)) in h/Mpc, so that
C_ell^ij = f_ell int dchi W_i W_j P_mm/chi^2 like for the internal lensing
kernels (same sign convention as the galaxy shear, the paper's kernels have
the opposite sign for both). One file per tomographic bin in z_d.

The lenses are read from an ascii file with the columns z_d, z_s and an
optional weight (e.g. a LensPop catalogue, or a grid of p(z_d, z_s) with the
probabilities as weights).

Noise in the [survey specs] of the line-of-sight bins:
    n_eff_lensing          = number of lenses in the bin per arcmin^2
    ellipticity_dispersion = sqrt((sigma_eta^2 + <|gamma_LOS|^2>)/2)
with the measurement noise sigma_eta of |gamma_LOS| and the sparsity variance
<|gamma_LOS|^2> of a single lens (paper eqs. 118-120), which is not small.

Example:
    python los_shear_kernel.py lenses.txt --config ../../../config.ini --zd_edges 0,0.5,3
"""
import argparse
import configparser
from os import path

import numpy as np
from astropy import units as u
from astropy.cosmology import w0waCDM

C_KMS = 299792.458


def cosmology(config_name):
    """The cosmology of OneCovariance (Setup) from the [cosmo] section."""
    config = configparser.ConfigParser()
    config.read(config_name)
    cosmo = config['cosmo'] if 'cosmo' in config else {}
    get = lambda key, default: float(cosmo.get(key, default))
    omega_m = get('omega_m', 0.315)
    return w0waCDM(H0=100*get('h', 0.674), Om0=omega_m, Ob0=get('omega_b', 0.049),
                   Ode0=get('omega_de', 1 - omega_m), w0=get('w0', -1), wa=get('wa', 0),
                   Neff=get('neff', 3.046), m_nu=[get('m_nu', 0), 0, 0]*u.eV, Tcmb0=get('tcmb0', 2.725))


def chi_K_los(chi, chi_d, chi_s):
    """chi K_LOS(chi; chi_d, chi_s) (paper eq. 12), chi along axis 0."""
    chi = chi[:, None]
    return np.where(chi <= chi_d, chi*(chi_s - chi)/chi_s + chi*(chi_d - chi)/chi_d,
                    np.where(chi <= chi_s, chi_d*(chi_s - chi)**2/(chi_s*(chi_s - chi_d)), 0.))


def main():
    parser = argparse.ArgumentParser(description='Line-of-sight shear kernels for OneCovariance')
    parser.add_argument('lenses', help='ascii file with the columns z_d, z_s and optionally a weight')
    parser.add_argument('--config', default='config.ini', help='OneCovariance config for the cosmology (default: config.ini)')
    parser.add_argument('--zd_edges', default=None, help='comma separated edges of the tomographic bins in z_d (default: one bin)')
    parser.add_argument('--nz', type=int, default=1000, help='number of redshifts of the tables (default: 1000)')
    parser.add_argument('--out', default=path.join(path.dirname(path.abspath(__file__)), '..', 'los_shear_kernel_'),
                        help='prefix of the output files, the bin number and .txt are appended')
    args = parser.parse_args()

    lenses = np.loadtxt(args.lenses, ndmin=2)
    z_d, z_s = lenses[:, 0], lenses[:, 1]
    weight = lenses[:, 2] if lenses.shape[1] > 2 else np.ones_like(z_d)
    if np.any(z_s <= z_d):
        raise ValueError('every lens needs z_s > z_d')
    cosmo = cosmology(args.config)
    to_chi = lambda z: cosmo.comoving_distance(z).value*cosmo.h  # Mpc/h
    zet = np.linspace(0, z_s.max(), args.nz)
    chi, chi_d, chi_s = to_chi(zet), to_chi(z_d), to_chi(z_s)
    prefactor = 1.5*cosmo.Om0*(100/C_KMS)**2*(1 + zet)
    edges = [z_d.min(), np.inf] if args.zd_edges is None else [float(e) for e in args.zd_edges.split(',')]
    for i_bin, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        sel = np.flatnonzero((z_d >= lo) & (z_d < hi))
        if len(sel) == 0:
            raise ValueError('no lenses with ' + str(lo) + ' <= z_d < ' + str(hi))
        kernel = sum(chi_K_los(chi, chi_d[chunk], chi_s[chunk]) @ weight[chunk]
                     for chunk in np.array_split(sel, max(1, len(sel)//1000)))
        np.savetxt(args.out + str(i_bin) + '.txt', np.column_stack((zet, prefactor*kernel/weight[sel].sum())),
                   header='z     W_LOS(chi(z)) [h/Mpc], ' + str(len(sel)) + ' entries with ' + str(lo) + ' <= z_d < ' + str(hi))
        print('bin ' + str(i_bin) + ': ' + str(len(sel)) + ' entries (' + str(round(weight[sel].sum()/weight.sum(), 4)) +
              ' of the weight) written to ' + args.out + str(i_bin) + '.txt')


if __name__ == '__main__':
    main()
