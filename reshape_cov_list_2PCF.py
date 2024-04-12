import gc
import time
from matplotlib import cm, pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import scipy
import matplotlib
import re
import configparser
import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
# matplotlib.use('Agg')


def extract_probes(probe_string, probe_names):
    """
    Extract probes from a string given a list of known probe names.

    Parameters:
    - probe_string: The string containing the concatenated probe names.
    - probe_names: A list of known probe names, sorted by length in descending order.

    Returns:
    - A list of extracted probe names.
    """
    # Create a regex pattern to match the probe names
    # Join the probe names into a pattern, separating them with '|'
    # The longer names should come first in the pattern to ensure they are matched correctly
    pattern = '|'.join(probe_names)

    # Find all matches of the pattern in the probe_string
    matched_probes = re.findall(pattern, probe_string)

    return matched_probes


cov_folder = '/home/cosmo/davide.sciotti/data/OneCovariance/output_2PCF_C01'
cl_input_folder = '/home/cosmo/davide.sciotti/data/CLOE_validation/output/v2.0.2/C01'

cfg = configparser.ConfigParser()
cfg.read(cov_folder + '/save_configs.ini')

zbins = 10
theta_bins = int(float(cfg['covTHETAspace settings']['theta_bins']))
theta_max = int(float(cfg['covTHETAspace settings']['theta_max']))

chunk_size = 5000000
load_mat_files = False
theta_unit = 'arcmin'

ind = mm.build_full_ind('triu', 'row-major', zbins)

probe_names = ['gg', 'gm', 'xip', 'xim']
df_header = pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=False, nrows=0)
print('df_header:')
print(df_header)
column_names = [
    '#obs', 'theta1', 'theta2', 's1', 's2', 'tomoi', 'tomoj', 'tomok', 'tomol',
    'cov', 'covg sva', 'covg mix', 'covg sn', 'covng', 'covssc',
]


probe_idx_dict = {
    'gg': 0,
    'gm': 1,
    'xip': 2,
    'xim': 3,
}


# ! get, show and reshape the .mat file, for a later check
if load_mat_files:

    print('Loading covariance matrix from .mat file...')
    start_time = time.perf_counter()

    cov_mat_fmt_2dcloe = np.genfromtxt(f'{cov_folder}/covariance_matrix.mat')
    print('Covariance matrix loaded in ', time.perf_counter() - start_time, ' seconds')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='original, 2d CLOE')

    corr_2dcloe = mm.cov2corr(cov_mat_fmt_2dcloe)

    mm.matshow(corr_2dcloe, log=False, title=' corr, 2d CLOE', matshow_kwargs={'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1})

    # np.savez_compressed(f'{cov_folder}/cov_tot_2dCLOE.npz', cov_mat_fmt_2dcloe)
    # np.savez_compressed(f'{cov_folder}/cov_tot_2ddav.npz', cov_mat_fmt_2ddav)

    # del cov_mat_fmt_2ddav, cov_mat_fmt_2dcloe
    # gc.collect()

# ! consistency check for the output cls
cl_ll_in = np.genfromtxt(f'{cl_input_folder}/Cij-LL-PyCCLforOneCov-C01.ascii')
cl_gl_in = np.genfromtxt(f'{cl_input_folder}/Cij-GL-PyCCLforOneCov-C01.ascii')
cl_gg_in = np.genfromtxt(f'{cl_input_folder}/Cij-GG-PyCCLforOneCov-C01.ascii')

cl_ll_out = np.genfromtxt(f'{cov_folder}/Cell_kappakappa.ascii')
cl_gl_out = np.genfromtxt(f'{cov_folder}/Cell_gkappa.ascii')
cl_gg_out = np.genfromtxt(f'{cov_folder}/Cell_gg.ascii')


ell = np.unique(cl_ll_out[:, 0])
print('nbl:', len(ell))


assert np.allclose(ell, np.unique(cl_ll_out[:, 0]), atol=0, rtol=1e-4), 'ell values are not the same'
# np.testing.assert_allclose(cl_ll_out, cl_ll_in, atol=0, rtol=1e-4)
# np.testing.assert_allclose(cl_gl_out, cl_gl_in, atol=0, rtol=1e-4)
# np.testing.assert_allclose(cl_gg_out, cl_gg_in, atol=0, rtol=1e-4)


# plt.plot(cl_ll_in[:, 3], label='in', ls='--')
# plt.plot(cl_ll_out[:, 3], label='out')
# plt.legend()


zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :]

# ! import _list covariance file
cov_g_10d = np.zeros((4, 4, theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_sva_10d = np.zeros((4, 4, theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_mix_10d = np.zeros((4, 4, theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_sn_10d = np.zeros((4, 4, theta_bins, theta_bins, zbins, zbins, zbins, zbins))

thetas = pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, usecols=['theta1'])['theta1'].unique()

theta_indices = {theta: idx for idx, theta in enumerate(thetas)}
assert len(thetas) == theta_bins, 'Number of thetas does not match the number of theta bins'

# read and print the header
with open(f'{cov_folder}/covariance_list.dat', 'r') as file:
    header = file.readline().strip()  # Read the first line and strip newline characters
print('.dat file header: ')
print(header)
header_list = re.split('\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t'))
assert column_names == header_list, 'column names from .dat file do not match with the expected ones'


print('Loading the dataframe in chunks...')

# ! load the dataframe in chunks - optimised version
start = time.perf_counter()
for df_chunk in pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

    # Use apply to vectorize the extraction of probes
    extracted_probes = df_chunk['#obs'].apply(lambda x: extract_probes(x, probe_names))
    probe_idxs = np.array([[probe_idx_dict[probe[0]], probe_idx_dict[probe[1]]] for probe in extracted_probes])

    theta1_indices = df_chunk['theta1'].map(theta_indices).values
    theta2_indices = df_chunk['theta2'].map(theta_indices).values

    z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values

    index_tuple = (probe_idxs[:, 0], probe_idxs[:, 1], theta1_indices, theta2_indices,
                   z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

    cov_g_10d[index_tuple] = df_chunk['cov'].values
    cov_sva_10d[index_tuple] = df_chunk['covg sva'].values
    cov_mix_10d[index_tuple] = df_chunk['covg mix'].values
    cov_sn_10d[index_tuple] = df_chunk['covg sn'].values

print('df optimized loaded in ', time.perf_counter() - start, ' seconds')


# let's test the 2d conversion against the .mat file, e.g. for gggg
cov_g_6d_gggg = cov_g_10d[0, 0, ...]
cov_g_4d_gggg = mm.cov_6D_to_4D(cov_g_6d_gggg, theta_bins, zpairs_auto, ind_auto)
cov_g_2d_gggg = mm.cov_4D_to_2D(cov_g_4d_gggg, block_index='ij')

n_elem_auto = theta_bins * zpairs_auto
assert n_elem_auto == cov_g_2d_gggg.shape[0], 'number of elements in the 2d cov matrix does not match the expected one'

if load_mat_files:
    mm.matshow(cov_mat_fmt_2dcloe[:n_elem_auto, :n_elem_auto], log=True)
    mm.matshow(cov_g_2d_gggg, log=True)

    mm.compare_arrays(cov_mat_fmt_2dcloe[:n_elem_auto, :n_elem_auto], cov_g_2d_gggg,
                      'cov_mat_fmt_2dcloe', 'cov_list_fmt_2d')

# save vectors of variances for Matteo
for probe_idx in range(4):
    cov_g_6d = cov_g_10d[probe_idx, probe_idx, ...]

    zpairs = zpairs_auto
    ind_here = ind_auto
    if probe_idx == 1:
        zpairs = zpairs_cross
        ind_here = ind_cross

    cov_g_4d = mm.cov_6D_to_4D(cov_g_6d, theta_bins, zpairs, ind_here)
    cov_g_2d = mm.cov_4D_to_2D(cov_g_4d, block_index='vincenzo')

    mm.matshow(cov_g_2d, log=True)
    variance = np.diag(cov_g_2d)
    np.savetxt(cov_folder + '/variance_' + probe_names[probe_idx] + '.dat', variance)

colors = cm.rainbow(np.linspace(0, 1, zbins))

# ! plot 2PCF and errors
xi_gg_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-GG-PyCCL-C01.dat')
xi_gl_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-GL-PyCCL-C01.dat')
xi_pp_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-Lplus-PyCCL-C01.dat')
xi_mm_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-Lminus-PyCCL-C01.dat')

theta_deg = xi_gg_2d[:, 0]
theta_arcmin = theta_deg * 60

if theta_unit == 'arcmin':
    theta_arr = theta_arcmin
elif theta_unit == 'deg':
    theta_arr = theta_deg
else:
    raise ValueError('theta unit not recognised')

xi_gg_2d = xi_gg_2d[:, 1:]
xi_gl_2d = xi_gl_2d[:, 1:]
xi_pp_2d = xi_pp_2d[:, 1:]
xi_mm_2d = xi_mm_2d[:, 1:]

xi_gg_3D = mm.cl_2D_to_3D_symmetric(xi_gg_2d, theta_bins, zpairs_auto, zbins)
xi_gl_3D = mm.cl_2D_to_3D_asymmetric(xi_gl_2d, theta_bins, zbins=zbins, order='row-major')
xi_pp_3D = mm.cl_2D_to_3D_symmetric(xi_pp_2d, theta_bins, zpairs_auto, zbins)
xi_mm_3D = mm.cl_2D_to_3D_symmetric(xi_mm_2d, theta_bins, zpairs_auto, zbins)

cols = 2
rows = 2
fig, ax = plt.subplots(rows, cols, figsize=(12, 10))
for probe_idx, probe in zip((range(4)), (xi_gg_3D, xi_gl_3D, xi_pp_3D, xi_mm_3D)):

    row = probe_idx // cols
    col = probe_idx % cols

    # for zi in range(zbins):
    for zi in (5, ):

        cov_g_vs_theta = np.sqrt([cov_g_10d[probe_idx, probe_idx, theta_idx, theta_idx, zi, zi, zi, zi]
                                  for theta_idx in range(theta_bins)])
        cov_sva_vs_theta = np.sqrt([cov_sva_10d[probe_idx, probe_idx, theta_idx, theta_idx, zi, zi, zi, zi]
                                    for theta_idx in range(theta_bins)])
        cov_mix_vs_theta = np.sqrt([cov_mix_10d[probe_idx, probe_idx, theta_idx, theta_idx, zi, zi, zi, zi]
                                    for theta_idx in range(theta_bins)])
        cov_sn_vs_theta = np.sqrt([cov_sn_10d[probe_idx, probe_idx, theta_idx, theta_idx, zi, zi, zi, zi]
                                   for theta_idx in range(theta_bins)])

        # errorbars
        # ax[row, col].errorbar(theta_arcmin, xi_pp_3D[:, zi, zi], yerr=cov_vs_theta, label=f'z{zi}', c=colors[zi], alpha=0.5)

        # plot signal and error separately
        ax[row, col].plot(theta_arr, probe[:, zi, zi], label=f'z{zi}', c='tab:blue')
        # ax[row, col].plot(theta_arr, cov_g_vs_theta, label=f'z{zi}', c='tab:blue', ls='--')
        ax[row, col].plot(theta_arr, cov_sva_vs_theta, label=f'z{zi}, sva', c='tab:blue', ls=':')
        ax[row, col].plot(theta_arr, cov_mix_vs_theta, label=f'z{zi}, mix', c='tab:blue', ls='--')
        ax[row, col].plot(theta_arr, cov_sn_vs_theta, label=f'z{zi}, sn', c='tab:blue', ls='-.')

    ax[row, col].set_title(probe_names[probe_idx])
    ax[row, col].set_xlabel(f'theta [{theta_unit}]')
    ax[row, col].set_ylabel('2PCF')
    ax[row, col].set_yscale('log')
    ax[row, col].set_xscale('log')
ax[row, col].legend(bbox_to_anchor=(1.22, 1), loc='center right')

print('done in ', time.perf_counter() - start, ' seconds')
