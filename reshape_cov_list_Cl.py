import gc
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import scipy
import matplotlib
import configparser
import os
ROOT = os.getenv('ROOT')
sys.path.append(f'{ROOT}/Spaceborne')
import bin.my_module as mm
import bin.ell_values as ell_utils


cov_folder = '/home/cosmo/davide.sciotti/data/OneCovariance/output_Cl_C01'

cfg = configparser.ConfigParser()
cfg.read(cov_folder + '/save_configs.ini')

zbins = len(cfg['survey specs']['ellipticity_dispersion'].split(', '))
cl_cfg_nbl = int(float(cfg['covELLspace settings']['ell_bins']))
ellmax = int(float(cfg['covELLspace settings']['ell_max']))
ellmin = int(float(cfg['covELLspace settings']['ell_min']))
cl_input_folder = cfg['tabulated inputs files']['cell_directory']
cl_ll_name = cfg['tabulated inputs files']['cmm_file'].strip("['").strip("']")
cl_gl_name = cfg['tabulated inputs files']['cgm_file'].strip("['").strip("']")
cl_gg_name = cfg['tabulated inputs files']['cgg_file'].strip("['").strip("']")

chunk_size = 5000000
load_mat_files = True

column_names = [
    '#obs', 'ell1', 'ell2', 's1', 's2', 'tomoi', 'tomoj', 'tomok', 'tomol',
    'cov', 'covg sva', 'covg mix', 'covg sn', 'covng', 'covssc'
]

ind = mm.build_full_ind('triu', 'row-major', zbins)
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :]
ind_dict = {
    ('L', 'L'): ind_auto,
    ('G', 'L'): ind_cross,
    ('G', 'G'): ind_auto,
}

probe_idx_dict = {
    'm': 0,
    'g': 1,
}

probe_name_dict = {
    0: 'L',
    1: 'G',
}

probe_ordering = (('L', 'L'), ('G', 'L'), ('G', 'G'))
GL_or_LG = 'GL'


# ! consistency check for the input/output cls
cl_ll_in = np.genfromtxt(f'{cl_input_folder}/{cl_ll_name}')
cl_gl_in = np.genfromtxt(f'{cl_input_folder}/{cl_gl_name}')
cl_gg_in = np.genfromtxt(f'{cl_input_folder}/{cl_gg_name}')

cl_ll_out = np.genfromtxt(f'{cov_folder}/Cell_kappakappa.ascii')
cl_gl_out = np.genfromtxt(f'{cov_folder}/Cell_gkappa.ascii')
cl_gg_out = np.genfromtxt(f'{cov_folder}/Cell_gg.ascii')

cl_in_ells = np.unique(cl_ll_in[:, 0])
cl_out_ells = np.unique(cl_ll_out[:, 0])

assert np.allclose(cl_in_ells, cl_out_ells, atol=0, rtol=1e-4), 'ell values are not the same'
np.testing.assert_allclose(cl_ll_out, cl_ll_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gl_out, cl_gl_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gg_out, cl_gg_in, atol=0, rtol=1e-4)

print('nbl_cl_in:', len(cl_in_ells))
print('nbl_cl_out:', len(cl_out_ells))
print('nbl_cl_cfg:', cl_cfg_nbl, '\n')

# ! read and print the header, check that matches the one manually defined
with open(f'{cov_folder}/covariance_list.dat', 'r') as file:
    header = file.readline().strip()  # Read the first line and strip newline characters
print('.dat file header: ')
print(header)
header_list = re.split('\t', header.strip().replace('\t\t', '\t').replace('\t\t', '\t'))
assert column_names == header_list, 'column names from .dat file do not match with the expected ones'

# ! get, show and reshape the .mat file, for a later check
if load_mat_files:

    print('Loading covariance matrix from .mat file...')
    start_time = time.perf_counter()

    cov_mat_fmt_2dcloe = np.genfromtxt(f'{cov_folder}/covariance_matrix.mat')
    print('Covariance matrix loaded in ', time.perf_counter() - start_time, ' seconds')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='cov, 2dCLOE')

    corr_2dcloe = mm.cov2corr(cov_mat_fmt_2dcloe)

    mm.matshow(corr_2dcloe, log=False, title=' corr 2dCLOE',
               matshow_kwargs={'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1})

    # np.savez_compressed(f'{cov_folder}/cov_tot_2dCLOE.npz', cov_mat_fmt_2dcloe)
    # np.savez_compressed(f'{cov_folder}/cov_tot_2ddav.npz', cov_mat_fmt_2ddav)

    # del cov_mat_fmt_2ddav, cov_mat_fmt_2dcloe
    # gc.collect()

# ! load anche check ell values from the .dat covariance file
cov_ells = pd.read_csv(f'{cov_folder}/covariance_list.dat', usecols=['ell1'], delim_whitespace=True)['ell1'].unique()
ell_indices = {ell_out: idx for idx, ell_out in enumerate(cov_ells)}
# assert len(cov_ells) == nbl, 'number of ells in the list file does not match the number of ell bins'

# this is taken from OC (in cov_ell_space.py)
ellrange_clustering_ul = np.unique(np.geomspace(float(cfg['covELLspace settings']['ell_min_clustering']),
                                                float(cfg['covELLspace settings']['ell_max_clustering']),
                                                int(cfg['covELLspace settings']['ell_bins_clustering']) + 1).astype(int))
cov_ells_manual = np.exp(.5 * (np.log(ellrange_clustering_ul[1:])
                               + np.log(ellrange_clustering_ul[:-1])))
np.testing.assert_allclose(cov_ells, cov_ells_manual, atol=0, rtol=1e-1,
                           err_msg='ell values from the .dat file do not match with \
                           the ones computed manyally using OC recipe (to 1% tolerance)')

print('covariance computed at ell values:\n', cov_ells)
cov_nbl = len(cov_ells)

# compare ell edges - perfect match if I drop the cast to int
ell_bin_edges_sb = np.logspace(np.log10(ellmin), np.log10(ellmax), cov_nbl + 1)
ell_bin_edges_oc_float = np.unique(np.geomspace(float(cfg['covELLspace settings']['ell_min_clustering']),
                                                float(cfg['covELLspace settings']['ell_max_clustering']),
                                                int(cfg['covELLspace settings']['ell_bins_clustering']) + 1))
np.testing.assert_allclose(ell_bin_edges_sb, ell_bin_edges_oc_float, atol=0, rtol=1e-6)

ells_sb = (ell_bin_edges_sb[1:] + ell_bin_edges_sb[:-1]) / 2
ells_oc_float = np.exp(.5 * (np.log(ell_bin_edges_oc_float[1:])
                       + np.log(ell_bin_edges_oc_float[:-1])))  # it's the same if I take base 10 log

# ell_sb can also be obtained as
# ells_sb, _ = ell_utils.compute_ells(nbl=cov_nbl, ell_min=ellmin, ell_max=ellmax,
# recipe='ISTF', output_ell_bin_edges=False)
try:
    np.testing.assert_allclose(ells_sb, ells_oc_float, atol=0, rtol=1e-6)
except AssertionError:
    diff_oc = mm.percent_diff(ells_sb, cov_ells)
    print('ells_sb:\n', ells_sb)
    print('\nells_oc:\n', cov_ells)
    print('\nells_oc_float:\n', ells_oc_float)
    print('\npercent diff OneCov:\n', mm.percent_diff(ells_sb, cov_ells))
    print('\npercent diff OneCov, with float edges:\n', mm.percent_diff(ells_sb, ells_oc_float))

# ! import _list covariance file
cov_g_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_sva_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_mix_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_sn_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_ssc_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_cng_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))
cov_tot_10d = np.zeros((2, 2, 2, 2, cov_nbl, cov_nbl, zbins, zbins, zbins, zbins))


print('loading dataframe in chunks...')
start = time.perf_counter()
for df_chunk in pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

    # Vectorize the extraction of probe indices
    probe_idx_a = df_chunk['#obs'].str[0].map(probe_idx_dict).values
    probe_idx_b = df_chunk['#obs'].str[1].map(probe_idx_dict).values
    probe_idx_c = df_chunk['#obs'].str[2].map(probe_idx_dict).values
    probe_idx_d = df_chunk['#obs'].str[3].map(probe_idx_dict).values

    # Map 'ell' values to their corresponding indices
    ell1_idx = df_chunk['ell1'].map(ell_indices).values
    ell2_idx = df_chunk['ell2'].map(ell_indices).values

    # Compute z indices
    z_indices = df_chunk[['tomoi', 'tomoj', 'tomok', 'tomol']].sub(1).values

    # Vectorized assignment to the arrays
    index_tuple = (probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d, ell1_idx, ell2_idx,
                   z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

    cov_sva_10d[index_tuple] = df_chunk['covg sva'].values
    cov_mix_10d[index_tuple] = df_chunk['covg mix'].values
    cov_sn_10d[index_tuple] = df_chunk['covg sn'].values
    cov_g_10d[index_tuple] = df_chunk['covg sva'].values + df_chunk['covg mix'].values + df_chunk['covg sn'].values
    cov_ssc_10d[index_tuple] = df_chunk['covssc'].values
    cov_cng_10d[index_tuple] = df_chunk['covng'].values
    cov_tot_10d[index_tuple] = df_chunk['cov'].values

print(f"df loaded in {time.perf_counter() - start:.2f} seconds")


cov_10d_dict = {
    'SVA': cov_sva_10d,
    'MIX': cov_mix_10d,
    'SN': cov_sn_10d,
    'G': cov_g_10d,
    'SSC': cov_ssc_10d,
    'cNG': cov_cng_10d,
    'tot': cov_tot_10d,
}

for cov_term in cov_10d_dict.keys():

    print(f'working on {cov_term}')

    cov_10d = cov_10d_dict[cov_term]

    cov_llll_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 0, 0, ...], cov_nbl,
                                         zpairs_auto, zpairs_auto, ind_auto, ind_auto)
    cov_llgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 1, 0, ...], cov_nbl,
                                         zpairs_auto, zpairs_cross, ind_auto, ind_cross)
    cov_ggll_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 0, 0, ...], cov_nbl,
                                         zpairs_auto, zpairs_auto, ind_auto, ind_auto)
    cov_glgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 0, 1, 0, ...], cov_nbl,
                                         zpairs_cross, zpairs_cross, ind_cross, ind_cross)
    cov_gggl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 0, ...], cov_nbl,
                                         zpairs_auto, zpairs_cross, ind_auto, ind_cross)
    cov_gggg_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 1, ...], cov_nbl,
                                         zpairs_auto, zpairs_auto, ind_auto, ind_auto)

    cov_llgg_4d = np.transpose(cov_ggll_4d, (1, 0, 3, 2))
    cov_glll_4d = np.transpose(cov_llgl_4d, (1, 0, 3, 2))
    cov_glgg_4d = np.transpose(cov_gggl_4d, (1, 0, 3, 2))

    cov_10d_dict[cov_term][0, 0, 1, 1] = mm.cov_4D_to_6D_blocks(cov_llgg_4d, cov_nbl, zbins, ind_auto, ind_auto)
    cov_10d_dict[cov_term][1, 0, 0, 0] = mm.cov_4D_to_6D_blocks(cov_glll_4d, cov_nbl, zbins, ind_cross, ind_auto)
    cov_10d_dict[cov_term][1, 0, 1, 1] = mm.cov_4D_to_6D_blocks(cov_glgg_4d, cov_nbl, zbins, ind_cross, ind_auto)

    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLLL_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llll_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLGL_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llgl_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLGG_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llgg_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GLGL_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_glgl_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GLGG_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_glgg_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GGGG_4D_nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_gggg_4d)

    del cov_llll_4d, cov_llgl_4d, cov_llgg_4d, cov_glgl_4d, cov_glgg_4d, cov_gggg_4d, cov_ggll_4d, cov_glll_4d, cov_gggl_4d
    gc.collect()


# ! construct 2d Cov as you do in spaceborne, from input blocks
block_index = 'ij'
cov_filename = 'cov_tot_onecovariance_{probe_a:s}{probe_b:s}{probe_c:s}{probe_d:s}_4D_' + \
    f'nbl{cov_nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz'
cov_3x2pt_dict_8D_load = mm.load_cov_from_probe_blocks(cov_folder, cov_filename, probe_ordering)
cov_3x2pt_dict_10D_load = mm.cov_3x2pt_dict_8d_to_10d(cov_3x2pt_dict_8D_load, cov_nbl, zbins, ind_dict, probe_ordering)
cov_tot_3x2pt_4d_load = mm.cov_3x2pt_10D_to_4D(
    cov_3x2pt_dict_10D_load, probe_ordering, cov_nbl, zbins, ind.copy(), GL_or_LG)
cov_tot_3x2pt_2dcloe_load = mm.cov_4D_to_2DCLOE_3x2pt(cov_tot_3x2pt_4d_load, zbins, block_index=block_index)

# check that it matches the one constructed on the fly
cov_tot_3x2pt_4d = mm.cov_3x2pt_10D_to_4D(cov_tot_10d, probe_ordering, cov_nbl, zbins, ind.copy(), GL_or_LG)
cov_tot_3x2pt_2dcloe = mm.cov_4D_to_2DCLOE_3x2pt(cov_tot_3x2pt_4d, zbins, block_index=block_index)
mm.compare_arrays(cov_tot_3x2pt_2dcloe_load, cov_tot_3x2pt_2dcloe,
                  'cov_tot_3x2pt_2dcloe_load', 'cov_tot_3x2pt_2dcloe', log_array=True)

# compare against the mat format, *which has the gg, gl, ll order instead of ll, gl, gg*
n_elem_auto = cov_nbl * zpairs_auto
n_elem_cross = cov_nbl * zpairs_cross

cov_mat_fmt_2dcloe_llll = cov_mat_fmt_2dcloe[-n_elem_auto:, -n_elem_auto:]
cov_mat_fmt_2dcloe_glgl = cov_mat_fmt_2dcloe[n_elem_auto:n_elem_auto +
                                             n_elem_cross, n_elem_auto:n_elem_auto + n_elem_cross]
cov_mat_fmt_2dcloe_gggg = cov_mat_fmt_2dcloe[:n_elem_auto, :n_elem_auto]

cov_tot_3x2pt_2dcloe_llll = cov_tot_3x2pt_2dcloe[:n_elem_auto, :n_elem_auto]
cov_tot_3x2pt_2dcloe_glgl = cov_tot_3x2pt_2dcloe[n_elem_auto:n_elem_auto +
                                                 n_elem_cross, n_elem_auto:n_elem_auto + n_elem_cross]
cov_tot_3x2pt_2dcloe_gggg = cov_tot_3x2pt_2dcloe[-n_elem_auto:, -n_elem_auto:]

for cov_mat_fmt_block, cov_dat_fmt_bloc in zip((cov_mat_fmt_2dcloe_llll, cov_mat_fmt_2dcloe_glgl, cov_mat_fmt_2dcloe_gggg),
                                               (cov_tot_3x2pt_2dcloe_llll, cov_tot_3x2pt_2dcloe_glgl, cov_tot_3x2pt_2dcloe_gggg)):
    mm.compare_arrays(cov_mat_fmt_block, cov_mat_fmt_block,
                      'cov_mat_fmt_2dcloe_llll', 'cov_tot_3x2pt_2dcloe_llll', log_array=True)


# TODO do this fot the cls to have a visual check against the 2PCF errors

# ! plot 2PCF and errors
# xi_mm_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-Lminus-PyCCL-C01.dat')

print(cl_ll_out.shape)
assert False, 'stop here'

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
