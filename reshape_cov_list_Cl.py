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


def cl_oc_to_3d(probe_oc_name):

    cl_col_name = ['#ell', 'tomoi', 'tomoj', probe_oc_name]

    cl_out_ells = pd.read_csv(f'{cov_folder}/{probe_oc_name}.ascii',
                              usecols=['#ell'], delim_whitespace=True)['#ell'].unique()
    cl_out_ell_indices = {ell_out: idx for idx, ell_out in enumerate(cl_out_ells)}

    cl_out_oc_3d = np.zeros((len(cl_out_ells), zbins, zbins))

    df_cl = pd.read_csv(f'{cov_folder}/{probe_oc_name}.ascii', delim_whitespace=True, names=cl_col_name, skiprows=1)

    # Map 'ell' values to their corresponding indices
    ell_idx = df_cl['#ell'].map(cl_out_ell_indices).values

    # Compute z indices
    z_indices = df_cl[['tomoi', 'tomoj']].sub(1).values

    # Vectorized assignment to the arrays
    index_tuple = (ell_idx, z_indices[:, 0], z_indices[:, 1])

    cl_out_oc_3d[index_tuple] = df_cl[probe_oc_name].values
    return cl_out_oc_3d


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

    start_time = time.perf_counter()
    print('Loading covariance matrix from .mat file...')
    cov_mat_fmt = np.genfromtxt(f'{cov_folder}/covariance_matrix.mat')
    corr_mat_fmt = mm.cov2corr(cov_mat_fmt)
    print('Covariance matrix loaded in ', time.perf_counter() - start_time, ' seconds')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Display the logarithm of the covariance matrix
    cax1 = ax[0].matshow(np.log10(cov_mat_fmt), cmap='viridis')
    ax[0].set_title('Log10 Covariance .mat fmt')
    fig.colorbar(cax1, ax=ax[0])  # Add colorbar to the first plot

    # Display the correlation matrix
    cax2 = ax[1].matshow(corr_mat_fmt, cmap='RdBu_r', vmin=-1, vmax=1)
    ax[1].set_title('Correlation .mat fmt')
    fig.colorbar(cax2, ax=ax[1])  # Add colorbar to the second plot

    plt.show()


# ! load anche check ell values from the .dat covariance file
cov_ells = pd.read_csv(f'{cov_folder}/covariance_list.dat', usecols=['ell1'], delim_whitespace=True)['ell1'].unique()
cov_ell_indices = {ell_out: idx for idx, ell_out in enumerate(cov_ells)}
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
    ell1_idx = df_chunk['ell1'].map(cov_ell_indices).values
    ell2_idx = df_chunk['ell2'].map(cov_ell_indices).values

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

# ! do the same for the cls, to get a consistent plot of the signal +- errorbars
cl_oc_out_ll_3d = cl_oc_to_3d('Cell_kappakappa')
cl_oc_out_gl_3d = cl_oc_to_3d('Cell_gkappa')
cl_oc_out_gg_3d = cl_oc_to_3d('Cell_gg')


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

cov_mat_fmt_2dcloe_llll = cov_mat_fmt[-n_elem_auto:, -n_elem_auto:]
cov_mat_fmt_2dcloe_glgl = cov_mat_fmt[n_elem_auto:n_elem_auto +
                                      n_elem_cross, n_elem_auto:n_elem_auto + n_elem_cross]
cov_mat_fmt_2dcloe_gggg = cov_mat_fmt[:n_elem_auto, :n_elem_auto]

cov_tot_3x2pt_2dcloe_llll = cov_tot_3x2pt_2dcloe[:n_elem_auto, :n_elem_auto]
cov_tot_3x2pt_2dcloe_glgl = cov_tot_3x2pt_2dcloe[n_elem_auto:n_elem_auto +
                                                 n_elem_cross, n_elem_auto:n_elem_auto + n_elem_cross]
cov_tot_3x2pt_2dcloe_gggg = cov_tot_3x2pt_2dcloe[-n_elem_auto:, -n_elem_auto:]

for cov_mat_fmt_block, cov_dat_fmt_bloc in zip((cov_mat_fmt_2dcloe_llll, cov_mat_fmt_2dcloe_glgl, cov_mat_fmt_2dcloe_gggg),
                                               (cov_tot_3x2pt_2dcloe_llll, cov_tot_3x2pt_2dcloe_glgl, cov_tot_3x2pt_2dcloe_gggg)):
    mm.compare_arrays(cov_mat_fmt_block, cov_mat_fmt_block,
                      'cov_mat_fmt_2dcloe_llll', 'cov_tot_3x2pt_2dcloe_llll', log_array=True)


# ! plot Cl and errors
probe_names = ['Cl_LL', 'Cl_GL', 'Cl_GG']
cols = 3
rows = 1
fig, ax = plt.subplots(rows, cols, figsize=(15, 4))
for probe_idx, probe in zip((range(cols)), (cl_oc_out_ll_3d, cl_oc_out_gl_3d, cl_oc_out_gg_3d)):

    if probe_idx == 0:
        probe_idx_list = (0, 0, 0, 0)
    elif probe_idx == 1:
        probe_idx_list = (1, 0, 1, 0)
    elif probe_idx == 2:
        probe_idx_list = (1, 1, 1, 1)

    # for zi in range(zbins):
    for zi in (5, ):

        cov_g_vs_ell = np.sqrt([cov_g_10d[probe_idx_list[0], probe_idx_list[1], probe_idx_list[2], probe_idx_list[3],
                                          ell_idx, ell_idx, zi, zi, zi, zi] for ell_idx in range(cov_nbl)])
        cov_sva_vs_ell = np.sqrt([cov_sva_10d[probe_idx_list[0], probe_idx_list[1], probe_idx_list[2], probe_idx_list[3],
                                              ell_idx, ell_idx, zi, zi, zi, zi] for ell_idx in range(cov_nbl)])
        cov_mix_vs_ell = np.sqrt([cov_mix_10d[probe_idx_list[0], probe_idx_list[1], probe_idx_list[2], probe_idx_list[3],
                                              ell_idx, ell_idx, zi, zi, zi, zi] for ell_idx in range(cov_nbl)])
        cov_sn_vs_ell = np.sqrt([cov_sn_10d[probe_idx_list[0], probe_idx_list[1], probe_idx_list[2], probe_idx_list[3],
                                            ell_idx, ell_idx, zi, zi, zi, zi] for ell_idx in range(cov_nbl)])

        # errorbars
        # ax[col].errorbar(theta_arcmin, xi_pp_3D[:, zi, zi], yerr=cov_vs_ell, label=f'z{zi}', c=colors[zi], alpha=0.5)

        # plot signal and error separately
        # ax[probe_idx].plot(cl_out_ells, probe[:, zi, zi], label=f'z{zi}', c='red', marker='.')
        ax[probe_idx].plot(cov_ells, cov_g_vs_ell, label=f'z{zi}, g tot', c='k', ls='--', marker='.')
        ax[probe_idx].plot(cov_ells, cov_sva_vs_ell, label=f'z{zi}, sva', c='tab:green', ls=':', marker='.')
        ax[probe_idx].plot(cov_ells, cov_mix_vs_ell, label=f'z{zi}, mix', c='tab:orange', ls=':', marker='.')
        ax[probe_idx].plot(cov_ells, cov_sn_vs_ell, label=f'z{zi}, sn', c='tab:purple', ls=':', marker='.')

    ax[probe_idx].set_title(probe_names[probe_idx])
    ax[probe_idx].set_xlabel('$\ell$')
    ax[probe_idx].set_ylabel('$C(\ell)$')
    ax[probe_idx].set_yscale('log')
    ax[probe_idx].set_xscale('log')
ax[probe_idx].legend(bbox_to_anchor=(1.22, 1), loc='center right')


print('done in ', time.perf_counter() - start, ' seconds')
