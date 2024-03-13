import gc
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
# matplotlib.use('Agg')


cov_folder = '/home/cosmo/davide.sciotti/data/OneCovariance/output_ISTF_v2'

cfg = configparser.ConfigParser()
cfg.read(cov_folder + '/save_configs.ini')

zbins = len(cfg['survey specs']['ellipticity_dispersion'].split(', '))
nbl = int(float(cfg['covELLspace settings']['ell_bins_clustering']))
ellmax = int(float(cfg['covELLspace settings']['ell_max']))
cl_input_folder = cfg['tabulated inputs files']['cell_directory']
cl_ll_name = cfg['tabulated inputs files']['cmm_file'].strip("['").strip("']")
cl_gl_name = cfg['tabulated inputs files']['cgm_file'].strip("['").strip("']")
cl_gg_name = cfg['tabulated inputs files']['cgg_file'].strip("['").strip("']")


chunk_size = 5000000
load_mat_files = True


ind = mm.build_full_ind('triu', 'row-major', zbins)

column_names = [
    '#obs', 'ell1', 'ell2', 's1', 's2', 'tomoi', 'tomoj', 'tomok', 'tomol',
    'cov', 'covg sva', 'covg mix', 'covg sn', 'covng', 'covssc'
]

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

# ! get, show and reshape the .mat file, for a later check
if load_mat_files:

    print('Loading covariance matrix from .mat file...')
    start_time = time.perf_counter()

    cov_mat_fmt_2dcloe = np.genfromtxt(f'{cov_folder}/covariance_matrix.mat')
    print('Covariance matrix loaded in ', time.perf_counter() - start_time, ' seconds')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='original, 2d CLOE')

    corr_2dcloe = mm.cov2corr(cov_mat_fmt_2dcloe)

    mm.matshow(corr_2dcloe, log=False, title=' corr flipped, 2d CLOE',
               matshow_kwargs={'cmap': 'RdBu_r', 'vmin': -1, 'vmax': 1})

    # np.savez_compressed(f'{cov_folder}/cov_tot_2dCLOE.npz', cov_mat_fmt_2dcloe)
    # np.savez_compressed(f'{cov_folder}/cov_tot_2ddav.npz', cov_mat_fmt_2ddav)

    # del cov_mat_fmt_2ddav, cov_mat_fmt_2dcloe
    # gc.collect()

# ! consistency check for the output cls
cl_ll_in = np.genfromtxt(f'{cl_input_folder}/{cl_ll_name}')
cl_gl_in = np.genfromtxt(f'{cl_input_folder}/{cl_ll_name}')
cl_gg_in = np.genfromtxt(f'{cl_input_folder}/{cl_ll_name}')

cl_ll_out = np.genfromtxt(f'{cov_folder}/Cell_kappakappa.ascii')
cl_gl_out = np.genfromtxt(f'{cov_folder}/Cell_gkappa.ascii')
cl_gg_out = np.genfromtxt(f'{cov_folder}/Cell_gg.ascii')


ell = np.unique(cl_ll_out[:, 0])
print('nbl:', len(ell))

# assert False, 'there seems to be a problem with the ell bins, the output files doesnt have 32 bins!!'

assert np.allclose(ell, np.unique(cl_ll_out[:, 0]), atol=0, rtol=1e-4), 'ell values are not the same'
# np.testing.assert_allclose(cl_ll_out, cl_ll_in, atol=0, rtol=1e-4)
# np.testing.assert_allclose(cl_gl_out, cl_gl_in, atol=0, rtol=1e-4)
# np.testing.assert_allclose(cl_gg_out, cl_gg_in, atol=0, rtol=1e-4)


zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :]

# ! import _list covariance file
cov_g_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_sva_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_mix_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_sn_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_ssc_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_cng_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))
cov_tot_10d = np.zeros((2, 2, 2, 2, nbl, nbl, zbins, zbins, zbins, zbins))


ells = pd.read_csv(f'{cov_folder}/covariance_list.dat', usecols=['ell1'], delim_whitespace=True)['ell1'].unique()
ell_indices = {ell: idx for idx, ell in enumerate(ells)}
assert len(ells) == nbl, 'number of ells in the list file does not match the number of ell bins'

# start = time.perf_counter()
# for df_chunk in pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

#     print('entered chunk loop')

#     # ! get the individual terms from the list file
#     for index, row in tqdm(df_chunk.iterrows()):

#         probe_str = row['#obs']
#         probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d = probe_idx_dict[probe_str[0]
#                                                                             ], probe_idx_dict[probe_str[1]], probe_idx_dict[probe_str[2]], probe_idx_dict[probe_str[3]]

#         ell1_idx = ell_indices[row['ell1']]
#         ell2_idx = ell_indices[row['ell2']]
#         z1_idx, z2_idx, z3_idx, z4_idx = row['tomoi'] - 1, row['tomoj'] - 1, row['tomok'] - 1, row['tomol'] - 1

#         cov_sva_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                     ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sva']
#         cov_mix_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                     ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg mix']
#         cov_sn_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                    ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sn']
#         cov_g_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d, ell1_idx, ell2_idx,
#                   z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sva'] + row['covg mix'] + row['covg sn']
#         cov_ssc_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                     ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covssc']
#         cov_cng_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                     ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covng']
#         cov_tot_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
#                     ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['cov']
# print(f"Processed in {time.perf_counter() - start:.2f} seconds")


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

    cov_llll_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 0, 0, ...], nbl, zpairs_auto, zpairs_auto, ind_auto, ind_auto)
    cov_llgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[0, 0, 1, 0, ...], nbl, zpairs_auto, zpairs_cross, ind_auto, ind_cross)
    cov_ggll_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 0, 0, ...], nbl, zpairs_auto, zpairs_auto, ind_auto, ind_auto)
    cov_glgl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 0, 1, 0, ...], nbl,
                                         zpairs_cross, zpairs_cross, ind_cross, ind_cross)
    cov_gggl_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 0, ...], nbl, zpairs_auto, zpairs_cross, ind_auto, ind_cross)
    cov_gggg_4d = mm.cov_6D_to_4D_blocks(cov_10d[1, 1, 1, 1, ...], nbl, zpairs_auto, zpairs_auto, ind_auto, ind_auto)

    cov_llgg_4d = np.transpose(cov_ggll_4d, (1, 0, 3, 2))
    cov_glgg_4d = np.transpose(cov_gggl_4d, (1, 0, 3, 2))

    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLLL_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llll_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLGL_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llgl_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_LLGG_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_llgg_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GLGL_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_glgl_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GLGG_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_glgg_4d)
    np.savez_compressed(
        f'{cov_folder}/cov_{cov_term}_onecovariance_GGGG_4D_nbl{nbl}_ellmax{ellmax}_zbinsEP{zbins}.npz', cov_gggg_4d)

    del cov_llll_4d, cov_llgl_4d, cov_llgg_4d, cov_glgl_4d, cov_glgg_4d, cov_gggg_4d
    gc.collect()

# test mat vs list format
cov_tot_3x2pt_4d = mm.cov_3x2pt_10D_to_4D(cov_tot_10d, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
cov_tot_3x2pt_2dcloe = mm.cov_4D_to_2DCLOE_3x2pt(cov_tot_3x2pt_4d, zbins, block_index='vincenzo')

cov_llll_6d = cov_tot_10d[0, 0, 0, 0, ...]
cov_llll_4d = mm.cov_6D_to_4D(cov_llll_6d, nbl, zpairs_auto, ind_auto)
cov_llll_2d = mm.cov_4D_to_2D(cov_llll_4d, block_index='ij')

elem_auto = nbl * zpairs_auto
mm.compare_arrays(cov_mat_fmt_2dcloe[-elem_auto:, -elem_auto:], cov_llll_2d,
                  'cov_mat_fmt_2dcloe', 'cov_llll_2d', log_array=True)


print('done in ', time.perf_counter() - start, ' seconds')
