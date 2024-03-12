import gc
import time
from matplotlib import pyplot as plt
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


cov_folder = 'output_2pcf'
cl_input_folder = '/home/cosmo/davide.sciotti/data/CLOE_validation/output/v2.0.2/C01'
cfg = configparser.ConfigParser()
cfg.read(cov_folder + '/save_configs.ini')
theta_bins = int(float(cfg['covTHETAspace settings']['theta_bins']))
zbins = 10
theta_max = int(float(cfg['covTHETAspace settings']['theta_max']))

chunk_size = 500000
load_mat_files = True


ind = mm.build_full_ind('triu', 'row-major', zbins)

probe_names = ['g', 'm', 'xip', 'xim']
column_names = [
    '#obs', 'theta1', 'theta2', 's1', 's2', 'tomoi', 'tomoj', 'tomok', 'tomol',
    'cov', 'covg', 'covng', 'covssc',
]


probe_idx_dict = {
    'g': 0,
    'm': 1,
    'xip': 2,
    'xim': 3,
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

    cov_mat_fmt_2dcloe = np.fliplr(cov_mat_fmt_2dcloe)
    cov_mat_fmt_2dcloe = np.flipud(cov_mat_fmt_2dcloe)

    cov_mat_fmt_2ddav = mm.cov_2d_cloe_to_dav(cov_mat_fmt_2dcloe, theta_bins, zbins, 'ell', 'ell')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='flipped, 2d CLOE')
    mm.matshow(cov_mat_fmt_2ddav, log=True, title='flipped, 2d Dav')

    corr_2dcloe = mm.cov2corr(cov_mat_fmt_2dcloe)

    mm.matshow(corr_2dcloe, log=False, title=' corr flipped, 2d CLOE', matshow_kwargs={'cmap':'RdBu', 'vmin':-1, 'vmax':1}) 

    # np.savez_compressed(f'{cov_folder}/cov_tot_2dCLOE.npz', cov_mat_fmt_2dcloe)
    # np.savez_compressed(f'{cov_folder}/cov_tot_2ddav.npz', cov_mat_fmt_2ddav)

    # del cov_mat_fmt_2ddav, cov_mat_fmt_2dcloe
    # gc.collect()

# ! consistency check for the output cls

# cl_ll_in = np.genfromtxt(f'{cl_input_folder}/Cell_ll_CLOE_ccl.ascii')
# cl_gl_in = np.genfromtxt(f'{cl_input_folder}/Cell_gl_CLOE_ccl.ascii')
# cl_gg_in = np.genfromtxt(f'{cl_input_folder}/Cell_gg_CLOE_ccl.ascii')

cl_ll_in = np.genfromtxt(f'{cl_input_folder}/Cij-LL-PyCCLforOneCov-C01.ascii')
cl_gl_in = np.genfromtxt(f'{cl_input_folder}/Cij-GL-PyCCLforOneCov-C01.ascii')
cl_gg_in = np.genfromtxt(f'{cl_input_folder}/Cij-GG-PyCCLforOneCov-C01.ascii')

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


# plt.plot(cl_ll_in[:, 3], label='in', ls='--')
# plt.plot(cl_ll_out[:, 3], label='out')
# plt.legend()


zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_cross + zpairs_auto, :]

# ! import _list covariance file
cov_g_10d = np.zeros((2, 2, 2, 2, theta_bins, theta_bins, zbins, zbins, zbins, zbins))


start = time.perf_counter()
# df_chunk = pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1)
for df_chunk in pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

    print('entered chunk loop')

    thetas = df_chunk['theta1'].unique()
    theta_indices = {theta: idx for idx, theta in enumerate(thetas)}
    assert len(thetas) == theta_bins, 'number of thetas in the list file does not match the number of theta bins'

    # ! get the individual terms from the list file
    print('number of rows in the dataframe: ', df_chunk.shape[0])

    for index, row in tqdm(df_chunk.iterrows()):

        probe_str = row['#obs']
        extracted_probes = extract_probes(probe_str, probe_names)
        probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d = probe_idx_dict[extracted_probes[0]], probe_idx_dict[extracted_probes[1]], \
            probe_idx_dict[extracted_probes[2]], probe_idx_dict[extracted_probes[3]]

        theta1_idx = theta_indices[row['theta1']]
        theta2_idx = theta_indices[row['theta2']]
        z1_idx, z2_idx, z3_idx, z4_idx = row['tomoi'] - 1, row['tomoj'] - 1, row['tomok'] - 1, row['tomol'] - 1

        cov_g_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                  theta1_idx, theta2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg']

print('df loaded in ', time.perf_counter() - start, ' seconds')

# let's test the 2d conversion against the .mat file, e.g. for gggg
cov_g_6d = cov_g_10d[0, 0, 0, 0, ...]
cov_g_4d = mm.cov_6D_to_4D(cov_g_6d, theta_bins, zpairs_auto, ind)
cov_g_2d = mm.cov_4D_to_2D(cov_g_4d, zbins, block_index='vincenzo')

n_elem_gggg = theta_bins * zpairs_auto
assert n_elem_gggg == cov_g_2d.shape[0], 'number of elements in the 2d cov matrix does not match the expected one'

mm.compare_arrays(cov_mat_fmt_2dcloe[:n_elem_gggg, :n_elem_gggg], cov_g_2d,
                  'cov_mat_fmt_2dcloe', 'cov_list_fmt_2dcloe', log=True)

cov_10d_dict = {
    # 'SVA': cov_sva_10d,
    # 'MIX': cov_mix_10d,
    # 'SN': cov_sn_10d,
    'G': cov_g_10d,
    # 'SSC': cov_ssc_10d,
    # 'cNG': cov_cng_10d,
    # 'tot': cov_tot_10d,
}

# for cov_term in cov_10d_dict.keys():

#     print(f'working on {cov_term}')

#     for probe_idx in range(len(probe_names)):
#         for theta_idx in range(theta_bins):

#             cov_vec = cov_term[probe_idx, probe_idx, probe_idx, probe_idx, theta_idx, theta_idx, zi, zj, zk, zl]


# TODO the line below is wrong
# TODO check nbl issue
# TODO check the total cov reshaped in this way against the 2D outputted one

# cov_tot_3x2pt_4d = mm.cov_3x2pt_10D_to_4D(cov_tot_10d, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
# cov_tot_3x2pt_2dcloe = mm.cov_4D_to_2DCLOE_3x2pt(cov_tot_3x2pt_4d, zbins, block_index='vincenzo')

# mm.compare_arrays(cov_mat_fmt_2dcloe, cov_tot_3x2pt_2dcloe, 'cov_mat_fmt_2dcloe', 'cov_list_fmt_2dcloe', log=True)

print('done in ', time.perf_counter() - start, ' seconds')
