import gc
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne/bin')
import my_module as mm
import scipy
import matplotlib

# nbl = 20
# zbins = 10
# ellmax = 3000
# cov_folder = 'output_ISTF_v2'

matplotlib.use('Agg')

nbl = 32
zbins = 13
ellmax = 5000
load_mat_files = False
chunk_size = 500000
cov_folder = 'output_SPV3'

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
    cov_mat_fmt_2dcloe = np.genfromtxt(f'{cov_folder}/covariance_matrix.mat')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='original, 2d CLOE')

    cov_mat_fmt_2dcloe = np.fliplr(cov_mat_fmt_2dcloe)
    cov_mat_fmt_2dcloe = np.flipud(cov_mat_fmt_2dcloe)

    cov_mat_fmt_2ddav = mm.cov_2d_cloe_to_dav(cov_mat_fmt_2dcloe, nbl, zbins, 'ell', 'ell')

    mm.matshow(cov_mat_fmt_2dcloe, log=True, title='flipped, 2d CLOE')
    mm.matshow(cov_mat_fmt_2ddav, log=True, title='flipped, 2d Dav')

    np.savez_compressed(f'{cov_folder}/cov_tot_2dCLOE.npz', cov_mat_fmt_2dcloe)
    np.savez_compressed(f'{cov_folder}/cov_tot_2ddav.npz', cov_mat_fmt_2ddav)

    del cov_mat_fmt_2ddav, cov_mat_fmt_2dcloe
    gc.collect()

# ! consistency check for the output cls
cl_input_folder = 'input/inputs_SPV3'

cl_ll_in = np.genfromtxt(f'{cl_input_folder}/Cell_ll_CLOE.ascii')
cl_gl_in = np.genfromtxt(f'{cl_input_folder}/Cell_gl_CLOE.ascii')
cl_gg_in = np.genfromtxt(f'{cl_input_folder}/Cell_gg_CLOE.ascii')

cl_ll_out = np.genfromtxt(f'{cov_folder}/Cell_kappakappa.ascii')
cl_gl_out = np.genfromtxt(f'{cov_folder}/Cell_gkappa.ascii')
cl_gg_out = np.genfromtxt(f'{cov_folder}/Cell_gg.ascii')


ell = np.unique(cl_ll_in[:, 0])

print('nbl:', len(ell))

assert np.allclose(ell, np.unique(cl_ll_out[:, 0]), atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_ll_out, cl_ll_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gl_out, cl_gl_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gg_out, cl_gg_in, atol=0, rtol=1e-4)

plt.plot(cl_ll_out[:, 3], label='out')
plt.plot(cl_ll_in[:, 3], label='in', ls='--')


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


# Initialize an empty DataFrame or any other structure to hold aggregated results if needed
aggregated_results = None  # Example, adjust based on your needs

# df_chunk = pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1)

for df_chunk in pd.read_csv(f'{cov_folder}/covariance_list.dat', delim_whitespace=True, names=column_names, skiprows=1, chunksize=chunk_size):

    print('entered chunk loop')
    
    ells = df_chunk['ell1'].unique()
    ell_indices = {ell: idx for idx, ell in enumerate(ells)}
    assert len(ells) == nbl, 'number of ells in the list file does not match the number of ell bins'

    # ! get the individual terms from the list file
    print('number of rows in the dataframe: ', df_chunk.shape[0])

    for index, row in tqdm(df_chunk.iterrows()):

        probe = row['#obs']
        probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d = probe_idx_dict[probe[0]
                                                                            ], probe_idx_dict[probe[1]], probe_idx_dict[probe[2]], probe_idx_dict[probe[3]]

        ell1_idx = ell_indices[row['ell1']]
        ell2_idx = ell_indices[row['ell2']]
        z1_idx, z2_idx, z3_idx, z4_idx = row['tomoi'] - 1, row['tomoj'] - 1, row['tomok'] - 1, row['tomol'] - 1

        cov_sva_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                    ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sva']
        cov_mix_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                    ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg mix']
        cov_sn_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                   ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sn']
        cov_g_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d, ell1_idx, ell2_idx,
                  z1_idx, z2_idx, z3_idx, z4_idx] = row['covg sva'] + row['covg mix'] + row['covg sn']
        cov_ssc_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                    ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covssc']
        cov_cng_10d[probe_idx_a, probe_idx_b, probe_idx_c, probe_idx_d,
                    ell1_idx, ell2_idx, z1_idx, z2_idx, z3_idx, z4_idx] = row['covng']


cov_10d_dict = {
    'SVA': cov_sva_10d,
    'MIX': cov_mix_10d,
    'SN': cov_sn_10d,
    'G': cov_g_10d,
    'SSC': cov_ssc_10d,
    'cNG': cov_cng_10d,

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


cov_tot_10d = np.sum([cov_10d_dict[key] for key in cov_10d_dict.keys()])
cov_tot_3x2pt_4d = mm.cov_3x2pt_10D_to_4D(cov_tot_10d, probe_ordering, nbl, zbins, ind.copy(), GL_or_LG)
cov_tot_3x2pt_2d = mm.cov_4D_to_2D(cov_tot_3x2pt_4d, block_index='vincenzo', optimize=True)

mm.matshow(cov_tot_3x2pt_2d, log=True)
