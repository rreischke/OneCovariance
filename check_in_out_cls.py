import numpy as np
import matplotlib.pyplot as plt

input_folder = 'input/inputs_SPV3'
output_folder = 'output_SPV3'

cl_ll_in = np.genfromtxt(f'{input_folder}/Cell_ll.ascii')
cl_gl_in = np.genfromtxt(f'{input_folder}/Cell_gl.ascii')
cl_gg_in = np.genfromtxt(f'{input_folder}/Cell_gg.ascii')

cl_ll_out = np.genfromtxt(f'{output_folder}/Cell_kappakappa.ascii')
cl_gl_out = np.genfromtxt(f'{output_folder}/Cell_gkappa.ascii')
cl_gg_out = np.genfromtxt(f'{output_folder}/Cell_gg.ascii')


ell = np.unique(cl_ll_in[:, 0])

print('nbl:', len(ell))

assert np.allclose(ell, np.unique(cl_ll_out[:, 0]), atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_ll_out, cl_ll_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gl_out, cl_gl_in, atol=0, rtol=1e-4)
np.testing.assert_allclose(cl_gg_out, cl_gg_in, atol=0, rtol=1e-4)

plt.plot(cl_ll_out[:, 3], label='out')
plt.plot(cl_ll_in[:, 3], label='in', ls='--')
