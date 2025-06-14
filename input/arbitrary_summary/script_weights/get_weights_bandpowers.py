import numpy as np
from scipy.special import jv
import time
import levin
import argparse


def main():
    parser = argparse.ArgumentParser(description='Calculates fourier weights for bandpowers')
    parser.add_argument('-n', '--nthread', type=int, default=10, help='number of threads used (default is 10)')
    parser.add_argument('-nf', '--nfourier', type=int, default=int(1e4), help='number of Fourier modes at which the weights are calculated (default is 1e4)')
    parser.add_argument('-nt', '--ntheta', type=int, default=int(1e4), help='number of theta at which the realspace weights are calculated (default is 1e4)')

    parser.add_argument('-dt_mm', '--delta_ln_theta_mm', type=float, default=0.5, help='apodisation_log_width for lensing(default is 0.5)')
    parser.add_argument('-tlo_mm', '--theta_lo_mm', type=float, default=0.5, help='lower limit for angular range in arcmin for lensing(default is 0.5)')
    parser.add_argument('-tup_mm', '--theta_up_mm', type=float, default=300, help='upper limit for angular range in arcmin for lensing(default is 300)')
    parser.add_argument('-llo_mm', '--L_min_mm', type=float, default=100, help='Minimum bandpower multipole for lensing (default is 100)')
    parser.add_argument('-lup_mm', '--L_max_mm', type=float, default=1500, help='Maximum bandpower multipole for lensing (default is 1500)')
    parser.add_argument('-lb_mm', '--L_bins_mm', type=int, default=8, help='Number of bandpower multipole bins for lensing (default is 8)')
    parser.add_argument('-lt_mm', '--L_type_mm', type=str, default='log', help='Type if binning for bandpower multipoles for lensing (default is log)')

    parser.add_argument('-dt_gm', '--delta_ln_theta_gm', type=float, default=0.5, help='apodisation_log_width for ggl(default is 0.5)')
    parser.add_argument('-tlo_gm', '--theta_lo_gm', type=float, default=0.5, help='lower limit for angular range in arcmin for ggl(default is 0.5)')
    parser.add_argument('-tup_gm', '--theta_up_gm', type=float, default=300, help='upper limit for angular range in arcmin for ggl(default is 300)')
    parser.add_argument('-llo_gm', '--L_min_gm', type=float, default=100, help='Minimum bandpower multipole for ggl (default is 100)')
    parser.add_argument('-lup_gm', '--L_max_gm', type=float, default=1500, help='Maximum bandpower multipole for ggl (default is 1500)')
    parser.add_argument('-lb_gm', '--L_bins_gm', type=int, default=8, help='Number of bandpower multipole bins for ggl (default is 8)')
    parser.add_argument('-lt_gm', '--L_type_gm', type=str, default='log', help='Type if binning for bandpower multipoles for ggl (default is log)')

    parser.add_argument('-dt_gg', '--delta_ln_theta_gg', type=float, default=0.5, help='apodisation_log_width for clustering(default is 0.5)')
    parser.add_argument('-tlo_gg', '--theta_lo_gg', type=float, default=0.5, help='lower limit for angular range in arcmin for clustering(default is 0.5)')
    parser.add_argument('-tup_gg', '--theta_up_gg', type=float, default=300, help='upper limit for angular range in arcmin for clustering(default is 300)')
    parser.add_argument('-llo_gg', '--L_min_gg', type=float, default=100, help='Minimum bandpower multipole for clustering (default is 100)')
    parser.add_argument('-lup_gg', '--L_max_gg', type=float, default=1500, help='Maximum bandpower multipole for clustering (default is 1500)')
    parser.add_argument('-lb_gg', '--L_bins_gg', type=int, default=8, help='Number of bandpower multipole bins for clustering (default is 8)')
    parser.add_argument('-lt_gg', '--L_type_gg', type=str, default='log', help='Type if binning for bandpower multipoles for clustering (default is log)')

    args = parser.parse_args()

    num_cores = args.nthread
    N_fourier = args.nfourier # at how many ells should the Bessel functions be evaluated
    n_theta_bins = args.ntheta # How many theta bins used for evaluation
    fourier_ell = np.geomspace(1,1e4,N_fourier) 

    delta_ln_theta_mm = args.delta_ln_theta_mm
    theta_lo_mm = args.theta_lo_mm
    theta_up_mm = args.theta_up_mm
    L_min_mm = args.L_min_mm
    L_max_mm = args.L_max_mm
    L_bins_mm = args.L_bins_mm
    L_type_mm = args.L_type_mm

    delta_ln_theta_gm = args.delta_ln_theta_gm
    theta_lo_gm = args.theta_lo_gm
    theta_up_gm = args.theta_up_gm
    L_min_gm = args.L_min_gm
    L_max_gm = args.L_max_gm
    L_bins_gm = args.L_bins_gm
    L_type_gm = args.L_type_gm

    delta_ln_theta_gg = args.delta_ln_theta_gg
    theta_lo_gg = args.theta_lo_gg
    theta_up_gg = args.theta_up_gg
    L_min_gg = args.L_min_gg
    L_max_gg = args.L_max_gg
    L_bins_gg = args.L_bins_gg
    L_type_gg = args.L_type_gg

    hdr_str_mm_plus_fourier = 'Bandpower weights for C_E in Fourier space\n'
    hdr_str_mm_plus_fourier += 'lowest theta boundary = ' + str(theta_lo_mm) + '\n'
    hdr_str_mm_plus_fourier += 'highest theta boundary = ' + str(theta_up_mm) + '\n'
    hdr_str_mm_plus_fourier += 'apodization beand width = ' + str(delta_ln_theta_mm) + '\n'
    hdr_str_mm_plus_fourier += 'minimum multipole = ' + str(L_min_mm) + '\n'
    hdr_str_mm_plus_fourier += 'maximum multipole = ' + str(L_max_mm) + '\n'
    hdr_str_mm_plus_fourier += 'number of bandpowers modes = ' + str(L_bins_mm) + '\n'
    hdr_str_mm_plus_fourier += 'binning type for multipoles = ' + str(L_type_mm) + '\n'
    hdr_str_mm_plus_fourier += 'ell      W(ell)'

    hdr_str_mm_minus_fourier = 'Bandpower weights for C_B in Fourier space\n'
    hdr_str_mm_minus_fourier += 'lowest theta boundary = ' + str(theta_lo_mm) + '\n'
    hdr_str_mm_minus_fourier += 'highest theta boundary = ' + str(theta_up_mm) + '\n'
    hdr_str_mm_minus_fourier += 'apodization beand width = ' + str(delta_ln_theta_mm) + '\n'
    hdr_str_mm_minus_fourier += 'minimum multipole = ' + str(L_min_mm) + '\n'
    hdr_str_mm_minus_fourier += 'maximum multipole = ' + str(L_max_mm) + '\n'
    hdr_str_mm_minus_fourier += 'number of bandpowers modes = ' + str(L_bins_mm) + '\n'
    hdr_str_mm_minus_fourier += 'binning type for multipoles = ' + str(L_type_mm) + '\n'
    hdr_str_mm_minus_fourier += 'ell      W(ell)'


    hdr_str_mm_plus_real = 'Bandpower weights for C_E in Fourier space\n'
    hdr_str_mm_plus_real += 'lowest theta boundary = ' + str(theta_lo_mm) + '\n'
    hdr_str_mm_plus_real += 'highest theta boundary = ' + str(theta_up_mm) + '\n'
    hdr_str_mm_plus_real += 'apodization beand width = ' + str(delta_ln_theta_mm) + '\n'
    hdr_str_mm_plus_real += 'minimum multipole = ' + str(L_min_mm) + '\n'
    hdr_str_mm_plus_real += 'maximum multipole = ' + str(L_max_mm) + '\n'
    hdr_str_mm_plus_real += 'number of bandpowers modes = ' + str(L_bins_mm) + '\n'
    hdr_str_mm_plus_real += 'binning type for multipoles = ' + str(L_type_mm) + '\n'
    hdr_str_mm_plus_real += 'theta[arcmin]      R(theta)'

    hdr_str_mm_minus_real = 'Bandpower weights for C_B in Fourier space\n'
    hdr_str_mm_minus_real += 'lowest theta boundary = ' + str(theta_lo_mm) + '\n'
    hdr_str_mm_minus_real += 'highest theta boundary = ' + str(theta_up_mm) + '\n'
    hdr_str_mm_minus_real += 'apodization beand width = ' + str(delta_ln_theta_mm) + '\n'
    hdr_str_mm_minus_real += 'minimum multipole = ' + str(L_min_mm) + '\n'
    hdr_str_mm_minus_real += 'maximum multipole = ' + str(L_max_mm) + '\n'
    hdr_str_mm_minus_real += 'number of bandpowers modes = ' + str(L_bins_mm) + '\n'
    hdr_str_mm_minus_real += 'binning type for multipoles = ' + str(L_type_mm) + '\n'
    hdr_str_mm_minus_real += 'theta[arcmin]      R(theta)'


    hdr_str_gm_fourier = 'Bandpower weights for C of ggl in Fourier space\n'
    hdr_str_gm_fourier += 'lowest theta boundary = ' + str(theta_lo_gm) + '\n'
    hdr_str_gm_fourier += 'highest theta boundary = ' + str(theta_up_gm) + '\n'
    hdr_str_gm_fourier += 'apodization beand width = ' + str(delta_ln_theta_gm) + '\n'
    hdr_str_gm_fourier += 'minimum multipole = ' + str(L_min_gm) + '\n'
    hdr_str_gm_fourier += 'maximum multipole = ' + str(L_max_gm) + '\n'
    hdr_str_gm_fourier += 'number of bandpowers modes = ' + str(L_bins_gm) + '\n'
    hdr_str_gm_fourier += 'binning type for multipoles = ' + str(L_type_gm) + '\n'
    hdr_str_gm_fourier += 'ell      W(ell)'

    hdr_str_gm_real = 'Bandpower weights for C of ggl in Fourier space\n'
    hdr_str_gm_real += 'lowest theta boundary = ' + str(theta_lo_gm) + '\n'
    hdr_str_gm_real += 'highest theta boundary = ' + str(theta_up_gm) + '\n'
    hdr_str_gm_real += 'apodization beand width = ' + str(delta_ln_theta_gm) + '\n'
    hdr_str_gm_real += 'minimum multipole = ' + str(L_min_gm) + '\n'
    hdr_str_gm_real += 'maximum multipole = ' + str(L_max_gm) + '\n'
    hdr_str_gm_real += 'number of bandpowers modes = ' + str(L_bins_gm) + '\n'
    hdr_str_gm_real += 'binning type for multipoles = ' + str(L_type_gm) + '\n'
    hdr_str_gm_real += 'theta[arcmin]      R(theta)'

    hdr_str_gg_fourier = 'Bandpower weights for C of clustering in Fourier space\n'
    hdr_str_gg_fourier += 'lowest theta boundary = ' + str(theta_lo_gg) + '\n'
    hdr_str_gg_fourier += 'highest theta boundary = ' + str(theta_up_gg) + '\n'
    hdr_str_gg_fourier += 'apodization beand width = ' + str(delta_ln_theta_gg) + '\n'
    hdr_str_gg_fourier += 'minimum multipole = ' + str(L_min_gg) + '\n'
    hdr_str_gg_fourier += 'maximum multipole = ' + str(L_max_gg) + '\n'
    hdr_str_gg_fourier += 'number of bandpowers modes = ' + str(L_bins_gg) + '\n'
    hdr_str_gg_fourier += 'binning type for multipoles = ' + str(L_type_gg) + '\n'
    hdr_str_gg_fourier += 'ell      W(ell)'

    hdr_str_gg_real = 'Bandpower weights for C of clustering Fourier space\n'
    hdr_str_gg_real += 'lowest theta boundary = ' + str(theta_lo_gg) + '\n'
    hdr_str_gg_real += 'highest theta boundary = ' + str(theta_up_gg) + '\n'
    hdr_str_gg_real += 'apodization beand width = ' + str(delta_ln_theta_gg) + '\n'
    hdr_str_gg_real += 'minimum multipole = ' + str(L_min_gg) + '\n'
    hdr_str_gg_real += 'maximum multipole = ' + str(L_max_gg) + '\n'
    hdr_str_gg_real += 'number of bandpowers modes = ' + str(L_bins_gg) + '\n'
    hdr_str_gg_real += 'binning type for multipoles = ' + str(L_type_gg) + '\n'
    hdr_str_gg_real += 'theta[arcmin]      R(theta)'

    theta_min_gm = np.exp(np.log(theta_lo_gm) - delta_ln_theta_gm/2) # real lower limit after apodisation
    theta_max_gm = np.exp(np.log(theta_up_gm) + delta_ln_theta_gm/2) # real upper limit after apodisation
    theta_ul_bins_gm = np.geomspace(
                    theta_min_gm,
                    theta_max_gm,
                    n_theta_bins + 1)
    theta_bins_gm = np.exp(.5 * (np.log(theta_ul_bins_gm[1:])+ np.log(theta_ul_bins_gm[:-1])))
    theta_min_gg = np.exp(np.log(theta_lo_gg) - delta_ln_theta_gg/2) # real lower limit after apodisation
    theta_max_gg = np.exp(np.log(theta_up_gg) + delta_ln_theta_gg/2) # real upper limit after apodisation
    theta_ul_bins_gg = np.geomspace(
                    theta_min_gg,
                    theta_max_gg,
                    n_theta_bins + 1)
    theta_bins_gg = np.exp(.5 * (np.log(theta_ul_bins_gg[1:])+ np.log(theta_ul_bins_gg[:-1])))
    theta_min_mm = np.exp(np.log(theta_lo_mm) - delta_ln_theta_mm/2) # real lower limit after apodisation
    theta_max_mm = np.exp(np.log(theta_up_mm) + delta_ln_theta_mm/2) # real upper limit after apodisation
    theta_ul_bins_mm = np.geomspace(
                    theta_min_mm,
                    theta_max_mm,
                    n_theta_bins + 1)
    theta_bins_mm = np.exp(.5 * (np.log(theta_ul_bins_mm[1:])+ np.log(theta_ul_bins_mm[:-1])))

    def get_L_bins(L_type, L_min, L_max, nL_bins, L_list_boundary = None):
        '''
        This function returns the L bins and the corresponding bin boundaries

        Parameters:
        -----------
        L_type : string
            Do you want lin-/log-spaced L's or a list? Can be 'lin', 'log' or 'list'
        L_min : float
            Minimum angle (lowest bin boundary NOT the central value)
        L_max : float
            Maximum angle (higest bin boundary NOT the central value)
        nL_bins : integer
            How many L bins should there be?
        L_list_boundary : array
            Array of all L-bin boundaries (arbitrarily spaced)
        '''
        if L_type == 'lin':
            L_ul_bins = np.linspace(L_min, L_max, nL_bins + 1)
            L_bins = .5 * (L_ul_bins[1:] + L_ul_bins[:-1])
        if L_type == 'log':
            L_ul_bins = np.geomspace(L_min, L_max, nL_bins + 1)
            L_bins = np.exp(.5 * (np.log(L_ul_bins[1:])
                                    + np.log(L_ul_bins[:-1])))
        if L_type == 'list' and L_list_boundary is not None:
            L_ul_bins = L_list_boundary
            L_bins = .5 * (L_ul_bins[1:] + L_ul_bins[:-1])

        return L_bins, L_ul_bins

    def get_Hann_window(thetabins, delta_ln_theta, theta_lo, theta_up):
        """
        Precomputes the Hann window for the apodisation, yielding the function
        T(theta).
        """
        log_theta_bins = np.log(thetabins)
        T_of_theta = np.zeros(len(thetabins))
        xlo = np.log(theta_lo)
        xup = np.log(theta_up)
        for i_theta in range(len(thetabins)):
            x = log_theta_bins[i_theta]
            if x < xlo + delta_ln_theta/2.0:
                T_of_theta[i_theta] = np.cos(np.pi/2.*((x - (xlo + delta_ln_theta/2.))/delta_ln_theta))**2.0
            else:
                if x >= xlo + delta_ln_theta/2.0 and x < xup - delta_ln_theta/2.0: 
                    T_of_theta[i_theta] = 1.0
                else:
                    T_of_theta[i_theta] = np.cos(np.pi/2.*((x - (xup - delta_ln_theta/2.))/delta_ln_theta))**2.0
        return T_of_theta    

    def get_gpm(ell_bins,ell_ul_bins, thetabins, T_of_theta, Norm, type):
        """
        Precomputes the bandpower kernels and stores them in private
        variables
        """
        g_plus = np.zeros((len(ell_bins), len(thetabins)))
        g_minus = np.zeros((len(ell_bins), len(thetabins)))
        h_ell = np.zeros((len(ell_bins), len(thetabins)))
        for i_ell in range(len(ell_bins)):
            for i_theta in range(len(thetabins)):    
                theta_times_ell_lo = thetabins[i_theta]*ell_ul_bins[i_ell]/60/180*np.pi
                theta_times_ell_up = thetabins[i_theta]*ell_ul_bins[i_ell+1]/60/180*np.pi
                curly_G_minus_up = (theta_times_ell_up - 8./theta_times_ell_up)*jv(1, theta_times_ell_up) - 8*jv(2, theta_times_ell_up)
                curly_G_minus_lo = (theta_times_ell_lo - 8./theta_times_ell_lo)*jv(1, theta_times_ell_lo) - 8*jv(2, theta_times_ell_lo)
                g_plus[i_ell,i_theta] = 1./(thetabins[i_theta]/60/180*np.pi)**2*(theta_times_ell_up*jv(1, theta_times_ell_up) - theta_times_ell_lo*jv(1, theta_times_ell_lo))
                g_minus[i_ell,i_theta] = 1./(thetabins[i_theta]/60/180*np.pi)**2*(curly_G_minus_up - curly_G_minus_lo)
                h_ell[i_ell,i_theta] = - 1./(thetabins[i_theta]/60/180*np.pi)**2*(theta_times_ell_up*jv(1, theta_times_ell_up) - theta_times_ell_lo*jv(1, theta_times_ell_lo) + 2.*jv(0, theta_times_ell_up) - 2.*jv(0, theta_times_ell_lo))  
            if i_ell+1 < 10:
                filename_mm = "./../bandpowers/real_weight_bandpowers_mmE_0" + str(i_ell+1) + ".table"
                filename_gm = "./../bandpowers/real_weight_bandpowers_gm_0" + str(i_ell+1) + ".table"
                filename_gg = "./../bandpowers/real_weight_bandpowers_gg_0" + str(i_ell+1) + ".table"
            else:
                filename_mm = "./../bandpowers/real_weight_bandpowers_mmE_" + str(i_ell+1) + ".table"
                filename_gm = "./../bandpowers/real_weight_bandpowers_gm_" + str(i_ell+1) + ".table"
                filename_gg = "./../bandpowers/real_weight_bandpowers_gg_" + str(i_ell+1) + ".table"
            if type == 'mm':
                np.savetxt(filename_mm,np.array([thetabins,g_plus[i_ell, :]*T_of_theta*np.pi/Norm[i_ell]]).T, header=hdr_str_mm_plus_real)
            if type == 'gg':
                np.savetxt(filename_gg,np.array([thetabins,g_plus[i_ell, :]*T_of_theta*np.pi/Norm[i_ell]]).T, header=hdr_str_gg_real)
            if type == 'gm':
                np.savetxt(filename_gm,np.array([thetabins,h_ell[i_ell, :]*T_of_theta*2*np.pi/Norm[i_ell]]).T, header=hdr_str_gm_real)
            if i_ell+1 < 10:
                filename_mm = "./../bandpowers/real_weight_bandpowers_mmB_0" + str(i_ell+1) + ".table"
            else:
                filename_mm = "./../bandpowers/real_weight_bandpowers_mmB_" + str(i_ell+1) + ".table"
            if type == 'mm':
                np.savetxt(filename_mm,np.array([thetabins,g_minus[i_ell, :]*T_of_theta*np.pi/Norm[i_ell]]).T, header=hdr_str_mm_minus_real)
            
        
        return g_plus, g_minus, h_ell

    def __call_levin_many_args_WE(ells, ell_up, ell_lo, theta_range, T_of_theta, type):
        result_WEE = np.zeros(len(ells))
        result_WEB = np.zeros(len(ells))
        result_WnE = np.zeros(len(ells))


        lev = levin.Levin(2, 16, 32, 1e-6, 50, num_cores)
        lev.init_integral(theta_range, T_of_theta[:,None]*np.ones(num_cores)[None,:], True, False) 
        if type == "mm" or type == "gg":
            result_WEE = ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 0, 1, theta_range[0], theta_range[-1]))
            result_WEE -=ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 0, 1, theta_range[0], theta_range[-1]))
            result_WEE -=ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result_WEE +=ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None]*np.ones(num_cores)[None,:], True, False)
            result_WEE -=8.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 2, theta_range[0], theta_range[-1]))
            result_WEE +=8.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 2, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range**2)[:,None]*np.ones(num_cores)[None,:], True, False)
            result_WEE -=8.0/ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 1, theta_range[0], theta_range[-1]))
            result_WEE +=8.0/ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 1, theta_range[0], theta_range[-1]))

        if type == "mm":
            lev.init_integral(theta_range, T_of_theta[:,None]*np.ones(num_cores)[None,:], True, False)
            result_WEB = ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 0, 1, theta_range[0], theta_range[-1]))
            result_WEB -=ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 0, 1, theta_range[0], theta_range[-1]))
            result_WEB +=ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result_WEB -=ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None]*np.ones(num_cores)[None,:], True, True)
            result_WEB +=8.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 2, theta_range[0], theta_range[-1]))
            result_WEB -=8.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 2, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range**2)[:,None]*np.ones(num_cores)[None,:], True, True)
            result_WEB +=8.0/ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 4, 1, theta_range[0], theta_range[-1]))
            result_WEB -=8.0/ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 4, 1, theta_range[0], theta_range[-1]))
        
        if type == "gm":
            lev.init_integral(theta_range, T_of_theta[:,None]*np.ones(num_cores)[None,:], True, False) 
            result_WnE = -ell_up*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 2, 1, theta_range[0], theta_range[-1]))
            result_WnE +=ell_lo*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 2, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None]*np.ones(num_cores)[None,:], True, False)
            result_WnE -=2.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_up, 2, 0, theta_range[0], theta_range[-1]))
            result_WnE +=2.0*np.nan_to_num(lev.double_bessel_many_args(
                ells, ell_lo, 2, 0, theta_range[0], theta_range[-1]))

        return result_WEE, result_WEB, result_WnE


    def calc_fourier_filters_bp(ell_bins,ell_ul_bins, ell_fourier_integral, thetabins, T_of_theta, Norm, type):
        """
        Function precomputing bandpower weight functions WEE, WEB and WnE for later integration
        """
        Wl_EE = np.zeros((len(ell_bins), len(ell_fourier_integral)))
        Wl_EB = np.zeros((len(ell_bins), len(ell_fourier_integral)))
        Wl_nE = np.zeros((len(ell_bins), len(ell_fourier_integral)))
        t0, tcomb = time.time(), 1
        tcombs = len(ell_bins)
        
        for i_ell in range(len(ell_bins)):
            Wl_EE[i_ell, :], Wl_EB[i_ell, :], Wl_nE[i_ell, :] = __call_levin_many_args_WE(ell_fourier_integral,
                                                                                ell_ul_bins[i_ell+1],
                                                                                ell_ul_bins[i_ell],
                                                                                thetabins/60/180*np.pi,
                                                                                T_of_theta, type)
            eta = (time.time()-t0) / \
                60 * (tcombs/tcomb-1)
            print('\rCalculating Fourier weights bandpowers '
                    + str(round(tcomb/tcombs*100, 1)) + '% in '
                    + str(round(((time.time()-t0)/60), 1)) +
                    'min  ETA '
                    'in ' + str(round(eta, 1)) + 'min', end="")
            tcomb += 1
            if i_ell+1 < 10:
                filename_mm = "./../bandpowers/fourier_weight_bandpowers_mmE_0" + str(i_ell+1) + ".table"
                filename_gm = "./../bandpowers/fourier_weight_bandpowers_gm_0" + str(i_ell+1) + ".table"
                filename_gg = "./../bandpowers/fourier_weight_bandpowers_gg_0" + str(i_ell+1) + ".table"
            else:
                filename_mm = "./../bandpowers/fourier_weight_bandpowers_mmE_" + str(i_ell+1) + ".table"
                filename_gm = "./../bandpowers/fourier_weight_bandpowers_gm_" + str(i_ell+1) + ".table"
                filename_gg = "./../bandpowers/fourier_weight_bandpowers_gg_" + str(i_ell+1) + "table"
            if type == "mm":
                np.savetxt(filename_mm,np.array([ell_fourier_integral,Wl_EE[i_ell, :]*np.pi/Norm[i_ell]]).T, header=hdr_str_mm_plus_fourier)
            if type == "gm":
                np.savetxt(filename_gm,np.array([ell_fourier_integral,Wl_nE[i_ell, :]*np.pi*2.0/Norm[i_ell]]).T, header=hdr_str_gm_fourier)
            if type == "gg":
                np.savetxt(filename_gg,np.array([ell_fourier_integral,Wl_EE[i_ell, :]*np.pi*2.0/Norm[i_ell]]).T, header=hdr_str_gg_fourier)
            
            if i_ell+1 < 10:
                filename_mm = "./../bandpowers/fourier_weight_bandpowers_mmB_0" + str(i_ell+1) + ".table"
            else:
                filename_mm = "./../bandpowers/fourier_weight_bandpowers_mmB_" + str(i_ell+1) + ".table"
            if type == "mm":
                np.savetxt(filename_mm,np.array([ell_fourier_integral,Wl_EB[i_ell, :]*np.pi/Norm[i_ell]]).T,  header=hdr_str_mm_minus_fourier)
        print("")
        

    ###Clustering###
    print("Clustering")
    L_bins, L_ul_bins = get_L_bins(L_type_gg,L_min_gg,L_max_gg,L_bins_gg)
    N_ell = np.log(L_ul_bins[1:]/L_ul_bins[:-1])
    T_of_theta = get_Hann_window(theta_bins_gg,delta_ln_theta_gg,theta_lo_gg,theta_up_gg)
    g_plus, g_minus, h_ell = get_gpm(L_bins,L_ul_bins, theta_bins_gg,T_of_theta, N_ell, "gg")
    calc_fourier_filters_bp(L_bins,L_ul_bins,fourier_ell,theta_bins_gg,T_of_theta, N_ell, "gg")

    ###GGL###
    print("GGL")
    L_bins, L_ul_bins = get_L_bins(L_type_gm,L_min_gm,L_max_gm,L_bins_gm)
    N_ell = np.log(L_ul_bins[1:]/L_ul_bins[:-1])
    T_of_theta = get_Hann_window(theta_bins_gm,delta_ln_theta_gm,theta_lo_gm,theta_up_gm)
    g_plus, g_minus, h_ell = get_gpm(L_bins,L_ul_bins, theta_bins_gm,T_of_theta, N_ell, "gm")
    calc_fourier_filters_bp(L_bins,L_ul_bins,fourier_ell,theta_bins_gm,T_of_theta, N_ell, "gm")

    ###Lensing###
    print("Lensing")
    L_bins, L_ul_bins = get_L_bins(L_type_mm,L_min_mm,L_max_mm,L_bins_mm)
    N_ell = np.log(L_ul_bins[1:]/L_ul_bins[:-1])
    T_of_theta = get_Hann_window(theta_bins_mm,delta_ln_theta_mm,theta_lo_mm,theta_up_mm)
    g_plus, g_minus, h_ell = get_gpm(L_bins,L_ul_bins, theta_bins_mm,T_of_theta, N_ell, "mm")
    calc_fourier_filters_bp(L_bins,L_ul_bins,fourier_ell,theta_bins_mm,T_of_theta, N_ell, "mm")


if __name__ == '__main__':
    main()