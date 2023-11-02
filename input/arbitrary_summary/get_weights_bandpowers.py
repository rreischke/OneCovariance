import numpy as np
from scipy.special import jv

N_fourier = int(1e4) # at how many ells should the Bessel functions be evaluated
fourier_ell = np.geomspace(1,1e5,N_fourier)
thetabins = 


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



__call_levin_many_args_WE_non_par(ells, ell_up, ell_lo, theta_range, T_of_theta):
        """
        Auxillary function for the calculation of the weight functions for the bandpowers.
        Carries out the integrals over the Bessel functions in parallel for many arguments.
        
        Parameter
        ---------
        ells : array
            Fourier multipole (\ell) where the Weights should be evaluated at.
        ell_up : float
            Upper limit of the bandpower interval
        ell_lo : float
            Lower limit of the bandpower interval
        theta_range : array
            Theta range over which the Integration is carried out.
        T_of_theta : array
            Window function to select theta range over which the band power is estimated.
            We use a Hann window by default. Must have the same length as theta_range.
        num_cores : array
            Number of cores used for the computation. 
        
        Returns
        -------
        result_WEE, result_WEB, result_WnE : arrays
            The 3 weight for bandpower in a single ell_band but at all ells.
            Have the same length as ells

        """
        result_WEE = np.zeros(len(ells))
        result_WEB = np.zeros(len(ells))
        result_WnE = np.zeros(len(ells))
        for i_ell in range(len(ells)):
            lev = levin.Levin(2, 16, 32, 1e-6, 50)
            lev.init_integral(theta_range, T_of_theta[:,None], True, False) 
            result = ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 0, 1, theta_range[0], theta_range[-1]))
            result -=ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 0, 1, theta_range[0], theta_range[-1]))
            result -=ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result +=ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None], True, False)
            result -=8.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 2, theta_range[0], theta_range[-1]))
            result +=8.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 2, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range**2)[:,None], True, False)
            result -=8.0/ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 1, theta_range[0], theta_range[-1]))
            result +=8.0/ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result_WEE[i_ell] = result

            lev.init_integral(theta_range, T_of_theta[:,None], True, False)
            result = ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 0, 1, theta_range[0], theta_range[-1]))
            result -=ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 0, 1, theta_range[0], theta_range[-1]))
            result +=ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result -=ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None], True, True)
            result +=8.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 2, theta_range[0], theta_range[-1]))
            result -=8.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 2, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range**2)[:,None], True, True)
            result +=8.0/ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 4, 1, theta_range[0], theta_range[-1]))
            result -=8.0/ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 4, 1, theta_range[0], theta_range[-1]))
            result_WEB[i_ell] = result

            lev.init_integral(theta_range, T_of_theta[:,None], True, False) 
            result = -ell_up*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 2, 1, theta_range[0], theta_range[-1]))
            result +=ell_lo*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 2, 1, theta_range[0], theta_range[-1]))
            lev.init_integral(theta_range, (T_of_theta/theta_range)[:,None], True, False)
            result -=2.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_up, 2, 0, theta_range[0], theta_range[-1]))
            result +=2.0*np.nan_to_num(lev.double_bessel(
                ells[i_ell], ell_lo, 2, 0, theta_range[0], theta_range[-1]))
            result_WnE[i_ell] = result
        return result_WEE, result_WEB, result_WnE
