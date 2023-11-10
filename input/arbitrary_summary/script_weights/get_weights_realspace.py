import numpy as np
from scipy.special import jv

N_fourier = int(1e4) # at how many ells should the Fourier space filter be evaluated
fourier_ell = np.geomspace(1,1e5,N_fourier) # define the ell, this range is more than enough
N_real = int(1e4) # at how many theta should the real space filter be evaluated
theta_type = 'log'


def top_hat(x,width_low, width_high, location):
    return (np.heaviside(x- (location-width_low),1)- np.heaviside(x-(location + width_high),1))/(width_high + width_low)
def get_theta_bins(theta_type, theta_min, theta_max, ntheta_bins, theta_list_boundary = None):
    '''
    This function returns the theta bins and the corresponding bin boundaries

    Parameters:
    -----------
    theta_type : string
        Do you want lin-/log-spaced theta's or a list? Can be 'lin', 'log' or 'list'
    theta_min : float
        Minimum angle (lowest bin boundary NOT the central value) in arcmin
    theta_max : float
        Maximum angle (higest bin boundary NOT the central value) in arcmin
    ntheta_bins : integer
        How many theta bins should there be?
    theta_list_boundary : array
        Array of all bin boundaries (arbitrarily spaced)
    '''
    if theta_type == 'lin':
        theta_ul_bins = np.linspace(theta_min, theta_max, ntheta_bins + 1)
        theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])
    if theta_type == 'log':
        theta_ul_bins = np.geomspace(theta_min, theta_max, ntheta_bins + 1)
        theta_bins = np.exp(.5 * (np.log(theta_ul_bins[1:])
                                + np.log(theta_ul_bins[:-1])))
    if theta_type == 'list' and theta_list_boundary is not None:
        theta_ul_bins = theta_list_boundary
        theta_bins = .5 * (theta_ul_bins[1:] + theta_ul_bins[:-1])

    return theta_bins, theta_ul_bins


# Define theta-range for w (i.e. gg)
theta_bins, theta_ul_bins = get_theta_bins(theta_type='log',theta_min = 0.5, theta_max = 300, ntheta_bins=9)
real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)
# Get fourier and real weights for w covariance
for i_theta in range(len(theta_bins)):
    theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
    theta_l = theta_ul_bins[i_theta]/60/180*np.pi
    K_gg = 2/(theta_u**2 - theta_l**2)/fourier_ell*(theta_u*jv(1,theta_u*fourier_ell) - theta_l*jv(1,theta_l*fourier_ell))
    if i_theta+1 < 10:
        filename_gg = "./../fourier_weight_realspace_cf_gg_0" + str(i_theta+1) + ".table"
    else:
        filename_gg = "./../fourier_weight_realspace_cf_gg_" + str(i_theta+1) + ".table"
    np.savetxt(filename_gg,np.array([fourier_ell,K_gg]).T)
    R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
    if i_theta+1 < 10:
        filename = "./../real_weight_realspace_cf_gg_0" + str(i_theta+1) + ".table"
    else:
        filename = "./../real_weight_realspace_cf_gg_" + str(i_theta+1) + ".table"
    np.savetxt(filename,np.array([real_theta,R]).T)

# Define theta-range for gamma_t (i.e. gm)
theta_bins, theta_ul_bins = get_theta_bins(theta_type='log',theta_min = 0.5, theta_max = 300, ntheta_bins=9)
real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)
# Get fourier and real weights for w covariance
for i_theta in range(len(theta_bins)):
    theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
    theta_l = theta_ul_bins[i_theta]/60/180*np.pi
    xu = fourier_ell*theta_u
    xl = fourier_ell*theta_l
     
    K_gm = 2/(xu**2 - xl**2)*(-xu*jv(1,xu) + xl*jv(1,xl) -2*jv(0,xu) + 2*jv(0,xl))
    if i_theta+1 < 10:
        filename_gm = "./../fourier_weight_realspace_cf_gm_0" + str(i_theta+1) + ".table"
    else:
        filename_gm = "./../fourier_weight_realspace_cf_gm_" + str(i_theta+1) + ".table"
    np.savetxt(filename_gm,np.array([fourier_ell,K_gm]).T)
    R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
    if i_theta+1 < 10:
        filename = "./../real_weight_realspace_cf_gm_0" + str(i_theta+1) + ".table"
    else:
        filename = "./../real_weight_realspace_cf_gm_" + str(i_theta+1) + ".table"
    np.savetxt(filename,np.array([real_theta,R]).T)

# Define theta-range for xip_m (i.e. mm)
theta_bins, theta_ul_bins = get_theta_bins(theta_type='log',theta_min = 0.5, theta_max = 300, ntheta_bins=9)
real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)
# Get fourier and real  weights for w covariance
for i_theta in range(len(theta_bins)):
    theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
    theta_l = theta_ul_bins[i_theta]/60/180*np.pi
    xu = fourier_ell*theta_u
    xl = fourier_ell*theta_l
    K_mm = 2/(xu**2 - xl**2)*(xu*jv(1,xu) - xl*jv(1,xl))
    if i_theta+1 < 10:
        filename_mm = "./../fourier_weight_realspace_cf_mm_p_0" + str(i_theta+1) + ".table"
    else:
        filename_mm = "./../ourier_weight_realspace_cf_mm_p_" + str(i_theta+1) + ".table"
    np.savetxt(filename_mm,np.array([fourier_ell,K_mm]).T)
    K_mm = 2/(xu**2 - xl**2)*((xu - 8/xu)*jv(1,xu) - 8*jv(2,xu) - (xl-8/xl)*jv(1,xl) + 8*jv(2,xl))
    if i_theta+1 < 10:
        filename_mm = "./../fourier_weight_realspace_cf_mm_m_0" + str(i_theta+1) + ".table"
    else:
        filename_mm = "./../fourier_weight_realspace_cf_mm_m_" + str(i_theta+1) + ".table"
    np.savetxt(filename_mm,np.array([fourier_ell,K_mm]).T)
    R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
    if i_theta+1 < 10:
        filename = "./../real_weight_realspace_cf_mm_p_0" + str(i_theta+1) + ".table"
    else:
        filename = "./../real_weight_realspace_cf_mm_p_" + str(i_theta+1) + ".table"
    np.savetxt(filename,np.array([real_theta,R]).T)
    if i_theta+1 < 10:
        filename = "./../real_weight_realspace_cf_mm_m_0" + str(i_theta+1) + ".table"
    else:
        filename = "./../real_weight_realspace_cf_mm_m_" + str(i_theta+1) + ".table"
    np.savetxt(filename,np.array([real_theta,R]).T)



    