import numpy as np
from scipy.special import jv
import argparse


def main():
    parser = argparse.ArgumentParser(description='Calculates fourier weights for rcf')
    parser.add_argument('-n', '--nthread', type=int, default=10, help='number of threads used (default is 10)')
    parser.add_argument('-nf', '--nfourier', type=int, default=int(1e5), help='number of Fourier modes at which the weights are calculated (default is 1e5)')
    parser.add_argument('-nt', '--ntheta', type=int, default=int(1e5), help='number of theta at which the realspace weights are calculated (default is 1e5)')

    parser.add_argument('-tlo_mm', '--theta_lo_mm', type=float, default=0.5, help='lower limit for angular range in arcmin for lensing(default is 0.5)')
    parser.add_argument('-tup_mm', '--theta_up_mm', type=float, default=300, help='upper limit for angular range in arcmin for lensing(default is 300)')
    parser.add_argument('-tb_mm', '--t_bins_mm', type=int, default=9, help='Number of theta bins for lensing (default is 8)')
    parser.add_argument('-tt_mm', '--t_type_mm', type=str, default='log', help='Type if binning for theta bins for lensing (default is log)')

    parser.add_argument('-tlo_gm', '--theta_lo_gm', type=float, default=0.5, help='lower limit for angular range in arcmin for ggl(default is 0.5)')
    parser.add_argument('-tup_gm', '--theta_up_gm', type=float, default=300, help='upper limit for angular range in arcmin for ggl(default is 300)')
    parser.add_argument('-tb_gm', '--t_bins_gm', type=int, default=9, help='Number of theta bins for ggl (default is 8)')
    parser.add_argument('-tt_gm', '--t_type_gm', type=str, default='log', help='Type if binning for theta bins for ggl (default is log)')

    parser.add_argument('-tlo_gg', '--theta_lo_gg', type=float, default=0.5, help='lower limit for angular range in arcmin for clustering(default is 0.5)')
    parser.add_argument('-tup_gg', '--theta_up_gg', type=float, default=300, help='upper limit for angular range in arcmin for clustering(default is 300)')
    parser.add_argument('-tb_gg', '--t_bins_gg', type=int, default=9, help='Number of theta bins for clustering (default is 8)')
    parser.add_argument('-tt_gg', '--t_type_gg', type=str, default='log', help='Type if binning for theta bins for clustering (default is log)')

    args = parser.parse_args()
    N_fourier = args.nfourier # at how many ells should the Fourier space filter be evaluated
    fourier_ell = np.geomspace(1,1e5,N_fourier) # define the ell, this range is more than enough
    N_real = args.ntheta # at how many theta should the real space filter be evaluated

    theta_type_gg = args.t_type_gg 
    theta_min_gg = args.theta_lo_gg 
    theta_max_gg = args.theta_up_gg 
    ntheta_bins_gg = args.t_bins_gg 

    theta_type_gm = args.t_type_gm 
    theta_min_gm = args.theta_lo_gm 
    theta_max_gm = args.theta_up_gm 
    ntheta_bins_gm = args.t_bins_gm 

    theta_type_mm = args.t_type_mm 
    theta_min_mm = args.theta_lo_mm 
    theta_max_mm = args.theta_up_mm 
    ntheta_bins_mm = args.t_bins_mm


    hdr_str_mm_plus_fourier = 'real space correlation function weights for Xi_+ in Fourier space\n'
    hdr_str_mm_plus_fourier += 'lowest theta boundary = ' + str(theta_min_mm) + '\n'
    hdr_str_mm_plus_fourier += 'highest theta boundary = ' + str(theta_max_mm) + '\n'
    hdr_str_mm_plus_fourier += 'number of theta bins = ' + str(ntheta_bins_mm) + '\n'
    hdr_str_mm_plus_fourier += 'type of binning = ' + str(theta_type_mm) + '\n'
    hdr_str_mm_plus_fourier += 'ell      W(ell)'

    hdr_str_mm_minus_fourier = 'real space correlation function weights for Xi_- in Fourier space\n'
    hdr_str_mm_minus_fourier += 'lowest theta boundary = ' + str(theta_min_mm) + '\n'
    hdr_str_mm_minus_fourier += 'highest theta boundary = ' + str(theta_max_mm) + '\n'
    hdr_str_mm_minus_fourier += 'number of theta bins = ' + str(ntheta_bins_mm) + '\n'
    hdr_str_mm_minus_fourier += 'type of binning = ' + str(theta_type_mm) + '\n'
    hdr_str_mm_minus_fourier += 'ell      W(ell)'


    hdr_str_mm_plus_real = 'real space correlation function weights for Xi_+ in Real space\n'
    hdr_str_mm_plus_real += 'lowest theta boundary = ' + str(theta_min_mm) + '\n'
    hdr_str_mm_plus_real += 'highest theta boundary = ' + str(theta_max_mm) + '\n'
    hdr_str_mm_plus_real += 'number of theta bins = ' + str(ntheta_bins_mm) + '\n'
    hdr_str_mm_plus_real += 'type of binning = ' + str(theta_type_mm) + '\n'
    hdr_str_mm_plus_real += 'theta[arcmin]      R(theta)'


    hdr_str_mm_minus_real = 'real space correlation function weights for Xi_- in Real space\n'
    hdr_str_mm_minus_real += 'lowest theta boundary = ' + str(theta_min_mm) + '\n'
    hdr_str_mm_minus_real += 'highest theta boundary = ' + str(theta_max_mm) + '\n'
    hdr_str_mm_minus_real += 'number of theta bins = ' + str(ntheta_bins_mm) + '\n'
    hdr_str_mm_minus_real += 'type of binning = ' + str(theta_type_mm) + '\n'
    hdr_str_mm_minus_real += 'theta[arcmin]      R(theta)'


    
    hdr_str_gm_real = 'real space correlation function weights for gamma_t in Real space\n'
    hdr_str_gm_real += 'lowest theta boundary = ' + str(theta_min_gm) + '\n'
    hdr_str_gm_real += 'highest theta boundary = ' + str(theta_max_gm) + '\n'
    hdr_str_gm_real += 'number of theta bins = ' + str(ntheta_bins_gm) + '\n'
    hdr_str_gm_real += 'type of binning = ' + str(theta_type_gm) + '\n'
    hdr_str_gm_real += 'theta[arcmin]      R(theta)'

    hdr_str_gm_fourier = 'real space correlation function weights for gamma_t in Fourier space\n'
    hdr_str_gm_fourier += 'lowest theta boundary = ' + str(theta_min_gm) + '\n'
    hdr_str_gm_fourier += 'highest theta boundary = ' + str(theta_max_gm) + '\n'
    hdr_str_gm_fourier += 'number of theta bins = ' + str(ntheta_bins_gm) + '\n'
    hdr_str_gm_fourier += 'type of binning = ' + str(theta_type_gm) + '\n'
    hdr_str_gm_fourier += 'ell      W(ell)'


    hdr_str_gg_real = 'real space correlation function weights for w in Real space\n'
    hdr_str_gg_real += 'lowest theta boundary = ' + str(theta_min_gg) + '\n'
    hdr_str_gg_real += 'highest theta boundary = ' + str(theta_max_gm) + '\n'
    hdr_str_gg_real += 'number of theta bins = ' + str(ntheta_bins_gm) + '\n'
    hdr_str_gg_real += 'type of binning = ' + str(theta_type_gm) + '\n'
    hdr_str_gg_real += 'theta[arcmin]      R(theta)'

    hdr_str_gg_fourier = 'real space correlation function weights for w in Fourier space\n'
    hdr_str_gg_fourier += 'lowest theta boundary = ' + str(theta_min_gg) + '\n'
    hdr_str_gg_fourier += 'highest theta boundary = ' + str(theta_max_gg) + '\n'
    hdr_str_gg_fourier += 'number of theta bins = ' + str(ntheta_bins_gg) + '\n'
    hdr_str_gg_fourier += 'type of binning = ' + str(theta_type_gg) + '\n'
    hdr_str_gg_fourier += 'ell      W(ell)'


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
    theta_bins, theta_ul_bins = get_theta_bins(theta_type=theta_type_gg,theta_min = theta_min_gg, theta_max = theta_max_gg, ntheta_bins=ntheta_bins_gg)
    real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)
    # Get fourier and real weights for w covariance
    for i_theta in range(len(theta_bins)):
        theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
        theta_l = theta_ul_bins[i_theta]/60/180*np.pi
        K_gg = 2/(theta_u**2 - theta_l**2)/fourier_ell*(theta_u*jv(1,theta_u*fourier_ell) - theta_l*jv(1,theta_l*fourier_ell))
        if i_theta+1 < 10:
            filename_gg = "./../rcf/fourier_weight_realspace_cf_gg_0" + str(i_theta+1) + ".table"
        else:
            filename_gg = "./../rcf/fourier_weight_realspace_cf_gg_" + str(i_theta+1) + ".table"
        np.savetxt(filename_gg,np.array([fourier_ell,K_gg]).T, header = hdr_str_gg_fourier)
        R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
        if i_theta+1 < 10:
            filename = "./../rcf/real_weight_realspace_cf_gg_0" + str(i_theta+1) + ".table"
        else:
            filename = "./../rcf/real_weight_realspace_cf_gg_" + str(i_theta+1) + ".table"
        np.savetxt(filename,np.array([real_theta,R]).T, header = hdr_str_gg_real)

    # Define theta-range for gamma_t (i.e. gm)
    theta_bins, theta_ul_bins = get_theta_bins(theta_type=theta_type_gm,theta_min = theta_min_gm, theta_max = theta_max_gm, ntheta_bins=ntheta_bins_gm)
    real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)
    # Get fourier and real weights for w covariance
    for i_theta in range(len(theta_bins)):
        theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
        theta_l = theta_ul_bins[i_theta]/60/180*np.pi
        xu = fourier_ell*theta_u
        xl = fourier_ell*theta_l
        
        K_gm = 2/(xu**2 - xl**2)*(-xu*jv(1,xu) + xl*jv(1,xl) -2*jv(0,xu) + 2*jv(0,xl))
        if i_theta+1 < 10:
            filename_gm = "./../rcf/fourier_weight_realspace_cf_gm_0" + str(i_theta+1) + ".table"
        else:
            filename_gm = "./../rcf/fourier_weight_realspace_cf_gm_" + str(i_theta+1) + ".table"
        np.savetxt(filename_gm,np.array([fourier_ell,K_gm]).T, header = hdr_str_gm_fourier)
        R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
        if i_theta+1 < 10:
            filename = "./../rcf/real_weight_realspace_cf_gm_0" + str(i_theta+1) + ".table"
        else:
            filename = "./../rcf/real_weight_realspace_cf_gm_" + str(i_theta+1) + ".table"
        np.savetxt(filename,np.array([real_theta,R]).T, header = hdr_str_gm_real)

    # Define theta-range for xip_m (i.e. mm)
    theta_bins, theta_ul_bins = get_theta_bins(theta_type=theta_type_mm,theta_min = theta_min_mm, theta_max = theta_max_mm, ntheta_bins=ntheta_bins_mm)
    real_theta = np.geomspace(theta_ul_bins[0], theta_ul_bins[-1], N_real)

    # Get fourier and real  weights for w covariance
    for i_theta in range(len(theta_bins)):
        theta_u = theta_ul_bins[i_theta+1]/60/180*np.pi
        theta_l = theta_ul_bins[i_theta]/60/180*np.pi
        xu = fourier_ell*theta_u
        xl = fourier_ell*theta_l
        K_mm = 2/(xu**2 - xl**2)*(xu*jv(1,xu) - xl*jv(1,xl))
        if i_theta+1 < 10:
            filename_mm = "./../rcf/fourier_weight_realspace_cf_mm_p_0" + str(i_theta+1) + ".table"
        else:
            filename_mm = "./../rcf/fourier_weight_realspace_cf_mm_p_" + str(i_theta+1) + ".table"
        np.savetxt(filename_mm,np.array([fourier_ell,K_mm]).T, header = hdr_str_mm_plus_fourier)
        K_mm = 2/(xu**2 - xl**2)*((xu - 8/xu)*jv(1,xu) - 8*jv(2,xu) - (xl-8/xl)*jv(1,xl) + 8*jv(2,xl))
        if i_theta+1 < 10:
            filename_mm = "./../rcf/fourier_weight_realspace_cf_mm_m_0" + str(i_theta+1) + ".table"
        else:
            filename_mm = "./../rcf/fourier_weight_realspace_cf_mm_m_" + str(i_theta+1) + ".table"
        np.savetxt(filename_mm,np.array([fourier_ell,K_mm]).T, header = hdr_str_mm_minus_fourier)
        R = top_hat(real_theta, theta_bins[i_theta] - theta_ul_bins[i_theta], theta_ul_bins[i_theta+1] - theta_bins[i_theta],  theta_bins[i_theta])/(real_theta)* (60*180/np.pi)**2/np.sqrt(2)
        if i_theta+1 < 10:
            filename = "./../rcf/real_weight_realspace_cf_mm_p_0" + str(i_theta+1) + ".table"
        else:
            filename = "./../rcf/real_weight_realspace_cf_mm_p_" + str(i_theta+1) + ".table"
        np.savetxt(filename,np.array([real_theta,R]).T, header = hdr_str_mm_plus_real)
        if i_theta+1 < 10:
            filename = "./../rcf/real_weight_realspace_cf_mm_m_0" + str(i_theta+1) + ".table"
        else:
            filename = "./../rcf/real_weight_realspace_cf_mm_m_" + str(i_theta+1) + ".table"
        np.savetxt(filename,np.array([real_theta,R]).T, header = hdr_str_mm_minus_real)



if __name__ == '__main__':
    main()