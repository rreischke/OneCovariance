import time
import numpy as np
from scipy.signal import argrelextrema
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import simpson

from mpmath import mp
import mpmath
from scipy.special import p_roots
from scipy.special import eval_legendre

import levin

try:
    from onecov.cov_ell_space import CovELLSpace
except:
    from cov_ell_space import CovELLSpace


class CovCOSEBI(CovELLSpace):
    """
    This class calculates the covariance for COSEBI estimator of the
    shear-shear correlation. Inherits the functionality of the
    CovELLSpace class.

    Parameters
    ----------
    cov_dict : dictionary
        Specifies which terms of the covariance (Gaussian, non-Gaussian,
        super-sample covariance) should be calculated. To be passed from
        the read_input method of the Input class.
    obs_dict : dictionary
        with the following keys (To be passed from the read_input method
        of the Input class.)
        'observables' : dictionary
            Specifies which observables (cosmic shear, galaxy-galaxy
            lensing and/or clustering) should be calculated. Also,
            indicates whether cross-terms are evaluated.
        'ELLspace' : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        'THETAspace' : dictionary
            Specifies the exact details of the projection to real space,
            e.g., theta_min/max and the number of theta bins to be
            calculated.
        'COSEBIs' : dictionary
            Specifies the exact details of the projection to COSEBIs,
            e.g. the number of modes to be calculated.
        'bandpowers' : dictionary
            Specifies the exact details of the projection to bandpowers,
            e.g. the ell modes and their spacing.
    cosmo_dict : dictionary
        Specifies all cosmological parameters. To be passed from the
        read_input method of the Input class.
    bias_dict : dictionary
        Specifies all the information about the bias model. To be passed
        from the read_input method of the Input class.
    hod_dict : dictionary
        Specifies all the information about the halo occupation
        distribution used. This defines the shot noise level of the
        covariance and includes the mass bin definition of the different
        galaxy populations. To be passed from the read_input method of
        the Input class.
    survey_params_dict : dictionary
        Specifies all the information unique to a specific survey.
        Relevant values are the effective number density of galaxies for
        all tomographic bins as well as the ellipticity dispersion for
        galaxy shapes. To be passed from the read_input method of the
        Input class.
    prec : dictionary
        with the following keys (To be passed from the read_input method
        of the Input class.)
        'hm' : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        'powspec' : dictionary
            Contains precision information about the power spectra, this
            includes k-range and spacing.
        'trispec' : dictionary
            Contains precision information about the trispectra, this
            includes k-range and spacing and the desired precision
            limits.
    read_in_tables : dictionary
        with the following keys (To be passed from the read_input method
        of the FileInput class.)
        'zclust' : dictionary
            Look-up table for the clustering redshifts and their number
            of tomographic bins. Relevant for clustering and galaxy-
            galaxy lensing estimators.
        'zlens' : dictionary
            Look-up table for the lensing redshifts and their number of
            tomographic bins. Relevant for cosmic shear and galaxy-
            galaxy lensing estimators.
        'survey_area_clust' : dictionary
            Contains information about the survey footprint that is read
            from the fits file of the survey (processed by healpy).
            Possible keys: 'area', 'ell', 'a_lm'
        'Pxy' : dictionary
            default : None
            Look-up table for the power spectra (matter-matter, tracer-
            tracer, matter-tracer, optional).
        'Cxy' : dictionary
            default : None
            Look-up table for the C_ell projected power spectra (matter-
            matter, tracer- tracer, matter-tracer, optional).
        'effbias' : dictionary
            default : None
            Look-up table for the effective bias as a function of
            redshift(optional).
        'mor' : dictionary
            default : None
            Look-up table for the mass-observable relation (optional).
        'occprob' : dictionary
            default : None
            Look-up table for the occupation probability as a function
            of halo mass per galaxy sample (optional).
        'occnum' : dictionary
            default : None
            Look-up table for the occupation number as a function of
            halo mass per galaxy sample (optional).
        'tri' : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        'cosebis' : dictionary
            Look-up tables for calculating COSEBIs. Currently the code
            cannot generate the roots and normalizations for the kernel
            functions itself, so either the kernel's roots and
            normalizations or the kernel (W_n) itself must be given.
            Possible keys: 'wn_log_ell', 'wn_log', 'wn_lin_ell',
                           'wn_lin', 'norms', 'roots'

    Attributes
    ----------
    see CovELLSpace class

    Example :
    ---------
    from cov_input import Input, FileInput
    from cov_cosebis import CovCOSEBI
    inp = Input()
    covterms, observables, output, cosmo, bias, hod, survey_params, \
        prec = inp.read_input()
    fileinp = FileInput()
    read_in_tables = fileinp.read_input(inp.config_name)
    covcosebi = CovCOSEBI(covterms, observables, output, cosmo, bias,
        hod, survey_params, prec, read_in_tables)

    """

    def __init__(self,
                 cov_dict,
                 obs_dict,
                 output_dict,
                 cosmo_dict,
                 bias_dict,
                 iA,
                 hod_dict,
                 survey_params_dict,
                 prec,
                 read_in_tables):
        CovELLSpace.__init__(self,
                             cov_dict,
                             obs_dict,
                             output_dict,
                             cosmo_dict,
                             bias_dict,
                             iA,
                             hod_dict,
                             survey_params_dict,
                             prec,
                             read_in_tables)
        
        self.__get_WXY(obs_dict)
        self.ell_limits = []
        for mode in range(self.total_modes):
            limits_at_mode = np.array(self.wn_ells[argrelextrema(self.wn_kernels[mode], np.less)[0][:]])[::30]
            #limits_at_mode_append = np.zeros(len(limits_at_mode) + 2)
            limits_at_mode_append = np.zeros(len(limits_at_mode[(limits_at_mode >  self.wn_ells[1]) & (limits_at_mode < self.wn_ells[-2])]) + 2)
            limits_at_mode_append[1:-1] = limits_at_mode[(limits_at_mode >  self.wn_ells[1]) & (limits_at_mode < self.wn_ells[-2])]
            limits_at_mode_append[0] = self.wn_ells[0]
            limits_at_mode_append[-1] = self.wn_ells[-1]
            self.ell_limits.append(limits_at_mode_append)
        self.levin_int = levin.Levin(0, 16, 32, obs_dict['COSEBIs']['En_acc'] / np.sqrt(len(self.ell_limits[0][:])), 20, self.num_cores)
        self.levin_int.init_w_ell(self.wn_ells, np.array(self.wn_kernels).T)
        if self.gg or self.gm:
            save_n_eff_clust = survey_params_dict['n_eff_clust']
        if self.mm or self.gm:
            save_n_eff_lens = survey_params_dict['n_eff_lens']
        if self.sample_dim > 1:
            if self.gg or self.gm:
                survey_params_dict['n_eff_clust'] = self.Ngal.T/self.arcmin2torad2
            if self.mm or self.gm:
                survey_params_dict['n_eff_lens'] = survey_params_dict['n_eff_lens'][:, None]
        else:
            if self.gg or self.gm:
                survey_params_dict['n_eff_clust'] = survey_params_dict['n_eff_clust'][:, None]
            if self.mm or self.gm:
               survey_params_dict['n_eff_lens'] = survey_params_dict['n_eff_lens'][:, None]  
        self.dnpair_gg, self.dnpair_gm, self.dnpair_mm, self.theta_gg, self.theta_gm, self.theta_mm  = self.get_dnpair([self.gg, self.gm, self.mm],
                                                                                                                        self.theta_integral,
                                                                                                                        survey_params_dict,
                                                                                                                        read_in_tables['npair'])        
        self.array_En_modes = None
        self.array_En_g_modes = None                                                                                                                
        if self.mm:
            self.array_En_modes = np.arange(1,self.En_modes + 1, 1)
        if self.gg or self.gm:
            self.array_En_g_modes = np.arange(1,self.En_g_modes + 1, 1)
        
        if self.gg or self.gm:    
            survey_params_dict['n_eff_clust'] = save_n_eff_clust
        if self.mm or self.gm:
            survey_params_dict['n_eff_lens'] = save_n_eff_lens
        if self.cov_dict['gauss']:
            self.E_mode_gg, self.E_mode_gm, self.E_mode_mm = self.calc_E_mode()


    def __get_wn_kernels(self,
                         cosebi_tabs):

        for wn_style in ['log', 'lin']:
            if cosebi_tabs['wn_'+wn_style] is not None:
                return cosebi_tabs['wn_'+wn_style+'_ell'], \
                    cosebi_tabs['wn_'+wn_style]

        raise Exception("ConfigError: To calculate the COSEBI " +
                        "covariance the W_n kernels must be provided as an " +
                        "external table. Must be included in [tabulated inputs " +
                        "files] as 'wn_log_file' or 'wn_lin_file' to go on.")

        # insert calc of Wn at some point
    
    def __get_wngg_kernels(self,
                         cosebi_tabs):
        if cosebi_tabs['wn_gg'] is not None:
            return cosebi_tabs['wn_gg_ell'], \
                cosebi_tabs['wn_gg']
    
    def __J(self, k,j,zmax):
        J = mp.gamma(j+1) - mp.gammainc(j+1,-k*zmax)
        k_power = mp.power(-k,j+1.)
        J = mp.fdiv(J,k_power)
        return J

    def __tplus(self, tmin,tmax,n,norm,root,ntheta):
        theta=np.logspace(np.log10(tmin),np.log10(tmax),ntheta)
        tplus=np.zeros((ntheta,2))
        tplus[:,0]=theta
        z=np.log(theta/tmin)
        result=1.
        for r in range(n+1):
            result*=(z-root[r])
        result*=norm
        tplus[:,1]=result
        return tplus

    def __tminus(self, tmin,tmax,n,norm,root,tp,theta,ntheta=10000):
        tplus_func=interp1d(np.log(tp[:,0]/tmin),tp[:,1])
        rtminus = np.zeros_like(tp)
        rtminus[:,0] = tp[:,0]
        z=np.log(theta/tmin)
        rtminus[:,1]= tplus_func(z)
        lev = levin.Levin(0, 16, 32, 1e-8, 200, self.num_cores)
        wide_theta = np.linspace(theta[0]*0.999, theta[-1]*1.001,int(1e4))
        lev.init_w_ell(np.log(wide_theta/tmin), np.ones_like(wide_theta)[:,None])
        z=np.log(theta/tmin)
        for i_z, val_z in enumerate(z[1:]):
            y = np.linspace(z[0],val_z, 1000)
            integrand = tplus_func(y)*(np.exp(2*(y-val_z)) - 3*np.exp(4*(y-val_z)))
            limits_at_mode = np.array(y[argrelextrema(integrand, np.less)[0][:]])
            limits_at_mode_append = np.zeros(len(limits_at_mode) + 2)
            if len(limits_at_mode) != 0:
                limits_at_mode_append[1:-1] = limits_at_mode
            limits_at_mode_append[0] = y[0]
            limits_at_mode_append[-1] = y[-1]
            
            lev.init_integral(y,integrand[:,None],False,False)
            rtminus[i_z, 1] +=  4*lev.cquad_integrate_single_well(limits_at_mode_append,0)[0]
        return rtminus

    def __get_W_ell(self, theta, tp, ell):
        lev_w = levin.Levin(0, 16, 32, 1e-8, 200, self.num_cores)
        lev_w.init_integral(theta,(theta*tp)[:,None]*np.ones(self.num_cores)[None,:],True,True)
        return lev_w.single_bessel_many_args(ell,0,theta[0],theta[-1])

    def __get_Un(self, tmax, tmin, theta, Nmax):
        delta_theta = tmax - tmin
        bart = (tmin + tmax)/2
        result = np.zeros((Nmax,len(theta)))
        result[0, :] = (12*bart/delta_theta**2*(theta-bart) - 1)/np.sqrt(2*(1+12*bart**2/delta_theta**2))/delta_theta**2
        for i in range(1,Nmax):
            result[i, :] = 1/delta_theta**2*np.sqrt((2*(i+1)+1)/2)*eval_legendre(i+1,2*(theta-bart)/delta_theta)
        return result
        
    def __get_Qn(self, Un, theta, Nmax):
        theta_integral_weight = np.ones((len(theta),len(theta)))
        theta_integral_weight = np.triu(theta_integral_weight)[None, :, :]*np.ones(Nmax)[:, None, None]
        theta_integral_weight += (np.diag(np.ones_like(theta)))[None,:,:]
        return 2/theta**2*simpson(theta_integral_weight*theta[None,:,None]*Un[:,:,None], x = theta,axis = 1) - Un

    def __get_Wpsi_ell(self, n, Un, theta, ell):
        lev = levin.Levin(0, 16, 32, 1e-8, 200, self.num_cores)
        lev.init_integral(theta,(theta*Un)[:,None]*np.ones(self.num_cores)[None, :],True,True)
        Wngg = np.zeros_like(ell)
        Wngg = lev.single_bessel_many_args(ell,0,theta[0],theta[-1])
        return Wngg
    '''
    def __get_WXY(self,obs_dict):
        ell_min = 1
        ell_max = 1e5
        N_ell = int(1e5)
        N_theta = int(1e4)
        get_W_ell_COSEBI_well = True
        mp.dps = 160
        arcmintorad = 1./60./180.*np.pi
        tmin_mm = 1e6
        tmax_mm = 0
        tmin_gg = 1e6
        tmax_gg = 0
        Nmax_mm = 0
        Nmax_gg = 0
        if self.mm:
            Nmax_mm = obs_dict['COSEBIs']['En_modes_lensing']
            tmin_mm = obs_dict['COSEBIs']['theta_min_lensing']
            tmax_mm = obs_dict['COSEBIs']['theta_max_lensing']
            tmin_mm *= arcmintorad
            tmax_mm *= arcmintorad
            theta_mm = np.geomspace(tmin_mm,tmax_mm, N_theta)
        if self.gg or self.gm:
            Nmax_gg = obs_dict['COSEBIs']['En_modes_clustering']
            tmin_gg = obs_dict['COSEBIs']['theta_min_clustering']
            tmax_gg = obs_dict['COSEBIs']['theta_max_clustering']
            tmin_gg *= arcmintorad
            tmax_gg *= arcmintorad
            theta_gg = np.geomspace(tmin_gg,tmax_gg, N_theta)
            theta_gm = theta_gg
        self.theta_integral = np.geomspace(min(tmin_gg/arcmintorad,tmin_mm/arcmintorad),max(tmax_gg/arcmintorad,tmax_mm/arcmintorad), 1000) 
        self.total_modes = 0
        self.En_g_modes = 0 
        self.En_modes = 0
        if self.gg or self.gm:
            self.total_modes += Nmax_gg
            self.En_g_modes = Nmax_gg
        if self.mm:
            self.total_modes += Nmax_mm
            self.En_modes = Nmax_mm
        zmax = mp.log(tmax_mm/tmin_mm)
        self.wn_ells = np.geomspace(ell_min, ell_max, N_ell)
        if Nmax_mm > 0:
            coeff_j = mp.matrix(Nmax_mm+1,Nmax_mm+2)
            for i in range(Nmax_mm+1):
                coeff_j[i,i+1] = mp.mpf(1.)
            nn = 1
            aa = [self.__J(2,0,zmax),self.__J(2,1,zmax)],[self.__J(4,0,zmax),self.__J(4,1,zmax)]
            bb = [-self.__J(2,nn+1,zmax),-self.__J(4,nn+1,zmax)]
            coeff_j_ini = mp.lu_solve(aa,bb)
            coeff_j[1,0] = coeff_j_ini[0]
            coeff_j[1,1] = coeff_j_ini[1]
            for nn in np.arange(2,Nmax_mm+1):
                aa = mp.matrix(int(nn+1))
                bb = mp.matrix(int(nn+1),1)
                for m in np.arange(1,nn): 
                    for j in range(0,nn+1):           
                        for i in range(0,m+2): 
                            aa[m-1,j] += self.__J(1,i+j,zmax)*coeff_j[m,i]        
                    for i in range(0,m+2): 
                        bb[int(m-1)] -= self.__J(1,i+nn+1,zmax)*coeff_j[m,i]
                for j in range(nn+1):
                    aa[nn-1,j] = self.__J(2,j,zmax) 
                    aa[nn,j]   = self.__J(4,j,zmax) 
                    bb[int(nn-1)] = -self.__J(2,nn+1,zmax)
                    bb[int(nn)]   = -self.__J(4,nn+1,zmax)
                temp_coeff = mp.lu_solve(aa,bb)
                coeff_j[nn,:len(temp_coeff)] = temp_coeff[:,0].T
            coeff_j = coeff_j[1:,:]
            Nn = []
            for nn in np.arange(1,Nmax_mm+1):
                temp_sum = mp.mpf(0)
                for i in range(nn+2):
                    for j in range(nn+2):
                        temp_sum += coeff_j[nn-1,i]*coeff_j[nn-1,j]*self.__J(1,i+j,zmax)
                temp_Nn = (mp.expm1(zmax))/(temp_sum)
                temp_Nn = mp.sqrt(mp.fabs(temp_Nn))
                Nn.append(temp_Nn)
            rn = []
            for nn in range(1,Nmax_mm+1):
                rn.append(mpmath.polyroots(coeff_j[nn-1,:nn+2][::-1],maxsteps=500,extraprec=100))
        self.wn_kernels = np.zeros((self.total_modes, len(self.wn_ells)))
        self.Tn_p = []
        self.Tn_m = []
        self.Qn = []
        self.Un = []
        if self.mm:
            t0, tcomb = time.time(), 1
            tcombs = Nmax_mm
            for nn in range(1,Nmax_mm+1):
                n = nn-1
                tpn = self.__tplus(tmin_mm,tmax_mm,nn,Nn[n],rn[n], N_theta)
                theta = tpn[:,0]
                tmn = self.__tminus(tmin_mm,tmax_mm,1,Nn[n],rn[n], tpn, theta)
                self.wn_kernels[nn - 1, :] = self.__get_W_ell(theta,tpn[:,1], self.wn_ells)
                self.Tn_p.append(UnivariateSpline(tpn[:,0]/arcmintorad,tpn[:,1], k=2, s=0, ext=0))
                self.Tn_m.append(UnivariateSpline(tmn[:,0]/arcmintorad,tmn[:,1], k=2, s=0, ext=0))
                eta = (time.time()-t0) / \
                    60 * (tcombs/tcomb-1)
                print('\rCalculating weights for lensing COSEBI covariance '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
        if self.gg or self.gm:
            print("")
            t0, tcomb = time.time(), 1
            tcombs = Nmax_gg
            Ungg = self.__get_Un(tmax_gg, tmin_gg, theta_gg, Nmax_gg)
            Qngg = self.__get_Qn(Ungg, theta_gm, Nmax_gg)
            for nn in range(1,Nmax_gg+1):
                self.wn_kernels[nn - 1  + self.En_modes, :] = self.__get_Wpsi_ell(nn - 1, Ungg[nn-1,:], theta_gg, self.wn_ells)
                self.Un.append(UnivariateSpline(theta_gg/arcmintorad,  Ungg[nn-1,:]/self.arcmin2torad2, k=2, s=0, ext=1))
                self.Qn.append(UnivariateSpline(theta_gg/arcmintorad,  Qngg[nn-1,:]/self.arcmin2torad2, k=2, s=0, ext=1))
                eta = (time.time()-t0) / \
                    60 * (tcombs/tcomb-1)
                print('\rCalculating weights for clustering COSEBI covariance '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
    '''
    def __get_WXY(self,obs_dict):
        ell_min = 1
        ell_max = 1e5
        N_ell = int(1e5)
        N_theta = int(1e4)
        get_W_ell_COSEBI_well = True
        mp.dps = 160
        arcmintorad = 1./60./180.*np.pi
        tmin_mm = 1e6
        tmax_mm = 0
        tmin_gg = 1e6
        tmax_gg = 0
        Nmax_mm = 0
        Nmax_gg = 0
        if self.mm:
            Nmax_mm = obs_dict['COSEBIs']['En_modes_lensing']
            tmin_mm = obs_dict['COSEBIs']['theta_min_lensing']
            tmax_mm = obs_dict['COSEBIs']['theta_max_lensing']
            tmin_mm *= arcmintorad
            tmax_mm *= arcmintorad
            theta_mm = np.geomspace(tmin_mm,tmax_mm, N_theta)
        if self.gg or self.gm:
            Nmax_gg = obs_dict['COSEBIs']['En_modes_clustering']
            tmin_gg = obs_dict['COSEBIs']['theta_min_clustering']
            tmax_gg = obs_dict['COSEBIs']['theta_max_clustering']
            tmin_gg *= arcmintorad
            tmax_gg *= arcmintorad
            theta_gg = np.geomspace(tmin_gg,tmax_gg, N_theta)
            theta_gm = theta_gg
        self.theta_integral = np.geomspace(min(tmin_gg/arcmintorad,tmin_mm/arcmintorad),max(tmax_gg/arcmintorad,tmax_mm/arcmintorad), 1000) 
        self.total_modes = 0
        self.En_g_modes = 0 
        self.En_modes = 0
        if self.gg or self.gm:
            self.total_modes += Nmax_gg
            self.En_g_modes = Nmax_gg
        if self.mm:
            self.total_modes += Nmax_mm
            self.En_modes = Nmax_mm
        zmax = mp.log(tmax_mm/tmin_mm)
        self.wn_ells = np.geomspace(ell_min, ell_max, N_ell)
        if Nmax_mm > 0:
            coeff_j = mp.matrix(Nmax_mm+1,Nmax_mm+2)
            for i in range(Nmax_mm+1):
                coeff_j[i,i+1] = mp.mpf(1.)
            nn = 1
            aa = [self.__J(2,0,zmax),self.__J(2,1,zmax)],[self.__J(4,0,zmax),self.__J(4,1,zmax)]
            bb = [-self.__J(2,nn+1,zmax),-self.__J(4,nn+1,zmax)]
            coeff_j_ini = mp.lu_solve(aa,bb)
            coeff_j[1,0] = coeff_j_ini[0]
            coeff_j[1,1] = coeff_j_ini[1]
            for nn in np.arange(2,Nmax_mm+1):
                aa = mp.matrix(int(nn+1))
                bb = mp.matrix(int(nn+1),1)
                for m in np.arange(1,nn): 
                    for j in range(0,nn+1):           
                        for i in range(0,m+2):
                            if obs_dict['COSEBIs']['dimensionless_cosebi']:
                                aa[m-1,j] += self.__J(2,i+j,zmax)*coeff_j[m,i]    
                            else:
                                aa[m-1,j] += self.__J(1,i+j,zmax)*coeff_j[m,i]        
                    for i in range(0,m+2):
                        if obs_dict['COSEBIs']['dimensionless_cosebi']:
                            bb[int(m-1)] -= self.__J(2,i+nn+1,zmax)*coeff_j[m,i]
                        else:    
                            bb[int(m-1)] -= self.__J(1,i+nn+1,zmax)*coeff_j[m,i]
                for j in range(nn+1):
                    aa[nn-1,j] = self.__J(2,j,zmax) 
                    aa[nn,j]   = self.__J(4,j,zmax) 
                    bb[int(nn-1)] = -self.__J(2,nn+1,zmax)
                    bb[int(nn)]   = -self.__J(4,nn+1,zmax)
                temp_coeff = mp.lu_solve(aa,bb)
                coeff_j[nn,:len(temp_coeff)] = temp_coeff[:,0].T
            coeff_j = coeff_j[1:,:]
            Nn = []
            for nn in np.arange(1,Nmax_mm+1):
                temp_sum = mp.mpf(0)
                for i in range(nn+2):
                    for j in range(nn+2):
                        if obs_dict['COSEBIs']['dimensionless_cosebi']:
                            temp_sum += coeff_j[nn-1,i]*coeff_j[nn-1,j]*self.__J(2,i+j,zmax)
                        else:
                            temp_sum += coeff_j[nn-1,i]*coeff_j[nn-1,j]*self.__J(1,i+j,zmax)
                if obs_dict['COSEBIs']['dimensionless_cosebi']:
                    tbar = (tmax_mm + tmin_mm)/2
                    BB = (tmax_mm - tmin_mm)/(tmax_mm + tmin_mm)
                    temp_Nn = 1/(temp_sum)
                    temp_Nn = mp.sqrt(mp.fabs(temp_Nn))*np.sqrt(tbar**2*BB/tmin_mm**2)
                else:
                    temp_Nn = (mp.expm1(zmax))/(temp_sum)
                    temp_Nn = mp.sqrt(mp.fabs(temp_Nn))
                Nn.append(temp_Nn)
            rn = []
            for nn in range(1,Nmax_mm+1):
                rn.append(mpmath.polyroots(coeff_j[nn-1,:nn+2][::-1],maxsteps=500,extraprec=100))
        self.wn_kernels = np.zeros((self.total_modes, len(self.wn_ells)))
        self.Tn_p = []
        self.Tn_m = []
        self.Qn = []
        self.Un = []
        if self.mm:
            t0, tcomb = time.time(), 1
            tcombs = Nmax_mm
            for nn in range(1,Nmax_mm+1):
                n = nn-1
                tpn = self.__tplus(tmin_mm,tmax_mm,nn,Nn[n],rn[n], N_theta)
                theta = tpn[:,0]
                tmn = self.__tminus(tmin_mm,tmax_mm,1,Nn[n],rn[n], tpn, theta)
                self.wn_kernels[nn - 1, :] = self.__get_W_ell(theta,tpn[:,1], self.wn_ells)
                self.Tn_p.append(UnivariateSpline(tpn[:,0]/arcmintorad,tpn[:,1], k=2, s=0, ext=0))
                self.Tn_m.append(UnivariateSpline(tmn[:,0]/arcmintorad,tmn[:,1], k=2, s=0, ext=0))
                eta = (time.time()-t0) / \
                    60 * (tcombs/tcomb-1)
                print('\rCalculating weights for lensing COSEBI covariance '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
        if self.gg or self.gm:
            print("")
            t0, tcomb = time.time(), 1
            tcombs = Nmax_gg
            Ungg = self.__get_Un(tmax_gg, tmin_gg, theta_gg, Nmax_gg)
            Qngg = self.__get_Qn(Ungg, theta_gm, Nmax_gg)
            for nn in range(1,Nmax_gg+1):
                self.wn_kernels[nn - 1  + self.En_modes, :] = self.__get_Wpsi_ell(nn - 1, Ungg[nn-1,:], theta_gg, self.wn_ells)
                self.Un.append(UnivariateSpline(theta_gg/arcmintorad,  Ungg[nn-1,:]/self.arcmin2torad2, k=2, s=0, ext=1))
                self.Qn.append(UnivariateSpline(theta_gg/arcmintorad,  Qngg[nn-1,:]/self.arcmin2torad2, k=2, s=0, ext=1))
                eta = (time.time()-t0) / \
                    60 * (tcombs/tcomb-1)
                print('\rCalculating weights for clustering COSEBI covariance '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1

    def __get_Tn_pm(self,
                    cosebi_tabs,
                    covCOSEBIsettings,
                    obs_dict):
        self.Tn_p = []
        self.Tn_m = []
        self.Qn = []
        self.Un = []
        
        if cosebi_tabs['Tn_p'] is not None:
            for i_mode in range(len(cosebi_tabs['Tn_p'][:,0])):
                self.Tn_p.append(UnivariateSpline(cosebi_tabs['Tn_pm_theta'],  cosebi_tabs['Tn_p'][i_mode,:], k=2, s=0, ext=1))
                self.Tn_m.append(UnivariateSpline(cosebi_tabs['Tn_pm_theta'],  cosebi_tabs['Tn_m'][i_mode,:], k=2, s=0, ext=1))
            if cosebi_tabs['Tn_pm_theta'][0] > covCOSEBIsettings['theta_min'] or cosebi_tabs['Tn_pm_theta'][-1] < covCOSEBIsettings['theta_max']:
                print("Warning: To calculate the shot noise contribution for COSEBI "+
                    "I will have to extrapolate Tn_pm. "+
                    "Please check the angular support for Tn_pm file." + 
                    "Should be from " + str(covCOSEBIsettings['theta_min']) + " to " + 
                    str(covCOSEBIsettings['theta_max']) + ", but is only given in " +
                    str(cosebi_tabs['Tn_pm_theta'][0]) + " to " + str(cosebi_tabs['Tn_pm_theta'][-1]) + ".")
        else:
            if obs_dict['observables']['ggl'] or obs_dict['observables']['cosmic_shear']:
                raise Exception("ConfigError: To calculate the COSEBI " +
                            "covariance the Tn_pm kernels must be provided as an " +
                            "external table. Must be included in [tabulated inputs " +
                            "files] as 'Tn_plus_file' and 'Tn_minus_file' to go on.")
            
        if cosebi_tabs['Qn'] is not None:
            for i_mode in range(len(cosebi_tabs['Qn'][:,0])):
                self.Qn.append(UnivariateSpline(cosebi_tabs['Qn_theta'],  cosebi_tabs['Qn'][i_mode,:], k=2, s=0, ext=0))
            if cosebi_tabs['Qn_theta'][0] > covCOSEBIsettings['theta_min'] or cosebi_tabs['Qn_theta'][-1] < covCOSEBIsettings['theta_max']:
                print("Warning: To calculate the shot noise contribution for COSEBI "+
                    "I will have to extrapolate Q_n. "+
                    "Please check the angular support for Q_n file." + 
                    "Should be from " + str(covCOSEBIsettings['theta_min']) + " to " + 
                    str(covCOSEBIsettings['theta_max']) + ", but is only given in " +
                    str(cosebi_tabs['Qn_theta'][0]) + " to " + str(cosebi_tabs['Qn_theta'][-1]) + ".")
        else:
            if obs_dict['observables']['ggl'] or obs_dict['observables']['clustering']:   
                raise Exception("ConfigError: To calculate the COSEBI " +
                            "covariance the Qn kernels must be provided as an " +
                            "external table. Must be included in [tabulated inputs " +
                            "files] as 'Qn_file' to go on.")
        
        if cosebi_tabs['Un'] is not None:
            for i_mode in range(len(cosebi_tabs['Un'][:,0])):
                self.Un.append(UnivariateSpline(cosebi_tabs['Un_theta'],  cosebi_tabs['Un'][i_mode,:], k=2, s=0, ext=0))
            if cosebi_tabs['Un_theta'][0] > covCOSEBIsettings['theta_min'] or cosebi_tabs['Un_theta'][-1] < covCOSEBIsettings['theta_max']:
                print("Warning: To calculate the shot noise contribution for COSEBI "+
                    "I will have to extrapolate U_n. "+
                    "Please check the angular support for U_n file." + 
                    "Should be from " + str(covCOSEBIsettings['theta_min']) + " to " + 
                    str(covCOSEBIsettings['theta_max']) + ", but is only given in " +
                    str(cosebi_tabs['Un_theta'][0]) + " to " + str(cosebi_tabs['Un_theta'][-1]) + ".")
        else:
            if obs_dict['observables']['ggl'] or obs_dict['observables']['clustering']: 
                raise Exception("ConfigError: To calculate the COSEBI " +
                            "covariance the Un kernels must be provided as an " +
                            "external table. Must be included in [tabulated inputs " +
                            "files] as 'Un_file' to go on.")

    def calc_E_mode(self):
        """
        Calculates the E-mode signal of the COSEBis in all tomographic bin
        combination and all tracers specified using the Gauss Legendre integrator.

        Returns
        -------
        E_mode_mm : array
            An numpy array containing all E modes for cosmic shear with
            the following shape:
            (E_mode order, sample_bin, tomo_bin_i, tomo_bin_j)

        E_mode_gm : array
            An numpy array containing all E modes for GGL with
            the following shape:
            (E_mode order, sample_bin, tomo_bin_i, tomo_bin_j)

        E_mode_gg : array
            An numpy array containing all E modes for GGL with
            the following shape:
            (E_mode order, sample_bin, sample_bin, tomo_bin_i, tomo_bin_j)

        """

        E_mode_mm = np.zeros((self.En_modes, self.sample_dim,
                              self.n_tomo_lens, self.n_tomo_lens))
        E_mode_gm = np.zeros((self.En_g_modes, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_lens))
        E_mode_gg = np.zeros((self.En_g_modes, self.sample_dim, self.sample_dim,
                              self.n_tomo_clust, self.n_tomo_clust))
        
        if (self.mm or self.gm):
            t0, tcomb = time.time(), 1
            tcombs = self.En_modes
            original_shape = self.Cell_mm[0, :, :, :].shape
            flat_length = self.n_tomo_lens**2*self.sample_dim
            Cell_mm_flat = np.reshape(self.Cell_mm, (len(
                self.ellrange), flat_length))
            for mode in range(self.En_modes):
                self.levin_int.init_integral(self.ellrange, Cell_mm_flat*self.ellrange[:,None], True, True)
                E_mode_mm[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[mode][:], mode)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rCOSEBI E-mode calculation for lensing at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
            print(" ")

        if (self.gm or (self.gg and self.mm and self.cross_terms)):
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes
            original_shape = self.Cell_gm[0, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens*self.n_tomo_clust
            Cell_gm_flat = np.reshape(self.Cell_gm, (len(
                self.ellrange), flat_length))
            for mode in range(self.En_g_modes):
                self.levin_int.init_integral(self.ellrange, Cell_gm_flat*self.ellrange[:,None], True, True)
                E_mode_gm[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[mode + self.En_modes][:], mode + self.En_modes)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rCOSEBI E-mode calculation for GGL at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
                
            print(" ")

        if (self.gg or self.gm):
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes
            original_shape = self.Cell_gg[0, :, :, :, :].shape
            flat_length = self.sample_dim**2*self.n_tomo_clust**2
            Cell_gg_flat = np.reshape(self.Cell_gg, (len(
                self.ellrange), flat_length))
            for mode in range(self.En_g_modes):
                self.levin_int.init_integral(self.ellrange, Cell_gg_flat*self.ellrange[:,None], True, True)
                E_mode_gg[mode,:,:,:] = 1 / 2 / np.pi * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[mode + self.En_modes][:], mode + self.En_modes)),original_shape)
                eta = (time.time()-t0)/60 * (tcombs/tcomb-1)
                print('\rCOSEBI E-mode calculation for clustering at '
                        + str(round(tcomb/tcombs*100, 1)) + '% in '
                        + str(round(((time.time()-t0)/60), 1)) +
                        'min  ETA '
                        'in ' + str(round(eta, 1)) + 'min', end="")
                tcomb += 1
            print(" ")

        return E_mode_gg, E_mode_gm, E_mode_mm

    def calc_covCOSEBI(self,
                       obs_dict,
                       output_dict,
                       bias_dict,
                       hod_dict,
                       survey_params_dict,
                       prec,
                       read_in_tables):
        """
        Calculates the full covariance for the defined observables 
        for the COSEBIs as specified in the config file.

        Parameters
        ----------
        obs_dict : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
            'observables' : dictionary
                Specifies which observables (cosmic shear, galaxy-galaxy
                lensing and/or clustering) should be calculated. Also,
                indicates whether cross-terms are evaluated.
            'ELLspace' : dictionary
                Specifies the exact details of the projection to ell space.
                The projection from wavefactor k to angular scale ell is
                done first, followed by the projection to real space in this
                class
            'THETAspace' : dictionary
                Specifies the exact details of the projection to real space,
                e.g., theta_min/max and the number of theta bins to be
                calculated.
            'COSEBIs' : dictionary
                Specifies the exact details of the projection to COSEBIs,
                e.g. the number of modes to be calculated.
            'bandpowers' : dictionary
                Specifies the exact details of the projection to bandpowers,
                e.g. the ell modes and their spacing.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be 
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from 
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        prec : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
            'hm' : dictionary
                Contains precision information about the HaloModel (also,
                see hmf documentation by Steven Murray), this includes mass
                range and spacing for the mass integrations in the halo
                model.
            'powspec' : dictionary
                Contains precision information about the power spectra, this
                includes k-range and spacing.
            'trispec' : dictionary
                Contains precision information about the trispectra, this
                includes k-range and spacing and the desired precision
                limits.
        read_in_tables : dictionary
            with the following keys (To be passed from the read_input method
            of the FileInput class.)
            'zclust' : dictionary
                Look-up table for the clustering redshifts and their number
                of tomographic bins. Relevant for clustering and galaxy-
                galaxy lensing estimators.
            'zlens' : dictionary
                Look-up table for the lensing redshifts and their number of
                tomographic bins. Relevant for cosmic shear and galaxy-
                galaxy lensing estimators.
            'survey_area_clust' : dictionary
                Contains information about the survey footprint that is read
                from the fits file of the survey (processed by healpy).
                Possible keys: 'area', 'ell', 'a_lm'
            'Pxy' : dictionary
                default : None
                Look-up table for the power spectra (matter-matter, tracer-
                tracer, matter-tracer, optional).
            'Cxy' : dictionary
                default : None
                Look-up table for the C_ell projected power spectra (matter-
                matter, tracer- tracer, matter-tracer, optional).
            'effbias' : dictionary
                default : None
                Look-up table for the effective bias as a function of
                redshift(optional).
            'mor' : dictionary
                default : None
                Look-up table for the mass-observable relation (optional).
            'occprob' : dictionary
                default : None
                Look-up table for the occupation probability as a function
                of halo mass per galaxy sample (optional).
            'occnum' : dictionary
                default : None
                Look-up table for the occupation number as a function of
                halo mass per galaxy sample (optional).
            'tri' : dictionary
                Look-up table for the trispectra (for all combinations of
                matter 'm' and tracer 'g', optional).
                Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
                'gggm', 'gggg'
            'cosebis' : dictionary
                Look-up tables for calculating COSEBIs. Currently the code
                cannot generate the roots and normalizations for the kernel
                functions itself, so either the kernel's roots and
                normalizations or the kernel (W_n) itself must be given.
                Possible keys: 'wn_log_ell', 'wn_log', 'wn_lin_ell',
                                'wn_lin', 'norms', 'roots'

        Returns
        -------
        gauss, nongauss, ssc : list of arrays
            each with 10 entries for the observables 
                ['EEgggg', 'EEgggm', 'EEggmm', 'EBggmm', 'EEgmgm', 'EEmmgm',
                 'EBmmgm', 'EEmmmm', 'EEBmmmm', 'BBmmmm']
            each entry with shape (m_modes, n_modes, 
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """
        print("Calculating covariance for COSEBIs E_n")
        if not self.cov_dict['split_gauss']:
            if self.csmf:
                gauss_EEgggg, gauss_EEgggm, gauss_EEggmm, gauss_EBggmm, \
                    gauss_EEgmgm, gauss_EEmmgm, gauss_EBmmgm, \
                    gauss_EEmmmm, gauss_EBmmmm, \
                    gauss_BBmmmm, \
                    gauss_EEgggg_sn, gauss_EEgmgm_sn, gauss_EEmmmm_sn, gauss_BBmmmm_sn, \
                    csmf_COSEBI_auto, csmf_COSEBI_gg, csmf_COSEBI_gm, csmf_COSEBI_mmE, csmf_COSEBI_mmB = \
                    self.covCOSEBI_gaussian(obs_dict,
                                            survey_params_dict)
                gauss = [gauss_EEgggg + gauss_EEgggg_sn, gauss_EEgggm, gauss_EEggmm, gauss_EBggmm,
                        gauss_EEgmgm + gauss_EEgmgm_sn, gauss_EEmmgm, gauss_EBmmgm,
                        gauss_EEmmmm + gauss_EEmmmm_sn, gauss_EBmmmm,
                        gauss_BBmmmm + gauss_BBmmmm_sn,
                        csmf_COSEBI_auto, csmf_COSEBI_gg, csmf_COSEBI_gm, csmf_COSEBI_mmE, csmf_COSEBI_mmB]
            else:
                gauss_EEgggg, gauss_EEgggm, gauss_EEggmm, gauss_EBggmm, \
                    gauss_EEgmgm, gauss_EEmmgm, gauss_EBmmgm, \
                    gauss_EEmmmm, gauss_EBmmmm, \
                    gauss_BBmmmm, \
                    gauss_EEgggg_sn, gauss_EEgmgm_sn, gauss_EEmmmm_sn, gauss_BBmmmm_sn = \
                    self.covCOSEBI_gaussian(obs_dict,
                                            survey_params_dict)
                gauss = [gauss_EEgggg + gauss_EEgggg_sn, gauss_EEgggm, gauss_EEggmm, gauss_EBggmm,
                        gauss_EEgmgm + gauss_EEgmgm_sn, gauss_EEmmgm, gauss_EBmmgm,
                        gauss_EEmmmm + gauss_EEmmmm_sn, gauss_EBmmmm,
                        gauss_BBmmmm + gauss_BBmmmm_sn]
        else:
            gauss = self.covCOSEBI_gaussian(obs_dict,
                                            survey_params_dict)

        nongauss = self.covCOSEBI_non_gaussian(obs_dict['ELLspace'],
                                               survey_params_dict,
                                               output_dict,
                                               bias_dict,
                                               hod_dict,
                                               prec,
                                               read_in_tables['tri'])
        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            ssc = []
            for i_list in range(len(nongauss)):
                if nongauss[i_list] is not None:
                    ssc.append(nongauss[i_list]*0)
                else:
                    ssc.append(None)
        else:     
            ssc = self.covCOSEBI_ssc(obs_dict['ELLspace'],
                                    survey_params_dict,
                                    output_dict,
                                    bias_dict,
                                    hod_dict,
                                    prec,
                                    read_in_tables['tri'])

        return list(gauss), list(nongauss), list(ssc)

    def covCOSEBI_gaussian(self,
                           obs_dict,
                           survey_params_dict):
        """
        Calculates the Gaussian (disconnected) covariance for COSEBIs.


        Parameters
        ----------
        obs_dict : dictionary
            with the following keys (To be passed from the read_input method
            of the Input class.)
            'observables' : dictionary
                Specifies which observables (cosmic shear, galaxy-galaxy
                lensing and/or clustering) should be calculated. Also,
                indicates whether cross-terms are evaluated.
            'ELLspace' : dictionary
                Specifies the exact details of the projection to ell space.
                The projection from wavefactor k to angular scale ell is
                done first, followed by the projection to real space in this
                class
            'THETAspace' : dictionary
                Specifies the exact details of the projection to real space,
                e.g., theta_min/max and the number of theta bins to be
                calculated.
            'bandpowers' : dictionary
                Specifies the exact details for the bandpower covariance.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        
        Returns
        -------
        Depending whether split_gaussian is true or not.
        split_gauss == True:
            gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
            gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
            gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
            gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
            gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
            gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
            gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
            gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
            gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
            gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn : list of arrays

            each with shape (number of modes, number of modes,
                             sample bins, sample bins,
                             n_tomo_clust/lens, n_tomo_clust/lens,
                             n_tomo_clust/lens, n_tomo_clust/lens)
        
        split_gauss == False:
            gaussCOSEBIgggg, gaussCOSEBIgggm, gaussCOSEBIEggmm, \
            gaussCOSEBIBggmm, gaussCOSEBIgmgm, gaussCOSEBIEmmgm, \
            gaussCOSEBIBmmgm, gaussCOSEBIEEmmmm, gaussCOSEBIEBmmmm, gaussCOSEBIBBmmmm,\
            gaussCOSEBIgggg_sn, gaussCOSEBIgmgm_sn, gaussCOSEBIEEmmmm_sn, gaussCOSEBIBBmmmm_sn

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the diagonal terms of the covariance
        matrix, one needs to add gauss_xy + gauss_xy_sn. They
        are kept separated.

        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        print("Calculating gaussian covariance for COSEBIS")

        if self.csmf:
            gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn, \
                csmf_COSEBI_auto, csmf_COSEBI_gg, csmf_COSEBI_gm, csmf_COSEBI_mmE, csmf_COSEBI_mmB = \
                self.__covCOSEBI_split_gaussian(obs_dict,
                                                survey_params_dict)
        else:
            gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn = \
                self.__covCOSEBI_split_gaussian(obs_dict,
                                                survey_params_dict)
        if self.csmf:
            if not self.cov_dict['split_gauss']:
                gaussCOSEBIgggg = gaussCOSEBIgggg_sva + gaussCOSEBIgggg_mix
                gaussCOSEBIgggm = gaussCOSEBIgggm_sva + gaussCOSEBIgggm_mix
                gaussCOSEBIEggmm = gaussCOSEBIEggmm_sva + gaussCOSEBIEggmm_mix
                gaussCOSEBIBggmm = gaussCOSEBIBggmm_sva + gaussCOSEBIBggmm_mix
                gaussCOSEBIgmgm = gaussCOSEBIgmgm_sva + gaussCOSEBIgmgm_mix
                gaussCOSEBIEmmgm = gaussCOSEBIEmmgm_sva + gaussCOSEBIEmmgm_mix
                gaussCOSEBIBmmgm = gaussCOSEBIBmmgm_sva + gaussCOSEBIBmmgm_mix
                gaussCOSEBIEEmmmm = gaussCOSEBIEEmmmm_sva + gaussCOSEBIEEmmmm_mix
                gaussCOSEBIEBmmmm = gaussCOSEBIEBmmmm_sva + gaussCOSEBIEBmmmm_mix
                gaussCOSEBIBBmmmm = gaussCOSEBIBBmmmm_sva + gaussCOSEBIBBmmmm_mix
                return gaussCOSEBIgggg, gaussCOSEBIgggm, gaussCOSEBIEggmm, \
                    gaussCOSEBIBggmm, gaussCOSEBIgmgm, gaussCOSEBIEmmgm, \
                    gaussCOSEBIBmmgm, gaussCOSEBIEEmmmm, gaussCOSEBIEBmmmm, gaussCOSEBIBBmmmm,\
                    gaussCOSEBIgggg_sn, gaussCOSEBIgmgm_sn, gaussCOSEBIEEmmmm_sn, gaussCOSEBIBBmmmm_sn, \
                    csmf_COSEBI_auto, csmf_COSEBI_gg, csmf_COSEBI_gm, csmf_COSEBI_mmE, csmf_COSEBI_mmB 
            else:
                return gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                    gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                    gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                    gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                    gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                    gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                    gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                    gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                    gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                    gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn, \
                    csmf_COSEBI_auto, csmf_COSEBI_gg, csmf_COSEBI_gm, csmf_COSEBI_mmE, csmf_COSEBI_mmB
        else:
            if not self.cov_dict['split_gauss']:
                gaussCOSEBIgggg = gaussCOSEBIgggg_sva + gaussCOSEBIgggg_mix
                gaussCOSEBIgggm = gaussCOSEBIgggm_sva + gaussCOSEBIgggm_mix
                gaussCOSEBIEggmm = gaussCOSEBIEggmm_sva + gaussCOSEBIEggmm_mix
                gaussCOSEBIBggmm = gaussCOSEBIBggmm_sva + gaussCOSEBIBggmm_mix
                gaussCOSEBIgmgm = gaussCOSEBIgmgm_sva + gaussCOSEBIgmgm_mix
                gaussCOSEBIEmmgm = gaussCOSEBIEmmgm_sva + gaussCOSEBIEmmgm_mix
                gaussCOSEBIBmmgm = gaussCOSEBIBmmgm_sva + gaussCOSEBIBmmgm_mix
                gaussCOSEBIEEmmmm = gaussCOSEBIEEmmmm_sva + gaussCOSEBIEEmmmm_mix
                gaussCOSEBIEBmmmm = gaussCOSEBIEBmmmm_sva + gaussCOSEBIEBmmmm_mix
                gaussCOSEBIBBmmmm = gaussCOSEBIBBmmmm_sva + gaussCOSEBIBBmmmm_mix
                return gaussCOSEBIgggg, gaussCOSEBIgggm, gaussCOSEBIEggmm, \
                    gaussCOSEBIBggmm, gaussCOSEBIgmgm, gaussCOSEBIEmmgm, \
                    gaussCOSEBIBmmgm, gaussCOSEBIEEmmmm, gaussCOSEBIEBmmmm, gaussCOSEBIBBmmmm,\
                    gaussCOSEBIgggg_sn, gaussCOSEBIgmgm_sn, gaussCOSEBIEEmmmm_sn, gaussCOSEBIBBmmmm_sn
            else:
                return gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                    gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                    gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                    gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                    gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                    gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                    gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                    gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                    gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                    gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn

    def __covCOSEBI_split_gaussian(self,
                                   obs_dict,
                                   survey_params_dict):
        """
        Calculates the Gaussian (disconnected) covariance for COSEBIs
        between two observables.

        Parameters
        ----------
        ccovCOSEBIsettings : dictionary
            Specifies the exact details of the COSEBI calculation,
            e.g., theta_min/max and the number of modes to be
            calculated.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.

        Returns
        -------
        gaussCOSEBIgggg, gaussCOSBEIgggm, gaussCOSEBIggmm, \
        gaussCOSEBIgmgm, gaussCOSEBImmgm, gaussCOSEBImmmm, \
        gaussCOSEBIgggg_sn, gaussCOSEBIgmgm_sn, gaussCOSEBImmmm_sn : list of
                                                            arrays
            with shape (number of modes, number of modes,
                        sample bins, sample bins,
                        n_tomo_clust/lens, n_tomo_clust/lens,
                        n_tomo_clust/lens, n_tomo_clust/lens)

        Note :
        ------
        The shot-noise terms are denoted with '_sn'. To get the full
        covariance contribution to the pure kappa-kappa ('mmmm'),
        tracer-tracer ('gggg'), and kappa-tracer ('gmgm') terms, one
        needs to add gaussCOSEBIxxyy + gaussCOSEBIxxyy_sn. They are kept
        separate for a later numerical integration.

        """

        if not self.cov_dict['gauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        save_entry = self.cov_dict['split_gauss']
        self.cov_dict['split_gauss'] = True
        if self.csmf:
            gaussELLgggg_sva, gaussELLgggg_mix, _, \
            gaussELLgggm_sva, gaussELLgggm_mix, _, \
            gaussELLggmm_sva, _, _, \
            gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
            gaussELLmmgm_sva, gaussELLmmgm_mix, _, \
            gaussELLmmmm_sva, gaussELLmmmm_mix, _, \
            csmf_auto, csmf_gg, csmf_gm, csmf_mm = self.covELL_gaussian(
                obs_dict['ELLspace'], survey_params_dict, False)
        else:
            gaussELLgggg_sva, gaussELLgggg_mix, _, \
                gaussELLgggm_sva, gaussELLgggm_mix, _, \
                gaussELLggmm_sva, _, _, \
                gaussELLgmgm_sva, gaussELLgmgm_mix, _, \
                gaussELLmmgm_sva, gaussELLmmgm_mix, _, \
                gaussELLmmmm_sva, gaussELLmmmm_mix, _ = self.covELL_gaussian(
                    obs_dict['ELLspace'], survey_params_dict, False)
        self.cov_dict['split_gauss'] = save_entry
        gaussCOSEBIgggg_sva = None
        gaussCOSEBIgggg_mix = None
        gaussCOSEBIgggg_sn = None

        gaussCOSEBIgggm_sva = None
        gaussCOSEBIgggm_mix = None
        gaussCOSEBIgggm_sn = None

        gaussCOSEBIEggmm_sva = None
        gaussCOSEBIEggmm_mix = None
        gaussCOSEBIEggmm_sn = None
        gaussCOSEBIBggmm_sva = None
        gaussCOSEBIBggmm_mix = None
        gaussCOSEBIBggmm_sn = None



        gaussCOSEBIgmgm_sva = None
        gaussCOSEBIgmgm_mix = None
        gaussCOSEBIgmgm_sn = None

        gaussCOSEBIEmmgm_sva = None
        gaussCOSEBIEmmgm_mix = None
        gaussCOSEBIEmmgm_sn = None
        gaussCOSEBIBmmgm_sva = None
        gaussCOSEBIBmmgm_mix = None
        gaussCOSEBIBmmgm_sn = None

        gaussCOSEBIEEmmmm_sva = None
        gaussCOSEBIEEmmmm_mix = None
        gaussCOSEBIEEmmmm_sn = None
        gaussCOSEBIEBmmmm_sva = None
        gaussCOSEBIEBmmmm_mix = None
        gaussCOSEBIEBmmmm_sn = None
        gaussCOSEBIBBmmmm_sva = None
        gaussCOSEBIBBmmmm_mix = None
        gaussCOSEBIBBmmmm_sn = None

        if self.gg or self.gm:
            kron_delta_tomo_clust = np.diag(np.ones(self.n_tomo_clust))
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))
        if self.mm or self.gm:
            kron_delta_tomo_lens = np.diag(survey_params_dict['ellipticity_dispersion']**2)
            kron_delta_mass_bins = np.diag(np.ones(self.sample_dim))
        if self.gg:
            gaussCOSEBIgggg_sva = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            gaussCOSEBIgggg_mix = np.zeros_like(gaussCOSEBIgggg_sva)
            gaussCOSEBIgggg_sn = np.zeros_like(gaussCOSEBIgggg_sva)
            original_shape = gaussCOSEBIgggg_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            gaussELL_sva_flat = np.reshape(gaussELLgggg_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgggg_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode + self.En_modes][:]
                    if len(self.ell_limits[m_mode + self.En_modes][:]) < len(self.ell_limits[n_mode + self.En_modes][:]):
                        local_ell_limit = self.ell_limits[n_mode + self.En_modes][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgggg_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgggg_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                    else:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgggg_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_clust']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                        
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    Tpm_product = self.Un[m_mode](self.theta_gg)*self.Un[n_mode](self.theta_gg)*self.arcmin2torad2**2
                    integrand = (Tpm_product*self.theta_gg**2)[:,None, None, None]/self.dnpair_gg    
                    aux_gg_sn = simpson(integrand, x = self.theta_gg,axis=0)/self.arcmin2torad2**2
                    gaussCOSEBIgggg_sn[n_mode, m_mode, :, :, :, :, :, :] = (kron_delta_tomo_clust[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_clust[None, None, None, :, None, :]
                                                                            + kron_delta_tomo_clust[None, None, :, None, None, :]
                                                                            * kron_delta_tomo_clust[None, None, None, :, :, None]) \
                                                                            * kron_delta_mass_bins[:, :, None, None, None, None] \
                                                                            * (np.ones(self.n_tomo_clust)[None, :]**2*np.ones(self.n_tomo_clust)[:, None]**2*aux_gg_sn)[None, :, :, :, None, None]
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'gggg term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1

            print(" ")
        else:
            gaussCOSEBIgggg_sva = 0
            gaussCOSEBIgggg_mix = 0
            gaussCOSEBIgggg_sn = 0

        if self.gg and self.gm and self.cross_terms:
            gaussCOSEBIgggm_sn = 0
            gaussCOSEBIgggm_sva = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            gaussCOSEBIgggm_mix = np.zeros_like(gaussCOSEBIgggm_sva)
            original_shape = gaussCOSEBIgggm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens*self.n_tomo_clust**3
            gaussELL_sva_flat = np.reshape(gaussELLgggm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgggm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode + self.En_modes][:]
                    if len(self.ell_limits[m_mode + self.En_modes][:]) < len(self.ell_limits[n_mode + self.En_modes][:]):
                        local_ell_limit = self.ell_limits[n_mode + self.En_modes][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        survey_area = max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])   
                        gaussCOSEBIgggm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgggm_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                    else:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        survey_area = max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl'])   
                        gaussCOSEBIgggm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                    
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'gggm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print(" ")
        else:
            gaussCOSEBIgggm_sva = 0
            gaussCOSEBIgggm_mix = 0
            gaussCOSEBIgggm_sn = 0

        if self.gg and self.mm and self.cross_terms:
            gaussCOSEBIEggmm_sva = np.zeros(
                (self.En_g_modes, self.En_modes, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            gaussCOSEBIBggmm_sva = np.zeros_like(gaussCOSEBIEggmm_sva)
            gaussCOSEBIEggmm_mix = 0
            gaussCOSEBIEggmm_sn = 0
            gaussCOSEBIBggmm_mix = 0
            gaussCOSEBIBggmm_sn = 0
            original_shape = gaussCOSEBIEggmm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            gaussELL_sva_flat = np.reshape(gaussELLggmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes*self.En_modes
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_modes):
                    local_ell_limit = self.ell_limits[m_mode + self.En_modes][:]
                    if len(self.ell_limits[m_mode + self.En_modes][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                    survey_area = max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens'])
                    gaussCOSEBIEggmm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) *  np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode)),original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'ggmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            print(" ")
        else:
            gaussCOSEBIEggmm_sva = 0
            gaussCOSEBIEggmm_mix = 0
            gaussCOSEBIEggmm_sn = 0
            gaussCOSEBIBggmm_sva = 0
            gaussCOSEBIBggmm_mix = 0
            gaussCOSEBIBggmm_sn = 0
        if self.gm:
            gaussCOSEBIgmgm_sva = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            gaussCOSEBIgmgm_mix = np.zeros_like(gaussCOSEBIgmgm_sva)
            gaussCOSEBIgmgm_sn = np.zeros_like(gaussCOSEBIgmgm_sva)
            original_shape = gaussCOSEBIgmgm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**2*self.n_tomo_lens**2
            gaussELL_sva_flat = np.reshape(gaussELLgmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLgmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode + self.En_modes][:]
                    if len(self.ell_limits[m_mode + self.En_modes][:]) < len(self.ell_limits[n_mode + self.En_modes][:]):
                        local_ell_limit = self.ell_limits[n_mode + self.En_modes][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgmgm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgmgm_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)
                    else:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIgmgm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_ggl']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode + self.En_modes, n_mode + self.En_modes)),original_shape)

                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    Tpm_product = self.Qn[m_mode](self.theta_gm)*self.Qn[n_mode](self.theta_gm)*self.arcmin2torad2**2
                    integrand = (Tpm_product*self.theta_gm**2)[:,None, None, None]/self.dnpair_gm
                    aux_gm_sn = simpson(integrand, x = self.theta_gm,axis=0)/self.arcmin2torad2**2
                    gaussCOSEBIgmgm_sn[n_mode, m_mode, :, :, :, :, :, :] = (kron_delta_tomo_clust[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_lens[None, None, None, :, None, :]) \
                                                                            * kron_delta_mass_bins[:,:, None, None, None, None] \
                                                                            * (aux_gm_sn)[None, :, :, :, None, None]
                    
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'gmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            adding = self.gaussELLgmgm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.E_mode_gm[:, None, :, None, :, :, None, None]*self.E_mode_gm[None, :, None, :, None, None, :, :])
            gaussCOSEBIgmgm_sva[:, :, :, :, :, :, :, :] = gaussCOSEBIgmgm_sva[:, :, :, :, :, :, :, :] + adding[:, :, :, :, :, :, :, :]
            print("")
        else:
            gaussCOSEBIgmgm_sva = 0
            gaussCOSEBIgmgm_mix = 0
            gaussCOSEBIgmgm_sn = 0

        if self.gm and self.mm and self.cross_terms:
            gaussCOSEBIEmmgm_sn = 0
            gaussCOSEBIBmmgm_sn = 0
            gaussCOSEBIEmmgm_sva = np.zeros(
                (self.En_modes, self.En_g_modes, 1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            gaussCOSEBIEmmgm_mix = np.zeros_like(gaussCOSEBIEmmgm_sva)
            gaussCOSEBIBmmgm_sva = np.zeros_like(gaussCOSEBIEmmgm_sva)
            gaussCOSEBIBmmgm_mix = np.zeros_like(gaussCOSEBIEmmgm_sva)
            original_shape = gaussCOSEBIEmmgm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens**3*self.n_tomo_clust
            gaussELL_sva_flat = np.reshape(gaussELLmmgm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLmmgm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes*self.En_modes
            for m_mode in range(self.En_modes):
                for n_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode + self.En_modes][:]):
                        local_ell_limit = self.ell_limits[n_mode + self.En_modes][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        survey_area = max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens'])
                        gaussCOSEBIEmmgm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.En_modes)),original_shape)
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIEmmgm_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.En_modes)),original_shape)
                    else:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        survey_area = max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens'])
                        gaussCOSEBIEmmgm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_area/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode + self.En_modes)),original_shape)

                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'mmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            adding = self.gaussELLmmgm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.E_mode_mm[:, None, :, None, :, :, None, None]*self.E_mode_gm[None, :, None, :, None, None, :, :])
            gaussCOSEBIEmmgm_sva[:, :, 0, :, :, :, :, :] = gaussCOSEBIEmmgm_sva[:, :, 0, :, :, :, :, :] + adding[:, :, 0, :, :, :, :, :]
            print("")
        else:
            gaussCOSEBIEmmgm_sva = 0
            gaussCOSEBIEmmgm_mix = 0
            gaussCOSEBIEmmgm_sn = 0
            gaussCOSEBIBmmgm_sva = 0
            gaussCOSEBIBmmgm_mix = 0
            gaussCOSEBIBmmgm_sn = 0

        if self.mm:
            gaussCOSEBIEEmmmm_sva = np.zeros(
                (self.En_modes, self.En_modes, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            gaussCOSEBIEEmmmm_mix = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIEEmmmm_sn = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIEBmmmm_sva = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIEBmmmm_mix = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIEBmmmm_sn = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIBBmmmm_sva = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIBBmmmm_mix = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            gaussCOSEBIBBmmmm_sn = np.zeros_like(gaussCOSEBIEEmmmm_sva)
            

            original_shape = gaussCOSEBIEEmmmm_sva[0, 0, :, :, :, :, :, :].shape
            flat_length = self.n_tomo_lens**4
            gaussELL_sva_flat = np.reshape(gaussELLmmmm_sva, (len(self.ellrange), len(
                self.ellrange), flat_length))
            gaussELL_mix_flat = np.reshape(gaussELLmmmm_mix, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_modes**2
            for m_mode in range(self.En_modes):
                for n_mode in range(self.En_modes):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    if len(self.ell_limits[m_mode][:]) < len(self.ell_limits[n_mode][:]):
                        local_ell_limit = self.ell_limits[n_mode][:]
                    if self.cov_dict['split_gauss']:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIEEmmmm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIEEmmmm_mix[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    else:
                        self.levin_int.init_integral(self.ellrange, np.moveaxis(np.diagonal(gaussELL_sva_flat + gaussELL_mix_flat)*self.ellrange,0,-1), True, True)
                        gaussCOSEBIEEmmmm_sva[m_mode, n_mode, :, :, :, :, :, :] = 1./2./np.pi/(survey_params_dict['survey_area_lens']/self.deg2torad2) * np.reshape(np.array(self.levin_int.cquad_integrate_double_well(local_ell_limit, m_mode, n_mode)),original_shape)
                    eta = (time.time()-t0) / \
                        60 * (tcombs/tcomb-1)
                    Tpm_product = self.Tn_p[m_mode](self.theta_mm)*self.Tn_p[n_mode](self.theta_mm) + self.Tn_m[m_mode](self.theta_mm)*self.Tn_m[n_mode](self.theta_mm)
                    integrand = (Tpm_product*self.theta_mm**2)[:,None, None, None]/self.dnpair_mm
                    aux_mm_sn = simpson(integrand, x = self.theta_mm,axis=0)/self.arcmin2torad2**2
                    gaussCOSEBIEEmmmm_sn[m_mode, n_mode, :, :, :, :, :, :] = (kron_delta_tomo_lens[None, None, :, None, :, None]
                                                                            * kron_delta_tomo_lens[None, None, None, :, None, :]
                                                                            + kron_delta_tomo_lens[None, None, :, None, None, :]
                                                                            * kron_delta_tomo_lens[None, None, None, :, :, None]) \
                                                                            * aux_mm_sn[None, :, :, :, None, None]/2.
                    gaussCOSEBIBBmmmm_sn[m_mode, n_mode, :, :, :, :, :, :] = gaussCOSEBIEEmmmm_sn[m_mode, n_mode, :, :, :, :, :, :]
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'mmmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            adding = self.gaussELLmmmm_sva_mult_shear_bias[None, None, :, : ,: , :, : ,:]*(self.E_mode_mm[:, None, :, None, :, :, None, None]*self.E_mode_mm[None, :, None, :, None, None, :, :])
            gaussCOSEBIEEmmmm_sva[:, :, 0, 0, :, :, :, :] = gaussCOSEBIEEmmmm_sva[:, :, 0, 0, :, :, :, :] + adding[:, :, 0, 0, :, :, :, :]
            print("")
        else:
            gaussCOSEBIEEmmmm_sva = 0
            gaussCOSEBIEEmmmm_mix = 0
            gaussCOSEBIEEmmmm_sn = 0
            gaussCOSEBIEBmmmm_sva = 0
            gaussCOSEBIEBmmmm_mix = 0
            gaussCOSEBIEBmmmm_sn = 0
            gaussCOSEBIBBmmmm_sva = 0
            gaussCOSEBIBBmmmm_mix = 0
            gaussCOSEBIBBmmmm_sn = 0

        if self.csmf:
            if self.gg:
                csmf_COSEBIgg = np.zeros((self.En_g_modes, len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_clust))
                original_shape = csmf_gg[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust**2*self.n_tomo_csmf
                csmf_COSEBI_flat = np.reshape(csmf_gg, (len(self.ellrange), flat_length))
                for m_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode + + self.En_modes][:]
                    self.levin_int.init_integral(self.ellrange, csmf_COSEBI_flat, True, True)
                    csmf_COSEBIgg[m_mode, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(local_ell_limit, m_mode + self.En_modes)),original_shape)
            else:
                csmf_COSEBIgg = 0
            if self.gm:
                csmf_COSEBIgm = np.zeros((self.En_g_modes, len(self.log10csmf_mass_bins), self.sample_dim, self.n_tomo_csmf, self.n_tomo_clust, self.n_tomo_lens))
                original_shape = csmf_gm[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins) *self.sample_dim*self.n_tomo_clust*self.n_tomo_lens*self.n_tomo_csmf
                csmf_COSEBI_flat = np.reshape(csmf_gm, (len(self.ellrange), flat_length))
                for m_mode in range(self.En_g_modes):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    self.levin_int.init_integral(self.ellrange, csmf_COSEBI_flat, True, True)
                    csmf_COSEBIgm[m_mode , :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(local_ell_limit, m_mode + self.En_modes)),original_shape)
            else:
                csmf_COSEBIgm = 0
            if self.mm:
                csmf_COSEBImmE = np.zeros((self.En_modes, len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                csmf_COSEBImmB = np.zeros((self.En_modes, len(self.log10csmf_mass_bins), 1, self.n_tomo_csmf, self.n_tomo_lens, self.n_tomo_lens))
                original_shape = csmf_mm[0, :, :, :, :, :].shape
                flat_length = len(self.log10csmf_mass_bins)*self.n_tomo_lens**2*self.n_tomo_csmf
                csmf_COSEBI_flat = np.reshape(csmf_mm, (len(self.ellrange), flat_length))
                for m_mode in range(self.En_modes):
                    local_ell_limit = self.ell_limits[m_mode][:]
                    self.levin_int.init_integral(self.ellrange, csmf_COSEBI_flat, True, True)
                    csmf_COSEBImmE[m_mode, :, :, :, :, :] = 1./(2.0*np.pi) * np.reshape(np.array(self.levin_int.cquad_integrate_single_well(local_ell_limit, m_mode)),original_shape)
            else:
                csmf_COSEBImmE, csmf_COSEBImmB = 0, 0

        if self.csmf:
            return gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn, \
                csmf_auto, csmf_COSEBIgg, csmf_COSEBIgm, csmf_COSEBImmE, csmf_COSEBImmB
        else:
            return gaussCOSEBIgggg_sva, gaussCOSEBIgggg_mix, gaussCOSEBIgggg_sn, \
                gaussCOSEBIgggm_sva, gaussCOSEBIgggm_mix, gaussCOSEBIgggm_sn, \
                gaussCOSEBIEggmm_sva, gaussCOSEBIEggmm_mix, gaussCOSEBIEggmm_sn, \
                gaussCOSEBIBggmm_sva, gaussCOSEBIBggmm_mix, gaussCOSEBIBggmm_sn, \
                gaussCOSEBIgmgm_sva, gaussCOSEBIgmgm_mix, gaussCOSEBIgmgm_sn, \
                gaussCOSEBIEmmgm_sva, gaussCOSEBIEmmgm_mix, gaussCOSEBIEmmgm_sn, \
                gaussCOSEBIBmmgm_sva, gaussCOSEBIBmmgm_mix, gaussCOSEBIBmmgm_sn, \
                gaussCOSEBIEEmmmm_sva, gaussCOSEBIEEmmmm_mix, gaussCOSEBIEEmmmm_sn, \
                gaussCOSEBIEBmmmm_sva, gaussCOSEBIEBmmmm_mix, gaussCOSEBIEBmmmm_sn, \
                gaussCOSEBIBBmmmm_sva, gaussCOSEBIBBmmmm_mix, gaussCOSEBIBBmmmm_sn
    
    
    def covCOSEBI_non_gaussian(self,
                               covELLspacesettings,
                               survey_params_dict,
                               output_dict,
                               bias_dict,
                               hod_dict,
                               prec,
                               tri_tab):
        """
        Calculates the non-Gaussian covariance between all observables for the
        COSEBIS as specified in the config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        
        Returns
        -------
            nongauss_BPgggg, nongauss_BPgggm, nongauss_BPEggmm, nongauss_BPBggmm, \
            nongauss_BPgmgm, nongauss_BPEmmgm, nongauss_BPBmmgm, nongauss_BPEEmmmm, \
            nongauss_BPEBmmmm, nongauss_BPBBmmmm : list of arrays

            each entry with shape (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        
        """

        if not self.cov_dict['nongauss']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covCOSEBI_4pt_projection(covELLspacesettings,
                                                  survey_params_dict,
                                                  output_dict,
                                                  bias_dict,
                                                  hod_dict,
                                                  prec,
                                                  tri_tab,
                                                  True)

    def covCOSEBI_ssc(self,
                      covELLspacesettings,
                      survey_params_dict,
                      output_dict,
                      bias_dict,
                      hod_dict,
                      prec,
                      tri_tab):
        """
        Calculates the super sample covariance between all observables for
        COSEBI as specified in the config file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        
        Returns
        -------
            nongauss_EEgggg, nongauss_EEgggm, nongauss_EEggmm, nongauss_EBggmm, \
            nongauss_EEgmgm, nongauss_EEmmgm, nongauss_EBmmgm, nongauss_EEmmmm, \
            nongauss_EBmmmm, nongauss_BBmmmm : list of arrays

            each entry with shape (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """
        if not self.cov_dict['ssc']:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            return self.__covCOSEBI_4pt_projection(covELLspacesettings,
                                                  survey_params_dict,
                                                  output_dict,
                                                  bias_dict,
                                                  hod_dict,
                                                  prec,
                                                  tri_tab,
                                                  False)          

    def __covCOSEBI_4pt_projection(self,
                               covELLspacesettings,
                               survey_params_dict,
                               output_dict,
                               bias_dict,
                               hod_dict,
                               prec,
                               tri_tab,
                               connected):
        """
        Auxillary function to integrate four point functions from ell space to
        COSEBI space for all observables specified in the input file.

        Parameters
        ----------
        covELLspacesettings : dictionary
            Specifies the exact details of the projection to ell space.
            The projection from wavefactor k to angular scale ell is
            done first, followed by the projection to real space in this
            class
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies
            for all tomographic bins as well as the ellipticity
            dispersion for galaxy shapes. To be passed from the
            read_input method of the Input class.
        output_dict : dictionary
            Specifies whether a file for the trispectra should be
            written to save computational time in the future. Gives the
            full path and name for the output file. To be passed from
            the read_input method of the Input class.
        bias_dict : dictionary
            Specifies all the information about the bias model. To be passed
            from the read_input method of the Input class.
        hod_dict : dictionary
            Specifies all the information about the halo occupation
            distribution used. This defines the shot noise level of the
            covariance and includes the mass bin definition of the different
            galaxy populations. To be passed from the read_input method of
            the Input class.
        survey_params_dict : dictionary
            Specifies all the information unique to a specific survey.
            Relevant values are the effective number density of galaxies for
            all tomographic bins as well as the ellipticity dispersion for
            galaxy shapes. To be passed from the read_input method of the
            Input class.
        hm_prec : dictionary
            Contains precision information about the HaloModel (also,
            see hmf documentation by Steven Murray), this includes mass
            range and spacing for the mass integrations in the halo
            model.
        tri_tab : dictionary
            Look-up table for the trispectra (for all combinations of
            matter 'm' and tracer 'g', optional).
            Possible keys: '', 'z', 'mmmm', 'mmgm', 'gmgm', 'ggmm',
            'gggm', 'gggg'
        connected : boolean
            If True the trispectrum is considered
            If False the SSC is considered

        Returns
        -------
            nongauss_EEgggg, nongauss_EEgggm, nongauss_EEggmm, nongauss_EBggmm, \
            nongauss_EEgmgm, nongauss_EEmmgm, nongauss_EBmmgm, nongauss_EEmmmm, \
            nongauss_EBmmmm, nongauss_BBmmmm
        
            each entry with shape (ell bins, ell bins,
                                   sample bins, sample bins,
                                   no_tomo_clust\lens, no_tomo_clust\lens,
                                   no_tomo_clust\lens, no_tomo_clust\lens)
        """

        nongaussCOSEBIEgggg = None
        nongaussCOSEBIEgggm = None
        nongaussCOSEBIEggmm = None
        nongaussCOSEBIBggmm = None
        nongaussCOSEBIEgmgm = None
        nongaussCOSEBIEmmgm = None
        nongaussCOSEBIBmmgm = None
        nongaussCOSEBIEEmmmm = None
        nongaussCOSEBIEBmmmm = None
        nongaussCOSEBIBBmmmm = None
        self.levin_int.update_Levin(0, 16, 32,1e-3,1e-4)
        
        if self.cov_dict['ssc'] and self.cov_dict['nongauss'] and (not self.cov_dict['split_gauss']):
            nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab)
            nongaussELLgggg1, nongaussELLgggm1, nongaussELLggmm1, nongaussELLgmgm1, nongaussELLmmgm1, nongaussELLmmmm1 = self.covELL_ssc(
                    bias_dict, hod_dict, prec, survey_params_dict, covELLspacesettings)
            if self.gg:
                nongaussELLgggg = nongaussELLgggg/(survey_params_dict['survey_area_clust'] / self.deg2torad2) + nongaussELLgggg1
            if self.gg and self.gm and self.cross_terms:
                nongaussELLgggm = nongaussELLgggm/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) + nongaussELLgggm1
            if self.gg and self.mm and self.cross_terms:
                nongaussELLggmm = nongaussELLggmm/(max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2) + nongaussELLggmm1
            if self.gm:
                nongaussELLgmgm = nongaussELLgmgm/(survey_params_dict['survey_area_ggl'] / self.deg2torad2) + nongaussELLgmgm1
            if self.mm and self.gm and self.cross_terms:
                nongaussELLmmgm = nongaussELLmmgm/(max(survey_params_dict['survey_area_lens'],survey_params_dict['survey_area_ggl']) / self.deg2torad2) + nongaussELLmmgm1
            if self.mm:
                nongaussELLmmmm = nongaussELLmmmm/(survey_params_dict['survey_area_lens'] / self.deg2torad2) + nongaussELLmmmm1
            connected = False
        else:
            if (connected):
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_non_gaussian(
                    covELLspacesettings, output_dict, bias_dict, hod_dict, prec, tri_tab)
            else:
                nongaussELLgggg, nongaussELLgggm, nongaussELLggmm, nongaussELLgmgm, nongaussELLmmgm, nongaussELLmmmm = self.covELL_ssc(
                    bias_dict, hod_dict, prec, survey_params_dict, covELLspacesettings)
        

        if self.gg:
            nongaussCOSEBIEgggg = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust))
            original_shape = nongaussCOSEBIEgggg[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**4
            nongaussELL_flat = np.reshape(nongaussELLgggg, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode + self.En_modes][:], n_mode + self.En_modes))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongaussCOSEBIEgggg[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode + self.En_modes][:], m_mode + self.En_modes)),original_shape)
                    if connected:
                        nongaussCOSEBIEgggg[n_mode, m_mode, :, :, :, :, :, :] /= (survey_params_dict['survey_area_clust'] / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian gggg term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
        else:
            nongaussCOSEBIEgggg = 0
            

        if self.gg and self.gm and self.cross_terms:
            nongaussCOSEBIEgggm = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens))
            original_shape = nongaussCOSEBIEgggm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_clust**3*self.n_tomo_lens
            nongaussELL_flat = np.reshape(nongaussELLgggm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode + self.En_modes][:], n_mode + self.En_modes))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongaussCOSEBIEgggm[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode + self.En_modes][:], m_mode + self.En_modes)),original_shape)
                    if connected:
                        nongaussCOSEBIEgggm[m_mode, n_mode, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_ggl']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian gggm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
        else:
            nongaussCOSEBIEgggm = 0

        if self.gg and self.mm and self.cross_terms:
            nongaussCOSEBIEggmm = np.zeros(
                (self.En_g_modes, self.En_modes, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            nongaussCOSEBIBggmm = np.zeros(
                (self.En_g_modes, self.En_modes, self.sample_dim, 1, self.n_tomo_clust, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_lens))
            original_shape = nongaussCOSEBIEggmm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            nongaussELL_flat = np.reshape(nongaussELLggmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_modes*self.En_g_modes
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode + self.En_modes][:], n_mode + self.En_modes))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongaussCOSEBIEggmm[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongaussCOSEBIEggmm[m_mode, n_mode, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_clust'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian ggmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
        else:
            nongaussCOSEBIEggmm = 0
            nongaussCOSEBIBggmm = 0

        if self.gm:
            nongaussCOSEBIEgmgm = np.zeros(
                (self.En_g_modes, self.En_g_modes, self.sample_dim, self.sample_dim, self.n_tomo_clust, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            original_shape = nongaussCOSEBIEgmgm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.sample_dim*self.n_tomo_lens**2*self.n_tomo_clust**2
            nongaussELL_flat = np.reshape(nongaussELLgmgm, (len(self.ellrange), len(self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes**2
            for m_mode in range(self.En_g_modes):
                for n_mode in range(self.En_g_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode + self.En_modes][:], n_mode + self.En_modes))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongaussCOSEBIEgmgm[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode + self.En_modes][:], m_mode + self.En_modes)),original_shape)
                    if connected: 
                        nongaussCOSEBIEgmgm[m_mode, n_mode, :, :, :, :, :, :] /= (survey_params_dict['survey_area_ggl']) / self.deg2torad2
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian gmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
        else:
            nongaussCOSEBIEgmgm = 0

        if self.gm and self.mm and self.cross_terms:
            nongaussCOSEBIEmmgm = np.zeros(
                (self.En_modes, self.En_g_modes, 1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))
            nongaussCOSEBIBmmgm = np.zeros(
                (self.En_modes, self.En_g_modes, 1, self.sample_dim, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_clust, self.n_tomo_lens))   
            original_shape = nongaussCOSEBIEmmgm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.sample_dim*self.n_tomo_lens**3*self.n_tomo_clust
            nongaussELL_flat = np.reshape(nongaussELLmmgm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_g_modes*self.En_modes
            for m_mode in range(self.En_modes):
                for n_mode in range(self.En_g_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode + self.En_modes][:], n_mode + self.En_modes))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, False)
                    nongaussCOSEBIEmmgm[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongaussCOSEBIEmmgm[m_mode, n_mode, :, :, :, :, :, :] /= (max(survey_params_dict['survey_area_ggl'],survey_params_dict['survey_area_lens']) / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian mmgm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
        else:
            nongaussCOSEBIEmmgm = 0
            nongaussCOSEBIBmmgm = 0

        if self.mm:
            nongaussCOSEBIEEmmmm = np.zeros(
                (self.En_modes, self.En_modes, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            nongaussCOSEBIEBmmmm = np.zeros(
                (self.En_modes, self.En_modes, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            nongaussCOSEBIBBmmmm = np.zeros(
                (self.En_modes, self.En_modes, 1, 1, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens, self.n_tomo_lens))
            original_shape = nongaussCOSEBIEEmmmm[0, 0, :, :, :, :, :, :].shape
            flat_length = self.n_tomo_lens**4
            nongaussELL_flat = np.reshape(nongaussELLmmmm, (len(self.ellrange), len(
                self.ellrange), flat_length))
            t0, tcomb = time.time(), 1
            tcombs = self.En_modes**2
            for m_mode in range(self.En_modes):
                for n_mode in range(self.En_modes):
                    inner_integral = np.zeros((len(self.ellrange), flat_length))
                    for i_ell in range(len(self.ellrange)):
                        self.levin_int.init_integral(self.ellrange, nongaussELL_flat[i_ell, :, :]*self.ellrange[:, None], True, True)
                        inner_integral[i_ell, :] = np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[n_mode][:], n_mode))
                    self.levin_int.init_integral(self.ellrange, inner_integral*self.ellrange[:, None], True, True)
                    nongaussCOSEBIEEmmmm[m_mode, n_mode, :, :, :, :, :, :] = 1.0/(4.0*np.pi**2)*np.reshape(np.array(self.levin_int.cquad_integrate_single_well(self.ell_limits[m_mode][:], m_mode)),original_shape)
                    if connected:
                        nongaussCOSEBIEEmmmm[m_mode, n_mode, :, :, :, :, :, :] /= (survey_params_dict['survey_area_lens'] / self.deg2torad2)
                    eta = (time.time()-t0) / \
                            60 * (tcombs/tcomb-1)
                    print('\rCOSEBI E-mode covariance calculation for the '
                            'nonGaussian mmmm term '
                            + str(round(tcomb/tcombs*100, 1)) + '% in '
                            + str(round(((time.time()-t0)/60), 1)) +
                            'min  ETA '
                            'in ' + str(round(eta, 1)) + 'min', end="")
                    tcomb += 1
            
            print("")
            
        else:
            nongaussCOSEBIEEmmmm = 0
            nongaussCOSEBIEBmmmm = 0
            nongaussCOSEBIBBmmmm = 0

        return nongaussCOSEBIEgggg, nongaussCOSEBIEgggm, nongaussCOSEBIEggmm, nongaussCOSEBIBggmm, \
              nongaussCOSEBIEgmgm, nongaussCOSEBIEmmgm, nongaussCOSEBIBmmgm, nongaussCOSEBIEEmmmm, \
              nongaussCOSEBIEBmmmm, nongaussCOSEBIBBmmmm
