/* --- includes --- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <complex.h>

#define _PI_ 3.14159265358979323846
#define FLAG_NOGAL -1  
#define FLAG_OUTSIDE -1  
#define SQUARE(x) ((x)*(x))
#define mymax(x,y) ((x) >= (y)) ? (x) : (y)
#define mymin(x,y) ((x) < (y)) ? (x) : (y)


////////////////////////////////////////////
/// FUNCTIONS RELATED TO SPATIAL HASHING ///
////////////////////////////////////////////
void build_spatialhash(double *pos_1, double *pos_2, int ngal,
    double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2,
    int *result){

    int npix, npixs_with_gals, pix_1, pix_2, index, noutside;
    int nrelpix, cumsum;
    int noutsiders, index_raw, index_red;
    int ind_gal, ind_pix;

    int start_isoutside, start_matcher, start_bounds, start_pixgals, start_ngalinpix;

    // First step: Allocate number of galaxies per pixel
    // ngals_in_pix = [ngals_in_pix1, ngals_in_pix2, ..., ngals_in_pix-1]
    // s.t. sum(ngals_in_poix) == ngal_tot --> at most ngal_tot non-zero elements
    npix = mask_n1*mask_n2;
    start_isoutside = 0;
    start_matcher = ngal;
    start_bounds = ngal+npix;
    start_pixgals = ngal+npix+ngal+1;
    start_ngalinpix=ngal+npix+ngal+1+ngal;

    int test = 213;
    npixs_with_gals = 0;
    noutside = 0;
    for (ind_gal=0; ind_gal<ngal; ind_gal++){
        pix_1 = (int) floor((pos_1[ind_gal]-mask_min1)/mask_d1);
        pix_2 = (int) floor((pos_2[ind_gal]-mask_min2)/mask_d2);
        index = pix_2*mask_n1+pix_1;
        if (pix_1 > mask_n1 || pix_2 > mask_n2 || pix_1<0 || pix_2<0){
            result[start_isoutside+ind_gal] = 0;//true
            noutside += 1;}
        else{
            if (result[start_ngalinpix+index] == 0){npixs_with_gals+=1;}
            result[start_isoutside+ind_gal] = 1;//false
            result[start_ngalinpix+index] += 1;
        }
    } 

    // Second step: Allocate pixels with galaxies in them and their bounds
    // index_matcher = [flag_nogal, ..., 0, ..., 1, 2, ..., nrelpixs, flag_nogal, ...]
    //     --> length npix
    // pixs_galind_bounds = [0, ngals_in_pix_a, ngals_in_pix_a + ngals_in_pix_b, ..., ngal_tot, g.a.r.b.a.g.e]
    //     --> length ngal+1
    nrelpix = 0;
    cumsum = 0;
    result[start_bounds+0] = 0;
    for (ind_pix=0; ind_pix<npix; ind_pix++){
        if (result[start_ngalinpix+ind_pix] == 0){result[start_matcher+ind_pix] = FLAG_NOGAL;}
        else{
            result[start_matcher+ind_pix] = nrelpix;
            result[start_bounds+nrelpix+1] = result[start_bounds+nrelpix] + result[start_ngalinpix+ind_pix];
            nrelpix += 1;
        }
    }

    // Third step: Put galaxy indices into pixels
    // pix_gals = [gal1_in_pix_a, ..., gal-1_in_pix_a, ..., gal1_in_pix_n, ..., gal-1_in_pix_n, e.m.p.t.y.g.a.l.s]
    //     --> length ngal
    noutsiders = 0;
    for (ind_gal=0; ind_gal<ngal; ind_gal++){
        if (result[start_isoutside+ind_gal] == 0){
            result[start_pixgals+ngal-noutsiders-1] = FLAG_OUTSIDE;
            noutsiders += 1;
        }
        else{
            pix_1 = (int) floor((pos_1[ind_gal]-mask_min1)/mask_d1);
            pix_2 = (int) floor((pos_2[ind_gal]-mask_min2)/mask_d2);
            index_raw = pix_2*mask_n1+pix_1;
            index_red = result[start_matcher+index_raw];
            index = result[start_bounds+index_red] + result[start_ngalinpix+index_raw]-1;
            result[start_pixgals+index] =  ind_gal;
            result[start_ngalinpix+index_raw] -= 1;
        }
    }
}

void reducecat(double *w, double *pos_1, double *pos_2, double *scalarquants, int ngal, int nscalarquants,
               double mask_d1, double mask_d2, double mask_min1, double mask_min2, int mask_n1, int mask_n2, int shuffle,
               double *w_red, double *pos1_red, double *pos2_red, double *scalarquants_red, int ngal_red){
    
    // Build spatial hash
    int npix = mask_n1*mask_n2;
    int start_isoutside = 0;
    int start_matcher = ngal;
    int start_bounds = ngal+npix;
    int start_pixgals = ngal+npix+ngal+1;
    int start_ngalinpix=ngal+npix+ngal+1+ngal;
    int *spatialhash = calloc(2*npix+3*ngal+1, sizeof(int));
    build_spatialhash(pos_1, pos_2, ngal,
                      mask_d1, mask_d2, mask_min1, mask_min2, mask_n1, mask_n2,
                      spatialhash);
    
    // Allocate pixelized catalog from spatial hash
    int ind_pix1, ind_pix2, ind_red, lower, upper, ind_inpix, ind_gal, elscalarquant;
    double tmppos_1, tmppos_2, tmpw, shift_1, shift_2;
    double *tmpscalarquants;
    int rseed=42;
    srand(rseed);  
    for (ind_pix1=0; ind_pix1<mask_n1; ind_pix1++){
        for (ind_pix2=0; ind_pix2<mask_n2; ind_pix2++){
            ind_red = spatialhash[start_matcher + ind_pix2*mask_n1 + ind_pix1];
            if (ind_red==FLAG_NOGAL){continue;}
            lower = spatialhash[start_bounds+ind_red];
            upper = spatialhash[start_bounds+ind_red+1];
            tmpw = 0;
            tmppos_1 = 0;
            tmppos_2 = 0;
            tmpscalarquants = calloc(nscalarquants, sizeof(double));            
            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                ind_gal = spatialhash[start_pixgals+ind_inpix];
                tmpw += w[ind_gal];
                tmppos_1 += w[ind_gal]*pos_1[ind_gal];
                tmppos_2 += w[ind_gal]*pos_2[ind_gal];
                for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                    tmpscalarquants[elscalarquant] +=  w[ind_gal]*scalarquants[elscalarquant*ngal+ind_gal];
                }
            }
            if (tmpw==0){continue;}
            w_red[ngal_red] = tmpw;
            if (shuffle==0){
                pos1_red[ngal_red] = tmppos_1/tmpw;
                pos2_red[ngal_red] = tmppos_2/tmpw;}
            else{
                shift_1 = ((double)rand()/(double)(RAND_MAX)) * mask_d1;
                shift_2 = ((double)rand()/(double)(RAND_MAX)) * mask_d2;
                pos1_red[ngal_red] = mask_min1+ind_pix1*mask_d1 + shift_1;
                pos2_red[ngal_red] = mask_min2+ind_pix2*mask_d2 + shift_2;}
            for (elscalarquant=0; elscalarquant<nscalarquants; elscalarquant++){
                // Here we need to average each of the quantities in order to retain the correct
                // normalization of the NPCF - i.e. for a polar field we would have
                // Upsilon_n,pix ~ w_pix * G1_pix * G2_pix
                //               ~ w_pix * (w_pix * shape_pix * g1) * (w_pix * shape_pix * g2)
                //               ~ w_pix^3 * shape_pix^2
                // This means that shape_pix should be independent of the size of the pixel, i.e. that
                // we should normalize shape_pix ~ (sum_i w_i*gamma_i) / (sum_i w_i)
                scalarquants_red[elscalarquant*ngal+ngal_red] =  tmpscalarquants[elscalarquant]/tmpw;
            }
            ngal_red += 1;
            free(tmpscalarquants);
        }
    }
}


/// ALLOCATION OF THE TRIPLET COUNTS VIA MULTIPOLES IN THE TREE-BASED APPROXIMATION ///
void alloc_triplets_tree_xipxipcov(
    int *isinner, double *weight, double *pos1, double *pos2, int *zbins, int nbinsz, int ngal, 
    int nresos, double *reso_redges, int *ngal_resos, 
    double *weight_resos, double *pos1_resos, double *pos2_resos, int *zbin_resos,
    int *index_matcher, int *pixs_galind_bounds, int *pix_gals, 
    double pix1_start, double pix1_d, int pix1_n, double pix2_start, double pix2_d, int pix2_n,
    int nmin, int nmax, double rmin, double rmax, double *rbins, int nbinsr, int dccorr, 
    int nthreads, 
    double *bin_centers, double *wwcounts, double *w2wcounts, double complex *w2wwcounts, double complex *wwwcounts){
    
    // Index shift for the Gamman
    int _gamma_zshift = nbinsr*nbinsr;
    int _gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
    int _gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
    
    double *totcounts = calloc(nbinsz*nbinsr, sizeof(double));
    double *totnorms = calloc(nbinsz*nbinsr, sizeof(double));
    
    // Allocate Gns
    // We do this in parallel as follows:
    // * Split survey in 2*nthreads equal area stripes along x-axis
    // * Do two parallelized iterations over galaxies
    //   - In first one only consider galaxies within stripes of even number
    //   - In second one only consider galaxies within stripes of odd number
    // --> We avoid race conditions in calling the spatial hash arrays. - This
    //    is explicitly made sure by (re)setting nthreads in the python layer.
    for (int odd=0; odd<2; odd++){
        
        // Temporary arrays that are allocated in parallel and later reduced
        double *tmpwcounts = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double *tmpwnorms = calloc(nthreads*nbinsz*nbinsr, sizeof(double));
        double *tmpwwcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
        double *tmpw2wcounts = calloc(nthreads*nbinsz*nbinsz*nbinsr, sizeof(double));
        double complex *tmpGammans = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        double complex *tmpGammans_norm = calloc(nthreads*_gamma_compshift, sizeof(double complex));
        #pragma omp parallel for num_threads(nthreads)
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            int gamma_zshift = nbinsr*nbinsr;
            int gamma_nshift = _gamma_zshift*nbinsz*nbinsz*nbinsz;
            int gamma_compshift = (nmax-nmin+1)*_gamma_nshift;
            
            int npix_hash = pix1_n*pix2_n;
            int *rshift_index_matcher = calloc(nresos, sizeof(int));
            int *rshift_pixs_galind_bounds = calloc(nresos, sizeof(int));
            int *rshift_pix_gals = calloc(nresos, sizeof(int));
            double *reso_redges2 = calloc(nresos+1, sizeof(double));
            reso_redges2[0] = reso_redges[0]*reso_redges[0];
            for (int elreso=1;elreso<nresos;elreso++){
                rshift_index_matcher[elreso] = rshift_index_matcher[elreso-1] + npix_hash;
                rshift_pixs_galind_bounds[elreso] = rshift_pixs_galind_bounds[elreso-1] + ngal_resos[elreso-1]+1;
                rshift_pix_gals[elreso] = rshift_pix_gals[elreso-1] + ngal_resos[elreso-1];
                reso_redges2[1+elreso] = reso_redges[1+elreso]*reso_redges[1+elreso];
            }
                
            for (int ind_gal=0; ind_gal<ngal; ind_gal++){
                // Check if galaxy falls in stripe used in this process
                double p11, p12, w1, e11, e12;
                int zbin1, innergal;
                #pragma omp critical
                {p11 = pos1[ind_gal];
                p12 = pos2[ind_gal];
                w1 = weight[ind_gal];
                zbin1 = zbins[ind_gal];
                innergal = isinner[ind_gal];}
                if (innergal==0){continue;}
                int thisstripe = 2*thisthread + odd;
                int galstripe = (int) floor((p11-pix1_start)/pix1_d * (2*nthreads)/pix1_n);
                if (thisstripe != galstripe){continue;}
                
                int ind_pix1, ind_pix2, ind_inpix, ind_gal2;
                int ind_red, lower, upper; 
                double  p21, p22, w2, z2;
                double rel1, rel2, dist, dist2;
                double complex wshape;
                int nnvals, nnvals_norm, nextn, nzero;
                double complex nphirot, twophirotc, nphirotc, phirot, phirotc, phirotm, phirotp, phirotn;
                
                // Check how many ns we need for Gn
                // Gns have shape (nnvals, nbinsz, nbinsr)
                nnvals = nmax+1;
                double complex *nextGns =  calloc(nnvals*nbinsr*nbinsz, sizeof(double complex));
                double complex *nextG2ns =  calloc(nbinsz*nbinsr, sizeof(double complex));

                int ind_rbin, ind_wwbin, rbin;
                int zrshift, normzrshift, _normzrshift;
                int nbinszr = nbinsz*nbinsr;
                int nbinsz2r = nbinsz*nbinsz*nbinsr;
                double drbin = (log(rmax)-log(rmin))/(nbinsr);
                
                for (int elreso=0;elreso<nresos;elreso++){
                    int pix1_lower = mymax(0, (int) floor((p11 - (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_lower = mymax(0, (int) floor((p12 - (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));
                    int pix1_upper = mymin(pix1_n-1, (int) floor((p11 + (reso_redges[elreso+1]+pix1_d) - pix1_start)/pix1_d));
                    int pix2_upper = mymin(pix2_n-1, (int) floor((p12 + (reso_redges[elreso+1]+pix2_d) - pix2_start)/pix2_d));

                    for (ind_pix1=pix1_lower; ind_pix1<pix1_upper; ind_pix1++){
                        for (ind_pix2=pix2_lower; ind_pix2<pix2_upper; ind_pix2++){
                            ind_red = index_matcher[rshift_index_matcher[elreso] + ind_pix2*pix1_n + ind_pix1];
                            if (ind_red==-1){continue;}
                            lower = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red];
                            upper = pixs_galind_bounds[rshift_pixs_galind_bounds[elreso]+ind_red+1];
                            for (ind_inpix=lower; ind_inpix<upper; ind_inpix++){
                                ind_gal2 = rshift_pix_gals[elreso] + pix_gals[rshift_pix_gals[elreso]+ind_inpix];
                                p21 = pos1_resos[ind_gal2];
                                p22 = pos2_resos[ind_gal2];
                                w2 = weight_resos[ind_gal2];
                                z2 = zbin_resos[ind_gal2];
                                rel1 = p21 - p11;
                                rel2 = p22 - p12;
                                dist2 = rel1*rel1 + rel2*rel2;
                                dist = sqrt(dist2);
                                if(dist < reso_redges[elreso] || dist >= reso_redges[elreso+1]){continue;}
                                //dist = sqrt(dist2);
                                if (rbins[0] < 0){
                                    rbin = (int) floor(log(dist/rmin)/drbin);
                                }
                                else{
                                    rbin=0;
                                    while(rbins[rbin+1] < dist){rbin+=1;}
                                }
                                
                                
                                phirot = (rel1+I*rel2)/dist * fabs(rel1)/rel1;
                                //if (rel1<0){phirot*=-1;}
                                //phirotc = conj(phirot);
                                //twophirotc = phirotc*phirotc;
                                //double dphi = atan2(rel2,rel1);
                                //phirot = cexp(I*dphi);
                                zrshift = z2*nbinsr + rbin;
                                ind_rbin = thisthread*nbinszr + zrshift;
                                ind_wwbin = thisthread*nbinsz2r+zbin1*nbinszr+zrshift;
                                nphirot = 1+I*0;
                                tmpwcounts[ind_rbin] += w1*w2*dist; 
                                tmpwnorms[ind_rbin] += w1*w2; 
                                tmpwwcounts[ind_wwbin] += w1*w2; 
                                tmpw2wcounts[ind_wwbin] += w1*w1*w2; 
                                nextGns[zrshift] += w2*nphirot;  
                                nextG2ns[zrshift] += w2*w2;
                                nphirot *= phirot;
                                
                                for (nextn=1;nextn<nmax+1;nextn++){
                                    nextGns[zrshift+nextn*nbinszr] += w2*nphirot;  
                                    nphirot *= phirot;
                                }
                            }
                        }
                    }
                }
                
                // Now update the Gammans
                // tmpGammas have shape (nthreads, nmax+1, nzcombis3, r*r, 4)
                // Gns have shape (nnvals, nbinsz, nbinsr)
                //int nonzero_tmpGammas = 0;
                double complex wsq, w1_sq, w0;
                int thisnshift, r12shift;
                int gammashift1, gammashift;
                int ind_norm;
                int thisn, elb1, elb2, zbin2, zbin3, zcombi;
                w1_sq = w1*w1;
                for (thisn=0; thisn<nmax-nmin+1; thisn++){
                    ind_norm = thisn*nbinszr;
                    thisnshift = thisthread*gamma_compshift + thisn*gamma_nshift;
                    for (zbin2=0; zbin2<nbinsz; zbin2++){
                        for (elb1=0; elb1<nbinsr; elb1++){
                            normzrshift = ind_norm + zbin2*nbinsr + elb1;
                            wsq = w1_sq * nextGns[normzrshift];
                            w0 = w1 * nextGns[normzrshift];
                            for (zbin3=0; zbin3<nbinsz; zbin3++){
                                zcombi = zbin1*nbinsz*nbinsz+zbin2*nbinsz+zbin3;
                                gammashift1 = thisnshift + zcombi*gamma_zshift;
                                // Double counting correction
                                if (zbin1==zbin2 && zbin1==zbin3 && dccorr==1){
                                    zrshift = zbin2*nbinsr + elb1;
                                    r12shift = elb1*nbinsr+elb1;
                                    gammashift = gammashift1 + r12shift;
                                    tmpGammans[gammashift] -= w1_sq*nextG2ns[zrshift];
                                    tmpGammans_norm[gammashift] -= w1*nextG2ns[zrshift];
                                }
                                // Nominal allocation
                                _normzrshift = ind_norm+zbin3*nbinsr;
                                for (elb2=0; elb2<nbinsr; elb2++){
                                    normzrshift = _normzrshift + elb2;
                                    gammashift = gammashift1 + elb1*nbinsr+elb2;
                                    //phirotm = h0*nextGns[ind_mnm3 + zrshift];
                                    tmpGammans[gammashift] += wsq*conj(nextGns[normzrshift]);
                                    tmpGammans_norm[gammashift] += w0*conj(nextGns[normzrshift]);
                                    //if(thisthread==0 && ind_gal%1000==0){
                                    //    if (cabs(tmpGammans[gammashift] )>1e-5){nonzero_tmpGammas += 1;}
                                    //}
                                }
                            }
                        }
                    }
                }
                
                free(nextGns);
                free(nextG2ns);
                nextGns = NULL;
                nextG2ns = NULL;
            }
            
            free(rshift_index_matcher);
            free(rshift_pixs_galind_bounds);
            free(rshift_pix_gals);
            free(reso_redges2);
        }
        
        // Accumulate the Gamman
        #pragma omp parallel for num_threads(nthreads)
        for (int thisn=0; thisn<nmax-nmin+1; thisn++){
            int itmpGamma, iGamma, _iGamma;
            for (int thisthread=0; thisthread<nthreads; thisthread++){
                for (int zcombi=0; zcombi<nbinsz*nbinsz*nbinsz; zcombi++){
                    for (int elb1=0; elb1<nbinsr; elb1++){
                        _iGamma = thisn*_gamma_nshift + zcombi*_gamma_zshift + elb1*nbinsr;
                        for (int elb2=0; elb2<nbinsr; elb2++){
                            iGamma = _iGamma + elb2;
                            itmpGamma = iGamma + thisthread*_gamma_compshift;
                            w2wwcounts[iGamma] += tmpGammans[itmpGamma];
                            wwwcounts[iGamma] += tmpGammans_norm[itmpGamma];
                        }
                    }
                }
            }
        }
        
        
        // Accumulate the paircounts
        int threadshift, zzrind;
        int nbinszr = nbinsz*nbinsr;
        int nbinsz2r = nbinsz*nbinsz*nbinsr;
        for (int thisthread=0; thisthread<nthreads; thisthread++){
            threadshift = thisthread*nbinsz2r;
            for (int elbinz1=0; elbinz1<nbinsz; elbinz1++){
                for (int elbinz2=0; elbinz2<nbinsz; elbinz2++){
                    for (int elbinr=0; elbinr<nbinsr; elbinr++){
                        zzrind = elbinz1*nbinszr+elbinz2*nbinsr+elbinr;
                        wwcounts[zzrind] += tmpwwcounts[threadshift+zzrind];
                        w2wcounts[zzrind] += tmpw2wcounts[threadshift+zzrind];
                    }
                }
            }
        }
        
        
        // Update the bin distances and weights
        for (int elbinz=0; elbinz<nbinsz; elbinz++){
            for (int elbinr=0; elbinr<nbinsr; elbinr++){
                int tmpind = elbinz*nbinsr + elbinr;
                for (int thisthread=0; thisthread<nthreads; thisthread++){
                    int tshift = thisthread*nbinsz*nbinsr; 
                    totcounts[tmpind] += tmpwcounts[tshift+tmpind];
                    totnorms[tmpind] += tmpwnorms[tshift+tmpind];
                    
                }
            }
        }
        free(tmpwcounts);
        free(tmpwnorms);
        free(tmpwwcounts);
        free(tmpw2wcounts);
        free(tmpGammans);
        free(tmpGammans_norm); 
        tmpwcounts = NULL;
        tmpwnorms = NULL;
        tmpwwcounts = NULL;
        tmpw2wcounts = NULL;
        tmpGammans = NULL;
        tmpGammans_norm = NULL;
    }
    
    // Get bin centers
    for (int elbinz=0; elbinz<nbinsz; elbinz++){
        for (int elbinr=0; elbinr<nbinsr; elbinr++){
            int tmpind = elbinz*nbinsr + elbinr;
            if (totnorms[tmpind] != 0){
                bin_centers[tmpind] = totcounts[tmpind]/totnorms[tmpind];
            }
            
        }
    } 
    free(totcounts);
    free(totnorms);
    totcounts = NULL;
    totnorms = NULL;
}