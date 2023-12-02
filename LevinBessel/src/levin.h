#ifndef LEVIN_H
#define LEVIN_H

#include <vector>
#include <numeric>
#include <omp.h>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <algorithm>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>



#include <cmath>
#include <thread>

typedef std::vector<std::vector<double>> result_Cl_type;

/**
 * This class implements integrals over Bessel functions and their products using
 * Levin's method. The integrals need to be of the type
 * \f[
 *  I = \int\mathrm{d}x \langle F,w \rangle (x)
 * \f]
 * where \f$ w \f$ is highly oscillatory and \f$ F \f$ isn't. The angular brackets
 * define a scalar product.
 */

class Levin
{
private:
  static const double min_interval;
  static const double tol_abs;
  static const double min_sv;
  static const double kernel_overlap_eps;
  uint N_thread_max = std::thread::hardware_concurrency();
  std::vector<uint> ell;
  uint d;
  uint type;
  uint number_of_modes = 0;
  uint n_split_rs;
  uint col, nsub;
  uint number_x_values;
  uint number_integrals;
  double x_max, x_min, xmax_weight, xmin_weight;
  bool logx;
  std::vector<bool> logy;
  std::vector<double> slope, slope0;
  double relative_tol;
  bool already_called;

  std::vector<gsl_interp_accel *> acc_integrand;
  std::vector<gsl_spline *> spline_integrand;
  std::vector<std::vector<gsl_interp_accel *>> acc_w_ell;
  std::vector<std::vector<gsl_spline *>> spline_w_ell;

  std::vector<gsl_interp_accel *> acc_cov_Gauss, acc_cov_SSC;
  std::vector<gsl_spline *> spline_cov_Gauss, spline_cov_SSC;
  std::vector<gsl_spline2d *> spline_cov_non_Gauss;
  std::vector<gsl_interp_accel *> acc_non_cov_Gauss_k1;
  std::vector<gsl_interp_accel *> acc_non_cov_Gauss_k2;
  std::vector<double> cov_R_radii;
  std::vector<double> ell_w_ell;

  uint sample_size;
  double k_max;
  double k_min;
  double converged;
  uint *int_cov_R_m_bin, *int_cov_R_n_bin, *int_cov_R_i_R, *int_cov_R_j_R, *int_cov_R_ell1, *int_cov_R_ell2, *int_index_integral;
  double *int_cov_R_non_Gauss_outer_k;

  double *int_k_single_bessel;
  uint *int_ell_single_bessel;

  double *int_k1_double_bessel, *int_k2_double_bessel;
  uint *int_ell1_double_bessel, *int_ell2_double_bessel;

  uint *int_m_mode, *int_n_mode;
  
  double gslIntegratecquad(double (*fc)(double, void *), double a, double b);

public:
  /**
   * The constructor requires the type of integral to be carried out. All integrals should be of the type:
   * \f[
   *    I = \int_a^b \langle F,w\rangle (x)\;,
   * \f]
   * with \f$ w \f$ being oscillatory and \f$ F \f$ not. The brackets define a scalar product. The following cases are currently implemented
   *  - \p type  \p = 0, corresponds to integrals \f$ I(k) \int_a^b f(x;k)J_\ell (xk) \f$ with a cylindrical Bessel function \f$ J_\ell (xk) \f$.
   *  - \p type  \p = 1, corresponds to integrals \f$ I(k) \int_a^b f(x;k)j_\ell (xk) \f$ with a spherical Bessel function \f$ j_\ell (xk) \f$.
   *  - \p type  \p = 2, corresponds to integrals \f$ I(k) \int_a^b f(x;k)J_{\ell_1} (xk)j_{\ell_2} (xk) \f$ with a cylindrical Bessel function \f$ J_\ell (xk) \f$.
   *  - \p type  \p = 3, corresponds to integrals \f$ I(k) \int_a^b f(x;k)j_{\ell_1} (xk)j_{\ell_2} (xk) \f$ with a spherical Bessel function \f$ j_\ell (xk) \f$.
   */
  Levin(uint type1, uint col1 = 8, uint nsub1 = 16, double relative_tol1 = 1e-6, uint n_split_rs1 = 50, uint Nthread = 4);

  /**
   * Destructor: clean up all allocated memory.
   */
  ~Levin();

  /**
   *  Sets everything up depending on the type
   */
  void setup(uint type);

  /**
   * Updates the parameters used for the integration
   *
   * @param type1
   * @param col1
   * @param nsub1
   * @param relative_tol1
   */
  void update_Levin(uint type1, uint col1, uint nsub1, double relative_tol1, double converged1);

  /**
   * General initialization of the ingegral to be carried out. x is an array of all the x values in the domain
   * and the integrand has to be a matrix of the shape length(x), number of different integrands over the same
   * domain. The integrand is then splined accordingly.
   *
   * @param x
   * @param integrand
   */
  void init_integral(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx1, bool logy1);

  void init_w_ell(std::vector<double> ell, std::vector<std::vector<double>> w_ells);

  std::vector<double> get_w_ell(std::vector<double> ell, uint m_mode);

  std::vector<double> get_integrand(std::vector<double> x, uint j);

  double call_integrand(double x, uint i);

  void init_cov_R_space_Gaussian(std::vector<std::vector<std::vector<double>>> cov_k_space_Gaussian, std::vector<double> k, std::vector<double> r);

  void init_cov_R_space_SSC(std::vector<std::vector<std::vector<double>>> cov_k_space_SSC, std::vector<double> k, std::vector<double> r);

  void init_cov_R_space_NonGaussian(std::vector<std::vector<std::vector<std::vector<double>>>> cov_k_space_NonGaussian, std::vector<double> k, std::vector<double> r);

  /**
   *  Define the vector \f$ w \f$ for the integration of a single oscillatory function and returning the i-th component.
   */
  double w_single(double x, double k, uint ell, uint i);

  /**
   *  Define the vector \f$ w \f$ for the integration of a product of two oscillatory functions and returning the i-th component.
   */
  double w_double(double x, double k1, double k2, uint ell_1, uint ell_2, uint i);

  /**
   *  Define the matrix \f$w^\prime = A w \f$ for the integration of a single oscillatory function and returning the i,j component.
   */
  double A_matrix_single(uint i, uint j, double x, double k, uint ell);

  /**
   *  Define the matrix \f$w^\prime = A w \f$ for the integration of a product of two oscillatory functions and returning the i,j component.
   */
  double A_matrix_double(uint i, uint j, double x, double k1, double k2, uint ell_1, uint ell_2);

  /**
   *  Setting the nodes at the col collocation points in the interval \f$ A,B \f$  (see Levin) and returning the nodes as a list.
   */
  std::vector<double> setNodes(double A, double B, uint col);

  /**
   * Returns the \f$m\f$-th basis function in the interval \f$ A,B \f$  at position \f$x\f$ (see Levin)
   */
  double basis_function(double A, double B, double x, uint m);

  /**
   * Returns the derivative of the \f$m\f$-th basis function in the interval \f$ A,B \f$  at position \f$x\f$.
   */
  double basis_function_prime(double A, double B, double x, uint m);

  /**
   * Solves the linear system of equations for a single oscillatory function in the interval \f$ A,B \f$ at col collactation points with corresponding nodes x_j. The system is
   * specified by providing the tomographic bin, i_tomo, the wavenumber, k, multipole ell and whether the linear or nonlinear version of the
   * power spectrum should be used. The solution to the LSE is returned as a list.
   **/
  std::vector<double> solve_LSE_single(double (*function)(double, void *), double A, double B, uint col, std::vector<double> x_j, double k, uint ell);

  /**
   * Solves the linear system of equations for a product of two oscillatory functions in the interval \f$ A,B \f$ at col collactation points with corresponding nodes x_j. The system is
   * specified by providing the tomographic bin, i_tomo, the wavenumber, k, multipole ell and whether the linear or nonlinear version of the
   * power spectrum should be used. The solution to the LSE is returned as a list.
   **/
  std::vector<double> solve_LSE_double(double (*function)(double, void *), double A, double B, uint col, std::vector<double> x_j, double k1, double k2, uint ell_1, uint ell_2);

  /**
   * Returns the \f$i \f$-th component of the vector \f$ p \f$ given the solution to the LSE, c.
   **/
  double p(double A, double B, uint i, double x, uint col, std::vector<double> c);

  /**
   * Integrates
   * \f[
   *  I(k) = \int\mathrm{d}x \langle F,w\rangle
   * \f]
   * in an interval \f$ A,B \f$ with \p col collocation points. Here only a single oscillatory function is considered. The estimate of the integral is returned.
   **/
  double integrate_single(double (*function)(double, void *), double A, double B, uint col, double k, uint ell);

  /**
   * Integrates
   * \f[
   *  I(k) = \int\mathrm{d}x \langle F,w\rangle
   * \f]
   * in an interval \f$ A,B \f$ with \p col collocation points. Here a product of two oscillatory functions is considered. The estimate of the integral is returned.
   **/
  double integrate_double(double (*function)(double, void *), double A, double B, uint col, double k1, double k2, uint ell_1, uint ell_2);

  /**
   * Iterates over the integral by bisectiong the interval with the largest error until convergence or a maximum number of bisections, smax, is reached.
   * The final result is returned. This is for a single oscillatory function.
   **/
  double iterate_single(double (*function)(double, void *), double A, double B, uint col, double k, uint ell, uint smax, bool verbose);

  /**
   * Iterates over the integral by bisectiong the interval with the largest error until convergence or a maximum number of bisections, smax, is reached.
   * The final result is returned. This is for a product of two oscillatory functions.
   **/
  double iterate_double(double (*function)(double, void *), double A, double B, uint col, double k1, double k2, uint ell_1, uint ell_2, uint smax, bool verbose);

  /**
   *  Return the maximum index of a list.
   */
  uint findMax(const std::vector<double> vec);

  /**
   * Returns the integral of the function \p function depending on the value for \f$ k\f$ and \f$ \ell\f$ (compare with the cases in the constructor) in the interval
   * \p a and \p b for a single oscillatory function.
   */
  double levin_integrate_bessel_single(double (*function)(double, void *), double k, uint ell, double a, double b);

  /**
   * Returns the integral of the function \p function depending on the value for \f$ k\f$ and \f$ \ell\f$ (compare with the cases in the constructor) in the interval
   * \p a and \p b for a product of oscillatory functions.
   */
  double levin_integrate_bessel_double(double (*function)(double, void *), double k1, double k2, uint ell_1, uint ell_2, double a, double b);

  std::vector<double> single_bessel(double k, uint ell, double a, double b);

  static double single_bessel_integrand(double, void *);

  std::vector<double> double_bessel(double k1, double k2, uint ell_1, uint ell_2, double a, double b);

  std::vector<double> double_bessel_many_args(std::vector<double> k1, double k2, uint ell_1, uint ell_2, double a, double b);


  static double double_bessel_integrand(double, void *);

  std::vector<double> single_bessel_many_args(std::vector<double> k, uint ell, double a, double b);

  static double cquad_integrand(double x, void *p);

  std::vector<double> cquad_integrate(std::vector<double> limits);

  static double cquad_integrand_single_well(double x, void *p);

  std::vector<double> cquad_integrate_single_well(std::vector<double> limits, uint m_mode);

  static double cquad_integrand_double_well(double x, void *p);

  std::vector<double> cquad_integrate_double_well(std::vector<double> limits, uint m_mode, uint n_mode);
  
  static double integrand(double x, void *p);

  static double cov_R_Gaussian_integrand(double, void *);

  static double cov_R_SSC_integrand(double, void *);

  static double cov_R_NonGaussian_inner_integrand(double, void *);

  static double cov_R_NonGaussian_outer_integrand(double, void *);

  std::vector<std::vector<std::vector<std::vector<double>>>> cov_R_get_gauss(bool cross, uint ell_1, uint ell_2);

  std::vector<std::vector<std::vector<std::vector<double>>>> cov_R_get_ssc(bool cross, uint ell_1, uint ell_2);

  std::vector<std::vector<std::vector<std::vector<double>>>> cov_R_get_nongauss(bool cross, uint ell_1, uint ell_2);
};

#endif