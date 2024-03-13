#include "levin.h"
#include <iomanip>

const double Levin::min_interval = 1.e-4;
const double Levin::tol_abs = 1.0e-200;
const double Levin::min_sv = 1.0e-10;

Levin::Levin(uint type1, uint col1, uint nsub1, double relative_tol1, uint n_split_rs1, uint Nthread)
{
    type = type1;
    col = col1;
    nsub = nsub1;
    relative_tol = relative_tol1;
    already_called = false;
    number_integrals = 0;
    n_split_rs = n_split_rs1;
    converged = 1e-4;
    N_thread_max = Nthread;
    setup(type);
}

Levin::~Levin()
{
    for (uint i = 0; i < number_integrals; i++)
    {
        gsl_spline_free(spline_integrand.at(i));
        gsl_interp_accel_free(acc_integrand.at(i));
    }
    for (uint j = 0; j < N_thread_max; j++)
    {
        if (number_of_modes != 0)
        {
            for (uint i = 0; i < number_of_modes; i++)
            {
                gsl_spline_free(spline_w_ell.at(j).at(i));
                gsl_interp_accel_free(acc_w_ell.at(j).at(i));
            }
            if (j == 0)
            {
                delete int_m_mode;
                delete int_n_mode;
            }
        }
    }
    if (type < 2)
    {
        delete int_k_single_bessel;
        delete int_ell_single_bessel;
    }
    else
    {
        delete int_k1_double_bessel;
        delete int_ell1_double_bessel;
        delete int_k2_double_bessel;
        delete int_ell2_double_bessel;
    }
}

void Levin::setup(uint type)
{
    if (type < 2)
    {
        d = 2;
        int_k_single_bessel = new double[N_thread_max];
        int_ell_single_bessel = new uint[N_thread_max];
    }
    else
    {
        d = 4;
        int_k1_double_bessel = new double[N_thread_max];
        int_ell1_double_bessel = new uint[N_thread_max];
        int_k2_double_bessel = new double[N_thread_max];
        int_ell2_double_bessel = new uint[N_thread_max];
    }
}

void Levin::update_Levin(uint type1, uint col1, uint nsub1, double relative_tol1, double converged1)
{
    type = type1;
    col = col1;
    nsub = nsub1;
    relative_tol = relative_tol1;
    converged = converged1;
    setup(type);
}

void Levin::init_w_ell(std::vector<double> ell, std::vector<std::vector<double>> w_ells)
{
    ell_w_ell = ell;
    for (uint j = 0; j < N_thread_max; j++)
    {
        spline_w_ell.push_back(std::vector<gsl_spline *>());
        acc_w_ell.push_back(std::vector<gsl_interp_accel *>());
        int_m_mode = new uint[N_thread_max];
        int_n_mode = new uint[N_thread_max];
        number_of_modes = w_ells.at(0).size();
        std::vector<double> y_value(ell.size());
        for (uint i = 0; i < number_of_modes; i++)
        {
            spline_w_ell.at(j).push_back(gsl_spline_alloc(gsl_interp_steffen, ell.size()));
            acc_w_ell.at(j).push_back(gsl_interp_accel_alloc());
        }
        for (uint a = 0; a < number_of_modes; a++)
        {
#pragma omp parallel for num_threads(N_thread_max)
            for (uint i = 0; i < ell.size(); i++)
            {
                y_value.at(i) = w_ells.at(i).at(a);
            }
            gsl_spline_init(spline_w_ell.at(j).at(a), &ell[0], &y_value[0], ell.size());
        }
        xmax_weight = ell.back();
        xmin_weight = ell.at(0);
    }
}

void Levin::init_integral(std::vector<double> x, std::vector<std::vector<double>> integrand, bool logx1, bool logy1)
{
    number_x_values = x.size();
    uint number_integrals_save = number_integrals;
    number_integrals = integrand.at(0).size();
    logx = logx1;
    if (!already_called)
    {
        for (uint i = 0; i < number_integrals; i++)
        {
            logy.push_back(logy1);
            slope.push_back(0);
            slope0.push_back(0);
        }
    }
    if (number_integrals_save != number_integrals)
    {
        for (uint i = 0; i < number_integrals_save; i++)
        {
            gsl_spline_free(spline_integrand.at(i));
            gsl_interp_accel_free(acc_integrand.at(i));
        }
        already_called = false;
        spline_integrand.clear();
        acc_integrand.clear();
        logy.clear();
        slope.clear();
        slope0.clear();
    }
    if (!already_called)
    {
        for (uint i = 0; i < number_integrals; i++)
        {
            spline_integrand.push_back(gsl_spline_alloc(gsl_interp_steffen, number_x_values));
            acc_integrand.push_back(gsl_interp_accel_alloc());
            logy.push_back(logy1);
            slope.push_back(0.0);
            slope0.push_back(0.0);
        }
        already_called = true;
    }
    if (number_of_modes != 0)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint a = 0; a < number_integrals; a++)
        {
            gsl_spline_free(spline_integrand.at(a));
            gsl_interp_accel_free(acc_integrand.at(a));
            spline_integrand.at(a) = gsl_spline_alloc(gsl_interp_steffen, number_x_values);
            acc_integrand.at(a) = gsl_interp_accel_alloc();
        }
    }
    for (uint i = 0; i < number_x_values; i++)
    {
        if (logx)
        {
            x.at(i) = log(x.at(i));
        }
    }

    if (logy.at(0))
    {
        for (uint a = 0; a < number_integrals; a++)
        {
            for (uint i = 0; i < number_x_values; i++)
            {
                if (integrand.at(i).at(a) <= 0.0)
                {
                    logy.at(a) = false;
                    break;
                }
            }
        }
    }
    for (uint a = 0; a < number_integrals; a++)
    {
        std::vector<double> y_value(number_x_values);
        for (uint i = 0; i < number_x_values; i++)
        {
            if (logy.at(a))
            {
                y_value.at(i) = log(integrand.at(i).at(a));
            }
            else
            {
                y_value.at(i) = integrand.at(i).at(a);
            }
        }
        gsl_spline_init(spline_integrand.at(a), &x[0], &y_value[0], number_x_values);
        slope.at(a) = gsl_spline_eval_deriv(spline_integrand.at(a), x.at(number_x_values - 1), acc_integrand.at(a));
        slope0.at(a) = gsl_spline_eval_deriv(spline_integrand.at(a), x.at(0), acc_integrand.at(a));
        if (slope.at(a) > 0)
        {
            slope.at(a) = 0;
        }
    }
    x_max = x.at(number_x_values - 1);
    x_min = x.at(0);
    if (number_of_modes != 0)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint a = 0; a < number_integrals; a++)
        {
            std::vector<double> new_x(2 * x.size());
            std::vector<double> new_y(2 * x.size());
            for (uint i = 0; i < 2 * x.size(); i++)
            {
                if (logx)
                {
                    new_x.at(i) = exp(log(ell_w_ell.at(0)) + (log(ell_w_ell.back()) - log(ell_w_ell.at(0))) / (2 * x.size() - 1) * i);
                }
                else
                {
                    new_x.at(i) = (ell_w_ell.at(0)) + ((ell_w_ell.back()) - (ell_w_ell.at(0))) / (2 * x.size() - 1) * i;
                }
            }
            new_y = get_integrand(new_x, a);
            gsl_spline_free(spline_integrand.at(a));
            gsl_interp_accel_free(acc_integrand.at(a));
            for (uint i = 0; i < 2 * x.size(); i++)
            {
                if (logx)
                {
                    new_x.at(i) = log(new_x.at(i));
                }
                if (logy.at(a))
                {
                    new_y.at(i) = log(new_y.at(i));
                }
            }
            spline_integrand.at(a) = gsl_spline_alloc(gsl_interp_steffen, 2 * number_x_values);
            acc_integrand.at(a) = gsl_interp_accel_alloc();
            gsl_spline_init(spline_integrand.at(a), &new_x[0], &new_y[0], 2 * number_x_values);
        }
    }
}

std::vector<double> Levin::get_w_ell(std::vector<double> ell, uint m_mode)
{
    std::vector<double> result(ell.size());
    for (uint i = 0; i < ell.size(); i++)
    {
        result.at(i) = gsl_spline_eval(spline_w_ell.at(0).at(m_mode), ell.at(i), acc_w_ell.at(0).at(m_mode));
    }
    return result;
}

std::vector<double> Levin::get_integrand(std::vector<double> x, uint j)
{
    std::vector<double> result(x.size());
    for (uint i = 0; i < x.size(); i++)
    {
        if (logx)
        {
            x.at(i) = log(x.at(i));
        }
        if (x.at(i) < x_max && x.at(i) > x_min)
        {
            result.at(i) = gsl_spline_eval(spline_integrand.at(j), x.at(i), acc_integrand.at(j));
        }
        else
        {
            if (x.at(i) >= x_max)
            {
                result.at(i) = gsl_spline_eval(spline_integrand.at(j), x_max, acc_integrand.at(j));
                result.at(i) += slope.at(j) * (x.at(i) - x_max);
            }
            else
            {
                result.at(i) = gsl_spline_eval(spline_integrand.at(j), x_min, acc_integrand.at(j));
                result.at(i) += slope0.at(j) * (x.at(i) - x_min);
            }
        }
        if (logy.at(j))
        {
            result.at(i) = exp(result.at(i));
        }
    }
    return result;
}

double Levin::call_integrand(double x, uint i)
{
    double result = 0;
    if (logx)
    {
        x = log(x);
    }
    result = gsl_spline_eval(spline_integrand.at(i), x, acc_integrand.at(i));
    if (logy.at(i))
    {
        result = exp(result);
    }
    return result;
}

void Levin::init_cov_R_space_Gaussian(std::vector<std::vector<std::vector<double>>> cov_k_space_Gaussian, std::vector<double> k, std::vector<double> r)
{
    cov_R_radii = r;
    sample_size = cov_k_space_Gaussian.at(0).at(0).size();
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        spline_cov_Gauss.push_back(gsl_spline_alloc(gsl_interp_steffen, k.size()));
        acc_cov_Gauss.push_back(gsl_interp_accel_alloc());
    }
    k_min = k.at(0);
    k_max = k.at(k.size() - 1);
    std::vector<double> y_value(k.size());
    for (uint i = 0; i < k.size(); i++)
    {
        k.at(i) = log(k.at(i));
    }
    for (uint a = 0; a < sample_size; a++)
    {
        for (uint b = 0; b < sample_size; b++)
        {
            for (uint i = 0; i < k.size(); i++)
            {
                y_value.at(i) = log(cov_k_space_Gaussian.at(i).at(a).at(b));
            }
            gsl_spline_init(spline_cov_Gauss.at(a * sample_size + b), &k[0], &y_value[0], k.size());
        }
    }
}

void Levin::init_cov_R_space_SSC(std::vector<std::vector<std::vector<double>>> cov_k_space_SSC, std::vector<double> k, std::vector<double> r)
{
    cov_R_radii = r;
    sample_size = cov_k_space_SSC.at(0).at(0).size();
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        spline_cov_SSC.push_back(gsl_spline_alloc(gsl_interp_steffen, k.size()));
        acc_cov_SSC.push_back(gsl_interp_accel_alloc());
    }
    k_min = k.at(0);
    k_max = k.at(k.size() - 1);
    std::vector<double> y_value(k.size());
    for (uint i = 0; i < k.size(); i++)
    {
        k.at(i) = log(k.at(i));
    }
    for (uint a = 0; a < sample_size; a++)
    {
        for (uint b = 0; b < sample_size; b++)
        {
            for (uint i = 0; i < k.size(); i++)
            {
                y_value.at(i) = cov_k_space_SSC.at(i).at(a).at(b);
            }
            gsl_spline_init(spline_cov_SSC.at(a * sample_size + b), &k[0], &y_value[0], k.size());
        }
    }
}

void Levin::init_cov_R_space_NonGaussian(std::vector<std::vector<std::vector<std::vector<double>>>> cov_k_space_NonGaussian, std::vector<double> k, std::vector<double> r)
{
    cov_R_radii = r;
    sample_size = cov_k_space_NonGaussian.at(0).at(0).at(0).size();
    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        spline_cov_non_Gauss.push_back(gsl_spline2d_alloc(T, k.size(), k.size()));
        acc_non_cov_Gauss_k1.push_back(gsl_interp_accel_alloc());
        acc_non_cov_Gauss_k2.push_back(gsl_interp_accel_alloc());
    }
    k_min = k.at(0);
    k_max = k.at(k.size() - 1);
    std::vector<double> z_value(k.size() * k.size());
    for (uint i = 0; i < k.size(); i++)
    {
        k.at(i) = log(k.at(i));
    }

    for (uint a = 0; a < sample_size; a++)
    {
        for (uint b = 0; b < sample_size; b++)
        {
            for (uint i = 0; i < k.size(); i++)
            {
                for (uint j = 0; j < k.size(); j++)
                {
                    z_value.at(i * k.size() + j) = log(cov_k_space_NonGaussian.at(i).at(j).at(a).at(b));
                }
            }
            gsl_spline2d_init(spline_cov_non_Gauss.at(a * sample_size + b), &z_value[0], &k[0], &k[0], k.size(), k.size());
        }
    }
}

double Levin::w_single(double x, double k, uint ell, uint i)
{
    gsl_set_error_handler_off();
    gsl_sf_result r;
    int status;
    if (type == 0)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_Jn_e(ell, x * k, &r);
            break;
        case 1:
            status = gsl_sf_bessel_Jn_e(ell + 1, x * k, &r);
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute cylindrical Bessel function for ell=" << ell << std::endl;
        }
    }
    if (type == 1)
    {
        switch (i)
        {
        case 0:
            // status = gsl_sf_bessel_jl_e(ell, x * k, &r);
            // break;
            return gsl_sf_bessel_jl(ell, x * k);
        case 1:
            // status = gsl_sf_bessel_jl_e(ell + 1, x * k, &r);
            // break;
            return gsl_sf_bessel_jl(ell + 1, x * k);
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell << std::endl;
        }
    }
    return r.val;
}

double Levin::w_double(double x, double k1, double k2, uint ell_1, uint ell_2, uint i)
{
    double result = 0.0;
    gsl_sf_result r;
    int status;
    if (type == 2)
    {
        switch (i)
        {
        case 0:
            status = gsl_sf_bessel_Jn_e(ell_1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2, x * k2, &r);
            result *= r.val;
            break;
        case 1:
            status = gsl_sf_bessel_Jn_e(ell_1 + 1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2, x * k2, &r);
            result *= r.val;
            break;
        case 2:
            status = gsl_sf_bessel_Jn_e(ell_1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2 + 1, x * k2, &r);
            result *= r.val;
            break;
        case 3:
            status = gsl_sf_bessel_Jn_e(ell_1 + 1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_Jn_e(ell_2 + 1, x * k2, &r);
            result *= r.val;
            break;
        default:
            return 0.0;
        }
        if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute cylindrical Bessel function for ell=" << ell_2 << std::endl;
        }
    }
    if (type == 3)
    {
        switch (i)
        {

        case 0:
            /**status = gsl_sf_bessel_jl_e(ell_1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2, x * k2, &r);
            result *= r.val;*/
            result = gsl_sf_bessel_jl(ell_2, x * k2) * gsl_sf_bessel_jl(ell_1, x * k1);
            break;
        case 1:
            /*status = gsl_sf_bessel_jl_e(ell_1 + 1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2, x * k2, &r);
            result *= r.val;*/
            result = gsl_sf_bessel_jl(ell_2, x * k2) * gsl_sf_bessel_jl(ell_1 + 1, x * k1);
            break;
        case 2:
            /**status = gsl_sf_bessel_jl_e(ell_1, x * k1, &r);
            result = r.val;
            status = gsl_sf_bessel_jl_e(ell_2 + 1, x * k2, &r);
            result *= r.val;**/
            result = gsl_sf_bessel_jl(ell_2 + 1, x * k2) * gsl_sf_bessel_jl(ell_1, x * k1);
            break;
        case 3:
            /**
                status = gsl_sf_bessel_jl_e(ell_1 + 1, x * k1, &r);
                result = r.val;
                status = gsl_sf_bessel_jl_e(ell_2 + 1, x * k2, &r);
                result *= r.val;*/
            result = gsl_sf_bessel_jl(ell_2 + 1, x * k2) * gsl_sf_bessel_jl(ell_1 + 1, x * k1);
            break;
        default:
            return 0.0;
        }
        /*if (status != GSL_SUCCESS)
        {
            std::cerr << "Failed to compute spherical Bessel function for ell=" << ell_2 << std::endl;
        }*/
    }
    return result;
}

double Levin::A_matrix_single(uint i, uint j, double x, double k, uint ell)
{
    switch (type)
    {
    case 0:
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell) / x;
        }
        if (i * j == 1)
        {
            return -(ell + 1.0) / x;
        }
        if (i < j)
        {
            return -k;
        }
        else
        {
            return k;
        }
    case 1:
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell) / x;
        }
        if (i * j == 1)
        {
            return -(ell + 2.0) / x;
        }
        if (i < j)
        {
            return -k;
        }
        else
        {
            return k;
        }
    default:
        return 0.0;
    }
}

double Levin::A_matrix_double(uint i, uint j, double x, double k1, double k2, uint ell_1, uint ell_2)
{
    switch (type)
    {
    case 2:
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return static_cast<double>(ell_1 + ell_2) / x;
        }
        if (i == 1 && j == 1)
        {
            return (static_cast<double>(ell_2) - static_cast<double>(ell_1) - 1.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2) - 1.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return -(ell_1 + ell_2 + 2.0) / x;
        }
        if ((i == 1 && j == 0) || (i == 3 && j == 2))
        {
            return k1;
        }
        if ((i == 2 && j == 0) || (i == 3 && j == 1))
        {
            return k2;
        }
        if ((i == 0 && j == 1) || (i == 2 && j == 3))
        {
            return -k1;
        }
        if ((i == 0 && j == 2) || (i == 1 && j == 3))
        {
            return -k2;
        }

    case 3:
        if (i + j == 3)
        {
            return 0.0;
        }
        if (i == 0 && j == 0)
        {
            return (ell_1 + ell_2) / x;
        }
        if (i == 1 && j == 1)
        {
            return -(static_cast<double>(ell_1) - static_cast<double>(ell_2) + 2.0) / x;
        }
        if (i == 2 && j == 2)
        {
            return (static_cast<double>(ell_1) - static_cast<double>(ell_2) - 2.0) / x;
        }
        if (i == 3 && j == 3)
        {
            return -(static_cast<double>(ell_1) + static_cast<double>(ell_2) + 4.0) / x;
        }
        if ((i == 1 && j == 0) || (i == 3 && j == 2))
        {
            return k1;
        }
        if ((i == 2 && j == 0) || (i == 3 && j == 1))
        {
            return k2;
        }
        if ((i == 0 && j == 1) || (i == 2 && j == 3))
        {
            return -k1;
        }
        if ((i == 0 && j == 2) || (i == 1 && j == 3))
        {
            return -k2;
        }
    default:
        return 0.0;
    }
}

std::vector<double> Levin::setNodes(double A, double B, uint col)
{
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    for (uint j = 0; j < n; j++)
    {
        x_j[j] = A + j * (B - A) / (n - 1);
    }
    return x_j;
}

double Levin::basis_function(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 1.0;
    }
    return pow((x - (A + B) / 2) / (B - A), m);
}

double Levin::basis_function_prime(double A, double B, double x, uint m)
{
    if (m == 0)
    {
        return 0.0;
    }
    if (m == 1)
    {
        return 1.0 / (B - A);
    }
    return m / (B - A) * pow((x - (A + B) / 2.) / (B - A), (m - 1));
}

std::vector<double> Levin::solve_LSE_single(double (*function)(double, void *), double A, double B, uint col, std::vector<double> x_j, double k, uint ell)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    if (type >= 2)
    {
        std::cerr << "Please check the type you want to integrate in the constructor (<2 required for this function)" << std::endl;
    }
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, (*function)(x_j[j], this));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_single(q, i, x_j[j], k, ell) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

std::vector<double> Levin::solve_LSE_double(double (*function)(double, void *), double A, double B, uint col, std::vector<double> x_j, double k1, double k2, uint ell_1, uint ell_2)
{
    uint n = (col + 1) / 2;
    n *= 2;
    gsl_vector *F_stacked = gsl_vector_alloc(d * n);
    gsl_vector *c = gsl_vector_alloc(d * n);
    if (type <= 1)
    {
        std::cerr << "Please check the type you want to integrate in the constructor (>1 required for this function)" << std::endl;
    }
    for (uint j = 0; j < d * n; j++)
    {
        if (j < n)
        {
            gsl_vector_set(F_stacked, j, (*function)(x_j[j], this));
        }
        else
        {
            gsl_vector_set(F_stacked, j, 0.0);
        }
    }
    gsl_matrix *matrix_G = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_set_zero(matrix_G);
    for (uint i = 0; i < d; i++)
    {
        for (uint j = 0; j < n; j++)
        {
            for (uint q = 0; q < d; q++)
            {
                for (uint m = 0; m < n; m++)
                {
                    double LSE_coeff = A_matrix_double(q, i, x_j[j], k1, k2, ell_1, ell_2) * basis_function(A, B, x_j[j], m);
                    if (q == i)
                    {
                        LSE_coeff += basis_function_prime(A, B, x_j[j], m);
                    }
                    gsl_matrix_set(matrix_G, i * n + j, q * n + m, LSE_coeff);
                }
            }
        }
    }
    gsl_matrix *U = gsl_matrix_alloc(d * n, d * n);
    gsl_matrix_memcpy(U, matrix_G);
    int s;
    gsl_permutation *P = gsl_permutation_alloc(d * n);
    gsl_linalg_LU_decomp(matrix_G, P, &s);
    gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
    int lu = gsl_linalg_LU_solve(matrix_G, P, F_stacked, c);
    if (lu) // in case solution via LU decomposition fails, proceed with SVD
    {
        gsl_matrix *V = gsl_matrix_alloc(d * n, d * n);
        gsl_vector *S = gsl_vector_alloc(d * n);
        gsl_vector *aux = gsl_vector_alloc(d * n);
        gsl_linalg_SV_decomp(U, V, S, aux);
        int i = d * n - 1;
        while (i > 0 && gsl_vector_get(S, i) < min_sv * gsl_vector_get(S, 0))
        {
            gsl_vector_set(S, i, 0.);
            --i;
        }
        gsl_linalg_SV_solve(U, V, S, F_stacked, c);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(aux);
    }
    std::vector<double> result(d * n);
    for (uint j = 0; j < d * n; j++)
    {
        result[j] = gsl_vector_get(c, j);
    }
    gsl_matrix_free(U);
    gsl_vector_free(F_stacked);
    gsl_vector_free(c);
    gsl_permutation_free(P);
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(matrix_G);
    return result;
}

double Levin::p(double A, double B, uint i, double x, uint col, std::vector<double> c)
{
    uint n = (col + 1) / 2;
    n *= 2;
    double result = 0.0;
    for (uint m = 0; m < n; m++)
    {
        result += c[i * n + m] * basis_function(A, B, x, m);
    }
    return result;
}

double Levin::integrate_single(double (*function)(double, void *), double A, double B, uint col, double k, uint ell)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    x_j = setNodes(A, B, col);
    c = solve_LSE_single(function, A, B, col, x_j, k, ell);
    for (uint i = 0; i < d; i++)
    {
        result += p(A, B, i, B, col, c) * w_single(B, k, ell, i) - p(A, B, i, A, col, c) * w_single(A, k, ell, i);
    }
    return result;
}

double Levin::integrate_double(double (*function)(double, void *), double A, double B, uint col, double k1, double k2, uint ell_1, uint ell_2)
{
    double result = 0.0;
    uint n = (col + 1) / 2;
    n *= 2;
    std::vector<double> x_j(n);
    std::vector<double> c(n);
    x_j = setNodes(A, B, col);
    c = solve_LSE_double(function, A, B, col, x_j, k1, k2, ell_1, ell_2);
    for (uint i = 0; i < d; i++)
    {
        result += p(A, B, i, B, col, c) * w_double(B, k1, k2, ell_1, ell_2, i) - p(A, B, i, A, col, c) * w_double(A, k1, k2, ell_1, ell_2, i);
    }
    return result;
}

double Levin::iterate_single(double (*function)(double, void *), double A, double B, uint col, double k, uint ell, uint smax, bool verbose)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_single(function, A, B, col / 2, k, ell);
    double I_full = integrate_single(function, A, B, col, k, ell);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*  if (verbose)
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(relative_tol * abs(result), tol_abs))
        {
            // if (verbose)
            // {
            //     std::cerr << "converged!" << std::endl;
            // }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                }
                return result;
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        I_half = integrate_single(function, x_sub.at(i - 1), x_sub.at(i), col / 2, k, ell);
        I_full = integrate_single(function, x_sub.at(i - 1), x_sub.at(i), col, k, ell);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_single(function, x_sub.at(i), x_sub.at(i + 1), col / 2, k, ell);
        I_full = integrate_single(function, x_sub.at(i), x_sub.at(i + 1), col, k, ell);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

double Levin::iterate_double(double (*function)(double, void *), double A, double B, uint col, double k1, double k2, uint ell_1, uint ell_2, uint smax, bool verbose)
{
    std::vector<double> intermediate_results;
    if (B - A < min_interval)
    {
        return 0.0;
    }
    double borders[2] = {A, B};
    std::vector<double> x_sub(borders, borders + 2);
    double I_half = integrate_double(function, A, B, col / 2, k1, k2, ell_1, ell_2);
    double I_full = integrate_double(function, A, B, col, k1, k2, ell_1, ell_2);
    uint sub = 1;
    double previous = I_half;
    std::vector<double> approximations(1, I_full);
    std::vector<double> error_estimates(1, fabs(I_full - I_half));
    double result = I_full;
    while (sub <= smax + 1)
    {
        result = 0.0;
        for (uint i = 0; i < approximations.size(); i++)
        {
            result += approximations.at(i);
        }
        /*if (verbose)
        {
            std::cerr << "estimate: " << std::scientific << result << std::endl
                      << sub << " subintervals: " << std::endl;
            for (uint i = 0; i < approximations.size(); ++i)
            {
                std::cerr << "[" << x_sub[i] << "," << x_sub[i + 1] << "]: " << approximations[i] << " (" << error_estimates[i] << ")" << std::endl;
                std::cerr << std::endl;
            }
        }*/
        intermediate_results.push_back(result);
        if (abs(result - previous) <= GSL_MAX(relative_tol * abs(result), tol_abs))
        {
            if (verbose)
            {
                std::cerr << "converged!" << std::endl;
            }
            return result;
        }
        previous = result;
        sub++;
        uint i = 1;
        while (true)
        {
            i = findMax(error_estimates) + 1;
            if (error_estimates[i - 1] < 0)
            {
                if (verbose)
                {
                    std::cerr << "subintervals too narrow for further bisection!" << std::endl;
                    return result;
                }
            }
            if (x_sub[i] - x_sub[i - 1] > min_interval)
            {
                break;
            }
            error_estimates.at(i - 1) = -1.0;
        }
        x_sub.insert(x_sub.begin() + i, (x_sub.at(i - 1) + x_sub.at(i)) / 2.);
        I_half = integrate_double(function, x_sub.at(i - 1), x_sub.at(i), col / 2, k1, k2, ell_1, ell_2);
        I_full = integrate_double(function, x_sub.at(i - 1), x_sub.at(i), col, k1, k2, ell_1, ell_2);
        approximations.at(i - 1) = I_full;
        error_estimates.at(i - 1) = fabs(I_full - I_half);
        I_half = integrate_double(function, x_sub.at(i), x_sub.at(i + 1), col / 2, k1, k2, ell_1, ell_2);
        I_full = integrate_double(function, x_sub.at(i), x_sub.at(i + 1), col, k1, k2, ell_1, ell_2);
        approximations.insert(approximations.begin() + i, I_full);
        error_estimates.insert(error_estimates.begin() + i, fabs(I_full - I_half));
    }
    if (verbose)
    {
        std::cerr << "maximum number of subintervals reached!" << std::endl;
    }
    return result;
}

uint Levin::findMax(const std::vector<double> vec)
{
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

double Levin::levin_integrate_bessel_single(double (*function)(double, void *), double k, uint ell, double a, double b)
{
    return iterate_single(function, a, b, col, k, ell, nsub, false);
}

double Levin::levin_integrate_bessel_double(double (*function)(double, void *), double k1, double k2, uint ell_1, uint ell_2, double a, double b)
{
    return iterate_double(function, a, b, col, k1, k2, ell_1, ell_2, nsub, false);
}

double Levin::integrand(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    double result = 0.0;
    if (lp->logx)
    {
        x = log(x);
    }
    if (x <= lp->x_max && x >= lp->x_min)
    {
        result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        if (lp->logy[lp->int_index_integral[tid]])
        {
            result = exp(result);
        }
    }
    return result;
}

double Levin::single_bessel_integrand(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    double result = 0.0;
    if (lp->logx)
    {
        x = log(x);
        if (x <= lp->x_max && x >= lp->x_min)
        {
            result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        }
        x = exp(x);
    }
    else
    {
        if (x <= lp->x_max && x >= lp->x_min)
        {
            result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        }
    }
    if (lp->logy[lp->int_index_integral[tid]])
    {
        result = exp(result);
    }
    double bessel = 0.0;
    if (lp->type == 0)
    {
        bessel = gsl_sf_bessel_Jn(lp->int_ell_single_bessel[tid], x * lp->int_k_single_bessel[tid]);
    }
    else
    {
        bessel = gsl_sf_bessel_jl(lp->int_ell_single_bessel[tid], x * lp->int_k_single_bessel[tid]);
    }
    return result * bessel;
}

std::vector<double> Levin::single_bessel(double k, uint ell, double a, double b)
{

    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(number_integrals);
    if (k * b > 1000)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i = 0; i < number_integrals; i++)
        {
            uint tid = omp_get_thread_num();
            int_index_integral[tid] = i;
            result.at(i) = iterate_single(integrand, a, b, col, k, ell, nsub, false);
        }
    }
    else
    {
        gsl_error_handler_t *old_handler = gsl_set_error_handler_off();
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i = 0; i < number_integrals; i++)
        {
            uint tid = omp_get_thread_num();
            int_ell_single_bessel[tid] = ell;
            int_k_single_bessel[tid] = k;
            int_index_integral[tid] = i;
            uint N_sum = uint(n_split_rs / 10);
            if (N_sum < 2)
            {
                N_sum = 2;
            }
            result.at(i) = 0.0;
            for (uint j = 0; j < N_sum; j++)
            {
                double al = exp(log(a) + (log(b) - log(a)) / (N_sum)*j);
                double bl = exp(log(a) + (log(b) - log(a)) / (N_sum) * (j + 1));
                result.at(i) += gslIntegratecquad(single_bessel_integrand, al, bl);
            }
        }
        gsl_set_error_handler(old_handler);
    }
    delete int_index_integral;
    return result;
}

double Levin::double_bessel_integrand(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    double result = 0.0;
    if (lp->logx)
    {
        x = log(x);
        if (x <= lp->x_max && x >= lp->x_min)
        {
            result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        }
        x = exp(x);
    }
    else
    {
        if (x <= lp->x_max && x >= lp->x_min)
        {

            result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        }
    }
    if (lp->logy[lp->int_index_integral[tid]])
    {
        result = exp(result);
    }
    double bessel = 0.0;
    if (lp->type == 2)
    {
        bessel = gsl_sf_bessel_Jn(lp->int_ell1_double_bessel[tid], x * lp->int_k1_double_bessel[tid]) * gsl_sf_bessel_Jn(lp->int_ell2_double_bessel[tid], x * lp->int_k2_double_bessel[tid]);
    }
    else
    {
        bessel = gsl_sf_bessel_jl(lp->int_ell1_double_bessel[tid], x * lp->int_k1_double_bessel[tid]) * gsl_sf_bessel_jl(lp->int_ell2_double_bessel[tid], x * lp->int_k2_double_bessel[tid]);
    }
    return result * bessel;
}

std::vector<double> Levin::double_bessel(double k1, double k2, uint ell_1, uint ell_2, double a, double b)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(number_integrals);
    if (k1 * b > 1000 && k2 * b > 1000)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i = 0; i < number_integrals; i++)
        {
            uint tid = omp_get_thread_num();
            int_index_integral[tid] = i;
            result.at(i) = iterate_double(integrand, a, b, col, k1, k2, ell_1, ell_2, nsub, false);
        }
    }
    else
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i = 0; i < number_integrals; i++)
        {
            uint tid = omp_get_thread_num();
            int_ell1_double_bessel[tid] = ell_1;
            int_k1_double_bessel[tid] = k1;
            int_ell2_double_bessel[tid] = ell_2;
            int_k2_double_bessel[tid] = k2;
            int_index_integral[tid] = i;
            result.at(i) = 0.0;
            uint N_sum = n_split_rs;
            for (uint j = 0; j < N_sum; j++)
            {
                double al = exp(log(a) + (log(b) - log(a)) / (N_sum)*j);
                double bl = exp(log(a) + (log(b) - log(a)) / (N_sum) * (j + 1));
                result.at(i) += gslIntegratecquad(double_bessel_integrand, al, bl);
            }
        }
    }
    delete int_index_integral;
    return result;
}

std::vector<double> Levin::double_bessel_many_args(std::vector<double> k1, double k2, uint ell_1, uint ell_2, double a, double b)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(k1.size());
#pragma omp parallel for schedule(dynamic) num_threads(N_thread_max)
    for (uint i_k1 = 0; i_k1 < k1.size(); i_k1++)
    {
        if (k1.at(i_k1) * b > 1000 && k2 * b > 1000)
        {
            uint tid = omp_get_thread_num();
            int_index_integral[tid] = tid;
            result.at(i_k1) = iterate_double(integrand, a, b, col, k1.at(i_k1), k2, ell_1, ell_2, nsub, false);
        }
        else
        {
            uint tid = omp_get_thread_num();
            int_ell1_double_bessel[tid] = ell_1;
            int_k1_double_bessel[tid] = k1.at(i_k1);
            int_ell2_double_bessel[tid] = ell_2;
            int_k2_double_bessel[tid] = k2;
            int_index_integral[tid] = tid;
            result.at(i_k1) = 0.0;
            uint N_sum = n_split_rs;
            for (uint j = 0; j < N_sum; j++)
            {
                double al = exp(log(a) + (log(b) - log(a)) / (N_sum)*j);
                double bl = exp(log(a) + (log(b) - log(a)) / (N_sum) * (j + 1));
                result.at(i_k1) += gslIntegratecquad(double_bessel_integrand, al, bl);
            }
        }
    }
    delete int_index_integral;
    return result;
}

std::vector<double> Levin::single_bessel_many_args(std::vector<double> k, uint ell, double a, double b)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(k.size());
#pragma omp parallel for num_threads(N_thread_max)
    for (uint ik = 0; ik < k.size(); ik++)
    {
        if (k.at(ik) * b > 1000)
        {
            uint tid = omp_get_thread_num();
            int_index_integral[tid] = tid;
            result.at(ik) = iterate_single(integrand, a, b, col, k.at(ik), ell, nsub, false);
        }
        else
        {
            uint tid = omp_get_thread_num();
            int_ell_single_bessel[tid] = ell;
            int_k_single_bessel[tid] = k.at(ik);
            int_index_integral[tid] = tid;
            uint N_sum = uint(n_split_rs / 10);
            if (N_sum < 2)
            {
                N_sum = 2;
            }
            result.at(ik) = 0.0;
            for (uint j = 0; j < N_sum; j++)
            {
                double al = exp(log(a) + (log(b) - log(a)) / (N_sum)*j);
                double bl = exp(log(a) + (log(b) - log(a)) / (N_sum) * (j + 1));
                result.at(ik) += gslIntegratecquad(single_bessel_integrand, al, bl);
            }
        }
    }
    delete int_index_integral;
    return result;
}

double Levin::cquad_integrand(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    double result = 0.0;
    if (lp->logx)
    {
        x = log(x);
        result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
        x = exp(x);
    }
    else
    {
        result = gsl_spline_eval(lp->spline_integrand.at(lp->int_index_integral[tid]), x, lp->acc_integrand.at(lp->int_index_integral[tid]));
    }
    if (lp->logy[lp->int_index_integral[tid]])
    {
        result = exp(result);
    }
    return result;
}

std::vector<double> Levin::cquad_integrate(std::vector<double> limits)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(number_integrals);
#pragma omp parallel for num_threads(N_thread_max)
    for (uint i = 0; i < number_integrals; i++)
    {
        uint tid = omp_get_thread_num();
        int_index_integral[tid] = i;
        result.at(i) = 0.0;
        for (uint j = 0; j < limits.size() - 1; j++)
        {
            result.at(i) += gslIntegratecquad(cquad_integrand, limits.at(j), limits.at(j + 1));
        }
    }
    delete int_index_integral;
    return result;
}

double Levin::cquad_integrand_single_well(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    if (lp->xmax_weight > x && lp->xmin_weight < x)
    {
        return lp->call_integrand(x, lp->int_index_integral[tid]) * gsl_spline_eval(lp->spline_w_ell.at(tid).at(lp->int_m_mode[tid]), x, lp->acc_w_ell.at(tid).at(lp->int_m_mode[tid]));
    }
    else
    {
        return 0.;
    }
}

std::vector<double> Levin::cquad_integrate_single_well(std::vector<double> limits, uint m_mode)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(number_integrals);
#pragma omp parallel for schedule(dynamic) num_threads(N_thread_max)
    for (uint i = 0; i < number_integrals; i++)
    {
        uint tid = omp_get_thread_num();
        int_index_integral[tid] = i;
        int_m_mode[tid] = m_mode;
        result.at(i) = 0.0;
        for (uint j = 0; j < limits.size() - 1; j++)
        {
            double result_aux = gslIntegratecquad(cquad_integrand_single_well, limits.at(j), limits.at(j + 1));
            result.at(i) += result_aux;
            if (abs(result_aux / result.at(i)) < converged && limits.at(j) > 1e3)
            {
                break;
            }
        }
    }
    delete int_index_integral;
    return result;
}

double Levin::cquad_integrand_double_well(double x, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    if (lp->xmax_weight > x && lp->xmin_weight < x)
    {
        return lp->call_integrand(x, lp->int_index_integral[tid]) * gsl_spline_eval(lp->spline_w_ell.at(tid).at(lp->int_m_mode[tid]), x, lp->acc_w_ell.at(tid).at(lp->int_m_mode[tid])) * gsl_spline_eval(lp->spline_w_ell.at(tid).at(lp->int_n_mode[tid]), x, lp->acc_w_ell.at(tid).at(lp->int_n_mode[tid]));
    }
    else
    {
        return 0.0;
    }
}

std::vector<double> Levin::cquad_integrate_double_well(std::vector<double> limits, uint m_mode, uint n_mode)
{
    int_index_integral = new uint[N_thread_max];
    std::vector<double> result(number_integrals);
#pragma omp parallel for schedule(dynamic) num_threads(N_thread_max)
    for (uint i = 0; i < number_integrals; i++)
    {
        uint tid = omp_get_thread_num();
        int_index_integral[tid] = i;
        int_m_mode[tid] = m_mode;
        int_n_mode[tid] = n_mode;

        for (uint j = 0; j < limits.size() - 1; j++)
        {
            double result_aux = gslIntegratecquad(cquad_integrand_double_well, limits.at(j), limits.at(j + 1));
            result.at(i) += result_aux;
            if (abs(result_aux / result.at(i)) < converged && limits.at(j) > 5e3)
            {
                break;
            }
        }
    }
    delete int_index_integral;
    return result;
}

double Levin::cov_R_Gaussian_integrand(double k, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    return k * exp(gsl_spline_eval(lp->spline_cov_Gauss.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid]), log(k), lp->acc_cov_Gauss.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid])));
}

double Levin::cov_R_SSC_integrand(double k, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    return k * gsl_spline_eval(lp->spline_cov_SSC.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid]), log(k), lp->acc_cov_SSC.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid]));
}
double Levin::cov_R_NonGaussian_inner_integrand(double k, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    return k * exp(gsl_spline2d_eval(lp->spline_cov_non_Gauss.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid]), log(lp->int_cov_R_non_Gauss_outer_k[tid]), log(k), lp->acc_non_cov_Gauss_k1.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid]), lp->acc_non_cov_Gauss_k2.at(lp->int_cov_R_m_bin[tid] * lp->sample_size + lp->int_cov_R_n_bin[tid])));
}

double Levin::cov_R_NonGaussian_outer_integrand(double k, void *p)
{
    uint tid = omp_get_thread_num();
    Levin *lp = static_cast<Levin *>(p);
    lp->int_cov_R_non_Gauss_outer_k[tid] = k;
    return k * lp->levin_integrate_bessel_double(cov_R_NonGaussian_inner_integrand, lp->cov_R_radii.at(lp->int_cov_R_i_R[tid]), lp->cov_R_radii.at(lp->int_cov_R_j_R[tid]), lp->int_cov_R_ell1[tid], lp->int_cov_R_ell2[tid], lp->k_min, lp->k_max);
}

std::vector<std::vector<std::vector<std::vector<double>>>> Levin::cov_R_get_gauss(bool cross, uint ell_1, uint ell_2)
{
    int_cov_R_m_bin = new uint[N_thread_max];
    int_cov_R_n_bin = new uint[N_thread_max];
    std::vector<std::vector<std::vector<std::vector<double>>>> result(cov_R_radii.size(), std::vector<std::vector<std::vector<double>>>(cov_R_radii.size(), std::vector<std::vector<double>>(sample_size, std::vector<double>(sample_size, 0.0))));
    if (!cross)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = i_R; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = m; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        result.at(i_R).at(j_R).at(m).at(n) = levin_integrate_bessel_double(cov_R_Gaussian_integrand, cov_R_radii.at(i_R), cov_R_radii.at(j_R), ell_1, ell_2, k_min, k_max);
                        result.at(j_R).at(i_R).at(m).at(n) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(i_R).at(j_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(j_R).at(i_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = 0; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = 0; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        result.at(i_R).at(j_R).at(m).at(n) = levin_integrate_bessel_double(cov_R_Gaussian_integrand, cov_R_radii.at(i_R), cov_R_radii.at(j_R), ell_1, ell_2, k_min, k_max);
                    }
                }
            }
        }
    }
    delete int_cov_R_m_bin;
    delete int_cov_R_n_bin;
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        gsl_spline_free(spline_cov_Gauss.at(i));
        gsl_interp_accel_free(acc_cov_Gauss.at(i));
    }
    return result;
}

std::vector<std::vector<std::vector<std::vector<double>>>> Levin::cov_R_get_ssc(bool cross, uint ell_1, uint ell_2)
{
    int_cov_R_m_bin = new uint[N_thread_max];
    int_cov_R_n_bin = new uint[N_thread_max];
    std::vector<std::vector<std::vector<std::vector<double>>>> result(cov_R_radii.size(), std::vector<std::vector<std::vector<double>>>(cov_R_radii.size(), std::vector<std::vector<double>>(sample_size, std::vector<double>(sample_size, 0.0))));
    if (!cross)
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = i_R; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = m; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        result.at(i_R).at(j_R).at(m).at(n) = levin_integrate_bessel_double(cov_R_SSC_integrand, cov_R_radii.at(i_R), cov_R_radii.at(j_R), ell_1, ell_2, k_min, k_max);
                        result.at(j_R).at(i_R).at(m).at(n) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(i_R).at(j_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(j_R).at(i_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for num_threads(N_thread_max)
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = 0; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = 0; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        result.at(i_R).at(j_R).at(m).at(n) = levin_integrate_bessel_double(cov_R_SSC_integrand, cov_R_radii.at(i_R), cov_R_radii.at(j_R), ell_1, ell_2, k_min, k_max);
                    }
                }
            }
        }
    }
    delete int_cov_R_m_bin;
    delete int_cov_R_n_bin;
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        gsl_spline_free(spline_cov_SSC.at(i));
        gsl_interp_accel_free(acc_cov_SSC.at(i));
    }
    return result;
}

std::vector<std::vector<std::vector<std::vector<double>>>> Levin::cov_R_get_nongauss(bool cross, uint ell_1, uint ell_2)
{
    int_cov_R_m_bin = new uint[N_thread_max];
    int_cov_R_n_bin = new uint[N_thread_max];
    int_cov_R_i_R = new uint[N_thread_max];
    int_cov_R_j_R = new uint[N_thread_max];
    int_cov_R_ell1 = new uint[N_thread_max];
    int_cov_R_ell2 = new uint[N_thread_max];
    int_cov_R_non_Gauss_outer_k = new double[N_thread_max];
    std::vector<std::vector<std::vector<std::vector<double>>>> result(cov_R_radii.size(), std::vector<std::vector<std::vector<double>>>(cov_R_radii.size(), std::vector<std::vector<double>>(sample_size, std::vector<double>(sample_size, 0.0))));
    if (!cross)
    {
        // #pragma omp parallel for
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = i_R; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = m; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        int_cov_R_ell1[tid] = ell_1;
                        int_cov_R_ell2[tid] = ell_2;
                        int_cov_R_i_R[tid] = i_R;
                        int_cov_R_j_R[tid] = j_R;
                        result.at(i_R).at(j_R).at(m).at(n) = gslIntegratecquad(cov_R_NonGaussian_outer_integrand, k_min, k_max);
                        result.at(j_R).at(i_R).at(m).at(n) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(i_R).at(j_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                        result.at(j_R).at(i_R).at(n).at(m) = result.at(i_R).at(j_R).at(m).at(n);
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for
        for (uint i_R = 0; i_R < cov_R_radii.size(); i_R++)
        {
            for (uint j_R = 0; j_R < cov_R_radii.size(); j_R++)
            {
                for (uint m = 0; m < sample_size; m++)
                {
                    for (uint n = 0; n < sample_size; n++)
                    {
                        uint tid = omp_get_thread_num();
                        int_cov_R_m_bin[tid] = m;
                        int_cov_R_n_bin[tid] = n;
                        int_cov_R_ell1[tid] = ell_1;
                        int_cov_R_ell2[tid] = ell_2;
                        int_cov_R_i_R[tid] = i_R;
                        int_cov_R_j_R[tid] = j_R;
                        result.at(i_R).at(j_R).at(m).at(n) = gslIntegratecquad(cov_R_NonGaussian_outer_integrand, k_min, k_max);
                    }
                }
            }
        }
    }
    delete int_cov_R_m_bin;
    delete int_cov_R_n_bin;
    delete int_cov_R_i_R;
    delete int_cov_R_j_R;
    delete int_cov_R_ell1;
    delete int_cov_R_ell2;
    delete int_cov_R_non_Gauss_outer_k;
    for (uint i = 0; i < gsl_pow_2(sample_size); i++)
    {
        gsl_spline2d_free(spline_cov_non_Gauss.at(i));
        gsl_interp_accel_free(acc_non_cov_Gauss_k1.at(i));
        gsl_interp_accel_free(acc_non_cov_Gauss_k1.at(i));
    }
    return result;
}

double Levin::gslIntegratecquad(double (*fc)(double, void *), double a, double b)
{
    double tiny = 0.0;
    double tol = relative_tol;
    gsl_function gf;
    gf.function = fc;
    gf.params = this;
    double e, y;
    const uint n = 64;
    gsl_integration_cquad_workspace *w = gsl_integration_cquad_workspace_alloc(n);
    gsl_integration_cquad(&gf, a, b, tiny, tol, w, &y, &e, NULL);
    gsl_integration_cquad_workspace_free(w);
    return y;
}