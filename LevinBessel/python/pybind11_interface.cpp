#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

#include "./../src/levin.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;


typedef py::array_t<double, py::array::c_style | py::array::forcecast> darray;

// Piecewise cubic Hermite interpolant (in u = ln x) through (lx, Y[:, c]) with
// derivatives YP[:, c], continued linearly beyond lx with slope0 (below) and
// slope (above), evaluated at le; exp of it for the columns with positive[c]
// (interpolation of ln y). With derivative, also the derivatives with
// respect to u from the right and from the left (of exp(...) if positive):
// they differ only at the nodes of lx. This is the evaluation part of
// CovELLSpace._integrand_at (which prepares the arguments), done in parallel
// over the evaluation points.
py::tuple evaluate_hermite(darray lx_in, darray Y_in, darray YP_in, darray slope_in, darray slope0_in,
                           py::array_t<bool, py::array::c_style | py::array::forcecast> positive_in,
                           darray le_in, bool derivative, int n_threads)
{
     const ssize_t n = lx_in.shape(0), nc = Y_in.shape(1), m = le_in.shape(0);
     if (n < 2 || Y_in.shape(0) != n || YP_in.shape(0) != n || YP_in.shape(1) != nc
         || slope_in.shape(0) != nc || slope0_in.shape(0) != nc || positive_in.shape(0) != nc)
     {
          throw std::invalid_argument("evaluate_hermite: inconsistent shapes");
     }
     darray values({m, nc});
     darray d_right(derivative ? std::vector<ssize_t>{m, nc} : std::vector<ssize_t>{0, nc});
     darray d_left(derivative ? std::vector<ssize_t>{m, nc} : std::vector<ssize_t>{0, nc});
     const double *lx = lx_in.data(), *Y = Y_in.data(), *YP = YP_in.data(), *le = le_in.data();
     const double *slope = slope_in.data(), *slope0 = slope0_in.data();
     const bool *positive = positive_in.data();
     double *v = values.mutable_data();
     double *dr = derivative ? d_right.mutable_data() : nullptr;
     double *dl = derivative ? d_left.mutable_data() : nullptr;
     {
          py::gil_scoped_release release;
#pragma omp parallel for num_threads(std::max(n_threads, 1)) schedule(static)
          for (ssize_t e = 0; e < m; e++)
          {
               const double u = le[e];
               double *ve = v + e*nc;
               // values (the interval to the right at the nodes, as numpy.searchsorted(side='right'))
               const bool above = u >= lx[n - 1], below = u < lx[0];
               ssize_t i = 0;
               double h = 1.0, t = 0.0;
               if (!above && !below)
               {
                    i = std::min<ssize_t>(std::max<ssize_t>(std::upper_bound(lx, lx + n, u) - lx - 1, 0), n - 2);
                    h = lx[i + 1] - lx[i];
                    t = (u - lx[i])/h;
               }
               const double *Yi = Y + i*nc, *Yj = Y + (i + 1)*nc, *Pi = YP + i*nc, *Pj = YP + (i + 1)*nc;
               if (above)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = Y[(n - 1)*nc + c] + slope[c]*(u - lx[n - 1]);
               }
               else if (below)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = Y[c] + slope0[c]*(u - lx[0]);
               }
               else
               {
                    const double a0 = 2*t*t*t - 3*t*t + 1, a1 = 3*t*t - 2*t*t*t;
                    const double b0 = h*(t*t*t - 2*t*t + t), b1 = h*(t*t*t - t*t);
                    for (ssize_t c = 0; c < nc; c++)
                         ve[c] = (a0*Yi[c] + a1*Yj[c]) + (b0*Pi[c] + b1*Pj[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         ve[c] = std::exp(ve[c]);
               if (!derivative)
                    continue;
               // derivative from the right
               double *dre = dr + e*nc;
               if (above)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = slope[c];
               }
               else if (below)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = slope0[c];
               }
               else
               {
                    const double a0 = 6*(t*t - t)/h, a1 = 6*(t - t*t)/h;
                    const double b0 = 3*t*t - 4*t + 1, b1 = 3*t*t - 2*t;
                    for (ssize_t c = 0; c < nc; c++)
                         dre[c] = (a0*Yi[c] + a1*Yj[c]) + (b0*Pi[c] + b1*Pj[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         dre[c] *= ve[c];
               // derivative from the left: differs only at the nodes
               double *dle = dl + e*nc;
               if (!std::binary_search(lx, lx + n, u))
               {
                    std::copy(dre, dre + nc, dle);
                    continue;
               }
               const bool above_l = u > lx[n - 1], below_l = u <= lx[0];
               if (above_l)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = slope[c];
               }
               else if (below_l)
               {
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = slope0[c];
               }
               else
               {
                    const ssize_t k = std::min<ssize_t>(std::max<ssize_t>(std::lower_bound(lx, lx + n, u) - lx - 1, 0), n - 2);
                    const double hk = lx[k + 1] - lx[k], tk = (u - lx[k])/hk;
                    const double a0 = 6*(tk*tk - tk)/hk, a1 = 6*(tk - tk*tk)/hk;
                    const double b0 = 3*tk*tk - 4*tk + 1, b1 = 3*tk*tk - 2*tk;
                    const double *Yk = Y + k*nc, *Yl = Y + (k + 1)*nc, *Pk = YP + k*nc, *Pl = YP + (k + 1)*nc;
                    for (ssize_t c = 0; c < nc; c++)
                         dle[c] = (a0*Yk[c] + a1*Yl[c]) + (b0*Pk[c] + b1*Pl[c]);
               }
               for (ssize_t c = 0; c < nc; c++)
                    if (positive[c])
                         dle[c] *= ve[c];
          }
     }
     if (!derivative)
          return py::make_tuple(values);
     return py::make_tuple(values, d_right, d_left);
}

PYBIND11_MODULE(levin, m)
{
     m.doc() = "Compute integrals with Levin's method.";
     m.def("evaluate_hermite", &evaluate_hermite,
           "lx"_a, "Y"_a, "YP"_a, "slope"_a, "slope0"_a, "positive"_a, "le"_a, "derivative"_a, "n_threads"_a);
     py::class_<Levin>(m, "Levin")
         .def(py::init<uint, uint, uint, double, uint, uint>(),
              "type1"_a, "col1"_a, "nsub1"_a, "relative_tol1"_a, "n_split_rs1"_a, "Nthread"_a) // Keyword arguments
         .def("update_Levin", &Levin::update_Levin,
              "type1"_a, "col1"_a, "nsub1"_a, "relative_tol1"_a, "converged1"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("init_integral", &Levin::init_integral,
              "x"_a, "integrand"_a, "logx1"_a, "logy1"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("init_w_ell", &Levin::init_w_ell,
              "ell"_a, "w_ell"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_w_ell", &Levin::get_w_ell,
              "ell"_a, "mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("single_bessel", &Levin::single_bessel,
              "k"_a, "ell"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("double_bessel", &Levin::double_bessel,
              "k1"_a, "k2"_a, "ell_1"_a, "ell_2"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("double_bessel_many_args", &Levin::double_bessel_many_args,
              "k1"_a, "k2"_a, "ell_1"_a, "ell_2"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("get_integrand",&Levin::get_integrand,
              "x"_a, "j"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("set_kb_levin_min", &Levin::set_kb_levin_min, "kb"_a)
         .def("single_bessel_many_args", &Levin::single_bessel_many_args,
              "k"_a, "ell"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("single_bessel_many_args_diagonal", &Levin::single_bessel_many_args_diagonal,
               "k"_a, "ell"_a, "a"_a, "b"_a,
               py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate", &Levin::cquad_integrate,
              "limits"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_single_well", &Levin::cquad_integrate_single_well,
              "limits"_a, "m_mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_single_well_subset", &Levin::cquad_integrate_single_well_subset,
              "limits"_a, "m_modes"_a, "indices"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_double_well", &Levin::cquad_integrate_double_well,
              "limits"_a, "m_mode"_a, "n_mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_double_well_subset", &Levin::cquad_integrate_double_well_subset,
              "limits"_a, "m_modes"_a, "n_modes"_a, "indices"_a,
              py::call_guard<py::gil_scoped_release>());
}