#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

#include "./../src/levin.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(levin, m)
{
     m.doc() = "Compute integrals with Levin's method.";
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
         .def("single_bessel_many_args", &Levin::single_bessel_many_args,
              "k"_a, "ell"_a, "a"_a, "b"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate", &Levin::cquad_integrate,
              "limits"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_single_well", &Levin::cquad_integrate_single_well,
              "limits"_a, "m_mode"_a,
              py::call_guard<py::gil_scoped_release>())
         .def("cquad_integrate_double_well", &Levin::cquad_integrate_double_well,
              "limits"_a, "m_mode"_a, "n_mode"_a,
              py::call_guard<py::gil_scoped_release>());
}