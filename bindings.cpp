#include "math.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

namespace py = pybind11;

/// Bind a vector of given type and name to Python.
template <typename VT, typename Mod>
void bindVector(Mod &mod, std::string const &name) {
    py::class_<VT>(mod, name.c_str())
        .def(py::init([](std::size_t const size){ return VT(size); }))
        .def("__getitem__", py::overload_cast<std::size_t>(&VT::operator[]))
        .def("__setitem__", [](VT &vec, std::size_t const i,
                               typename VT::ElementType const x) {
                 vec[i] = x;
             })
        .def("__iter__", [](VT &vec) {
                return py::make_iterator(vec.begin(), vec.end());
            })
        .def("__len__", &VT::size)
        .def("__repr__", [](VT const &vec) {
                std::ostringstream oss;
                oss << vec;
                return oss.str();
            })
        ;    
}

PYBIND11_MODULE(cns, mod) {
    bindVector<Vector<double>>(mod, "DVector");
    bindVector<Vector<std::complex<double>>>(mod, "CDVector");
}
