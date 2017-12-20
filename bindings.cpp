#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include "math.hpp"

namespace py = pybind11;

struct Foo {
    int operator()() const {
        return 4;
    }
};

struct Bar {
    int operator()(std::function<int()> const &f) const {

        Matrix<double> m{3,3};
        std::cout << blaze::det(m) << '\n';
        
        std::cout << f() << '\n';
        return f();
    }
};

PYBIND11_MODULE(cns, mod) {
    py::class_<Foo>(mod, "Foo")
        .def(py::init<>())
        .def("__call__", &Foo::operator());

    py::class_<Bar>(mod, "Bar")
        .def(py::init<>())
        .def("__call__", &Bar::operator());
}
