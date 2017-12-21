#include <functional>

#include "math.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include <blaze/math/Row.h>
#include <blaze/math/CustomMatrix.h>

using blaze::DynamicMatrix;
using blaze::StaticVector;

namespace py = pybind11;

struct Foo {
    int operator()() const {
        return 4;
    }
};

struct Bar {
    int operator()(std::function<int()> const &f) const {

        Matrix<double> m{{3,3},{2,4}};
        std::cout << blaze::det(m) << '\n';
        
        // blaze::DynamicVector<double, true> v{1,5};
        
        blaze::DynamicVector<double> v{1,2,3,-1,-2,-3};
        blaze::DynamicVector<double> u{0,0,0,0,0,0};
        blaze::DynamicMatrix<double> space_matrix{{1,3},{4,7}};
        blaze::CustomMatrix<double,blaze::unaligned,blaze::unpadded,blaze::rowMajor> vit(&v[0], 2, 3);
        blaze::CustomMatrix<double,blaze::unaligned,blaze::unpadded,blaze::rowMajor> ujs(&u[0], 2, 3);
        ujs = space_matrix * vit;
        
        std::cout << ujs << '\n';
        std::cout << u << std::endl;
        
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
