#include "math.hpp"

#include <string>
#include <complex>

#include "core.hpp"
#include "../math.hpp"
#include "../tmp.hpp"


/// Internals for binding math routines and classes.
namespace {
    template <typename T>
    constexpr char const *typeName = "_";

    template <>
    constexpr char const *typeName<bool> = "B";

    template <>
    constexpr char const *typeName<int> = "I";

    template <>
    constexpr char const *typeName<double> = "D";

    template <>
    constexpr char const *typeName<std::complex<double>> = "CD";

    /// Returns the name for vectors in Python.
    template <typename T>
    std::string vecName() {
        return std::string{typeName<T>} + "Vector";
    }

    /// Returns the name for dense matrices in Python.
    template <typename T>
    std::string matName() {
        return std::string{typeName<T>} + "Matrix";
    }

    /// Returns the name for sparce matrices in Python.
    template <typename T>
    std::string sparseMatName() {
        return std::string{typeName<T>} + "SparseMatrix";
    }

    /// Bind __iadd_ operator if possible for two given types.
    template <typename LHS, typename RHS,
              typename = decltype(LHS{} += RHS{})>
    struct bindIAdd {
        template <typename CT>
        static void f(CT &cl) {
            cl.def("__iadd__", [](LHS &v, const RHS &w) {
                    return v += w;
                });
        }
    };
    /// Fallback that does not bind anything.
    template <typename LHS, typename RHS>
    struct bindIAdd<LHS, RHS> {
        template <typename CT>
        static void f(CT &UNUSED(vec)) { }
    };

    /// Bind __isub_ operator if possible for two given types.
    template <typename LHS, typename RHS,
              typename = decltype(LHS{} -= RHS{})>
    struct bindISub {
        template <typename CT>
        static void f(CT &cl) {
            cl.def("__isub__", [](LHS &v, const RHS &w) {
                    return v -= w;
                });
        }
    };
    /// Fallback that does not bind anything.
    template <typename LHS, typename RHS>
    struct bindISub<LHS, RHS> {
        template <typename CT>
        static void f(CT &UNUSED(vec)) { }
    };

    /// Bind __imul_ operator if possible for two given types.
    template <typename LHS, typename RHS,
              typename = decltype(LHS{} *= RHS{})>
    struct bindIMul {
        template <typename CT>
        static void f(CT &cl) {
            cl.def("__imul__", [](LHS &v, const RHS &w) {
                    return v *= w;
                });
        }
    };
    /// Fallback that does not bind anything.
    template <typename LHS, typename RHS>
    struct bindIMul<LHS, RHS> {
        template <typename CT>
        static void f(CT &UNUSED(vec)) { }
    };

    /// Bind __idiv_ operator if possible for two given types.
    template <typename LHS, typename RHS,
              typename = decltype(LHS{} /= RHS{})>
    struct bindIDiv {
        template <typename CT>
        static void f(CT &cl) {
            cl.def("__idiv__", [](LHS &v, const RHS &w) {
                    return v /= w;
                });
        }
    };
    /// Fallback that does not bind anything.
    template <typename LHS, typename RHS>
    struct bindIDiv<LHS, RHS> {
        template <typename CT>
        static void f(CT &UNUSED(vec)) { }
    };


    /// Bind operators to the vector type.
    /**
     * Only binds operations if the operation `typename VT::ElementType{} * ET{}` is well
     * formed. I.e. if left and right hand sides can be multiplied. This rules out
     * combining complex numbers and integers, for example.
     *
     * \tparam ET Elemental type for right hand side.
     * \tparam VT Vector type for left hand side.
     * \tparam CT Pybind11 class type.
     * \param vec Pybind11 vector class to bind to.
     */
    template <typename ET, typename VT, typename = void_t<>>
    struct bindVectorOps {
        template <typename CT>
        static void f(CT &UNUSED(vec)) { }
    };

    /// Overload that actually binds something.
    template <typename ET, typename VT>
    struct bindVectorOps<ET, VT, void_t<decltype(typename VT::ElementType{} * ET{})>> {
        template <typename CT>
        static void f(CT &vec) {
            // return vector type
            using RVT = Vector<decltype(typename VT::ElementType{} * ET{})>;

            // with Vector
            vec.def("__add__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v+w);
                });
            vec.def("__sub__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v-w);
                });
            vec.def("__mul__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v*w);
                });
            vec.def("__matmul__", [](const VT &v, const Vector<ET> &w) {
                    return (v, w);
                });
            vec.def("__truediv__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v/w);
                });
            bindIAdd<VT, Vector<ET>>::f(vec);
            bindISub<VT, Vector<ET>>::f(vec);
            bindIMul<VT, Vector<ET>>::f(vec);
            bindIDiv<VT, Vector<ET>>::f(vec);
                                                
            // with scalar
            vec.def("__mul__", [](const VT &v, const ET &x) {
                    return RVT(v*x);
                });
            vec.def("__rmul__", [](const VT &v, const ET &x) {
                    return RVT(x*v);
                });
            vec.def("__truediv__", [](const VT &v, const ET &x) {
                    return RVT(v/x);
                });
            bindIMul<VT, ET>::f(vec);
            bindIDiv<VT, ET>::f(vec);
        }
    };

    
    /// Bind a single vector.
    /**
     * \tparam ET Elemental type for the vector to bind.
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ET, typename ElementalTypes>
    struct bindVector{
        static void f(py::module &mod) {
            // make a new Pybind11 class and add basic functions
            using VT = Vector<ET>;

            auto &vec = py::class_<VT>(mod, vecName<ET>().c_str(), py::buffer_protocol{})
                .def(py::init([](const std::size_t size){ return VT(size); }))
                .def("__getitem__", py::overload_cast<std::size_t>(&VT::operator[]))
                .def("__setitem__", [](VT &vec, const std::size_t i,
                                       const typename VT::ElementType x) {
                         vec[i] = x;
                     })
                .def("__iter__", [](VT &vec) {
                        return py::make_iterator(vec.begin(), vec.end());
                    })
                .def("__len__", &VT::size)
                .def("__repr__", [](const VT &vec) {
                        std::ostringstream oss;
                        oss << vec;
                        return oss.str();
                    })
                .def_buffer([](VT &vec) {
                        return py::buffer_info{
                            &vec[0], sizeof(ET), py::format_descriptor<ET>::format(),
                            1, {vec.size()}, {sizeof(ET)}};
                    })
                ;

            // bind operators for all right hand sides
            foreach<ElementalTypes, bindVectorOps, VT>::f(vec);
        }
    };
}


void bindTensors(py::module &mod) {
    // TODO what about bool
    // TODO can we treat floordiv as well?

    using ElementalTypes = Types<int, double, std::complex<double>>;
    
    foreach<ElementalTypes, bindVector, ElementalTypes>::f(mod);
}
