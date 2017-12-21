#include "math.hpp"

#include <string>
#include <complex>

// include math before core to avoid name conflict on Macs
#include "../math.hpp"
#include "core.hpp"


/// Internals for binding math routines and classes.
namespace {
    template <typename T>
    constexpr char const *typeName = "_";

    template <>
    constexpr char const *typeName<bool> = "B";

    template <>
    constexpr char const *typeName<int> = "I";

    template <>
    constexpr char const *typeName<float> = "F";

    template <>
    constexpr char const *typeName<double> = "D";

    template <>
    constexpr char const *typeName<std::complex<float>> = "CF";

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


    /// Holds a list of types in a head ('Top'), tail ('Rem') structure.
    template <typename T, typename... Args>
    struct Types {
        using Top = T;
        using Rem = Types<Args...>;
    };

    /// Base case for Types template.
    template <typename T>
    struct Types<T> {
        using Top = T;
        using Rem = void;
    };


    /// Bind operators to the vector type.
    /**
     * \tparam CT Pybind11 class type.
     * \tparam VT Vector type for left hand side.
     * \tparam ET Elemental type for right hand side.
     * \param vec Pybind11 vector class to bind to.
     */
    template <typename CT, typename VT, typename ET>
    struct doVectorBindOps {
        static void f(CT &vec) {
            // return vector type
            using RVT = Vector<decltype(typename VT::ElementType{} * ET{})>;
            vec.def("__add__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v+w);
                });
            vec.def("__iadd__", [](VT &v, const Vector<ET> &w) {
                    return v += w;
                });

            vec.def("__sub__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v-w);
                });
            vec.def("__isub__", [](VT &v, const Vector<ET> &w) {
                    return v -= w;
                });

            vec.def("__mul__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v*w);
                });
            vec.def("__matmul__", [](const VT &v, const Vector<ET> &w) {
                    return (v, w);
                });
            vec.def("__mul__", [](const VT &v, const ET &x) {
                    return RVT(v*x);
                });
            vec.def("__rmul__", [](const VT &v, const ET &x) {
                    return RVT(x*v);
                });
            vec.def("__imul__", [](VT &v, const Vector<ET> &w) {
                    return v *= w;
                });
            vec.def("__imul__", [](VT &v, const ET &x) {
                    return v *= x;
                });

            // TODO can we treat floordiv as well?
            vec.def("__truediv__", [](const VT &v, const Vector<ET> &w) {
                    return RVT(v/w);
                });
            vec.def("__truediv__", [](const VT &v, const ET &x) {
                    return RVT(v/x);
                });
            vec.def("__trueidiv__", [](VT &v, const Vector<ET> &w) {
                    return v /= w;
                });
            vec.def("__trueidiv__", [](VT &v, const ET &x) {
                    return v /= x;
                });
        }
    };
    
    /// Iterate over elemental types and bind operators for each.
    /**
     * \tparam CT Pybind11 class type.
     * \tparam VT Vector type for left hand side.
     * \tparam ET Current elemental type for right hand side.
     * \tparam ElementalTypes Instance of Types template for remaining elemental types.
     * \param vec Pybind11 vector class to bind to.
     */
    template <typename CT, typename VT, typename ET, typename ElementalTypes>
    struct bindVectorOps {
        static void f(CT &vec) {
            doVectorBindOps<CT, VT, ET>::f(vec);
            // peel off one type and recurse
            bindVectorOps<CT, VT, typename ElementalTypes::Top,
                          typename ElementalTypes::Rem>::f(vec);
        }
    };

    /// Base case for iteration over elemental types.
    /**
     * \tparam CT Pybind11 class type.
     * \tparam VT Vector type for left hand side.
     * \tparam ET Current and last elemental type for right hand side.
     * \param vec Pybind11 vector class to bind to.
     */
    template <typename CT, typename VT, typename ET>
    struct bindVectorOps<CT, VT, ET, void> {
        static void f(CT &vec) {
            doVectorBindOps<CT, VT, ET>::f(vec);
        }
    };

    /// Bind a single vector.
    /**
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \tparam ET Elemental type for the vector to bind.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ElementalTypes, typename ET>
    struct bindVector{
        static void f(py::module &mod) {
            // make a new Pybind11 class and add basic functions
            using VT = Vector<ET>;
            auto &vec = py::class_<VT>(mod, vecName<ET>().c_str())
                .def(py::init([](const std::size_t size){ return VT(size); }))
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

            // bind operators
            bindVectorOps<decltype(vec), VT, typename ElementalTypes::Top,
                          typename ElementalTypes::Rem>::f(vec);
        }
    };

    /// Iterate over elemental types and bind a vector for each.
    /**
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \tparam CurT Elemental type for vector to use in this iteration.
     * \tparam Rest Remaining Elemental types for vectors.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ElementalTypes, typename CurT, typename... Rest>
    struct bindVectors {
        static void f(py::module &mod) {
            bindVector<ElementalTypes, CurT>::f(mod);
            bindVectors<ElementalTypes, Rest...>::f(mod);
        }
    };
    
    /// Base case for iteration over elemental types.
    /**
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \tparam CurT Elemental type for vector to use in this iteration.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ElementalTypes, typename CurT>
    struct bindVectors<ElementalTypes, CurT> {
        static void f(py::module &mod) {
            bindVector<ElementalTypes, CurT>::f(mod);
        }
    };

    /// Bind tensors for given elemental types.
    template <typename... ElementalTypes>
    struct bindTensorsImpl {
        static void f(py::module &mod) {
            bindVectors<Types<ElementalTypes...>, ElementalTypes...>::f(mod);
        }
    };
}


void bindTensors(py::module &mod) {
    // TODO bool and complex cause some problems
    bindTensorsImpl<int, float, double>::f(mod);
}
