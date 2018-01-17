#include "math.hpp"

#include <string>
#include <complex>
#include <cmath>

#include "core.hpp"
#include "operators.hpp"
#include "../math.hpp"
#include "../tmp.hpp"


/// Internals for binding math routines and classes.
namespace {
    // TODO change to prefixName
    /// Prefix for names of linear algebra classes.
    template <typename T>
    constexpr char typeName[] = "_";

    template <>
    constexpr char typeName<int>[] = "I";

    template <>
    constexpr char typeName<double>[] = "D";

    template <>
    constexpr char typeName<std::complex<double>>[] = "CD";
    
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

    /// Bind a given operation to class cls; this version does not bind anything.
    template <typename LHS, typename RHS, typename OP, typename = void>
    struct bindOp {
        template <typename CT>
        static void f(CT &&UNUSED(cls), const char * const UNUSED(name)) { }
    };
    /// Specialization to actually bind op if possible.
    // Enable this version if the operator can be called on lvalues of the elemental types
    // of LHS and RHS. The & in declval makes sure that we have lvalues.
    template <typename LHS, typename RHS, typename OP>
    struct bindOp<LHS, RHS, OP,
                  void_t<decltype(OP::f(std::declval<ElementType_t<LHS>&>(),
                                        std::declval<ElementType_t<RHS>&>()))>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](LHS &lhs, RHS &rhs) {
                    return blaze::evaluate(OP::f(lhs, rhs));
                });
        }
    };

    /// Bind __iadd__ operator if possible for two given types.
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

    /// Bind __isub__ operator if possible for two given types.
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

    /// Bind __imul__ operator if possible for two given types.
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

    /// Bind __idiv__ operator if possible for two given types.
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

    /// Perform Python truediv operation on vector and vector.
    template <typename VT1, typename VT2, typename ET1, typename ET2>
    struct truediv_vv {
        static auto f(const VT1 &vec1, const VT2 vec2) {
            using RVT = Vector<decltype(ET1{} / ET2{})>;
            return RVT(vec1/vec2);
        }
    };
    /// Perform Python truediv operation on vector of int and vector of int.
    template <typename VT1, typename VT2>
    struct truediv_vv<VT1, VT2, int, int> {
        static auto f(const VT1 &vec1, const VT2 vec2) {
            using RVT = typename VT1::template Rebind<double>::Other;
            return RVT(RVT(vec1) / vec2);
        }
    };

    /// Perform Python truediv operation on vector and scalar.
    template <typename VT, typename ET, typename ST>
    struct truediv_vs {
        static auto f(const VT &vec, const ST scalar) {
            using RVT = Vector<decltype(ET{} / ST{})>;
            return RVT(vec/scalar);
        }
    };
    /// Perform Python truediv operation on vector of int and int.
    template <typename VT>
    struct truediv_vs<VT, int, int> {
        static auto f(const VT &vec, const int scalar) {
            using RVT = typename VT::template Rebind<double>::Other;
            return RVT(vec/static_cast<double>(scalar));
        }
    };

    /// Perform Python floordiv on vector and scalar.
    template <typename VT, typename ET, typename ST, typename Enable = void>
    struct floordiv_vs {
        static auto f(const VT &vec, const ST scalar) {
            using RVT = Vector<decltype(ET{} / ST{})>;
            return RVT(blaze::floor(vec/scalar));
        }
    };
    /// Throws std::invalid_argument; floordiv not allowed with complex numbers.
    template <typename VT, typename ET, typename ST>
    struct floordiv_vs<VT, ET, ST,
                       std::enable_if_t<(IsSpecialization<std::complex, ET>::value
                                         || IsSpecialization<std::complex, ST>::value)>> {
        [[noreturn]] static auto f(const VT &UNUSED(vec), const ST UNUSED(scalar)) {
            throw std::invalid_argument("can't take floor of complex number.");
        }
    };
    /// Perform Python floordiv on vector of int and int.
    template <typename VT>
    struct floordiv_vs<VT, int, int, void> {
        static auto f(const VT &vec, const int scalar) {
            return VT(vec/scalar);
        }
    };

    /// Perform Python floordiv on vector and vector.
    template <typename VT1, typename VT2, typename ET1, typename ET2,
              typename Enable = void>
    struct floordiv_vv {
        static auto f(const VT1 &vec1, const VT2 vec2) {
            using RVT = Vector<decltype(ET1{} / ET2{})>;
            return RVT(blaze::floor(vec1/vec2));
        }
    };
    /// Throws std::invalid_argument; floordiv not allowed with complex numbers.
    template <typename VT1, typename VT2, typename ET1, typename ET2>
    struct floordiv_vv<VT1, VT2, ET1, ET2,
                       std::enable_if_t<(IsSpecialization<std::complex, ET1>::value
                                         || IsSpecialization<std::complex, ET2>::value)>> {
        [[noreturn]] static auto f(const VT1 &UNUSED(vec1), const VT2 UNUSED(vec2)) {
            throw std::invalid_argument("can't take floor of complex number.");
        }
    };
    /// Perform Python floordiv on vector of int and int.
    template <typename VT1, typename VT2>
    struct floordiv_vv<VT1, VT2, int, int, void> {
        static auto f(const VT1 &vec1, const VT2 vec2) {
            return VT1(vec1/vec2);
        }
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
    // template <typename ET, typename VT, typename = void_t<>>
    // struct bindVectorOps {
    //     template <typename CT>
    //     static void f(CT &&UNUSED(vec)) { }
    // };

    /// Overload that actually binds something.
    // template <typename ET, typename VT>
    // struct bindVectorOps<ET, VT, void_t<decltype(typename VT::ElementType{} * ET{})>> {

    template <typename ET, typename VT, typename = void_t<>>
    struct bindVectorOps {
        
        template <typename CT>
        static void f(CT &&vec) {
            // return vector type
            // using RVT = Vector<decltype(typename VT::ElementType{} * ET{})>;

            // with Vector
            bindOp<VT, Vector<ET>, bind::op::add>::f(vec, "__add__");
            bindOp<VT, Vector<ET>, bind::op::sub>::f(vec, "__sub__");
            bindOp<VT, Vector<ET>, bind::op::mul>::f(vec, "__mul__");

            // vec.def("__matmul__", [](const VT &v, const Vector<ET> &w) {
            //         return (v, w);
            //     });
            // vec.def("__truediv__", truediv_vv<VT, Vector<ET>, typename VT::ElementType, ET>::f);
            // vec.def("__floordiv__", floordiv_vv<VT, Vector<ET>, typename VT::ElementType, ET>::f);
            // // bindIAdd<VT, Vector<ET>>::f(vec);
            // bindOp<VT, Vector<ET>, bind::op::iadd>::f(vec);
            // bindISub<VT, Vector<ET>>::f(vec);
            // bindIMul<VT, Vector<ET>>::f(vec);
            // bindIDiv<VT, Vector<ET>>::f(vec);

            // // with scalar
            bindOp<VT, ET, bind::op::mul>::f(vec, "__mul__");
            // vec.def("__rmul__", [](const VT &v, const ET &x) {
            //         return RVT(x*v);
            //     });
            // vec.def("__truediv__", truediv_vs<VT, typename VT::ElementType, ET>::f);
            // vec.def("__floordiv__", floordiv_vs<VT, typename VT::ElementType, ET>::f);

            // bindIMul<VT, ET>::f(vec);
            // bindIDiv<VT, ET>::f(vec);
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

            auto vec = py::class_<VT>{mod, vecName<ET>().c_str(), py::buffer_protocol{}}
                .def(py::init([](const std::size_t size){ return VT(size); }))
                .def(py::init([](py::buffer &buf) {
                            const py::buffer_info binfo = buf.request();
                            if (binfo.format != py::format_descriptor<ET>::format())
                                throw std::runtime_error("Incompatible buffer format: mismatched elemental data type");
                            if (binfo.ndim != 1)
                                throw std::runtime_error("Wrong buffer dimention to construct vector.");
                            return VT(binfo.shape.at(0), static_cast<ET const *>(binfo.ptr));
                        }))
                .def(py::init([](const std::vector<ET> &arr) {
                            return VT(arr.size(), &arr[0]);
                        }))
                .def("__getitem__", py::overload_cast<std::size_t>(&VT::operator[]))
                .def("__setitem__", [](VT &vec, const std::size_t i,
                                       const typename VT::ElementType x) {
                         vec[i] = x;
                     })
                .def("__iter__", [](VT &vec) {
                        return py::make_iterator(vec.begin(), vec.end());
                    }, py::keep_alive<0, 1>())
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

    /// Create Python wrappers around Tensors for different datatypes.
    void defineWrapperClasses(py::module &mod) {
        py::exec(R"(
class Vector:
    def __new__(self, *args, dtype=float, **kwargs):
        import cnxx
        if dtype == float:
            return cnxx.DVector(*args, **kwargs)
        if dtype == int:
            return cnxx.IVector(*args, **kwargs)
        if dtype == complex:
            return cnxx.CDVector(*args, **kwargs)
)", py::globals(), mod.attr("__dict__"));
    }
}

namespace bind {
    
    void bindTensors(py::module &mod) {
        // TODO int format_descriptor does not match

        using ElementalTypes = Types<int, double, std::complex<double>>;
    
        foreach<ElementalTypes, bindVector, ElementalTypes>::f(mod);

        defineWrapperClasses(mod);
    }
}
