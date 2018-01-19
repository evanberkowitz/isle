#include "math.hpp"

#include <string>
#include <complex>
#include <cmath>

#include "core.hpp"
#include "../math.hpp"
#include "../tmp.hpp"






    
    

    // /// Perform Python truediv operation on vector and vector.
    // template <typename VT1, typename VT2, typename ET1, typename ET2>
    // struct truediv_vv {
    //     static auto f(const VT1 &vec1, const VT2 vec2) {
    //         using RVT = Vector<decltype(ET1{} / ET2{})>;
    //         return RVT(vec1/vec2);
    //     }
    // };
    // /// Perform Python truediv operation on vector of int and vector of int.
    // template <typename VT1, typename VT2>
    // struct truediv_vv<VT1, VT2, int, int> {
    //     static auto f(const VT1 &vec1, const VT2 vec2) {
    //         using RVT = typename VT1::template Rebind<double>::Other;
    //         return RVT(RVT(vec1) / vec2);
    //     }
    // };

    // /// Perform Python truediv operation on vector and scalar.
    // template <typename VT, typename ET, typename ST>
    // struct truediv_vs {
    //     static auto f(const VT &vec, const ST scalar) {
    //         using RVT = Vector<decltype(ET{} / ST{})>;
    //         return RVT(vec/scalar);
    //     }
    // };
    // /// Perform Python truediv operation on vector of int and int.
    // template <typename VT>
    // struct truediv_vs<VT, int, int> {
    //     static auto f(const VT &vec, const int scalar) {
    //         using RVT = typename VT::template Rebind<double>::Other;
    //         return RVT(vec/static_cast<double>(scalar));
    //     }
    // };

    // /// Perform Python floordiv on vector and scalar.
    // template <typename VT, typename ET, typename ST, typename Enable = void>
    // struct floordiv_vs {
    //     static auto f(const VT &vec, const ST scalar) {
    //         using RVT = Vector<decltype(ET{} / ST{})>;
    //         return RVT(blaze::floor(vec/scalar));
    //     }
    // };
    // /// Throws std::invalid_argument; floordiv not allowed with complex numbers.
    // template <typename VT, typename ET, typename ST>
    // struct floordiv_vs<VT, ET, ST,
    //                    std::enable_if_t<(IsSpecialization<std::complex, ET>::value
    //                                      || IsSpecialization<std::complex, ST>::value)>> {
    //     [[noreturn]] static auto f(const VT &UNUSED(vec), const ST UNUSED(scalar)) {
    //         throw std::invalid_argument("can't take floor of complex number.");
    //     }
    // };
    // /// Perform Python floordiv on vector of int and int.
    // template <typename VT>
    // struct floordiv_vs<VT, int, int, void> {
    //     static auto f(const VT &vec, const int scalar) {
    //         return VT(vec/scalar);
    //     }
    // };

    // /// Perform Python floordiv on vector and vector.
    // template <typename VT1, typename VT2, typename ET1, typename ET2,
    //           typename Enable = void>
    // struct floordiv_vv {
    //     static auto f(const VT1 &vec1, const VT2 vec2) {
    //         using RVT = Vector<decltype(ET1{} / ET2{})>;
    //         return RVT(blaze::floor(vec1/vec2));
    //     }
    // };
    // /// Throws std::invalid_argument; floordiv not allowed with complex numbers.
    // template <typename VT1, typename VT2, typename ET1, typename ET2>
    // struct floordiv_vv<VT1, VT2, ET1, ET2,
    //                    std::enable_if_t<(IsSpecialization<std::complex, ET1>::value
    //                                      || IsSpecialization<std::complex, ET2>::value)>> {
    //     [[noreturn]] static auto f(const VT1 &UNUSED(vec1), const VT2 UNUSED(vec2)) {
    //         throw std::invalid_argument("can't take floor of complex number.");
    //     }
    // };
    // /// Perform Python floordiv on vector of int and int.
    // template <typename VT1, typename VT2>
    // struct floordiv_vv<VT1, VT2, int, int, void> {
    //     static auto f(const VT1 &vec1, const VT2 vec2) {
    //         return VT1(vec1/vec2);
    //     }
    // };




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


    enum class Op {
        add, sub, mul, rmul, iadd, isub, imul, dot
    };


    template <Op op, typename LHS, typename RHS, typename = void>
    struct bindOp {
        template <typename CT>
        static void f(CT &&UNUSED(cls), const char * const UNUSED(name)) { }
    };


    template <typename LHS, typename RHS>
    struct bindOp<Op::add, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()+std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()+std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs + rhs);
                });
        }
    };

    template <typename LHS, typename RHS>
    struct bindOp<Op::sub, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()-std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()-std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs - rhs);
                });
        }
    };

    template <typename LHS, typename RHS>
    struct bindOp<Op::mul, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()*std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs * rhs);
                });
        }
    };

    template <typename LHS, typename RHS>
    struct bindOp<Op::rmul, LHS, RHS,
                  void_t<decltype(std::declval<RHS&>()*std::declval<LHS&>()),
                         decltype(std::declval<ElementType_t<RHS>&>()*std::declval<ElementType_t<LHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const RHS &rhs, const LHS &lhs) {
                    return blaze::evaluate(lhs * rhs);
                });
        }
    };

    
    template <typename LHS, typename RHS>
    struct bindOp<Op::iadd, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()+=std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()+=std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](LHS &lhs, const RHS &rhs) {
                    return lhs += rhs;
                });
        }
    };

    template <typename LHS, typename RHS>
    struct bindOp<Op::isub, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()-=std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()-=std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs -= rhs);
                });
        }
    };

    // additional test for operator* because blaze uses it internally and
    // operator*=(std::complex<double>, int) exists but operator*(std::complex<double>, int)
    // does not
    template <typename LHS, typename RHS>
    struct bindOp<Op::imul, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()*=std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()*=std::declval<ElementType_t<RHS>&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs *= rhs);
                });
        }
    };


    
    template <typename LHS, typename RHS>
    struct bindOp<Op::dot, LHS, RHS,
                  void_t<decltype(blaze::dot(std::declval<LHS&>(), std::declval<RHS&>())),
                         decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()
                                  + std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>())>>
    {
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::dot(lhs, rhs);
                });
        }
    };



    
    template <typename ET, typename VT>
    struct bindVectorOps {
        template <typename CT>
        static void f(CT &&vec) {
            // with Vector
            bindOp<Op::add, VT, Vector<ET>>::f(vec, "__add__");
            bindOp<Op::sub, VT, Vector<ET>>::f(vec, "__sub__");
            bindOp<Op::mul, VT, Vector<ET>>::f(vec, "__mul__");
            bindOp<Op::rmul, VT, Vector<ET>>::f(vec, "__rmul__");

            bindOp<Op::iadd, VT, Vector<ET>>::f(vec, "__iadd__");
            bindOp<Op::isub, VT, Vector<ET>>::f(vec, "__isub__");
            bindOp<Op::imul, VT, Vector<ET>>::f(vec, "__imul__");

            bindOp<Op::dot, VT, Vector<ET>>::f(vec, "__matmul__");
            
            // vec.def("__truediv__", truediv_vv<VT, Vector<ET>, typename VT::ElementType, ET>::f);
            // vec.def("__floordiv__", floordiv_vv<VT, Vector<ET>, typename VT::ElementType, ET>::f);
            

            // // with scalar
            bindOp<Op::mul, VT, ET>::f(vec, "__mul__");
            bindOp<Op::rmul, VT, ET>::f(vec, "__rmul__");
            bindOp<Op::imul, VT, ET>::f(vec, "__imul__");
            // vec.def("__truediv__", truediv_vs<VT, typename VT::ElementType, ET>::f);
            // vec.def("__floordiv__", floordiv_vs<VT, typename VT::ElementType, ET>::f);

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
