#include "math.hpp"

#include <string>
#include <complex>
#include <cmath>

#include "core.hpp"
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


    /// Labels for all algebraic operators.
    enum class Op {
        add, sub, mul, rmul, truediv, floordiv, iadd, isub, imul, dot
    };


    /// Bind an operator for given left- and righ-hand-sides.
    // This is the fallback which does not bind anything.
    // Specializations are enabled if operand types support the operation.
    template <Op op,  // identify operator
              typename LHS, typename RHS,  // types for operands
              typename = void>  // dummy for SFINAE
    struct bindOp {
        template <typename CT>
        static void f(CT &&UNUSED(cls), const char * const UNUSED(name)) { }
    };

    /// Bind operator add.
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

    /// Bind operator sub.
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

    /// Bind operator mul.
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

    /// Bind reverse mul operator.
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

    /// Bind operator truediv.
    template <typename LHS, typename RHS>
    struct bindOp<Op::truediv, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()/std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()/std::declval<ElementType_t<RHS>&>())>>
    {
        /// Indicate whether operands need to be converted to floating point numbers.
        static constexpr bool needConversion = std::is_same<ElementType_t<LHS>, int>::value
            && std::is_same<ElementType_t<RHS>, int>::value;

        /// Bind with convertion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<convert, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs / Rebind_t<RHS, double>(rhs));
                });
        }

        /// Bind without conversion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<!convert, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs / rhs);
                });
        }
    };

    /// Bind operator floordiv.
    template <typename LHS, typename RHS>
    struct bindOp<Op::floordiv, LHS, RHS,
                  void_t<decltype(std::declval<LHS&>()/std::declval<RHS&>()),
                         decltype(std::declval<ElementType_t<LHS>&>()/std::declval<ElementType_t<RHS>&>())>>
    {
        /// Indicate whether floor needs to be taken explicitly.
        static constexpr bool needFloor = !std::is_same<ElementType_t<LHS>, int>::value
            || !std::is_same<ElementType_t<RHS>, int>::value;
        /// Indicate whether it is possible to take floor.
        static constexpr bool floorAllowed = !IsSpecialization<std::complex,
                                                              ElementType_t<LHS>>::value
            && !IsSpecialization<std::complex, ElementType_t<RHS>>::value;

        /// Bind with implicit flooring.
        template <typename CT, bool floor = needFloor,
                  typename std::enable_if<!floor, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(lhs / rhs);
                });
        }

        /// Bind with explicit flooring.
        template <typename CT, bool floor = needFloor,
                  typename std::enable_if<floor && floorAllowed, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &lhs, const RHS &rhs) {
                    return blaze::evaluate(blaze::floor(lhs / rhs));
                });
        }

        /// Bind function that throws a std::invalid_argument because operands cannot be floored.
        template <typename CT, bool floor = needFloor,
                  std::enable_if_t<floor && !floorAllowed, int> = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, [](const LHS &UNUSED(lhs), const RHS &UNUSED(rhs)) {
                    throw std::invalid_argument("can't take floor of complex number.");
                });
        }
    };

    /// Bind inplace addition operator.
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

    /// Bind inplace subtraction operator.
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

    /// Bind inplace multiplication operator.
    // Additional test for operator* because blaze uses it internally and
    // operator*=(std::complex<double>, int) exists but operator*(std::complex<double>, int)
    // does not.
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

    /// Bind dot product ("operator").
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


    /// Bind all operators that are members of Vector.
    template <typename ET, typename VT>
    struct bindVectorOps {
        template <typename CT>
        static void f(CT &&vec) {
            // with Vector
            bindOp<Op::add, VT, Vector<ET>>::f(vec, "__add__");
            bindOp<Op::sub, VT, Vector<ET>>::f(vec, "__sub__");
            bindOp<Op::mul, VT, Vector<ET>>::f(vec, "__mul__");
            bindOp<Op::rmul, VT, Vector<ET>>::f(vec, "__rmul__");
            bindOp<Op::truediv, VT, Vector<ET>>::f(vec, "__truediv__");
            bindOp<Op::floordiv, VT, Vector<ET>>::f(vec, "__floordiv__");

            bindOp<Op::iadd, VT, Vector<ET>>::f(vec, "__iadd__");
            bindOp<Op::isub, VT, Vector<ET>>::f(vec, "__isub__");
            bindOp<Op::imul, VT, Vector<ET>>::f(vec, "__imul__");

            bindOp<Op::dot, VT, Vector<ET>>::f(vec, "__matmul__");
            bindOp<Op::dot, VT, Vector<ET>>::f(vec, "dot");

            // with scalar
            bindOp<Op::mul, VT, ET>::f(vec, "__mul__");
            bindOp<Op::rmul, VT, ET>::f(vec, "__rmul__");
            bindOp<Op::truediv, VT, ET>::f(vec, "__truediv__");
            bindOp<Op::floordiv, VT, ET>::f(vec, "__floordiv__");

            bindOp<Op::imul, VT, ET>::f(vec, "__imul__");
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
                .def(py::init([](const py::list &list) {
                            auto v = VT(list.size());
                            for (py::size_t i = 0; i < list.size(); ++i)
                                v[i] = list[i].cast<ET>();
                            return v;
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
