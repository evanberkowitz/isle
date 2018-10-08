#include "math.hpp"

#include <string>
#include <complex>
#include <cmath>

#include "core.hpp"
#include "../math.hpp"
#include "../tmp.hpp"

using namespace isle;

/// Internals for binding math routines and classes.
namespace {
    /// Prefix for names of linear algebra classes.
    template <typename T>
    constexpr char prefixName[] = "_";

    template <>
    constexpr char prefixName<int>[] = "I";

    template <>
    constexpr char prefixName<double>[] = "D";

    template <>
    constexpr char prefixName<std::complex<double>>[] = "CD";

    /// Returns the name for vectors in Python.
    template <typename T>
    std::string vecName() {
        return std::string{prefixName<T>} + "Vector";
    }

    /// Returns the name for dense matrices in Python.
    template <typename T>
    std::string matName() {
        return std::string{prefixName<T>} + "Matrix";
    }

    /// Returns the name for sparce matrices in Python.
    template <typename T>
    std::string sparseMatName() {
        return std::string{prefixName<T>} + "SparseMatrix";
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
        static void f(CT &&UNUSED(cls), const char * const UNUSED(name)) noexcept { }
    };


    /// Bind operator add.
    template <typename LHS, typename RHS>
    struct bindOp<Op::add, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()+std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()+std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform the addition.
        static auto exec(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs + rhs);
        }
    };

    /// Bind operator sub.
    template <typename LHS, typename RHS>
    struct bindOp<Op::sub, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()-std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()-std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform the subtraction.
        static auto exec(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs - rhs);
        }
    };

    /// Bind operator mul.
    template <typename LHS, typename RHS>
    struct bindOp<Op::mul, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()*std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform the multiplication.
        static auto exec(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs * rhs);
        }
    };

    /// Bind reverse mul operator.
    template <typename RHS, typename LHS>
    struct bindOp<Op::rmul, RHS, LHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()*std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform the multiplication.
        static auto exec(const RHS &rhs, const LHS &lhs) {
            return blaze::evaluate(lhs * rhs);
        }
    };

    /// Bind operator truediv.
    template <typename LHS, typename RHS>
    struct bindOp<Op::truediv, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()/std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()/std::declval<RHS&>())>>
    {
        /// Indicate whether operands need to be converted to floating point numbers.
        static constexpr bool needConversion = std::is_same<ElementType_t<LHS>, int>::value
            && std::is_same<ElementType_t<RHS>, int>::value;

        /// Bind with conversion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<convert, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, execWithConversion);
        }

        /// Perform division with conversion of rhs.
        static auto execWithConversion(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs / tmp::Rebind_t<RHS, double>(rhs));
        }

        /// Bind without conversion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<!convert, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, execWithoutConversion);
        }

        /// Perform division without conversion of rhs.
        static auto execWithoutConversion(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs / rhs);
        }
    };

    /// Bind operator floordiv.
    template <typename LHS, typename RHS>
    struct bindOp<Op::floordiv, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()/std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()/std::declval<RHS&>())>>
    {
        /// Indicate whether vectors need to be converted.
        static constexpr bool needConversion = std::is_same<ElementType_t<LHS>, int>::value
            && std::is_same<ElementType_t<RHS>, int>::value;
        /// Indicate whether it is possible to take floor.
        static constexpr bool floorAllowed = !tmp::IsSpecialization<std::complex,
                                                                    ElementType_t<LHS>>::value
            && !tmp::IsSpecialization<std::complex, ElementType_t<RHS>>::value;

        /// Bind with conversion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<convert, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, execWithConversion);
        }

        /// Perform division with conversion of rhs.
        static auto execWithConversion(const LHS &lhs, const RHS &rhs) {
            // convert to LHS to make sure to return with int as elementary type
            return LHS{blaze::floor(lhs / tmp::Rebind_t<RHS, double>(rhs))};
        }

        /// Bind without conversion.
        template <typename CT, bool convert = needConversion,
                  typename std::enable_if<!convert && floorAllowed, int>::type = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, execWithoutConversion);
        }

        /// Perform division without conversion of rhs.
        static auto execWithoutConversion(const LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(blaze::floor(lhs / rhs));
        }

        /// Bind function that throws a std::invalid_argument because operands cannot be floored.
        template <typename CT, bool convert = needConversion,
                  std::enable_if_t<!convert && !floorAllowed, int> = 0>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, throwInvalidArgument);
        }

        /// Throw an exception.
        static auto throwInvalidArgument(const LHS &UNUSED(lhs), const RHS &UNUSED(rhs)) {
            throw std::invalid_argument("can't take floor of complex number.");
        }
    };

    /// Bind inplace addition operator.
    template <typename LHS, typename RHS>
    struct bindOp<Op::iadd, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()+=std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()+=std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform inplace addition.
        static auto exec(LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs += rhs);
        }
    };

    /// Bind inplace subtraction operator.
    template <typename LHS, typename RHS>
    struct bindOp<Op::isub, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()-=std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()-=std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform inplace subtraction.
        static auto exec(LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs -= rhs);
        }
    };

    /// Bind inplace multiplication operator.
    // Additional test for operator* because blaze uses it internally and
    // operator*=(std::complex<double>, int) exists but operator*(std::complex<double>, int)
    // does not.
    template <typename LHS, typename RHS>
    struct bindOp<Op::imul, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()*=std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()),
                              decltype(std::declval<LHS&>()*=std::declval<RHS&>())>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform inplace multiplication.
        static auto exec(LHS &lhs, const RHS &rhs) {
            return blaze::evaluate(lhs *= rhs);
        }
    };

    /// Bind dot product ("operator").
    template <typename LHS, typename RHS>
    struct bindOp<Op::dot, LHS, RHS,
                  tmp::void_t<decltype(std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()
                                       + std::declval<ElementType_t<LHS>&>()*std::declval<ElementType_t<RHS>&>()),
                              decltype(blaze::dot(std::declval<LHS&>(), std::declval<RHS&>()))>>
    {
        /// Bind a Python function.
        template <typename CT>
        static void f(CT &&cls, const char * const name) {
            cls.def(name, exec);
        }

        /// Perform dot product.
        static auto exec(const LHS &lhs, const RHS &rhs) {
            return blaze::dot(lhs, rhs);
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

    /// Bind a vector with given elemental type.
    /**
     * \tparam ET Elemental type for the vector to bind.
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ET, typename ElementalTypes>
    struct bindVector {
        static void f(py::module &mod) {
            // make a new Pybind11 class and add basic functions
            using VT = Vector<ET>;

            auto cls = py::class_<VT>(mod, vecName<ET>().c_str(), py::buffer_protocol{})
                .def(py::init([](const std::size_t size){ return VT(size); }))
                .def(py::init([](py::buffer &buf) {
                            const py::buffer_info binfo = buf.request();
                            if (binfo.format != py::format_descriptor<ET>::format())
                                throw std::runtime_error("Incompatible buffer format: mismatched elemental data type");
                            if (binfo.ndim != 1)
                                throw std::runtime_error("Wrong buffer dimension to construct vector");

                            if (binfo.strides[0] == sizeof(ET))
                                // copy as is (w/o stride)
                                return VT(binfo.shape[0],
                                          static_cast<const ET *>(binfo.ptr));
                            else {
                                // copy with specific stride
                                VT vec(binfo.shape[0]);
                                const py::ssize_t stride = binfo.strides[0]/sizeof(ET);
                                for (py::ssize_t i = 0; i < binfo.shape[0]; ++i)
                                    vec[i] = static_cast<const ET*>(binfo.ptr)[i*stride];
                                return vec;
                            }
                        }))
                .def(py::init([](const py::list &list) {
                            VT v(list.size());
                            for (py::size_t i = 0; i < list.size(); ++i)
                                v[i] = list[i].cast<ET>();
                            return v;
                        }))
#ifdef NDEBUG
                .def("__getitem__", py::overload_cast<std::size_t>(&VT::operator[]))
#else
                .def("__getitem__", [](VT &vec, const std::size_t i) {
                        if (!(i < vec.size()))
                            throw std::out_of_range("Index out of range");
                        return vec[i];
                    })
#endif
                .def("__setitem__", [](VT &vec, const std::size_t i,
                                       const typename VT::ElementType x) {
#ifndef NDEBUG
                         if (!(i < vec.size()))
                             throw std::out_of_range("Index out of range");
#endif
                         vec[i] = x;
                     })
                .def("__iter__", [](VT &vec) {
                        return py::make_iterator(vec.begin(), vec.end());
                    }, py::keep_alive<0, 1>())
                .def("__len__", &VT::size)
                .def("__str__", [](const VT &vec) {
                        std::ostringstream oss;
                        oss << '[';
                        std::copy(vec.begin(), vec.end()-1,
                                  std::ostream_iterator<ET>(oss, ", "));
                        oss << vec[vec.size()-1] << ']';
                        return oss.str();
                    })
                .def_buffer([](VT &vec) {
                        return py::buffer_info{
                            &vec[0], sizeof(ET), py::format_descriptor<ET>::format(),
                            1, {vec.size()}, {sizeof(ET)}};
                    })
                .def("__neg__", [](const VT &vec) {
                        return blaze::evaluate(-vec);
                    })
                ;

            // allow implicit conversion from buffer to Vector
            py::implicitly_convertible<py::buffer, VT>();

            // bind operators for all right hand sides
            tmp::foreach<ElementalTypes, bindVectorOps, VT>::f(cls);

            // bind free function on VT
            mod.def("real", [](const VT &vec) {
                    return blaze::evaluate(blaze::real(vec));
                });
            mod.def("imag", [](const VT &vec) {
                    return blaze::evaluate(blaze::imag(vec));
                });
        }
    };


    /// Bind all operators that are members of Matrix.
    template <typename ET, typename MT>
    struct bindMatrixOps {
        template <typename CT>
        static void f(CT &&mat) {
            // with Matrix
            bindOp<Op::add, MT, Matrix<ET>>::f(mat, "__add__");
            bindOp<Op::sub, MT, Matrix<ET>>::f(mat, "__sub__");
            bindOp<Op::mul, MT, Matrix<ET>>::f(mat, "__mul__");

            bindOp<Op::iadd, MT, Matrix<ET>>::f(mat, "__iadd__");
            bindOp<Op::isub, MT, Matrix<ET>>::f(mat, "__isub__");
            bindOp<Op::imul, MT, Matrix<ET>>::f(mat, "__imul__");

            // with SparseMatrix
            bindOp<Op::add, MT, SparseMatrix<ET>>::f(mat, "__add__");
            bindOp<Op::sub, MT, SparseMatrix<ET>>::f(mat, "__sub__");
            bindOp<Op::mul, MT, SparseMatrix<ET>>::f(mat, "__mul__");

            bindOp<Op::imul, MT, SparseMatrix<ET>>::f(mat, "__imul__");

            // with Vector
            bindOp<Op::mul, MT, Vector<ET>>::f(mat, "__mul__");

            // with scalar
            bindOp<Op::mul, MT, ET>::f(mat, "__mul__");
            bindOp<Op::rmul, MT, ET>::f(mat, "__rmul__");
            bindOp<Op::truediv, MT, ET>::f(mat, "__truediv__");
            bindOp<Op::floordiv, MT, ET>::f(mat, "__floordiv__");

            bindOp<Op::imul, MT, ET>::f(mat, "__imul__");
        }
    };

    /// Bind a Matrix with given elemental type.
    /**
     * \tparam ET Elemental type for the Matrix to bind.
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ET, typename ElementalTypes>
    struct bindMatrix {
        static void f(py::module &mod) {
            // make a new Pybind11 class and add basic functions
            using MT = Matrix<ET>;

            static_assert(blaze::StorageOrder<MT>::value == blaze::rowMajor,
                          "Need row major storage order to bind matrices using buffer protocol.");

            auto cls = py::class_<MT>(mod, matName<ET>().c_str(), py::buffer_protocol{})
                .def(py::init([](const std::size_t nx, const std::size_t ny){
                            return MT(nx, ny);
                        }))
                .def(py::init([](const SparseMatrix<ET> &other){
                            return MT(other);
                        }))
                .def(py::init([](py::buffer &buf) {
                            const py::buffer_info binfo = buf.request();
                            if (binfo.format != py::format_descriptor<ET>::format())
                                throw std::runtime_error("Incompatible buffer format: mismatched elemental data type");
                            if (binfo.ndim != 2)
                                throw std::runtime_error("Wrong buffer dimension to construct matrix");

                            if (binfo.strides[0] == static_cast<py::ssize_t>(sizeof(ET))*binfo.shape[1]
                                && binfo.strides[1] == sizeof(ET))
                                // buffer in row-major format => copy as is
                                return MT(binfo.shape[0], binfo.shape[1],
                                          static_cast<const ET *>(binfo.ptr));
                            else {
                                // copy taking strides into account
                                MT mat(binfo.shape[0], binfo.shape[1]);
                                const py::ssize_t stride0 = binfo.strides[0]/sizeof(ET);
                                const py::ssize_t stride1 = binfo.strides[1]/sizeof(ET);
                                for (py::ssize_t i = 0; i < binfo.shape[0]; ++i) {
                                    for (py::ssize_t j = 0; j < binfo.shape[1]; ++j) {
                                        mat(i, j) = static_cast<const ET*>(binfo.ptr)[i*stride0 + j*stride1];
                                    }
                                }
                                return mat;
                            }
                        }))
                .def(py::init([](const py::list &list) {
                            MT mat(list.size(), list[0].cast<py::list>().size());
                            for (py::size_t i = 0; i < list.size(); ++i) {
                                const py::list &inner = list[i].cast<py::list>();
                                for (py::size_t j = 0; j < inner.size(); ++j)
                                    mat(i, j) = inner[j].cast<ET>();
                            }
                            return mat;
                        }))
                .def("__getitem__", [](const MT &mat,
                                       const std::tuple<std::size_t, std::size_t> &idxs) {
#ifndef NDEBUG
                         if (!(std::get<0>(idxs) < mat.rows()))
                             throw std::out_of_range("Row index out of range");
                         if (!(std::get<1>(idxs) < mat.columns()))
                             throw std::out_of_range("Column index out of range");
#endif
                         return mat(std::get<0>(idxs), std::get<1>(idxs));
                     })
                .def("__setitem__", [](MT &mat,
                                       const std::tuple<std::size_t, std::size_t> &idxs,
                                       const typename MT::ElementType x) {
#ifndef NDEBUG
                         if (!(std::get<0>(idxs) < mat.rows()))
                             throw std::out_of_range("Row index out of range");
                         if (!(std::get<1>(idxs) < mat.columns()))
                             throw std::out_of_range("Column index out of range");
#endif
                         mat(std::get<0>(idxs), std::get<1>(idxs)) = x;
                     })
                .def("row", [](MT &mat, std::size_t i) {
                        return py::make_iterator(mat.begin(i), mat.end(i));
                    }, py::keep_alive<0, 1>())
                .def("rows", &MT::rows)
                .def("columns", &MT::columns)
                .def("__str__", [](const MT &mat) {
                        std::ostringstream oss;
                        oss << '[';
                        // loop over rows
                        for (std::size_t ind = 0; ind < mat.rows()-1; ++ind) {
                            oss << '[';
                            // write columns
                            std::copy(mat.begin(ind), mat.end(ind)-1,
                                      std::ostream_iterator<ET>(oss, ", "));
                            oss << mat(ind, mat.columns()-1) << "],\n ";
                        }
                        // write last row
                        const std::size_t ind = mat.rows()-1;
                        oss << '[';
                        std::copy(mat.begin(ind), mat.end(ind)-1,
                                  std::ostream_iterator<ET>(oss, ", "));
                        oss << mat(ind, mat.columns()-1) << "]]";
                        return oss.str();
                    })
                .def_buffer([](MT &mat) {
                        // alignment adjusted number of columns
                        const std::size_t adjColumns = mat.rows()>1 ?
                                                         (&mat(1,0)-&mat(0,0)) :
                                                         mat.columns();
                        return py::buffer_info{
                            &mat(0, 0), sizeof(ET), py::format_descriptor<ET>::format(),
                            2, {mat.rows(), mat.columns()},
                            {sizeof(ET)*adjColumns, sizeof(ET)}};
                    })
                .def("__neg__", [](const MT &mat) {
                        return blaze::evaluate(-mat);
                    })
                ;

            // allow implicit conversion from buffer to Vector
            py::implicitly_convertible<py::buffer, MT>();

            // bind operators for all right hand sides
            tmp::foreach<ElementalTypes, bindMatrixOps, MT>::f(cls);

            // bind free function on MT
            mod.def("real", [](const MT &mat) {
                    return blaze::evaluate(blaze::real(mat));
                });
            mod.def("imag", [](const MT &mat) {
                    return blaze::evaluate(blaze::imag(mat));
                });
        }
    };


    /// Bind all operators that are members of SparseMatrix.
    template <typename ET, typename MT>
    struct bindSparseMatrixOps {
        template <typename CT>
        static void f(CT &&mat) {
            // with SparseMatrix
            bindOp<Op::add, MT, SparseMatrix<ET>>::f(mat, "__add__");
            bindOp<Op::sub, MT, SparseMatrix<ET>>::f(mat, "__sub__");
            bindOp<Op::mul, MT, SparseMatrix<ET>>::f(mat, "__mul__");

            bindOp<Op::imul, MT, SparseMatrix<ET>>::f(mat, "__imul__");

            // with Matrix
            bindOp<Op::add, MT, Matrix<ET>>::f(mat, "__add__");
            bindOp<Op::sub, MT, Matrix<ET>>::f(mat, "__sub__");
            bindOp<Op::mul, MT, Matrix<ET>>::f(mat, "__mul__");

            bindOp<Op::imul, MT, Matrix<ET>>::f(mat, "__imul__");

            // with Vector
            bindOp<Op::mul, MT, Vector<ET>>::f(mat, "__mul__");

            // with scalar
            bindOp<Op::mul, MT, ET>::f(mat, "__mul__");
            bindOp<Op::rmul, MT, ET>::f(mat, "__rmul__");
            bindOp<Op::truediv, MT, ET>::f(mat, "__truediv__");
            bindOp<Op::floordiv, MT, ET>::f(mat, "__floordiv__");

            bindOp<Op::imul, MT, ET>::f(mat, "__imul__");
        }
    };

    /// Bind a SparseMatrix with given elemental type.
    /**
     * \tparam ET Elemental type for the SparseMatrix to bind.
     * \tparam ElementalTypes Instance of Types template of all elemental types for the right hand side.
     * \param mod Pybind11 module to bind to.
     */
    template <typename ET, typename ElementalTypes>
    struct bindSparseMatrix {
        static void f(py::module &mod) {
            // make a new Pybind11 class and add basic functions
            using MT = SparseMatrix<ET>;

            auto cls = py::class_<MT>(mod, sparseMatName<ET>().c_str())
                .def(py::init([](const std::size_t nx, const std::size_t ny){
                            return MT(nx, ny);
                        }))
                .def(py::init([](const Matrix<ET> &other){
                            return MT(other);
                        }))
                .def("__getitem__", [](const MT &mat,
                                       const std::tuple<std::size_t, std::size_t> &idxs) {
                         if (mat.find(std::get<0>(idxs), std::get<1>(idxs))
                             != mat.end(std::get<0>(idxs)))
                             return mat(std::get<0>(idxs), std::get<1>(idxs));
                         else
                             throw std::invalid_argument("No matrix element at given indices");
                     })
                .def("__setitem__", [](MT &mat,
                                       const std::tuple<std::size_t, std::size_t> &idxs,
                                       const typename MT::ElementType x) {
#ifndef NDEBUG
                         if (!(std::get<0>(idxs) < mat.rows()))
                             throw std::out_of_range("Row index out of range");
                         if (!(std::get<1>(idxs) < mat.columns()))
                             throw std::out_of_range("Column index out of range");
#endif
                         mat.set(std::get<0>(idxs), std::get<1>(idxs), x);
                     })
                .def("erase", [](MT &mat,
                                 const std::size_t i, const std::size_t j) {
#ifndef NDEBUG
                         if (!(i < mat.rows()))
                             throw std::out_of_range("Row index out of range");
                         if (!(j < mat.columns()))
                             throw std::out_of_range("Column index out of range");
#endif
                         mat.erase(i, j);
                     })
                .def("row", [](MT &mat, std::size_t i) {
                        return bind::makeIndexValueIterator(mat.begin(i), mat.end(i));
                    }, py::keep_alive<0, 1>())
                .def("rows", &MT::rows)
                .def("columns", &MT::columns)
                .def("__str__", [](const MT &mat) {
                        std::ostringstream oss;
                        oss << '[';
                        // loop over rows
                        for (std::size_t ind = 0; ind < mat.rows(); ++ind) {
                            // write columns
                            for (auto it = mat.begin(ind); it != mat.end(ind); ++it){
                                oss << '(' << ind << ',' << it->index() << "): ";
                                oss << it->value() << ",\n ";
                            }
                        }
                        oss << '\n';
                        return oss.str();
                    })
                .def("__neg__", [](const MT &mat) {
                        return blaze::evaluate(-mat);
                    })
                ;

            // bind operators for all right hand sides
            tmp::foreach<ElementalTypes, bindSparseMatrixOps, MT>::f(cls);

            // bind free function on MT
            mod.def("real", [](const MT &mat) {
                    return blaze::evaluate(blaze::real(mat));
                });
            mod.def("imag", [](const MT &mat) {
                    return blaze::evaluate(blaze::imag(mat));
                });
        }
    };
}

namespace bind {
    void bindTensors(py::module &mod) {
        using ElementalTypes = tmp::Types<int, double, std::complex<double>>;

        tmp::foreach<ElementalTypes, bindVector, ElementalTypes>::f(mod);
        tmp::foreach<ElementalTypes, bindMatrix, ElementalTypes>::f(mod);
        tmp::foreach<ElementalTypes, bindSparseMatrix, ElementalTypes>::f(mod);

        mod.def("real", [](const int x) { return x; });
        mod.def("real", [](const double x) { return x; });
        mod.def("real", [](const std::complex<double> x) { return std::real(x); });
        mod.def("imag", [](const int UNUSED(x)) { return 0; });
        mod.def("imag", [](const double UNUSED(x)) { return 0; });
        mod.def("imag", [](const std::complex<double> x) { return std::imag(x); });

        mod.def("logdet", [](const Matrix<std::complex<double>> &mat) {
                return logdet(mat);
            });
        mod.def("logdet", [](const Matrix<double> &mat) {
                return logdet(mat);
            });
    }
}
