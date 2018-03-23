/** \file
 * \brief Wraps a round a math library and provides abstracted types and functions.
 *
 * The types do not provide a distinction between space and spacetime vectors / matrices.
 * Spacetime vectors are assumed to be encoded as a single vector with index
 * \f$(it) \equiv i n_{t} + t\f$, where \f$i\f$ is a space index, \f$t\f$ a time index, and
 * \f$n_{t}\f$ the number of time slices.
 */

#ifndef MATH_HPP
#define MATH_HPP

#include <type_traits>
#include <memory>

#include <blaze/Math.h>

#include "core.hpp"
#include "tmp.hpp"

namespace cnxx {
    /**
     * \brief A generic dense vector.
     * \tparam ET Element Type
     */
    template <typename ET>
    using Vector = blaze::DynamicVector<ET>;

    /**
     * \brief Holds spatial coordinates.
     * \tparam ET Element Type
     */
    template <typename ET>
    using Vec3 = blaze::StaticVector<ET, 3>;

    /**
     * \brief A generic dense matrix.
     * \tparam ET Element Type
     */
    template <typename ET>
    using Matrix = blaze::DynamicMatrix<ET>;

    /**
     * \brief A generic sparse matrix.
     * \tparam ET Element Type
     */
    template <typename ET>
    using SparseMatrix = blaze::CompressedMatrix<ET>;

    /**
     * \brief An identity matrix.
     * \tparam ET Element Type
     */
    template <typename ET>
    using IdMatrix = blaze::IdentityMatrix<ET>;

    /**
     * \brief A generic symmetric dense matrix.
     * \tparam ET Element Type
     */
    template <typename ET>
    using SymmetricMatrix = blaze::SymmetricMatrix<blaze::DynamicMatrix<ET>>;

    /**
     * \brief A generic symmetric sparse matrix.
     * \tparam ET Element Type
     */
    template <typename ET>
    using SymmetricSparseMatrix = blaze::SymmetricMatrix<blaze::CompressedMatrix<ET>>;

    // Convenience aliases.
    using IVector = Vector<int>;
    using DVector = Vector<double>;
    using CDVector = Vector<std::complex<double>>;

    using IMatrix = Matrix<int>;
    using DMatrix = Matrix<double>;
    using CDMatrix = Matrix<std::complex<double>>;

    using ISparseMatrix = SparseMatrix<int>;
    using DSparseMatrix = SparseMatrix<double>;
    using CDSparseMatrix = SparseMatrix<std::complex<double>>;


    /// Get the value type from a given compound type.
    /**
     * Falls back to give type if no specific overload or specialization exists.
     *
     * \see ElementType %ValueType does not retrieve the element type from containers.
     *                  This is done by %ElementType.
     */
    template <typename T>
    struct ValueType {
        using type = T;  ///< Deduced value type.
    };

    /// Overload for std::complex.
    template <typename T>
    struct ValueType<std::complex<T>> {
        using type = T;  ///< Deduced value type.
    };

    /// Helper alias for ValueType.
    template <typename T>
    using ValueType_t = typename ValueType<T>::type;


    /// Get the element type of a linear algebra type (vector, matrix).
    /**
     * Falls back to given type if it is arithmetic or std::complex.
     * Causes failure of a static assertion if the type is not recognized.
     *
     * \see ValueType %ElementType does not retrieve the value type from a compund type
     *                like std::complex but only operates on collections of elemental variables.
     */
    template <typename T, typename = void>
    struct ElementType {
        static_assert(tmp::AlwaysFalse_v<T>, "Cannot deduce element type.");
    };

    /// Overload for arithmetic types and std::complex.
    template <typename T>
    struct ElementType<T, std::enable_if_t<std::is_arithmetic<T>::value
                                           || tmp::IsSpecialization<std::complex, T>::value>> {
        using type = T;  ///< Deduced element type.
    };

    /// Overload for blaze::DynamicVector.
    template <typename ET, bool TF>
    struct ElementType<blaze::DynamicVector<ET, TF>> {
        using type = ET;  ///< Deduced element type.
    };

    /// Overload for blaze::DynamicMatrix.
    template <typename ET, bool TF>
    struct ElementType<blaze::DynamicMatrix<ET, TF>> {
        using type = ET;  ///< Deduced element type.
    };

    /// Overload for blaze::SparseMatrix.
    template <typename ET, bool TF>
    struct ElementType<blaze::CompressedMatrix<ET, TF>> {
        using type = ET;  ///< Deduced element type.
    };

    /// Convenience alias for ElementType.
    template <typename T>
    using ElementType_t = typename ElementType<T>::type;


    /// Variable template for pi up to long double precision.
    template <typename T>
    constexpr T pi = static_cast<T>(3.1415926535897932384626433832795028841971693993751058209749L);


    /// Return the flat spacetime coordinate for a given pair of space and time coordinates.
    /**
     * \param x Spatial coordinate.
     * \param t Temporal coordinate.
     * \param nx Number of spatial lattice sites.
     * \param nt Number of temporal lattice sites.
     */
    constexpr std::size_t spacetimeCoord(const std::size_t x,
                                         const std::size_t t,
                                         const std::size_t nx,
                                         const std::size_t UNUSED(nt)) noexcept {
        return t*nx + x;
    }


    /// Project a complex number to the first branch of the logarithm (-pi, pi].
    template <typename RT>
    std::complex<RT> toFirstLogBranch(const std::complex<RT> &x) {
        return {std::real(x), std::remainder(std::imag(x), 2*pi<RT>)};
    }


    /// Return a view on a spatial vector for given timeslice of a spacetime vector.
    template <typename VT>
    decltype(auto) spacevec(VT &&vec, const std::size_t t, const std::size_t nx)
        noexcept(ndebug) {
        // some rudimentary bounds check, no idea how to do this in general...
#ifndef NDEBUG
        if (t == static_cast<std::size_t>(-1))
            throw std::runtime_error("t is -1");
        if (t == static_cast<std::size_t>(-2))
            throw std::runtime_error("t is -2");
#endif
        return blaze::subvector(std::forward<VT>(vec), t*nx, nx);
    }

    /// Return a view on a spatial matrix for given timeslices of a spacetime matrix.
    template <typename MT>
    decltype(auto) spacemat(MT &&mat, const std::size_t tp, const std::size_t t,
                            const std::size_t nx) noexcept(ndebug) {
        // some rudimentary bounds check, no idea how to do this in general...
#ifndef NDEBUG
        if (tp == static_cast<std::size_t>(-1))
            throw std::runtime_error("tp is -1");
        if (tp == static_cast<std::size_t>(-2))
            throw std::runtime_error("tp is -2");
        if (t == static_cast<std::size_t>(-1))
            throw std::runtime_error("t is -1");
        if (t == static_cast<std::size_t>(-2))
            throw std::runtime_error("t is -2");
#endif
        return blaze::submatrix(std::forward<MT>(mat), tp*nx, t*nx, nx, nx);
    }


//     /// Multiply a space matrix with a space time vector.
//     /**
//      * Let \f$v, u\f$ be vectors in spacetime and \f$M\f$ a matrix in space.
//      * Furthermore, let \f$(it)\f$ denote a spacetime index comprised of the spatial index
//      * \f$i\f$ and time index \f$t\f$.<BR>
//      * This function computes
//      * \f[
//      *   u_{(it)} = M_{i,j} v_{(jt)}
//      * \f]
//      *
//      * \tparam MT Arbitrary matrix type.
//      * \tparam VT Dense vector type.
//      * \param spaceMatrix \f$M\f$
//      * \param spacetimeVector \f$v\f$
//      * \returns \f$u\f$
//      *
//      * \throws std::runtime_error
//      *  - `spaceMatrix` is not a square matrix.
//      *  - Length of `spacetimeVector` is not a multiple of the dimension of `spaceMatrix`.
//      *
//      *  Does not throw if macro `NDEBUG` is defined.
//      */
//     /*
//      * Works by wrapping input and output vectors in a blaze::CustomMatrix
//      * to treat a spacetime vector v_{(it)} as a matrix vm_{i,t}.
//      * Then m*v can be performed as m*vm.
//      */
//     template <typename MT, typename VT,
//               typename = std::enable_if_t<blaze::IsDenseVector<VT>::value, VT>>
//     auto spaceMatSpacetimeVec(const MT &spaceMatrix,
//                               const VT &spacetimeVector) noexcept(ndebug) {

//         // get lattice size
//         const auto nx = spaceMatrix.rows();
//         const auto nt = spacetimeVector.size() / nx;

// #ifndef NDEBUG
//         if (nx != spaceMatrix.columns())
//             throw std::runtime_error("Matrix is not square");
//         if (spacetimeVector.size() % nx != 0)
//             throw std::runtime_error("Matrix and vector size do not match");
// #endif

//         // return type, same as VT with adjusted element type
//         using RT = typename VT::template Rebind<
//             decltype(std::declval<typename MT::ElementType>()
//                      * std::declval<typename VT::ElementType>())
//             >::Other;

//         // space time matrix type for input vector
//         using STMV = blaze::CustomMatrix<std::add_const_t<typename VT::ElementType>,
//                                          blaze::unaligned, blaze::unpadded>;
//         // space time matrix type for returned vector
//         using STMR = blaze::CustomMatrix<typename RT::ElementType,
//                                          blaze::unaligned, blaze::unpadded>;

//         // do computation
//         RT result(spacetimeVector.size());
//         STMR{&result[0], nx, nt} = spaceMatrix * STMV{&spacetimeVector[0], nx, nt};
//         return result;
//     }

//     /// Dot a space vector into a space time vector.
//     /**
//      * Let \f$v\f$ be a vector in spacetime, and \f$x\f$ a vector in space, and \f$u\f$ be a vector in time.
//      * Furthermore, let \f$(it)\f$ denote a spacetime index comprised of the spatial index
//      * \f$i\f$ and time index \f$t\f$.<BR>
//      * This function computes
//      * \f[
//      *   u_{t} = x_i v_{(it)}
//      * \f]
//      *
//      * \tparam XT Arbitrary vector type.
//      * \tparam VT Dense vector type.
//      * \param spaceVector \f$x\f$
//      * \param spacetimeVector \f$v\f$
//      * \returns \f$u\f$
//      *
//      * \throws std::runtime_error
//      *  - Length of `spacetimeVector` is not a multiple of the dimension of `spaceVector`.
//      *
//      *  Does not throw if macro `NDEBUG` is defined.
//      */
//     /*
//      * Works by wrapping the input vector in a blaze::CustomMatrix
//      * to treat a spacetime vector v_{(it)} as a matrix vm_{i,t}.
//      * Then x*v can be performed as x*vm.
//      */
//     template <typename XT, typename VT,
//               typename = std::enable_if_t<blaze::IsDenseVector<VT>::value, VT>>
//     auto spaceVecSpacetimeVec(const XT &spaceVector,
//                               const VT &spacetimeVector) noexcept(ndebug) {

//         // get lattice size
//         const auto nx = spaceVector.size();
//         const auto nt = spacetimeVector.size() / nx;

// #ifndef NDEBUG
//         if (spacetimeVector.size() % nx != 0)
//             throw std::runtime_error("Matrix and vector size do not match");
// #endif

//         // return type, same as VT with adjusted element type
//         using RT = typename VT::template Rebind<
//             decltype(std::declval<typename XT::ElementType>()
//                      * std::declval<typename VT::ElementType>())
//             >::Other;

//         // space time matrix type for input vector
//         using STMV = blaze::CustomMatrix<std::add_const_t<typename VT::ElementType>,
//                                          blaze::unaligned, blaze::unpadded>;

//         // do computation
//         blaze::DynamicVector<RT> result;
//         result = spaceVector * STMV{&spacetimeVector[0], nx, nt};
//         return result;
//     }

    /// Invert a matrix in place.
    /**
     * \param mat Matrix to be inverted. Is replaced by the inverse.
     * \param ipiv Pivot indices. Must have at least `mat.rows()` elements
     *             but can be uninitialized.
     */
    inline void invert(Matrix<std::complex<double>> &mat, std::unique_ptr<int[]> &ipiv) {
        blaze::getrf(mat, ipiv.get());
        blaze::getri(mat, ipiv.get());
    }

    /// Compute the logarithm of the determinant of a dense matrix.
    /**
     * Note that the matrix is copied in order to leave the original unchanged.
     * See ilogdet() for an in-place version.
     * \tparam MT Specific matrix type.
     * \tparam SO Storage order of the matrix.
     * \param matrix Matrix to compute the determinant of; must be square.
     * \return \f$y = \log \det(\mathrm{mat})\f$ as a complex number
     *         projected onto the first Riemann sheet of the logarithm,
     *         i.e. \f$y \in (-\pi, \pi]\f$.
     */
    template <typename MT, bool SO>
    auto logdet(const blaze::DenseMatrix<MT, SO> &matrix) {
        using ET = ValueType_t<typename MT::ElementType>;
        MT mat{matrix};
        const std::size_t n = mat.rows();
#ifndef NDEBUG
        if (n != mat.columns())
            throw std::invalid_argument("Invalid non-square matrix provided");
#endif

        // pivot indices
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(n);
        // perform LU decomposition (mat = PLU)
        blaze::getrf(mat, ipiv.get());

        std::complex<ET> res = 0;
        std::int8_t detP = 1;
        for (std::size_t i = 0; i < n; ++i) {
            // determinant of pivot matrix P
            if (ipiv[i]-1 != blaze::numeric_cast<int>(i))
                detP = -detP;
            // log det of U (diagonal elements)
            res += std::log(std::complex<ET>{mat(i, i)});
        }
        // combine log dets and project to (-pi, pi]
        return toFirstLogBranch(res + (detP == 1 ? 0 : std::complex<ET>{0, pi<ET>}));
    }

    /// Compute the logarithm of the determinant of a dense matrix; overwrites the input.
    /**
     * \warning This version overwrites the input matrix. See logdet() for a version that
     *          does not change it.
     * \tparam MT Specific matrix type.
     * \tparam SO Storage order of the matrix.
     * \param matrix Matrix to compute the determinant of; must be square.
     * \return \f$y = \log \det(\mathrm{mat})\f$ as a complex number
     *         projected onto the first Riemann sheet of the logarithm,
     *         i.e. \f$y \in (-\pi, \pi]\f$.
     */
    template <typename MT, bool SO>
    auto ilogdet(blaze::DenseMatrix<MT, SO> &matrix) {
        using ET = ValueType_t<typename MT::ElementType>;
        const std::size_t n = (~matrix).rows();
#ifndef NDEBUG
        if (n != (~matrix).columns())
            throw std::invalid_argument("Invalid non-square matrix provided");
#endif

        // pivot indices
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(n);
        // perform LU decomposition (mat = PLU)
        blaze::getrf(~matrix, ipiv.get());

        std::complex<ET> res = 0;
        std::int8_t detP = 1;
        for (std::size_t i = 0; i < n; ++i) {
            // determinant of pivot matrix P
            if (ipiv[i]-1 != blaze::numeric_cast<int>(i))
                detP = -detP;
            // log det of U (diagonal elements)
            res += std::log(std::complex<ET>{(~matrix)(i, i)});
        }
        // combine log dets and project to (-pi, pi]
        return toFirstLogBranch(res + (detP == 1 ? 0 : std::complex<ET>{0, pi<ET>}));
    }

}  // namespace cnxx

#endif  // ndef MATH_HPP
