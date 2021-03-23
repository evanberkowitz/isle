/** \file
 * \brief Wraps around a math library and provides abstracted types and functions.
 */

#ifndef MATH_HPP
#define MATH_HPP

#include <type_traits>
#include <memory>

#include <blaze/Math.h>

#include "core.hpp"
#include "linear_algebra.hpp"
#include "tmp.hpp"
#include "profile.hpp"
#include "logging/logging.hpp"

namespace isle {

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

    /// Variable template for pi up to long double precision.
    template <typename T>
    constexpr T pi = static_cast<T>(3.1415926535897932384626433832795028841971693993751058209749L);

    /// Compute temporal lattice size from a spacetime vector and spatial lattice size.
    /**
     * \throws std::runtime_error if vector size is not a multiple of `nx` and not in release mode.
     * \param stVec An arbitrary spacetime vector.
     * \param nx spatial lattice size.
     * \return Number of time slices `nt = stVec.size() / nx`.
     */
    template <typename VT>
    std::size_t getNt(const VT &stVec, const std::size_t nx) noexcept(ndebug) {
#ifndef NDEBUG
        if (stVec.size() % nx != 0)
            throw std::runtime_error("Vector dimension does not match, expected a spacetime vector.");
#endif
        return stVec.size() / nx;
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
        if (blaze::size(vec) == 0)
            getLogger("cpp.math").warning("Size of vector is zero in spacevec.");
        return blaze::subvector(std::forward<VT>(vec), t*nx, nx);
#else
        return blaze::subvector(std::forward<VT>(vec), t*nx, nx, blaze::unchecked);
#endif
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
        if (blaze::rows(mat) == 0)
            getLogger("cpp.math").warning("Number of rows of matrix is zero in spacemat.");
        if (blaze::columns(mat) == 0)
            getLogger("cpp.math").warning("Number of columns of matrix is zero in spacemat.");
        return blaze::submatrix(std::forward<MT>(mat), tp*nx, t*nx, nx, nx);
#else
        return blaze::submatrix(std::forward<MT>(mat), tp*nx, t*nx, nx, nx, blaze::unchecked);
#endif
    }

    /// Invert a matrix in place.
    /**
     * \param mat Matrix to be inverted. Is replaced by the inverse.
     * \param ipiv Pivot indices. Must have at least `mat.rows()` elements
     *             but can be uninitialized.
     */
    template <typename ET>
    void invert(Matrix<ET> &mat, std::unique_ptr<int[]> &ipiv) {
        ISLE_PROFILE_NVTX_RANGE("invert");
        blaze::getrf(mat, ipiv.get());
        blaze::getri(mat, ipiv.get());
    }

    /// Compute eigenvalues to check whether a matrix is invertible.
    template <typename MT>
    bool isInvertible(MT mat, const double eps=1e-15) {
        isle::CDVector eigenvals;
        blaze::geev(mat, eigenvals);
        return blaze::min(blaze::abs(eigenvals)) > eps;
    }

    /// Matrix exponential of real, symmetrix matrices.
    /**
     * \throws std::runtime_error if `mat` has a zero eigenvalue.
     * \param mat Real, symmetric matrix to exponentiate.
     * \return Matrix exponential of mat.
     */
    inline DMatrix expmSym(const DMatrix &mat) {
        // compute eigenvalues evals and eigenvectors U
        DMatrix U = mat;
        DVector evals;
        blaze::syev(U, evals, 'V', 'U');

        // diagonalize mat and exponentiate
        DMatrix diag(mat.rows(), mat.columns(), 0);
        blaze::diagonal(diag) = blaze::exp(blaze::diagonal(U * mat * blaze::trans(U)));
        // transform back to non-diagonal matrix
        return blaze::trans(U) * diag * U;
    }

    /// Compute the logarithm of the determinant of a dense matrix; overwrites the input.
    /**
     * \warning This version overwrites the input matrix. See logdet() for a version that
     *          does not change it.
     * \tparam MT Specific matrix type, must be a blaze dense matrix.
     * \param matrix Matrix to compute the determinant of; must be square.
     * \return \f$y = \log \det(\mathrm{mat})\f$ as a complex number
     *         projected onto the first Riemann sheet of the logarithm,
     *         i.e. \f$y \in (-\pi, \pi]\f$.
     */
    template <typename MT>
    auto ilogdet(MT &matrix) {
        static_assert(blaze::IsDenseMatrix<MT>::value, "logdet needs dense matrices");
        ISLE_PROFILE_NVTX_RANGE("ilogdet");

        using ET = ValueType_t<typename MT::ElementType>;
        const std::size_t n = matrix.rows();
#ifndef NDEBUG
        if (n != matrix.columns())
            throw std::invalid_argument("Invalid non-square matrix provided");
#endif

        #ifdef USE_CUDA

        ilogdet_wrapper();

        #else // USE_CUDA

        // pivot indices
        std::unique_ptr<int[]> ipiv = std::make_unique<int[]>(n);
        // perform LU decomposition (mat = PLU)
        blaze::getrf(matrix, ipiv.get());

        std::complex<ET> res = 0;
        bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
        for (std::size_t i = 0; i < n; ++i) {
            // determinant of pivot matrix P
            if (ipiv[i]-1 != blaze::numeric_cast<int>(i)) {
                negDetP = !negDetP;
            }
            // log det of U (diagonal elements)
            res += std::log(std::complex<ET>{matrix(i, i)});
        }

        #endif // USE_CUDA

        // combine log dets and project to (-pi, pi]
        return toFirstLogBranch(res + (negDetP ? std::complex<ET>{0, pi<ET>} : 0));
    }

    /// Compute the logarithm of the determinant of a dense matrix.
    /**
     * Note that the matrix is copied in order to leave the original unchanged.
     * See ilogdet() for an in-place version.
     * \tparam MT Specific matrix type, must be a blaze dense matrix.
     * \param matrix Matrix to compute the determinant of; must be square.
     * \return \f$y = \log \det(\mathrm{mat})\f$ as a complex number
     *         projected onto the first Riemann sheet of the logarithm,
     *         i.e. \f$y \in (-\pi, \pi]\f$.
     */
    template <typename MT>
    auto logdet(const MT &matrix) {
        ISLE_PROFILE_NVTX_RANGE("logdet");
        static_assert(blaze::IsDenseMatrix<MT>::value, "logdet needs dense matrices");

        using ET = ValueType_t<typename MT::ElementType>;
        MT mat{matrix};  // need to copy here in order to disambiguate from overload for rvalues
        const auto n = mat.rows();
#ifndef NDEBUG
        if (n != mat.columns())
            throw std::invalid_argument("Invalid non-square matrix provided");
#endif

        // pivot indices
        auto ipiv = std::make_unique<int[]>(n);
        // perform LU-decomposition, afterwards matrix = PLU
        blaze::getrf(mat, ipiv.get());

        std::complex<ET> res = 0;
        bool negDetP = false;  // if true det(P) == -1, else det(P) == +1
        for (std::size_t i = 0; i < n; ++i) {
            // determinant of pivot matrix P
            if (ipiv[i]-1 != blaze::numeric_cast<int>(i)) {
                negDetP = !negDetP;
            }
            // log det of U (diagonal elements)
            res += std::log(std::complex<ET>{mat(i, i)});
        }
        // combine log dets and project to (-pi, pi]
        return toFirstLogBranch(res + (negDetP ? std::complex<ET>{0, pi<ET>} : 0));
    }

}  // namespace isle

#endif  // ndef MATH_HPP
