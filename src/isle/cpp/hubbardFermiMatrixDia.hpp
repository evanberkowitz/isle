/** \file
 * \brief Hubbard model fermion matrices with hopping matrix on the diagonal.
 */

#ifndef HUBBARD_FERMI_MATRIX_DIA_HPP
#define HUBBARD_FERMI_MATRIX_DIA_HPP

#include <vector>
#include <functional>

#include "math.hpp"
#include "lattice.hpp"
#include "cache.hpp"
#include "species.hpp"

namespace isle {

    /// Represents a fermion matrix \f$M\f$ for the Hubbard model with hopping matrix on the diagonal.
    /**
     * See <B>`docs/algorithms/hubbardFermiAction.pdf`</B> for definitions.
     * All member functions of this class are named after the corresponting matrices.
     *
     * `%HubbardFermiMatrixDia` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \mathrm{and}\, \sigma_\tilde{\kappa}\f$
     * as inputs and can construct the individual blocks \f$K, F, P, T^{+}, \mathrm{and}\, T^{-}\f$
     * or the full matrices \f$M, Q\f$ from them.
     *
     * Neither the full matrix nor any of its blocks are stored explicitly. Instead,
     * each block needs to be constructed when needed which might be expensive.
     *
     * The result of an LU-decomposition of \f$Q\f$ is stored in HubbardFermiMatrixDia::LU to save memory
     * and give easier access to the components compared to a `blaze::Matrix`.
     * The only exception are the inversed of \f$K\f$ which are cached after when the function
     * HubbardFermiMatrixDia::Kinv() is first called.
     *
     * \sa
     * Free functions operating on `%HubbardFermimatrixDia`:
     *  - std::complex<double> logdetM(const HubbardFermiMatrixDia &hfm, const CDVector &phi, Species species)
     *  - HubbardFermiMatrixDia::LU getQLU(const HubbardFermiMatrixDia &hfm, const CDVector &phi)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrixDia &hfm, const CDVector &phi)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrixDia::QLU &lu)
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrixDia &hfm, const CDVector &phi, const Vector<std::complex<double>> &rhs);
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrixDia::QLU &lu, const Vector<std::complex<double>> &rhs);
     *
     * See HubbardFermiMatrixExp for an alternative discretization.
     */
    class HubbardFermiMatrixDia {
    public:
        /// Store all necessary parameters to later construct the full fermion matrix.
        /**
         * \param kappaTile Hopping matrix \f$\tilde{\kappa}\f$.
         * \param muTilde Chemical potential \f$\tilde{\mu}\f$.
         * \param sigmaKappa Sign of hopping matrix in hole matrix.
         */
        HubbardFermiMatrixDia(const SparseMatrix<double> &kappaTilde,
                              double muTilde, std::int8_t sigmaKappa);

        HubbardFermiMatrixDia(const Lattice &lat, double beta,
                              double muTilde, std::int8_t sigmaKappa);

        HubbardFermiMatrixDia(const HubbardFermiMatrixDia &other);
        HubbardFermiMatrixDia &operator=(const HubbardFermiMatrixDia &other);
        // I am not motivated enough to implement those too.
        HubbardFermiMatrixDia(HubbardFermiMatrixDia &&) = delete;
        HubbardFermiMatrixDia &operator=(HubbardFermiMatrixDia &&) = delete;
        ~HubbardFermiMatrixDia() = default;

        /// Store the diagonal block K of matrix M in the parameter.
        /**
         * \param k Any old content is erased and the matrix is
         *          resized if need be.
         * \param species Select whether to construct for particles or holes.
         */
        void K(DSparseMatrix &k, Species species) const;

        /// Return the diagonal block matrix K of matrix M.
        /**
         * \param species Select whether to construct for particles or holes.
         */
        DSparseMatrix K(Species species) const;

        /// Return the inverse of the diagonal block matrix K of matrix M.
        /**
         * \param species Select whether to construct for particles or holes.
         */
        const DMatrix &Kinv(Species species) const;

        /// Store an off-diagonal block F of matrix M in the parameter.
        /**
         * \param f Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         * \param inv If `true` constructs the inverse of F.
         */
        void F(CDSparseMatrix &f, std::size_t tp, const CDVector &phi,
               Species species, bool inv=false) const;

        /// Return an off-diagonal block matrix F of matrix M.
        /**
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         * \param inv If `true` constructs the inverse of F.
         */
        CDSparseMatrix F(std::size_t tp, const CDVector &phi,
                         Species species, bool inv=false) const;

        /// Store the matrix \f$M\f$ in the parameter.
        /**
         * \param m Any old content is erased and the matrix is
         *          resized if need be.
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         */
        void M(CDSparseMatrix &m, const CDVector &phi, Species species) const;

        /// Return the matrix \f$M\f$.
        /**
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         */
        CDSparseMatrix M(const CDVector &phi, Species species) const;

        /// Store the block on the diagonal \f$P\f$ in the parameter.
        /**
         * \param p Block on the diagonal. Any old content is erased and the matrix is
         *          resized if need be.
         */
        void P(DSparseMatrix &p) const;

        /// Return the block on the diagonal \f$P\f$.
        DSparseMatrix P() const;

        /// Store the block on the lower subdiagonal \f$T^{+}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the lower subdiagonal. Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        void Tplus(CDSparseMatrix &T, std::size_t tp, const CDVector &phi) const;

        /// Return the block on the lower subdiagonal \f$T^{+}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        CDSparseMatrix Tplus(std::size_t tp, const CDVector &phi) const;

        /// Store the block on the upper subdiagonal \f$T^{-}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the upper subdiagonal.
         *          Any old content is erased and the matrix is resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        void Tminus(CDSparseMatrix &T, std::size_t tp, const CDVector &phi) const;

        /// Return the block on the upper subdiagonal \f$T^{-}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        CDSparseMatrix Tminus(std::size_t tp, const CDVector &phi) const;

        /// Store the full fermion matrix \f$Q\f$ in the parameter.
        /**
         * \param q Full fermion matrix. Any old content is erased and the matrix is
         *          resized if need be.
         * \param phi Auxilliary field.
         */
        void Q(CDSparseMatrix &q, const CDVector &phi) const;

        /// Return the full fermion matrix.
        /**
         * \param phi Auxilliary field.
         */
        CDSparseMatrix Q(const CDVector &phi) const;

        /// Update the hopping matrix
        void updateKappaTilde(const DSparseMatrix &kappaTilde);

        /// Update the chemical potential.
        void updateMuTilde(double muTilde);

        /// Return hopping matrix.
        const DSparseMatrix &kappaTilde() const noexcept;

        /// Return chemical potential.
        double muTilde() const noexcept;

        /// Return sign of kappa in hole matrix.
        std::int8_t sigmaKappa() const noexcept;

        /// Spatial size of the lattice.
        std::size_t nx() const noexcept;

        /// Result of an LU-decomposition of Q.
        /**
         * See documentation of HubbardFermiMatrixDia for the definition of
         * all member variables.
         *
         * If set up properly, the sizes of the member vectors are:
         *   nt | 1  | 2  | >2
         * -----|----|----|----
         * dinv | nt | nt | nt
         * u    | 0  | 1  | nt-1
         * l    | 0  | 1  | nt-1
         * v    | 0  | 0  | nt-2
         * h    | 0  | 0  | nt-2
         *
         * Use HubbardFermiMatrixDia::QLU::isConsistent() to check whether those conditions
         * are satisfied.
         */
        struct QLU {
            std::vector<CDMatrix> dinv; ///< \f$d^{-1}\f$, see definition of U.
            std::vector<CDMatrix> u; ///< See definition of U.
            std::vector<CDMatrix> v; ///< See definition of U.
            std::vector<CDMatrix> l; ///< See definition of L.
            std::vector<CDMatrix> h; ///< See definition of L.

            /// Reserves space for std::vectors but does not construct matrices.
            explicit QLU(std::size_t nt);

            /// Check whether an instance is set up properly, i.e. all vectors have consistent sizes.
            bool isConsistent() const;

            /// Reconstruct the fermion matrix as a dense matrix.
            CDMatrix reconstruct() const;
        };

    private:
        DSparseMatrix _kappa;  ///< Hopping matrix.
        double _mu;  ///< Chemical potential.
        std::int8_t _sigmaKappa;  ///< Sign of kappa in M^dag.

        /// K^-1 for particles.
        Cache<DMatrix, std::function<DMatrix()>> _kinvp;
        /// K^-1 for holes.
        Cache<DMatrix, std::function<DMatrix()>> _kinvh;

        void _invalidateKCaches() noexcept;
    };


    /// Perform an LU-decomposition of Q.
    /**
     * \param phi Auxilliary field.
     */
    HubbardFermiMatrixDia::QLU getQLU(const HubbardFermiMatrixDia &hfm,
                                      const CDVector &phi);

    /// Solve a system of equations \f$Q x = b\f$.
    /**
     * \param hfm Encodes matrix \f$Q\f$.
     * \param phi Auxilliary field.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     * \see `std::complex<double> solveQ(const HubbardFermiMatrixDia::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixDia &hfm,
                                        const CDVector &phi,
                                        const Vector<std::complex<double>> &rhs);

    /// Solve a system of equations \f$Q x = b\f$; use LU-decomposition directly.
    /**
     * \param lu LU-Decomposition of matrix \f$Q\f$.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixDia::QLU &lu,
                                        const Vector<std::complex<double>> &rhs);

    /// Compute \f$\log(\det(Q))\f$ by means of an LU-decomposition.
    /**
     * \param hfm %HubbardFermiMatrixDia to compute the determinant of.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q()))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdet(const HubbardFermiMatrixDia::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    std::complex<double> logdetQ(const HubbardFermiMatrixDia &hfm, const CDVector &phi);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition.
    /**
     * \param lu LU-decomposed Q.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see
     *      - `std::complex<double> logdetQ(const HubbardFermiMatrixDia &hfm)`
     *      - `std::complex<double> ilogdetQ(HubbardFermiMatrixDia::QLU &lu)`
     */
    std::complex<double> logdetQ(const HubbardFermiMatrixDia::QLU &lu);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition, overwrites input.
    /**
     * \warning This version operates in-place and overwrites the input parameter `lu`.
     *          See `std::complex<double> logdetQ(const HubbardFermiMatrixDia::QLU &lu)` for
     *          a version that preserves `lu`.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdetQ(const HubbardFermiMatrixDia &hfm)`
     */
    std::complex<double> ilogdetQ(HubbardFermiMatrixDia::QLU &lu);

    /// Compute \f$\log(\det(M))\f$.
    /**
     * \todo Is the new form stable for mu != 0?
     *       What about complex phi?
     * \param hfm %HubbardFermiMatrixDia to compute the determinant of.
     * \param phi Auxilliary field.
     * \param species Select whether to use particles or holes.
     * \return Value equivalent to `log(det(hfm.M()))` and projected onto the
     *         first branch of the logarithm.
     */
    std::complex<double> logdetM(const HubbardFermiMatrixDia &hfm, const CDVector &phi,
                                 Species species);

    /// Solve a system of equations \f$M x = b\f$.
    /**
     * Can be called for multiple right hand sides b in order to re-use parts of
     * the calculation.
     *
     * \note The shape of `rhs` and the return value may be counter intuitive.
     *       The first index is the number of the right hand side vector,
     *       the second index is space-time.
     *       That is space-time point (t, x) of right hand side i is `rhs(i, t*nx+x)`.
     *       That means, for instance, that the inverse of M is
     *       `transpose(solveM(hfm, phi, species, unity))`.
     *
     * \param hfm Represents matrix M which describes the system of equations.
     * \param phi Gauge configuration needed to construct M.
     * \param species Select whether to solve for particles or holes.
     * \param rhs Right hand sides b.
     * \returns Results x, same shape as rhs.
     */
    CDMatrix solveM(const HubbardFermiMatrixDia &hfm, const CDVector &phi,
                    const Species species, const CDMatrix &rhss);


}  // namespace isle

#endif  // ndef HUBBARD_FERMI_MATRIX_DIA_HPP
