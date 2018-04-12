/** \file
 * \brief Hubbard model fermion matrices.
 */

#ifndef HUBBARD_FERMI_MATRIX_HPP
#define HUBBARD_FERMI_MATRIX_HPP

#include "math.hpp"

#include <vector>

namespace cnxx {

    /// Mark particles and holes.
    enum class PH {PARTICLE, HOLE};

    /// Represents a fermion matrix for the Hubbard model.
    /**
     * ## Definitions
     * The fermion matrix is defined as
     \f{align}{
     {M(\phi, \tilde{\kappa}, \tilde{\mu})}_{x't';xt}
     &\equiv K_{x'x}\delta_{t't} - \mathcal{B}_{t'}{(F_{t'})}_{x'x}\delta_{t'(t+1)}.
     \f}
     * with
     \f{align}{
     K_{x'x} &= (1+\tilde{\mu})\delta_{x'x} - \kappa_{x'x}\\
     {(F_{t'})}_{x'x} &= e^{i\phi_{xt}}\delta_{x'x}
     \f}
     * These relations hold for particles, for holes, replace
     *  \f$\phi \rightarrow -\phi,\, \tilde{\kappa} \rightarrow \sigma_{\tilde{\kappa}}\tilde{\kappa},\, \tilde{\mu}\rightarrow -\tilde{\mu}\f$.
     *
     * The combined matrix is
     \f{align}{
     {Q(\phi, \tilde{\kappa}, \tilde{\mu}, \sigma_{\tilde{\kappa}})}_{x't',xt}
     &= {M(\phi, \tilde{\kappa}, \tilde{\mu})}_{x't',x''t''} {M^T(-\phi, \sigma_{\tilde{\kappa}}\tilde{\kappa}, -\tilde{\mu})}_{x''t'',xt}\\
     &= \delta_{t't}{(P)}_{x'x} + \delta_{t'(t+1)}{(T^+_{t'})}_{x'x} + \delta_{t(t'+1)}{(T^-_{t'})}_{x'x}
     \f}
     * with
     \f{align}{
     {P(\phi, \tilde{\kappa}, \tilde{\mu}, \sigma_{\tilde{\kappa}})}_{x'x} &\equiv (2-\tilde{\mu}^2)\delta_{x'x} - (\sigma_{\tilde{\kappa}}(1+\tilde{\mu}) + (1-\tilde{\mu}))\tilde{\kappa}_{x'x} + \sigma_{\tilde{\kappa}}{(\tilde{\kappa}^2)}_{x'x}\\
     {T^+_{t'}(\phi, \tilde{\kappa}, \tilde{\mu}, \sigma_{\tilde{\kappa}})}_{x'x} &\equiv \mathcal{B}_{t'}e^{i\phi_{x'(t'-1)}}[\sigma_{\tilde{\kappa}}\tilde{\kappa}_{x'x} - (1-\tilde{\mu})\delta_{x'x}]\\
     {T^-_{t'}(\phi, \tilde{\kappa}, \tilde{\mu}, \sigma_{\tilde{\kappa}})}_{x'x} &\equiv \mathcal{B}_{t'+1}e^{-i\phi_{xt'}}[\tilde{\kappa}_{x'x} - (1+\tilde{\mu})\delta_{x'x}]
     \f}
     *
     * Anti-periodic boundary conditions are encoded by
     \f{align}{
     \mathcal{B}_t =
     \begin{cases}
     +1,\quad 0 < t < N_t\\
     -1,\quad t = 0
     \end{cases}
     \f}
     *
     * Derivations and descriptions of algorithms can be found
     * in `docs/algorithms/hubbardFermiAction.pdf`.
     *
     * ## Usage
     * `%HubbardFermiMatrix` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \mathrm{and}\, \sigma_\tilde{\kappa}\f$
     * as inputs and can construct the individual blocks \f$K, F, P, T^{+}, \mathrm{and}\, T^{-}\f$
     * or the full matrices \f$M, Q\f$ from them.
     *
     * Neither the full matrix nor any of its blocks are stored explicitly. Instead,
     * each block needs to be constructed when needed. For this, `%HubbardFermiMatrix`
     * provides member functions with the same names as the matrices.
     *
     * The result of an LU-decomposition is stored in HubbardFermiMatrix::LU to save memory
     * and give easier access to the components compared to a `blaze::Matrix`.
     *
     * \sa
     * Free functions operating on `%HubbardFermimatrix`:
     *  - std::complex<double> logdetM(const HubbardFermiMatrix &hfm)
     *  - HubbardFermiMatrix::LU getQLU(const HubbardFermiMatrix &hfm)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrix &hfm)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrix::QLU &lu)
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrix &hfm, const Vector<std::complex<double>> &rhs);
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrix::QLU &lu, const Vector<std::complex<double>> &rhs);
     */
    class HubbardFermiMatrix {
    public:
        /// Store all necessary parameters to later construct the full fermion matrix.
        /**
         * \attention Input parameters `kappa` and `mu` are versions with tilde, i.e.
         *            scaled as \f$\tilde{\kappa} = \beta/N_t \kappa\f$.
         *
         * \param kappa Hopping matrix \f$\tilde{\kappa}\f$.
         * \param mu Chemical potential \f$\tilde{\mu}\f$.
         * \param sigmaKappa Sign of hopping matrix in hole matrix.
         */
        HubbardFermiMatrix(const SparseMatrix<double> &kappa,
                           double mu, std::int8_t sigmaKappa);

        HubbardFermiMatrix(const HubbardFermiMatrix &) = default;
        HubbardFermiMatrix &operator=(const HubbardFermiMatrix &) = default;
        HubbardFermiMatrix(HubbardFermiMatrix &&) = default;
        HubbardFermiMatrix &operator=(HubbardFermiMatrix &&) = default;
        ~HubbardFermiMatrix() = default;


        /// Store the diagonal block K of matrix M in the parameter.
        /**
         * \param k Any old content is erased and the matrix is
         *          resized if need be.
         * \param ph Select whether to construct for particles or holes.
         */        
        void K(DSparseMatrix &k, PH ph) const;

        /// Return the diagonal block matrix K of matrix M.
        /**
         * \param ph Select whether to construct for particles or holes.
         */
        DSparseMatrix K(PH ph) const;
        
        /// Store an off-diagonal block F of matrix M in the parameter.
        /**
         * \param f Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param ph Select whether to construct for particles or holes.
         * \param inv If `true` consttructs the inverse of F.
         */        
        void F(CDSparseMatrix &f, std::size_t tp, const CDVector &phi,
               PH ph, bool inv=false) const;

        /// Return an off-diagonal block matrix F of matrix M.
        /**
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param ph Select whether to construct for particles or holes.
         * \param inv If `true` consttructs the inverse of F.
         */
        CDSparseMatrix F(std::size_t tp, const CDVector &phi,
                         PH ph, bool inv=false) const;

        /// Store the matrix \f$M\f$ in the parameter.
        /**
         * \param m Any old content is erased and the matrix is
         *          resized if need be.
         * \param phi Auxilliary field.
         * \param ph Select whether to construct for particles or holes.
         */
        void M(CDSparseMatrix &m, const CDVector &phi, PH ph) const;

        /// Return the matrix \f$M\f$.
        /**
         * \param phi Auxilliary field.
         * \param ph Select whether to construct for particles or holes.
         */
        CDSparseMatrix M(const CDVector &phi, PH ph) const;

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
        void updateKappa(const DSparseMatrix &kappa);

        /// Update the chemical potential.
        void updateMu(double mu);

        /// Return hopping matrix.
        const DSparseMatrix &kappa() const noexcept;

        /// Return chemical potential.
        double mu() const noexcept;

        /// Return sign of kappa in hole matrix.
        std::int8_t sigmaKappa() const noexcept;

        /// Spatial size of the lattice.
        std::size_t nx() const noexcept;

        /// Result of an LU-decomposition of Q.
        /**
         * See documentation of HubbardFermiMatrix for the definition of
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
         * Use HubbardFermiMatrix::QLU::isConsistent() to check whether those conditions
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
        double _mu;              ///< Chemical potential.
        std::int8_t _sigmaKappa; ///< Sign of kappa in M^dag.
    };


    /// Perform an LU-decomposition of Q.
    /**
     * \param phi Auxilliary field.
     */
    HubbardFermiMatrix::QLU getQLU(const HubbardFermiMatrix &hfm, const CDVector &phi);

    /// Solve a system of equations \f$Q x = b\f$.
    /**
     * \param hfm Encodes matrix \f$Q\f$.
     * \param phi Auxilliary field.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     * \see `std::complex<double> solveQ(const HubbardFermiMatrix::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrix &hfm,
                                        const CDVector &phi,
                                        const Vector<std::complex<double>> &rhs);

    /// Solve a system of equations \f$Q x = b\f$; use LU-decomposition directly.
    /**
     * \param lu LU-Decomposition of matrix \f$Q\f$.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrix::QLU &lu,
                                        const Vector<std::complex<double>> &rhs);

    /// Compute \f$\log(\det(Q))\f$ by means of an LU-decomposition.
    /**
     * \param hfm %HubbardFermiMatrix to compute the determinant of.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q()))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdet(const HubbardFermiMatrix::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    std::complex<double> logdetQ(const HubbardFermiMatrix &hfm, const CDVector &phi);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition.
    /**
     * \param lu LU-decomposed Q.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see
     *      - `std::complex<double> logdetQ(const HubbardFermiMatrix &hfm)`
     *      - `std::complex<double> ilogdetQ(HubbardFermiMatrix::QLU &lu)`
     */
    std::complex<double> logdetQ(const HubbardFermiMatrix::QLU &lu);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition, overwrites input.
    /**
     * \warning This version operates in-place and overwrites the input parameter `lu`.
     *          See `std::complex<double> logdetQ(const HubbardFermiMatrix::QLU &lu)` for
     *          a version that preserves `lu`.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdetQ(const HubbardFermiMatrix &hfm)`
     */
    std::complex<double> ilogdetQ(HubbardFermiMatrix::QLU &lu);

    /// Compute \f$\log(\det(M))\f$ by means of an LU-decomposition.
    /**
     * \param hfm %HubbardFermiMatrix to compute the determinant of.
     * \param phi Auxilliary field.
     * \param ph Select whether to use particles or holes.
     * \return Value equivalent to `log(det(hfm.M()))` and projected onto the
     *         first branch of the logarithm.
     */
    std::complex<double> logdetM(const HubbardFermiMatrix &hfm, const CDVector &phi,
                                 PH ph);

}  // namespace cnxx

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
