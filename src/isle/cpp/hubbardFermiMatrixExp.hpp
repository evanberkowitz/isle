/** \file
 * \brief Hubbard model fermion matrices with hopping matrix in an exponential.
 */

#ifndef HUBBARD_FERMI_MATRIX_EXP_HPP
#define HUBBARD_FERMI_MATRIX_EXP_HPP

#include <vector>
#include <functional>

#include "math.hpp"
#include "lattice.hpp"
#include "cache.hpp"
#include "species.hpp"

namespace isle {

    /// Represents a fermion matrix \f$\hat{M}\f$ for the Hubbard model with hopping matrix in an exponential.
    /**
     * See <B>`docs/algorithms/hubbardFermiAction.pdf`</B> for definitions.
     * All member functions of this class are named after the corresponting matrices (sans the hat).
     *
     * `%HubbardFermiMatrixExp` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \mathrm{and}\, \sigma_\tilde{\kappa}\f$
     * as inputs and can construct the individual blocks \f$\hat{K}, \hat{F}, \hat{P}, \hat{T}^{+}, \mathrm{and}\, \hat{T}^{-}\f$
     * or the full matrices \f$\hat{M}, \hat{Q}\f$ from them.
     *
     * Neither the full matrices nor any of their blocks are stored explicitly. Instead,
     * each block is to be constructed when needed which might be expensive.
     * The only exception is \f$e^{-\tilde{\kappa}+\tilde{\mu}}\f$ which is cached
     * after it has been requested through HubbardFermiMatrixExp::expKappa() for the first time.
     *
     * The result of an LU-decomposition of \f$\hat{Q}\f$ is stored in HubbardFermiMatrixExp::LU
     * to save memory and give easier access to the components compared to a `blaze::Matrix`.
     *
     * \sa
     * Free functions operating on `%HubbardFermimatrixExp`:
     *  - std::complex<double> logdetM(const HubbardFermiMatrixExp &hfm, const CDVector &phi, Species species)
     *  - HubbardFermiMatrixExp::LU getQLU(const HubbardFermiMatrixExp &hfm, const CDVector &phi)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrixExp &hfm, const CDVector &phi)
     *  - std::complex<double> logdetQ(const HubbardFermiMatrixExp::QLU &lu)
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp &hfm, const CDVector &phi, const Vector<std::complex<double>> &rhs);
     *  - Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp::QLU &lu, const Vector<std::complex<double>> &rhs);
     *
     * See HubbardFermiMatrixDia for an alternative discretization.
     */
    class HubbardFermiMatrixExp {
    public:

        /// Store all necessary parameters to later construct the full fermion matrix.
        /**
         * \param kappaTilde Hopping matrix \f$\tilde{\kappa}\f$.
         * \param mu ChemicalTilde potential \f$\tilde{\mu}\f$.
         * \param sigmaKappa Sign of hopping matrix in hole matrix.
         */
        HubbardFermiMatrixExp(const SparseMatrix<double> &kappaTilde,
                              double muTilde, std::int8_t sigmaKappa);

        HubbardFermiMatrixExp(const Lattice &lat, double beta,
                              double muTilde, std::int8_t sigmaKappa);

        HubbardFermiMatrixExp(const HubbardFermiMatrixExp &) = default;
        HubbardFermiMatrixExp &operator=(const HubbardFermiMatrixExp &) = default;
        HubbardFermiMatrixExp(HubbardFermiMatrixExp &&) = default;
        HubbardFermiMatrixExp &operator=(HubbardFermiMatrixExp &&) = default;
        ~HubbardFermiMatrixExp() = default;

        /// Return the exponential of the hopping amtrix and chemical potential.
        /**
         * \returns \f$\exp(-\tilde{\kappa} + \tilde{\mu})\f$ for particles
         *          and \f$\exp(-\sigma_{\tilde{\kappa}}\tilde{\kappa} - \tilde{\mu})\f$ for holes.
         */
        const DMatrix &expKappa(const Species species) const;

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
        DMatrix Kinv(Species species) const;

        /// Return log(det(K^-1)).
        /**
         * \param species Select whether to construct for particles or holes.
         */
        std::complex<double> logdetKinv(Species species) const;

        /// Store an off-diagonal block F of matrix M in the parameter.
        /**
         * \param f Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         * \param inv If `true` constructs the inverse of F.
         */
        void F(CDMatrix &f, std::size_t tp, const CDVector &phi,
               Species species, bool inv=false) const;

        /// Return an off-diagonal block matrix F of matrix M.
        /**
         * \param tp Temporal row index t'.
         * \param phi Auxilliary field.
         * \param species Select whether to construct for particles or holes.
         * \param inv If `true` constructs the inverse of F.
         */
        CDMatrix F(std::size_t tp, const CDVector &phi,
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
        void P(DMatrix &p) const;

        /// Return the block on the diagonal \f$P\f$.
        DMatrix P() const;

        /// Store the block on the lower subdiagonal \f$T^{+}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the lower subdiagonal. Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        void Tplus(CDMatrix &T, std::size_t tp, const CDVector &phi) const;

        /// Return the block on the lower subdiagonal \f$T^{+}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        CDMatrix Tplus(std::size_t tp, const CDVector &phi) const;

        /// Store the block on the upper subdiagonal \f$T^{-}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the upper subdiagonal.
         *          Any old content is erased and the matrix is resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        void Tminus(CDMatrix &T, std::size_t tp, const CDVector &phi) const;

        /// Return the block on the upper subdiagonal \f$T^{-}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         * \param phi Auxilliary field.
         */
        CDMatrix Tminus(std::size_t tp, const CDVector &phi) const;

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
         * See documentation of HubbardFermiMatrixExp for the definition of
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
         * Use HubbardFermiMatrixExp::QLU::isConsistent() to check whether those conditions
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

        /// exp(kappaTilde) for particles.
        DMatrix _expKappap;
        /// exp(sigmaKappa*kappaTilde) for holes.
        DMatrix _expKappah;
    };


    /// Perform an LU-decomposition of Q.
    /**
     * \param phi Auxilliary field.
     */
    HubbardFermiMatrixExp::QLU getQLU(const HubbardFermiMatrixExp &hfm, const CDVector &phi);

    /// Solve a system of equations \f$Q x = b\f$.
    /**
     * \param hfm Encodes matrix \f$Q\f$.
     * \param phi Auxilliary field.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     * \see `std::complex<double> solveQ(const HubbardFermiMatrixExp::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp &hfm,
                                        const CDVector &phi,
                                        const Vector<std::complex<double>> &rhs);

    /// Solve a system of equations \f$Q x = b\f$; use LU-decomposition directly.
    /**
     * \param lu LU-Decomposition of matrix \f$Q\f$.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     */
    Vector<std::complex<double>> solveQ(const HubbardFermiMatrixExp::QLU &lu,
                                        const Vector<std::complex<double>> &rhs);

    /// Compute \f$\log(\det(Q))\f$ by means of an LU-decomposition.
    /**
     * \param hfm %HubbardFermiMatrixExp to compute the determinant of.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q()))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdet(const HubbardFermiMatrixExp::QLU &lu)` in case you
     *      already have the LU-decomposition of Q.
     */
    std::complex<double> logdetQ(const HubbardFermiMatrixExp &hfm, const CDVector &phi);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition.
    /**
     * \param lu LU-decomposed Q.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see
     *      - `std::complex<double> logdetQ(const HubbardFermiMatrixExp &hfm)`
     *      - `std::complex<double> ilogdetQ(HubbardFermiMatrixExp::QLU &lu)`
     */
    std::complex<double> logdetQ(const HubbardFermiMatrixExp::QLU &lu);

    /// Compute \f$\log(\det(Q))\f$ given an LU-decomposition, overwrites input.
    /**
     * \warning This version operates in-place and overwrites the input parameter `lu`.
     *          See `std::complex<double> logdetQ(const HubbardFermiMatrixExp::QLU &lu)` for
     *          a version that preserves `lu`.
     * \param phi Auxilliary field.
     * \return Value equivalent to `log(det(hfm.Q))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdetQ(const HubbardFermiMatrixExp &hfm)`
     */
    std::complex<double> ilogdetQ(HubbardFermiMatrixExp::QLU &lu);

    /// Compute \f$\log(\det(M))\f$.
    /**
     * \param hfm %HubbardFermiMatrixExp to compute the determinant of.
     * \param phi Auxilliary field.
     * \param species Select whether to use particles or holes.
     * \return Value equivalent to `log(det(hfm.M()))` and projected onto the
     *         first branch of the logarithm.
     */
    std::complex<double> logdetM(const HubbardFermiMatrixExp &hfm, const CDVector &phi,
                                 Species species);

    /// Solve a system of equations \f$M x = b\f$.
    /**
     * Can be called for multiple right hand sides b in order to re-use parts of
     * the calculation.
     *
     * \todo on-the-fly check in debug mode
     *
     * \param hfm Represents matrix M which describes the system of equations.
     * \param phi Gauge configuration needed to construct M.
     * \param species Select whether to solve for particles or holes.
     * \param rhs Right hand sides b.
     * \returns Results x, same length as rhs.
     */
    std::vector<CDVector> solveM(const HubbardFermiMatrixExp &hfm,
                                 const CDVector &phi,
                                 Species species,
                                 const std::vector<CDVector> &rhs);

}  // namespace isle

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
