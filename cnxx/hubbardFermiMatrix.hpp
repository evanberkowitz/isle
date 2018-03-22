/** \file
 * \brief Hubbard model fermion matrices.
 */

#ifndef HUBBARD_FERMI_MATRIX_HPP
#define HUBBARD_FERMI_MATRIX_HPP

#include "math.hpp"

#include <vector>

namespace cnxx {

    /// Represents a fermion matrix \f$Q\f$ for the Hubbard model.
    /**
     * \todo Move documentations to proper place.
     *
     * See notes ReWeighting in learning-physics.
     *
     * ## LU-decomposition
     * In the case of the fermion matrix, each element of \f$L\f$ and \f$U\f$ is a spacial
     * \f$N_x \times N_x\f$ matrix. Hence each inverse of \f$d\f$ requires a full
     * matrix inversion. This means that the algorithm requires \f$\mathrm{O}(N_t N_x^3)\f$
     * steps.
     * After the initial steps, all matrices except for most \f$u\f$'s are dense. For simplicity, all inversions
     * are performed using <TT>LAPACK</TT>.
     *
     * The algorithm for \f$N_t \in \{1,2\}\f$ is slightly different from the one shown
     * here but handled correctly by the implementation. See notes in HubbardFermiMatrix::LU
     * on the sizes of its members.
     *
     *
     * ## Solver
     * A linear system of equations \f$(M M^\dagger) x = b\f$ can be solved via an
     * LU-Decomposition and forward-/back-substitution.
     *
     * ### Matrix-Vector Equation
     * Solve a system of equations for a single right hand side, i.e. \f$x\f$ and \f$b\f$
     * are vetors. Start by solving \f$L y = b\f$:
     \f{align}{
     y_0 &= b_0\\
     y_i &= b_i - l_{i-1} y_{i-1}\quad \forall i \in [1, N_t-2]\\
     y_{N_t-1} &= b_{N_t-1} - l_{N_t-2} y_{N_t-2} - \textstyle\sum_{j=0}^{N_t-3}\, h_j y_j
     \f}
     * Then solve \f$Ux = y\f$:
     \f{align}{
     x_{N_t-1} &= d_{N_t-1}^{-1} y_{N_t-1}\\
     x_{N_t-2} &= d_{N_t-2}^{-1} (y_{N_t-2} - u_{N_t-2} x_{N_t-1})\\
     x_i &= d_i^{-1} (y_i - u_i x_{i+1} - v_i x_{N_t-1}) \quad \forall i \in [0, N_t-3]
     \f}
     * Since the inversed \f$d_i^{-1}\f$ have already been computed by the LU-factorization,
     * the solver alone requires only \f$\mathcal{O}(N_t N_x^2)\f$ steps.
     *
     * ### Matrix-Matrix Equation
     * Invert the hubbard fermi matrix by solving \f$(M M^\dagger) X^{-1} = \mathbb{I}\f$,
     * where \f$I\f$ is the \f$N_t N_x\f$ unit matrix.<BR>
     * Start by solving \f$L Y^{-1} = I\f$ and denote the elements of \f$Y^{-1}\f$
     * by \f$y_{ij}\f$; note that the \f$y_{ij}\f$ are spatial matrices.
     \f{align}{
     y_{0j} &= \delta_{0j}\\
     y_{ij} &= \begin{cases}
     \textstyle\prod_{k=j}^{i-1} (-l_{k}) & \mathrm{for}\quad i > j\\
     \delta_{ij} & \mathrm{for}\quad i\leq j
     \end{cases}, \quad\forall i \in [1, N_t-2]\\
     y_{(N_t-1)j} &= \delta_{(N_t-1)j} - \textstyle\sum_{k=j}^{N_t-3} h_{k} y_{kj} - l_{N_t-2} y_{(N_t-2)j}
     \f}
     * Then solve \f$U X^{-1} = Y^{-1}\f$:
     \f{align}{
     x_{(N_t-1)j} &= d_{N_t-1}^{-1}y_{(N_t-1)j}\\
     x_{(N_t-2)j} &= d_{N_t-2}^{-1}(y_{(N_t-2)j} - u_{N_t-2}x_{(N_t-1)j})\\
     x_{ij} &= d_{i}^{-1}(y_{ij} - u_{i}x_{(i+1)j} - v_{i}x_{(N_t-1)j}) \quad\forall i \in [0, N_t-3]
     \f}
     *
     * Most of those relations can be read off immediately, the others can be proven using
     * simple induction.<BR>
     * Apart from the last row, the \f$y\f$'s are independent from each other while the \f$x\f$'s
     * have to be computed iterating over rows from \f$N_t-1\f$ though 0. However, different
     * columns never mix.
     *
     *
     * ## Usage
     * `%HubbardFermiMatrix` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \mathrm{and} \sigma_\tilde{\kappa}\f$
     * as inputs and can construct the individual blocks \f$P, T^{+}, \mathrm{and} T^{-}\f$ or
     * the full matrix \f$Q\f$ from them.
     *
     * Neither the full matrix nor any of its blocks are stored explicitly. Instead,
     * each block needs to be constructed using P(), Tplus(), and Tminus() or Q() for the
     * full matrix. Note that these operations are fairly expensive.
     *
     * The result of an LU-decomposition is stored in HubbardFermiMatrix::LU to save memory
     * and give easier access to the components compared to a `blaze::Matrix`.
     *
     * \sa
     *  - HubbardFermiMatrix::LU getLU(const HubbardFermiMatrix &hfm)
     *  - std::complex<double> logdet(const HubbardFermiMatrix &hfm)
     *  - std::complex<double> logdet(const HubbardFermiMatrix::LU &lu)
     *  - Vector<std::complex<double>> solve(const HubbardFermiMatrix &hfm, const Vector<std::complex<double>> &rhs);
     *  - Vector<std::complex<double>> solve(const HubbardFermiMatrix::LU &lu, const Vector<std::complex<double>> &rhs);
     */
    class HubbardFermiMatrix {
    public:
        /// Store all necessary parameters to later construct the full fermion matrix.
        /**
         * \attention Input parameters `kappa` and `mu` are versions with tilde, i.e.
         *            scaled like \f$\tilde{\kappa} = \beta/N_t \kappa\f$.
         *
         * \param kappa Hopping matrix \f$\tilde{\kappa}\f$.
         * \param phi Auxilliary field \f$\phi\f$ from HS transformation.
         * \param mu Chemical potential \f$\tilde{\mu}\f$.
         * \param sigmaKappa Sign of hopping matrix in adjoint matrix.
         */
        HubbardFermiMatrix(const SparseMatrix<double> &kappa,
                           const Vector<std::complex<double>> &phi,
                           double mu, std::int8_t sigmaKappa);

        /// Store the matrix \f$M\f$ in the parameter.
        /**
         * \param m Any old content is erased and the matrix is
         *          resized if need be.
         */
        void M(SparseMatrix<std::complex<double>> &m, bool dagger) const;

        /// Return the matrix \f$M\f$.
        SparseMatrix<std::complex<double>> M(bool dagger) const;

        /// Store the block on the diagonal \f$P\f$ in the parameter.
        /**
         * \param p Block on the diagonal. Any old content is erased and the matrix is
         *          resized if need be.
         */
        void P(SparseMatrix<double> &p) const;

        /// Return the block on the diagonal \f$P\f$.
        SparseMatrix<double> P() const;

        /// Store the block on the lower subdiagonal \f$T^{+}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the lower subdiagonal. Any old content is erased and the matrix is
         *          resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         */
        void Tplus(SparseMatrix<std::complex<double>> &T, std::size_t tp) const;

        /// Return the block on the lower subdiagonal \f$T^{+}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         */
        SparseMatrix<std::complex<double>> Tplus(std::size_t tp) const;

        /// Store the block on the upper subdiagonal \f$T^{-}^{\dagger}_{t'}\f$ in a parameter.
        /**
         * Applies anti periodic boundary conditions.
         * \param T Block on the upper subdiagonal.
         *          Any old content is erased and the matrix is resized if need be.
         * \param tp Temporal row index \f$t'\f$.
         */
        void Tminus(SparseMatrix<std::complex<double>> &T, std::size_t tp) const;

        /// Return the block on the upper subdiagonal \f$T^{-}^{\dagger}_{t'}\f$.
        /**
         * \param tp Temporal row index \f$t'\f$.
         */
        SparseMatrix<std::complex<double>> Tminus(std::size_t tp) const;

        /// Store the full fermion matrix \f$Q\f$ in the parameter.
        /**
         * \param q Full fermion matrix. Any old content is erased and the matrix is
         *          resized if need be.
         */
        void Q(SparseMatrix<std::complex<double>> &q) const;

        /// Return the full fermion matrix.
        SparseMatrix<std::complex<double>> Q() const;

        /// Update the hopping matrix
        void updateKappa(const SparseMatrix<double> &kappa);

        /// Update the chemical potential.
        void updateMu(double mu);
        
        /// Update auxilliary HS field.
        void updatePhi(const Vector<std::complex<double>> &phi);

        /// Return number of spacial lattice sites; deduced from kappa.
        std::size_t nx() const noexcept;

        /// Return number of temporal lattice sites; deduced from kappa and phi.
        std::size_t nt() const noexcept;


        /// Result of an LU-decomposition.
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
         * Use HubbardFermiMatrix::LU::isConsistent() to check whether those conditions
         * are satisfied.
         */
        struct LU {
            std::vector<Matrix<std::complex<double>>> dinv; ///< \f$d^{-1}\f$, see definition of U.
            std::vector<Matrix<std::complex<double>>> u; ///< See definition of U.
            std::vector<Matrix<std::complex<double>>> v; ///< See definition of U.
            std::vector<Matrix<std::complex<double>>> l; ///< See definition of L.
            std::vector<Matrix<std::complex<double>>> h; ///< See definition of L.

            /// Reserves space for std::vectors but does not construct matrices.
            explicit LU(std::size_t nt);

            /// Check whether an instance is set up properly, i.e. all vectors have consistent sizes.
            bool isConsistent() const;

            /// Reconstruct the fermion matrix as a dense matrix.
            Matrix<std::complex<double>> reconstruct() const;
        };

    private:
        SparseMatrix<double> _kappa;  ///< Hopping matrix.
        Vector<std::complex<double>> _phi;  ///< Auxilliary HS field.
        double _mu;              ///< Chemical potential.
        std::int8_t _sigmaKappa; ///< Sign of kappa in M^dag.
    };


    /// Perform an LU-decomposition on a HubbardFermiMatrix.
    HubbardFermiMatrix::LU getLU(const HubbardFermiMatrix &hfm);

    /// Solve a system of equations \f$(M M^\dagger) x = b\f$.
    /**
     * See documentation of HubbardFermiMatrix for a description of the algorithm.
     * \param hfm Matrix \f$M M^\dagger\f$.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     * \see `std::complex<double> logdet(const HubbardFermiMatrix::LU &lu)` in case you
     *      already have the LU-decomposition.
     */
    Vector<std::complex<double>> solve(const HubbardFermiMatrix &hfm,
                                       const Vector<std::complex<double>> &rhs);

    /// Solve a system of equations \f$(M M^\dagger) x = b\f$; use LU-decomposition directly.
    /**
     * See documentation of HubbardFermiMatrix for a description of the algorithm.
     * \param lu LU-Decomposition of matrix \f$M M^\dagger\f$.
     * \param rhs Right hand side \f$b\f$.
     * \return Solution \f$x\f$.
     */
    Vector<std::complex<double>> solve(const HubbardFermiMatrix::LU &lu,
                                       const Vector<std::complex<double>> &rhs);

    /// Compute \f$\log(\det(M M^\dagger))\f$ by means of an LU-decomposition.
    /**
     * \param hfm %HubbardFermiMatrix to compute the determinant of.
     * \return Value equivalent to `log(det(hfm.MMdag()))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdet(const HubbardFermiMatrix::LU &lu)` in case you
     *      already have the LU-decomposition.
     */
    std::complex<double> logdet(const HubbardFermiMatrix &hfm);

    /// Compute \f$\log(\det(M M^\dagger))\f$ given an LU-decomposition.
    /**
     * \param lu LU-decomposed HubbardFermiMatrix.
     * \return Value equivalent to `log(det(hfm.reconstruct()))` and projected onto the
     *         first branch of the logarithm.
     * \see
     *      - `std::complex<double> logdet(const HubbardFermiMatrix &hfm)`
     *      - `std::complex<double> ilogdet(HubbardFermiMatrix::LU &lu)`
     */
    std::complex<double> logdet(const HubbardFermiMatrix::LU &lu);

    /// Compute \f$\log(\det(M M^\dagger))\f$ given an LU-decomposition, overwrites input.
    /**
     * \warning This version operates in-place and overwrites the input parameter `lu`.
     *          See `std::complex<double> logdet(const HubbardFermiMatrix::LU &lu)` for
     *          a version that preserves `lu`.
     * \param lu LU-decomposed HubbardFermiMatrix.
     * \return Value equivalent to `log(det(hfm.reconstruct()))` and projected onto the
     *         first branch of the logarithm.
     * \see `std::complex<double> logdet(const HubbardFermiMatrix &hfm)`
     */
    std::complex<double> ilogdet(HubbardFermiMatrix::LU &lu);

}  // namespace cnxx

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
