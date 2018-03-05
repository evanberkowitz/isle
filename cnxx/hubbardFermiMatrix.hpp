/** \file
 * \brief Hubbard model fermion matrices.
 */

#ifndef HUBBARD_FERMI_MATRIX_HPP
#define HUBBARD_FERMI_MATRIX_HPP

#include "math.hpp"

#include <vector>


/// Represents a fermion matrix \f$M M^\dagger\f$ for the Hubbard model.
/**
 * ## Definition
 * The fermion matrix is defined as
 \f[
 M[\phi,\tilde{\kappa}, \tilde{\mu}]M^\dagger[\phi,\sigma_{\tilde{\kappa}}\tilde{\kappa},\sigma_{\tilde{\mu}}\tilde{\mu}]
 = P[\phi,\tilde{\kappa},\tilde{\mu}] + Q[\phi,\tilde{\kappa},\tilde{\mu}] + Q^\dagger[\phi,\tilde{\kappa},\tilde{\mu}],
 \f]
 * where \f$\tilde{\kappa}\f$ is the hopping matrix, \f$\tilde{\mu}\f$ the chemical
 * potential. \f$\sigma_\tilde{\mu}\f$ should be -1 for real chemical potential and
 * \f$\sigma_\tilde{\kappa}\f$ can be +1 for tubes and -1 for buckyballs.
 *
 * The individual terms on the right hand side are
 \f{align}{
 {P[\phi,\tilde{\kappa},\tilde{\mu}]}_{x't';xt} &= \delta_{t',t} \big[\delta_{x',x} (2 + \sigma_{\tilde{\mu}}\tilde{\mu}^2 + (1+\sigma_{\tilde{\mu}})\tilde{\mu})
                                                 - \tilde{\kappa}_{x',x} (\sigma_{\tilde{\kappa}}(1+\tilde{\mu}) + 1 + \sigma_{\tilde{\mu}}\tilde{\mu})
                                                 + \sigma_{\tilde{\kappa}} {[\tilde{\kappa}\tilde{\kappa}]}_{x',x}\big],\\
 {Q[\phi,\tilde{\kappa},\tilde{\mu}]}_{x't';xt} &= \delta_{t',t+1} e^{i\phi_{x',t}}\left[\sigma_{\tilde{\kappa}}\tilde{\kappa}_{x',x} - \delta_{x',x}(1+\sigma_{\tilde{\mu}}\tilde{\mu})\right],\\
 {Q^\dagger[\phi,\tilde{\kappa},\tilde{\mu}]}_{x't';xt} &= \delta_{t'+1,t} e^{-i\phi_{x,t'}} \left[\tilde{\kappa}_{x',x} - \delta_{x',x} (1+\tilde{\mu})\right].
 \f}
 * 
 * The matrix can be expressed as a matrix in time where each element is a matrix in space.
 * Doing this gives the cyclic block tridiagonal matrix
 \f[
 M[\phi,\tilde{\kappa}, \tilde{\mu}]M^\dagger[\phi,\sigma_{\tilde{\kappa}}\tilde{\kappa},\sigma_{\tilde{\mu}}\tilde{\mu}] =
 \begin{pmatrix}
   P            & Q^\dagger_0 &         &         &          &              & Q_0        \\
   Q_1          & P       & Q^\dagger_1 &         &          &              &            \\
                & Q_2     & P       & Q^\dagger_2 &          &              &            \\
                &         & Q_3     & P       & \ddots   &              &            \\
                &         &         & \ddots  & \ddots   & Q^\dagger_{N_t-3} &            \\
                &         &         &         & Q_{N_t-2} & P            & Q^\dagger_{N_t-2}\\
   Q^\dagger_{N_t-1} &         &         &         &          & Q_{N_t-1}      & P
 \end{pmatrix},
 \f]
 * where the indices on \f$Q\f$ and \f$Q^\dagger\f$ are the row index \f$t'\f$.
 *
 *
 * ## LU-decomposition
 * Using its sparsity, a cyclic tridiagonal matrix can easily be LU-decomposed with a
 * special purpose algorithm. Use the ansatz
 \f[
   L =
  \begin{pmatrix}
    1   &     &        &        &        &\\
    l_0 & 1   &        &        &        &\\
        & l_1 & \ddots &        &        &\\
        &     & \ddots & 1      &        &\\
        &     &        & l_{N_t-3} & 1      & \\
    h_0 & h_1 & \cdots  & h_{N_t-3} & l_{N_t-2} & 1
  \end{pmatrix}
  \quad U =
  \begin{pmatrix}
    d_0 & u_0 &        &        &        & v_0    \\
        & d_1 & u_1    &        &        & v_1    \\
        &     & d_2    & \ddots &        & \vdots \\
        &     &        & \ddots & u_{N_t-3} & v_{N_t-3} \\
        &     &        &        & d_{N_t-2} & u_{N_t-2} \\
        &     &        &        &        & d_{N_t-1}
  \end{pmatrix}.
 \f]
 * The original matrix can be constructed from these via \f$M M^\dagger = LU\f$.
 * Note that the algorithm used here does not using pivoting.
 * The components of \f$L, U\f$ can be constructed using the following scheme.<BR>
 * Compute all except last \f$d, u, l\f$
 \f[
 \begin{matrix}
 d_0 = P                  & u_0 = Q_0^\dagger & l_0 = Q_1 d_{0}^{-1} & \\
 d_i = P - l_{i-1}u_{i-1} & u_i = Q_i^\dagger & l_i = Q_{i+1} d_{i}^{-1} & \forall i \in [1, N_t-3] \\
 d_{N_t-2} = P - l_{N_t-3} u_{N_t-3} & & &
 \end{matrix}
 \f]
 * Compute all \f$v, h\f$
 \f[
 \begin{matrix}
 v_0 = Q_0               & h_0 = Q_{N_t-1}^\dagger d_0^{-1} & \\
 v_i = - l_{i-1} v_{i-1} & h_i = - h_{i-1} u_{i-1} d_{i}^{-1} & \forall i \in [1, N_t-3]
 \end{matrix}
 \f]
 * Compute remaining \f$d, u, l\f$
 \f[
 u_{N_t-2} = Q_{N_t-2}^\dagger - l_{N_t-3} v_{N_t-3} \qquad l_{N_t-2} = (Q_{N_t-1} - h_{N_t-3} u_{N_t-3}) d_{N_t-2}^{-1}\\
 d_{N_t-1} = P - l_{N_t-2}u_{N_t-2} - \textstyle\sum_{i=0}^{N_t-3}\, h_i v_i
 \f]
 *
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
 * `%HubbardFermiMatrix` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \sigma_\tilde{\mu}, \mathrm{and} \sigma_\tilde{\kappa}\f$
 * as inputs and can construct the individual blocks \f$P, Q, \mathrm{and} Q^\dagger\f$ or
 * the full matrix \f$M M^\dagger\f$ from them.
 *
 * Neither the full matrix nor any of its blocks are stored explicitly. Instead,
 * each block needs to be constructed using P(), Q(), and Qdag() or MMdag() for the
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
     * \param kappa Hopping matrix \f$\tilde{\kappa}\f$.
     * \param phi Auxilliary field \f$\phi\f$ from HS transformation.
     * \param mu Chemical potential \f$\tilde{\mu}\f$.
     * \param sigmaMu Sign of chemical potential in adjoint matrix.
     * \param sigmaKappa Sign of hopping matrix in adjoint matrix.
     */
    HubbardFermiMatrix(const SymmetricSparseMatrix<double> &kappa,
                       const Vector<std::complex<double>> &phi,
                       double mu, std::int8_t sigmaMu, std::int8_t sigmaKappa);

    /// Overload for plain SparseMatrix kappa.
    HubbardFermiMatrix(const SparseMatrix<double> &kappa,
                       const Vector<std::complex<double>> &phi,
                       double mu, std::int8_t sigmaMu, std::int8_t sigmaKappa);

    /// Store the block on the diagonal \f$P\f$ in the parameter.
    /**
     * \param p Block on the diagonal. Any old content is erased and the matrix is
     *          resized if need be.
     */
    void P(SparseMatrix<double> &p) const;

    /// Return the block on the diagonal \f$P\f$.
    SparseMatrix<double> P() const;

    /// Store the block on the lower subdiagonal \f$Q_{t'}\f$ in a parameter.
    /**
     * \param q Block on the lower subdiagonal. Any old content is erased and the matrix is
     *          resized if need be.
     * \param tp Temporal row index \f$t'\f$.
     */
    void Q(SparseMatrix<std::complex<double>> &q, std::size_t tp) const;

    /// Return the block on the lower subdiagonal \f$Q_{t'}\f$.
    /**
     * \param tp Temporal row index \f$t'\f$.
     */
    SparseMatrix<std::complex<double>> Q(std::size_t tp) const;

    /// Store the block on the upper subdiagonal \f$Q^{\dagger}_{t'}\f$ in a parameter.
    /**
     * \param qdag Block on the upper subdiagonal.
     *             Any old content is erased and the matrix is resized if need be.
     * \param tp Temporal row index \f$t'\f$.
     */
    void Qdag(SparseMatrix<std::complex<double>> &qdag, std::size_t tp) const;

    /// Return the block on the upper subdiagonal \f$Q^{\dagger}_{t'}\f$.
    /**
     * \param tp Temporal row index \f$t'\f$.
     */
    SparseMatrix<std::complex<double>> Qdag(std::size_t tp) const;

    /// Store the full fermion matrix \f$M M^{\dagger}\f$ in the parameter.
    /**
     * \param mmdag Full fermion matrix. Any old content is erased and the matrix is
     *          resized if need be.
     */
    void MMdag(SparseMatrix<std::complex<double>> &mmdag) const;

    /// Return the full fermion matrix.
    SparseMatrix<std::complex<double>> MMdag() const;

    /// Update the hopping matrix.
    void updateKappa(const SymmetricSparseMatrix<double> &kappa);

    /// Update the hopping matrix
    void updateKappa(const SparseMatrix<double> &kappa);

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
    std::int8_t _sigmaMu;    ///< Sign of mu in M^dag.
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

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
