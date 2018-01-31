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
 * ## Usage
 * `%HubbardFermiMatrix` needs \f$\tilde{\kappa}, \phi, \tilde{\mu}, \sigma_\tilde{\mu}, \mathrm{and} \sigma_\tilde{\kappa}\f$
 * as inputs and can construct the individual blocks \f$P, Q, \mathrm{and} Q^\dagger\f$ or
 * the full matrix \f$M M^\dagger\f$ from them.
 *
 * Neither the full matrix nor any of its blocks are stored explicitly. Instead,
 * each block needs to be constructed using P(), Q(), and Qdag() or MMdag() for the
 * full matrix. Note that these operations are fairly expensive.
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


    struct LU {
        std::vector<Matrix<std::complex<double>>> d;
        std::vector<Matrix<std::complex<double>>> u;
        std::vector<Matrix<std::complex<double>>> v;
        std::vector<Matrix<std::complex<double>>> l;
        std::vector<Matrix<std::complex<double>>> h;

        explicit LU(std::size_t nt);

        Matrix<std::complex<double>> reconstruct() const;
    };

private:
    SparseMatrix<double> _kappa;  ///< Hopping matrix.
    Vector<std::complex<double>> _phi;  ///< Auxilliary HS field.
    double _mu;              ///< Chemical potential.
    std::int8_t _sigmaMu;    ///< Sign of mu in M^dag.
    std::int8_t _sigmaKappa; ///< Sign of kappa in M^dag.
};


HubbardFermiMatrix::LU getLU(const HubbardFermiMatrix &hfm);

std::complex<double> logdet(const HubbardFermiMatrix &hfm);
std::complex<double> logdet(const HubbardFermiMatrix::LU &lu);

#endif  // ndef HUBBARD_FERMI_MATRIX_HPP
