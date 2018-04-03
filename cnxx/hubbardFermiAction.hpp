/** \file
 * \brief Fermionic part of Hubbard action.
 */

#ifndef HUBBARD_FERMI_ACTION_HPP
#define HUBBARD_FERMI_ACTION_HPP

#include "action.hpp"
#include "hubbardFermiMatrix.hpp"

namespace cnxx {
    /// Fermion action for Hubbard model.
    /**
     * \todo Use PARDISO or custom solver instead of LAPACK for inversions.
     *
     * The action is
     \f[
     S_{\mathrm{HFA}} = - \log \det M[\phi,\tilde{\kappa}, \tilde{\mu}]M^\dagger[\phi,\sigma_{\tilde{\kappa}}\tilde{\kappa},\sigma_{\tilde{\mu}}\tilde{\mu}]
     \f]
     */
    class HubbardFermiAction : Action {
    public:
        /// Copy in a HubbardFermiMatrix.
        explicit HubbardFermiAction(const HubbardFermiMatrix &hfm) : _hfm{hfm} { }

        /// Construct from individual parameters of HubbardFermiMatrix.
        HubbardFermiAction(const SparseMatrix<double> &kappa,
                           double mu, std::int8_t sigmaKappa)
            : _hfm{kappa, Vector<std::complex<double>>{}, mu, sigmaKappa} { }

        HubbardFermiAction(const HubbardFermiAction &other) = default;
        HubbardFermiAction &operator=(const HubbardFermiAction &other) = default;
        HubbardFermiAction(HubbardFermiAction &&other) = default;
        HubbardFermiAction &operator=(HubbardFermiAction &&other) = default;
        ~HubbardFermiAction() override = default;

        /// Use a new HubbardFermiMatrix from now on.
        void updateHFM(const HubbardFermiMatrix &hfm);
        /// Use a new HubbardFermiMatrix from now on.
        void updateHFM(HubbardFermiMatrix &&hfm);

        /// Update auxilliary HS field.
        void updatePhi(const Vector<std::complex<double>> &phi);

        /// Evaluate the %Action for given auxilliary field phi.
        std::complex<double> eval(const Vector<std::complex<double>> &phi) override;

        /// Calculate force for given auxilliary field phi.
        Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) override;

    private:
        HubbardFermiMatrix _hfm;  ///< Stores all necessary parameters.
    };
}  // namespace cnxx

#endif  // ndef HUBBARD_FERMI_ACTION_HPP
