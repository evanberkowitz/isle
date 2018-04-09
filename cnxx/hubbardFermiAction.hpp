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
     * The action is
     \f[
     S_{\mathrm{HFA}} = - \log \det M(\phi, \tilde{\kappa}, \tilde{\mu}) M(-\phi, \sigma_{\tilde{\kappa}}\tilde{\kappa}, -\tilde{\mu}),
     \f]
     * see HubbardFermiMatrix for the definition of M.
     */
    class HubbardFermiAction : Action {
    public:
        /// Copy in a HubbardFermiMatrix.
        explicit HubbardFermiAction(const HubbardFermiMatrix &hfm) : _hfm{hfm} { }

        /// Construct from individual parameters of HubbardFermiMatrix.
        HubbardFermiAction(const SparseMatrix<double> &kappa,
                           double mu, std::int8_t sigmaKappa)
            : _hfm{kappa, mu, sigmaKappa} { }

        HubbardFermiAction(const HubbardFermiAction &other) = default;
        HubbardFermiAction &operator=(const HubbardFermiAction &other) = default;
        HubbardFermiAction(HubbardFermiAction &&other) = default;
        HubbardFermiAction &operator=(HubbardFermiAction &&other) = default;
        ~HubbardFermiAction() override = default;

        /// Use a new HubbardFermiMatrix from now on.
        void updateHFM(const HubbardFermiMatrix &hfm);
        /// Use a new HubbardFermiMatrix from now on.
        void updateHFM(HubbardFermiMatrix &&hfm);

        /// Evaluate the %Action for given auxilliary field phi.
        std::complex<double> eval(const CDVector &phi) override;

        /// Calculate force for given auxilliary field phi.
        CDVector force(const CDVector &phi) override;

    private:
        HubbardFermiMatrix _hfm;  ///< Stores all necessary parameters.
    };
}  // namespace cnxx

#endif  // ndef HUBBARD_FERMI_ACTION_HPP
