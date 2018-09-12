/** \file
 * \brief Fermionic part of Hubbard action.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_HPP
#define ACTION_HUBBARD_FERMI_ACTION_HPP

#include "action.hpp"
#include "../hubbardFermiMatrix.hpp"
#include "../lattice.hpp"

namespace isle {
    namespace action {
        /// Fermion action for Hubbard model.
        /**
         * The action is
         \f[
         S_{\mathrm{HFA}} = - \log \det M(\phi, \tilde{\kappa}, \tilde{\mu}) M(-\phi, \sigma_{\tilde{\kappa}}\tilde{\kappa}, -\tilde{\mu}),
         \f]
         * see HubbardFermiMatrix for the definition of M.
         *
         * Both variants of the algorithm for are implemented and can be chosen in
         * the constructor. The default is variant 1.
         * See <TT>docs/algorithm/hubbardFermiAction.pdf</TT>
         * for description and derivation of the algorithms.
         *
         * \warning Only supports `nt > 2`.
         */
        class HubbardFermiAction : Action {
        public:
            /// Specifies which variant of the algorithm gets used.
            enum class Variant { ONE, TWO };

            /// Copy in a HubbardFermiMatrix.
            explicit HubbardFermiAction(const HubbardFermiMatrix &hfm,
                                        const Variant variant=Variant::ONE)
                : _hfm{hfm}, _kp{hfm.K(Species::PARTICLE)},
                  _kh{hfm.K(Species::HOLE)}, _variant{variant} { }

            /// Construct from individual parameters of HubbardFermiMatrix.
            HubbardFermiAction(const SparseMatrix<double> &kappa,
                               double mu, std::int8_t sigmaKappa,
                               const Variant variant=Variant::ONE)
                : _hfm{kappa, mu, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)}, _variant{variant} { }

            HubbardFermiAction(const Lattice &lat, double beta,
                               double mu, std::int8_t sigmaKappa,
                               const Variant variant=Variant::ONE)
                : _hfm{lat, beta, mu, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)}, _variant{variant} { }

            HubbardFermiAction(const HubbardFermiAction &other) = default;
            HubbardFermiAction &operator=(const HubbardFermiAction &other) = default;
            HubbardFermiAction(HubbardFermiAction &&other) = default;
            HubbardFermiAction &operator=(HubbardFermiAction &&other) = default;
            ~HubbardFermiAction() override = default;

            /// Evaluate the %Action for given auxilliary field phi.
            std::complex<double> eval(const CDVector &phi) const override;

            /// Calculate force for given auxilliary field phi.
            CDVector force(const CDVector &phi) const override;

        private:
            const HubbardFermiMatrix _hfm;  ///< Stores all necessary parameters.
            const DSparseMatrix _kp;  ///< Matrix K for particles.
            const DSparseMatrix _kh;  ///< Matrix K for holes.
            const Variant _variant;  ///< Pick variant of the algorithm.
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_HPP
