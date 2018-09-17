/** \file
 * \brief Fermionic part of Hubbard action with an exponential hopping term.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_EXP_HPP
#define ACTION_HUBBARD_FERMI_ACTION_EXP_HPP


#include "action.hpp"
#include "../hubbardFermiMatrix.hpp"
#include "../lattice.hpp"

namespace isle {
    namespace action {
        /// Fermion action for Hubbard model with an exponential hopping term.
        /**
         * The action is
         \f[
         S_{\mathrm{HFA}} = - \log \det M'(\phi, \tilde{\kappa}, \tilde{\mu}) M'(-\phi, \sigma_{\tilde{\kappa}}\tilde{\kappa}, -\tilde{\mu}),
         \f]
         * see HubbardFermiMatrix for the definition of M'.
         *
         * This action can treat configurations as either in the spin or the
         * particle/hole basis.
         * This is controlled through the parameter `alpha` as
         * - `alpha == 0` - spin basis
         * - `alpha == 1` - particle/hole basis
         *
         * \warning Only supports `nt > 1`.
         *
         * See <TT>docs/algorithm/hubbardFermiAction.pdf</TT>
         * for description and derivation of the algorithms.
         */
        class HubbardFermiActionExp : Action {
        public:
            /// Copy in a HubbardFermiMatrix.
            explicit HubbardFermiActionExp(const HubbardFermiMatrix &hfm,
                                           std::int8_t alpha=1);

            /// Construct from individual parameters of HubbardFermiMatrix.
            HubbardFermiActionExp(const SparseMatrix<double> &kappaTilde,
                                  double muTilde, std::int8_t sigmaKappa,
                                  std::int8_t alpha=1);
            HubbardFermiActionExp(const Lattice &lat, double beta,
                                  double muTilde, std::int8_t sigmaKappa,
                                  std::int8_t alpha=1);

            HubbardFermiActionExp(const HubbardFermiActionExp &other) = default;
            HubbardFermiActionExp &operator=(const HubbardFermiActionExp &other) = default;
            HubbardFermiActionExp(HubbardFermiActionExp &&other) = default;
            HubbardFermiActionExp &operator=(HubbardFermiActionExp &&other) = default;
            ~HubbardFermiActionExp() override = default;

            /// Evaluate the %Action for given auxilliary field phi.
            std::complex<double> eval(const CDVector &phi) const override;

            /// Calculate force for given auxilliary field phi.
            CDVector force(const CDVector &phi) const override;

        private:
            const HubbardFermiMatrix _hfm;  ///< Stores all necessary parameters.
            DMatrix _expKappapInv;  ///< Inverse of exp(kappaTilde) for particles.
            DMatrix _expKappahInv;  ///< Inverse of exp(kappaTilde) for holes.
            const std::int8_t _alpha;  ///< _alpha=0: spin basis, _alpha=1: particle/hole basis.
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_EXP_HPP
