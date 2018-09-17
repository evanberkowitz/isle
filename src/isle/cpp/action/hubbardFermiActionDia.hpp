/** \file
 * \brief Fermion action for the Hubbard model with hopping matrix on the diagonal.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_DIA_HPP
#define ACTION_HUBBARD_FERMI_ACTION_DIA_HPP

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
         * This action can treat configurations as either in the spin or the
         * particle/hole basis.
         * This is controlled through the parameter `alpha` as
         * - `alpha == 0` - spin basis
         * - `alpha == 1` - particle/hole basis
         *
         * \warning Only supports `nt > 2`.
         */
        class HubbardFermiActionDia : Action {
        public:
            /// Specifies which variant of the algorithm gets used.
            enum class Variant { ONE, TWO };

            /// Copy in a HubbardFermiMatrix.
            explicit HubbardFermiActionDia(const HubbardFermiMatrix &hfm,
                                           const std::int8_t alpha=1,
                                           const Variant variant=Variant::ONE)
                : _hfm{hfm}, _kp{hfm.K(Species::PARTICLE)}, _kh{hfm.K(Species::HOLE)},
                  _alpha{alpha}, _variant{variant} { }

            /// Construct from individual parameters of HubbardFermiMatrix.
            HubbardFermiActionDia(const SparseMatrix<double> &kappa,
                                  double mu, std::int8_t sigmaKappa,
                                  const std::int8_t alpha=1,
                                  const Variant variant=Variant::ONE)
                : _hfm{kappa, mu, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)},
                  _alpha{alpha}, _variant{variant} { }

            HubbardFermiActionDia(const Lattice &lat, double beta,
                                  double mu, std::int8_t sigmaKappa,
                                  const std::int8_t alpha=1,
                                  const Variant variant=Variant::ONE)
                : _hfm{lat, beta, mu, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)},
                  _alpha{alpha}, _variant{variant} { }

            HubbardFermiActionDia(const HubbardFermiActionDia &other) = default;
            HubbardFermiActionDia &operator=(const HubbardFermiActionDia &other) = default;
            HubbardFermiActionDia(HubbardFermiActionDia &&other) = default;
            HubbardFermiActionDia &operator=(HubbardFermiActionDia &&other) = default;
            ~HubbardFermiActionDia() override = default;

            /// Evaluate the %Action for given auxilliary field phi.
            std::complex<double> eval(const CDVector &phi) const override;

            /// Calculate force for given auxilliary field phi.
            CDVector force(const CDVector &phi) const override;

        private:
            const HubbardFermiMatrix _hfm;  ///< Stores all necessary parameters.
            const DSparseMatrix _kp;  ///< Matrix K for particles.
            const DSparseMatrix _kh;  ///< Matrix K for holes.
            const std::int8_t _alpha;  ///< _alpha=0: spin basis, _alpha=1: particle/hole basis.
            const Variant _variant;  ///< Pick variant of the algorithm.
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_DIA_HPP
