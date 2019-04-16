/** \file
 * \brief Fermion action for the Hubbard.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_HPP
#define ACTION_HUBBARD_FERMI_ACTION_HPP

#include "action.hpp"
#include "../hubbardFermiMatrixDia.hpp"
#include "../hubbardFermiMatrixExp.hpp"
#include "../lattice.hpp"

namespace isle {
    namespace action {
        /// Indicate kind of hopping term for HubbardFermiAction.
        enum class HFAHopping { DIA, EXP };

        /// Indicate basis for HubbardFermiAction.
        enum class HFABasis { PARTICLE_HOLE, SPIN };

        /// Specifies which variant of the algorithm gets used for HubbardFermiAction.
        enum class HFAVariant { ONE, TWO };

        /// \cond DO_NOT_DOCUMENT
        namespace _internal {
            /// Type of HubbardFermiMatrix based on kind of hopping.
            template <HFAHopping HOPPING>
            struct HFM {
                using type = HubbardFermiMatrixDia;
            };
            template <>
            struct HFM<HFAHopping::EXP> {
                using type = HubbardFermiMatrixExp;
            };

            /// Check if the shortcut to compute det(M_hole) is possible.
            template <HFABasis BASIS>
            bool _holeShortcutPossible(const SparseMatrix<double> &hopping,
                                       const double muTilde,
                                       const std::int8_t sigmaKappa);


            template <> bool _holeShortcutPossible<HFABasis::PARTICLE_HOLE>(
                const SparseMatrix<double> &hopping,
                const double muTilde,
                const std::int8_t sigmaKappa);

            template <> bool _holeShortcutPossible<HFABasis::SPIN>(
                const SparseMatrix<double> &hopping,
                const double muTilde,
                const std::int8_t sigmaKappa);
        }
        /// \endcond DO_NOT_DOCUMENT

        /// Fermion action for Hubbard model.
        /**
         * The action is
         \f[
         S_{\mathrm{HFA}} = - \log \det M(\phi, \tilde{\kappa}, \tilde{\mu}) M(-\phi, \sigma_{\tilde{\kappa}}\tilde{\kappa}, -\tilde{\mu}),
         \f]
         * see HubbardFermiMatrixDia / HubbardFermiMatrixExp for the definition of M.
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
         * It is possible to compute the contribution from holes from the contribution
         * from particles iff the lattice is bipartite, mu=0, sigmaKappa=+1,
         * basis=SPIN, and phi is real.
         * If `allowShortcut=true` in the constructor and the parameters permit,
         * %HubbardFermiAction uses that shortcut.
         * \warning The field configuration is not checked.
         *          If you allow for the shortcut, you must ensure that phi is
         *          always real!
         *
         * \warning Only supports `nt > 2`.
         *
         * See <TT>docs/algorithm/hubbardFermiAction.pdf</TT>
         * for description and derivation of the algorithms.
         */
        template <HFAHopping HOPPING, HFAVariant VARIANT, HFABasis BASIS>
        class HubbardFermiAction : public Action {
        public:
            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut)
                : _hfm{kappaTilde, muTilde, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)},
                  _shortcutForHoles{allowShortcut
                                    && _internal::_holeShortcutPossible<BASIS>(
                                        kappaTilde, muTilde, sigmaKappa)}
            { }

            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const Lattice &lat, const double beta,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut)
                : _hfm{lat, beta, muTilde, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)},
                  _shortcutForHoles{allowShortcut
                                    && _internal::_holeShortcutPossible<BASIS>(
                                        lat.hopping(), muTilde, sigmaKappa)}
            { }

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
            /// Stores all necessary parameters.
            const typename _internal::HFM<HOPPING>::type _hfm;
            const DSparseMatrix _kp;  ///< Matrix K for particles.
            const DSparseMatrix _kh;  ///< Matrix K for holes.
            /// Can logdetM for holes be computed from logdetM from particles?
            const bool _shortcutForHoles;
        };

        // For each specialization, forward declare specializations of eval
        // and force before the explicit instantiation declarations below.
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        // all the instantiations we will ever need, but actually implement them in the .cpp
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>;

        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>;

    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_HPP
