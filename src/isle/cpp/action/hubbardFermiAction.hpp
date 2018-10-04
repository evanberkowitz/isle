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
         * \warning Only supports `nt > 2`.
         *
         * See <TT>docs/algorithm/hubbardFermiAction.pdf</TT>
         * for description and derivation of the algorithms.
         */
        template <HFAHopping HOPPING, HFAVariant VARIANT, std::int8_t ALPHA>
        class HubbardFermiAction : public Action {
            static_assert(ALPHA==0 || ALPHA==1, "Only alpha=0,1 supported by HubbardFermiAction");

        public:
            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                               const double muTilde, const std::int8_t sigmaKappa)
                : _hfm{kappaTilde, muTilde, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)} { }

            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const Lattice &lat, const double beta,
                               const double muTilde, const std::int8_t sigmaKappa)
                : _hfm{lat, beta, muTilde, sigmaKappa}, _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)} { }

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
        };

        // For each specialization, forward declare specializations of eval
        // and force before the explicit instantiation declarations below.
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 1>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 1>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 0>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 0>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 1>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 1>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 0>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 0>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 1>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 1>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 0>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 0>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 1>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 1>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 0>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 0>::force(
            const CDVector &phi) const;

        // all the instantiations we will ever need, but actually do this in the .cpp
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 1>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, 0>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 1>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, 0>;

        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 1>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, 0>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 1>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, 0>;

    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_HPP
