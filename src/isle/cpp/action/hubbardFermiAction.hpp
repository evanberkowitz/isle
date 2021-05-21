/** \file
 * \brief Fermion action for the Hubbard.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_HPP
#define ACTION_HUBBARD_FERMI_ACTION_HPP

#include "action.hpp"
#include "../hubbardFermiMatrixDia.hpp"
#include "../hubbardFermiMatrixExp.hpp"
#include "../lattice.hpp"
#include <torch/script.h>
#include <memory>
#include <iostream>


namespace isle {
    namespace action {
        /// Indicate kind of hopping term for HubbardFermiAction.
        enum class HFAHopping { DIA, EXP };

        /// Indicate basis for HubbardFermiAction.
        enum class HFABasis { PARTICLE_HOLE, SPIN };

        /// Specifies which algorithm gets used for HubbardFermiAction.
        /**
         * See documentation in docs/algorithm for more information.
         */
        enum class HFAAlgorithm { DIRECT_SINGLE, DIRECT_SQUARE, ML_APPROX_FORCE};

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

            /// Type used to the spatial matrix K in the fermion matrix.
            template <HFAHopping HOPPING>
            struct KMatrixType {
                using type = DSparseMatrix;
            };
            template <>
            struct KMatrixType<HFAHopping::EXP> {
                using type = IdMatrix<double>;
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
        template <HFAHopping HOPPING, HFAAlgorithm ALGORITHM, HFABasis BASIS>
        class HubbardFermiAction : public Action {
        public:
            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut)
                : _hfm{kappaTilde, muTilde, sigmaKappa},
                  _kp{_hfm.K(Species::PARTICLE)},
                  _kh{_hfm.K(Species::HOLE)},
                  _shortcutForHoles{allowShortcut
                                    && _internal::_holeShortcutPossible<BASIS>(
                                        kappaTilde, muTilde, sigmaKappa)}
            { }

            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const Lattice &lat, const double beta,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut)
                : _hfm{lat, beta, muTilde, sigmaKappa},
                  _kp{_hfm.K(Species::PARTICLE)},
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
            const typename _internal::KMatrixType<HOPPING>::type _kp;  ///< Matrix K for particles.
            const typename _internal::KMatrixType<HOPPING>::type _kh;  ///< Matrix K for holes.
            /// Can logdetM for holes be computed from logdetM from particles?
            const bool _shortcutForHoles;
            //torch::jit::script::Module _model;

        };

        // For each specialization, forward declare specializations of eval
        // and force before the explicit instantiation declarations below.
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::force(
            const CDVector &phi) const;



        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const;
        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::eval(
            const CDVector &phi) const;
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::force(
            const CDVector &phi) const;

        // all the instantiations we will ever need, but actually implement them in the .cpp
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>;

        extern template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>;
        extern template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>;

        template<>
        class  HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::ML_APPROX_FORCE, HFABasis::PARTICLE_HOLE>:public Action{
            public:

            

            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut,const std::string model_path,const double utilde);
                
                  
            /// Construct from individual parameters of HubbardFermiMatrix[Dia,Exp].
            HubbardFermiAction(const Lattice &lat, const double beta,
                               const double muTilde, const std::int8_t sigmaKappa,
                               const bool allowShortcut,std::string model_path,const double utilde);


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
            const typename _internal::HFM<HFAHopping::EXP>::type _hfm;
            const typename _internal::KMatrixType<HFAHopping::EXP>::type _kp;  ///< Matrix K for particles.
            const typename _internal::KMatrixType<HFAHopping::EXP>::type _kh;  ///< Matrix K for holes.
            /// Can logdetM for holes be computed from logdetM from particles?
            const bool _shortcutForHoles; 
            mutable torch::jit::script::Module _model;
            double _utilde;
                
                  
        };        

    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_HPP
