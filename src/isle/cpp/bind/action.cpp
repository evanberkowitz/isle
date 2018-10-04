#include "action.hpp"

#include "../action/action.hpp"
#include "../action/hubbardGaugeAction.hpp"
#include "../action/hubbardFermiAction.hpp"

using namespace pybind11::literals;
using namespace isle;
using namespace isle::action;

namespace bind {
    namespace {
        /// Trampoline class for isle::action::Action to allow Python classes to
        /// override its virtual members.
        struct ActionTramp : Action {
            std::complex<double> eval(const Vector<std::complex<double>> &phi) const override {
                PYBIND11_OVERLOAD_PURE(
                    std::complex<double>,
                    Action,
                    eval,
                    phi
                );
            }

            Vector<std::complex<double>> force(
                const Vector<std::complex<double>> &phi) const override {

                PYBIND11_OVERLOAD_PURE(
                    Vector<std::complex<double>>,
                    Action,
                    force,
                    phi
                );
            }
        };

        template <HFAHopping HOPPING, HFAVariant VARIANT, std::int8_t ALPHA,
                  typename A>
        void bindSpecificHFA(py::module &mod, const char * const name, A &action) {
            using HFA = HubbardFermiAction<HOPPING, VARIANT, ALPHA>;

            py::class_<HFA>(mod, name, action)
                .def(py::init<SparseMatrix<double>, double, std::int8_t>(),
                     "kappa"_a, "mu"_a, "sigmaKappa"_a)
                .def("eval", &HFA::eval)
                .def("force", &HFA::force)
                ;
        }

        /// Make a specific HubbardFermiAction controlled through run-time parameters.
        py::object makeHubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                                          const double muTilde,
                                          const std::int8_t sigmaKappa,
                                          const std::int8_t alpha,
                                          const HFAHopping hopping,
                                          const HFAVariant variant) {

            if (alpha == 1) {
                if (hopping == HFAHopping::DIA) {
                    if (variant == HFAVariant::ONE) {
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAVariant::ONE,
                                        1>(kappaTilde, muTilde, sigmaKappa));
                    } else {  // HFAVariant::TWO
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAVariant::TWO,
                                        1>(kappaTilde, muTilde, sigmaKappa));
                    }
                } else {  // HFAHopping::EXP
                    if (variant == HFAVariant::ONE) {
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAVariant::ONE,
                                        1>(kappaTilde, muTilde, sigmaKappa));
                    } else {  // HFAVariant::TWO
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAVariant::TWO,
                                        1>(kappaTilde, muTilde, sigmaKappa));
                    }
                }
            }

            else {  // alpha = 0
                if (hopping == HFAHopping::DIA) {
                    if (variant == HFAVariant::ONE) {
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAVariant::ONE,
                                        0>(kappaTilde, muTilde, sigmaKappa));
                    } else {  // HFAVariant::TWO
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAVariant::TWO,
                                        0>(kappaTilde, muTilde, sigmaKappa));
                    }
                } else {  // HFAHopping::EXP
                    if (variant == HFAVariant::ONE) {
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAVariant::ONE,
                                        0>(kappaTilde, muTilde, sigmaKappa));
                    } else {  // HFAVariant::TWO
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAVariant::TWO,
                                        0>(kappaTilde, muTilde, sigmaKappa));
                    }
                }
            }
        }

        /// Bind everything related to HubbardFermiActions.
        template <typename A>
        void bindHubbardFermiAction(py::module &mod, A &action) {
            // bind enums
            py::enum_<HFAVariant>(mod, "HFAVariant")
                .value("ONE", HFAVariant::ONE)
                .value("TWO", HFAVariant::TWO);

            py::enum_<HFAHopping>(mod, "HFAHopping")
                .value("DIA", HFAHopping::DIA)
                .value("EXP", HFAHopping::EXP);

            // bind all specific actions
            bindSpecificHFA<HFAHopping::DIA, HFAVariant::ONE, 1>(mod, "HubbardFermiActionDiaOneOne", action);
            bindSpecificHFA<HFAHopping::DIA, HFAVariant::ONE, 0>(mod, "HubbardFermiActionDiaOneZero", action);
            bindSpecificHFA<HFAHopping::DIA, HFAVariant::TWO, 1>(mod, "HubbardFermiActionDiaTwoOne", action);
            bindSpecificHFA<HFAHopping::DIA, HFAVariant::TWO, 0>(mod, "HubbardFermiActionDiaTwoZero", action);

            bindSpecificHFA<HFAHopping::EXP, HFAVariant::ONE, 1>(mod, "HubbardFermiActionExpOneOne", action);
            bindSpecificHFA<HFAHopping::EXP, HFAVariant::ONE, 0>(mod, "HubbardFermiActionExpOneZero", action);
            bindSpecificHFA<HFAHopping::EXP, HFAVariant::TWO, 1>(mod, "HubbardFermiActionExpTwoOne", action);
            bindSpecificHFA<HFAHopping::EXP, HFAVariant::TWO, 0>(mod, "HubbardFermiActionExpTwoZero", action);

            mod.def("makeHubbardFermiAction",
                    makeHubbardFermiAction,
                    "kappaTilde"_a, "muTilde"_a, "sigmaKappa"_a,
                    "alpha"_a=1, "hopping"_a=HFAHopping::DIA,
                    "variant"_a= HFAVariant::ONE);

            mod.def("makeHubbardFermiAction",
                    [] (const Lattice &lattice, const double beta,
                        const double muTilde, const std::int8_t sigmaKappa,
                        const std::int8_t alpha,
                        const HFAHopping hopping, const HFAVariant variant) {

                        return makeHubbardFermiAction(
                            lattice.hopping()*beta/lattice.nt(),
                            muTilde, sigmaKappa, alpha,
                            hopping, variant);
                    },
                    "lat"_a, "beta"_a, "muTilde"_a, "sigmaKappa"_a,
                    "alpha"_a=1, "hopping"_a=HFAHopping::DIA,
                    "variant"_a= HFAVariant::ONE);
        }
    }

    void bindActions(py::module &mod) {
        py::module actmod = mod.def_submodule("action", "Actions");

        py::class_<Action, ActionTramp> action{actmod, "Action"};
        action
            .def(py::init<>())
            .def("eval", &Action::eval)
            .def("force", &Action::force)
            ;

        py::class_<HubbardGaugeAction>{actmod, "HubbardGaugeAction", action}
            .def(py::init<double>())
            .def("eval", &HubbardGaugeAction::eval)
            .def("force", &HubbardGaugeAction::force)
            ;

        bindHubbardFermiAction(actmod, action);
    }
}
