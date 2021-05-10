#include "bind_action.hpp"

#include "../action/action.hpp"
#include "../action/hubbardGaugeAction.hpp"
#include "../action/hubbardFermiAction.hpp"
#include "../action/sumAction.hpp"

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

        void addAction(SumAction &sum, py::object &action) {
            try {
                // if action is a SumAction, add all its members
                SumAction *const otherSum = action.cast<SumAction*>();
                for (size_t i = 0; i < otherSum->size(); ++i)
                    sum.add((*otherSum)[i]);
            }
            catch (const py::cast_error &) {
                // if action is not a SumAction, just add it
                sum.add(action.cast<Action*>());
            }
        }

        auto bindBaseAction(py::module &mod) {
            return py::class_<Action, ActionTramp>(mod, "Action")
                .def(py::init<>())
                .def("eval", &Action::eval)
                .def("force", &Action::force)
                .def("__add__", [](py::object &self, py::object &other) {
                                    SumAction sum;
                                    addAction(sum, self);
                                    addAction(sum, other);
                                    return sum;
                                },
                    py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
                ;
        }

        template <typename A>
        void bindSumAction(py::module &mod, A &baseAction) {
            py::class_<SumAction>(mod, "SumAction", baseAction)
                .def(py::init<>())
                .def(py::init([](py::args args) {
                                  SumAction act;
                                  for (auto arg : args)
                                      act.add(arg.cast<Action*>());
                                  return act;
                              }),
                    py::keep_alive<1, 2>())
                .def("add", [](SumAction &self, Action *const action) {
                                self.add(action);
                            },
                    py::keep_alive<1, 2>())
                .def("__getitem__", py::overload_cast<std::size_t>(&SumAction::operator[]),
                     py::return_value_policy::reference_internal)
                .def("__len__", &SumAction::size)
                .def("clear", &SumAction::clear)
                .def("eval", &SumAction::eval)
                .def("force", &SumAction::force)
                ;
        }

        template <typename a>
        void bindHubbardGaugeAction(py::module &mod, a &action) {
            py::class_<HubbardGaugeAction>(mod, "HubbardGaugeAction", action)
                .def(py::init<double>())
                .def_readonly("utilde", &HubbardGaugeAction::utilde)
                .def("eval", &HubbardGaugeAction::eval)
                .def("force", &HubbardGaugeAction::force)
                ;
        }

        template <HFAHopping HOPPING, HFAAlgorithm ALGORITHM, HFABasis BASIS,
                  typename A>
        void bindSpecificHFA(py::module &mod, const char * const name, A &action) {
            using HFA = HubbardFermiAction<HOPPING, ALGORITHM, BASIS>;

            if constexpr (ALGORITHM == HFAAlgorithm::ML_APPROX_FORCE) {
                py::class_<HFA>(mod, name, action)
                    .def(py::init<SparseMatrix<double>, double, std::int8_t, bool, std::string>(),
                        "kappa"_a, "mu"_a, "sigmaKappa"_a, "allowShortcut"_a, "model_path"_a)
                    .def("eval", &HFA::eval)
                    .def("force", &HFA::force);
            } else{
                py::class_<HFA>(mod, name, action)
                    .def(py::init<SparseMatrix<double>, double, std::int8_t, bool>(),
                        "kappa"_a, "mu"_a, "sigmaKappa"_a, "allowShortcut"_a)
                    .def("eval", &HFA::eval)
                    .def("force", &HFA::force);
            }
        }

        /// Make a specific HubbardFermiAction controlled through run-time parameters.
        py::object makeHubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                                          const double muTilde,
                                          const std::int8_t sigmaKappa,
                                          const HFAHopping hopping,
                                          const HFABasis basis,
                                          const HFAAlgorithm algorithm,
                                          const bool allowShortcut) {

            if (basis == HFABasis::PARTICLE_HOLE) {
                if (hopping == HFAHopping::DIA) {
                    if (algorithm == HFAAlgorithm::DIRECT_SINGLE) {
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAAlgorithm::DIRECT_SINGLE,
                                        HFABasis::PARTICLE_HOLE>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    } else {  // HFAAlgorithm::DIRECT_SQUARE
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAAlgorithm::DIRECT_SQUARE,
                                        HFABasis::PARTICLE_HOLE>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    }
                } else {  // HFAHopping::EXP
                    if (algorithm == HFAAlgorithm::DIRECT_SINGLE) {
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAAlgorithm::DIRECT_SINGLE,
                                        HFABasis::PARTICLE_HOLE>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    } else if(algorithm == HFAAlgorithm::DIRECT_SQUARE){
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAAlgorithm::DIRECT_SQUARE,
                                        HFABasis::PARTICLE_HOLE>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    } else {
                        throw std::invalid_argument("makeHubbardFermiAction is not implemented for algorithm = HFAAlgorithm.ML_APPROX_FORCE");
                    }
                }
            } else {  // HFABasis::SPIN
                if (hopping == HFAHopping::DIA) {
                    if (algorithm == HFAAlgorithm::DIRECT_SINGLE) {
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAAlgorithm::DIRECT_SINGLE,
                                        HFABasis::SPIN>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    } else {  // HFAAlgorithm::DIRECT_SQUARE
                        return py::cast(HubbardFermiAction<HFAHopping::DIA,
                                        HFAAlgorithm::DIRECT_SQUARE,
                                        HFABasis::SPIN>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    }
                } else {  // HFAHopping::EXP
                    if (algorithm == HFAAlgorithm::DIRECT_SINGLE) {
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAAlgorithm::DIRECT_SINGLE,
                                        HFABasis::SPIN>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    } else {  // HFAAlgorithm::DIRECT_SQUARE
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAAlgorithm::DIRECT_SQUARE,
                                        HFABasis::SPIN>(kappaTilde, muTilde, sigmaKappa, allowShortcut));
                    }
                }
            }
        }

        ///Make HubbardFermiAction for ML_APPROX_FORCE Algorithm using run time parameters
        py::object makeHubbardFermiActionMLApprox(const SparseMatrix<double> &kappaTilde,
                                          const double muTilde,
                                          const std::int8_t sigmaKappa,
                                          const HFAHopping hopping,
                                          const HFABasis basis,
                                          const HFAAlgorithm algorithm,
                                          const bool allowShortcut,const std::string model_path) {  
            if (basis == HFABasis::PARTICLE_HOLE) {
                if (hopping == HFAHopping::EXP) {
                    if (algorithm == HFAAlgorithm::ML_APPROX_FORCE) {
                        return py::cast(HubbardFermiAction<HFAHopping::EXP,
                                        HFAAlgorithm::ML_APPROX_FORCE,
                                        HFABasis::PARTICLE_HOLE>(kappaTilde, muTilde, sigmaKappa, allowShortcut,model_path));
                         }
                    else{
                        throw std::invalid_argument("makeHubbardFermiActionMLApprox only for ML_APPROX_FORCE is defined ");
                    }
                                }
                else{
                    throw std::invalid_argument("makeHubbardFermiActionMLApprox only for EXP is defined ");
                    }
                }   
            else{
               throw std::invalid_argument("makeHubbardFermiActionMLApprox only for Particle_HOLE is defined "); 
            }                                      
         }

        /// Bind everything related to HubbardFermiActions.
        template <typename A>
        void bindHubbardFermiAction(py::module &mod, A &action) {
            // bind enums
            py::enum_<HFAAlgorithm>(mod, "HFAAlgorithm")
                .value("DIRECT_SINGLE", HFAAlgorithm::DIRECT_SINGLE)
                .value("DIRECT_SQUARE", HFAAlgorithm::DIRECT_SQUARE)
                .value("ML_APPROX_FORCE", HFAAlgorithm::ML_APPROX_FORCE);

            py::enum_<HFABasis>(mod, "HFABasis")
                .value("PARTICLE_HOLE", HFABasis::PARTICLE_HOLE)
                .value("SPIN", HFABasis::SPIN);

            py::enum_<HFAHopping>(mod, "HFAHopping")
                .value("DIA", HFAHopping::DIA)
                .value("EXP", HFAHopping::EXP);

            // bind all specific actions
            bindSpecificHFA<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>(mod, "HubbardFermiActionDiaDirsingleOne", action);
            bindSpecificHFA<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>(mod, "HubbardFermiActionDiaDirsingleZero", action);
            bindSpecificHFA<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>(mod, "HubbardFermiActionDiaDirsquareOne", action);
            bindSpecificHFA<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>(mod, "HubbardFermiActionDiaDirsquareZero", action);

            bindSpecificHFA<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>(mod, "HubbardFermiActionExpDirsingleOne", action);
            bindSpecificHFA<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>(mod, "HubbardFermiActionExpDirsingleZero", action);
            bindSpecificHFA<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>(mod, "HubbardFermiActionExpDirsquareOne", action);
            bindSpecificHFA<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>(mod, "HubbardFermiActionExpDirsquareZero", action);

            bindSpecificHFA<HFAHopping::EXP, HFAAlgorithm::ML_APPROX_FORCE, HFABasis::PARTICLE_HOLE>(mod, "HubbardFermiActionExpMLApproxOne", action);

            mod.def("makeHubbardFermiAction",
                    makeHubbardFermiAction,
                    "kappaTilde"_a, "muTilde"_a, "sigmaKappa"_a,
                    "hopping"_a=HFAHopping::DIA,
                    "basis"_a=HFABasis::PARTICLE_HOLE,
                    "algorithm"_a= HFAAlgorithm::DIRECT_SINGLE,
                    "allowShortcut"_a=false);

            mod.def("makeHubbardFermiAction",
                    [] (const Lattice &lattice, const double beta,
                        const double muTilde, const std::int8_t sigmaKappa,
                        const HFAHopping hopping, const HFABasis basis,
                        const HFAAlgorithm algorithm, const bool allowShortcut) {

                        return makeHubbardFermiAction(
                            lattice.hopping()*beta/lattice.nt(),
                            muTilde, sigmaKappa,
                            hopping, basis, algorithm, allowShortcut);
                    },
                    "lat"_a, "beta"_a, "muTilde"_a, "sigmaKappa"_a,
                    "hopping"_a=HFAHopping::DIA,
                    "basis"_a=HFABasis::PARTICLE_HOLE,
                    "algorithm"_a= HFAAlgorithm::DIRECT_SINGLE,
                    "allowShortcut"_a=false);

             mod.def("makeHubbardFermiActionMLApprox",
                    makeHubbardFermiActionMLApprox,
                    "kappaTilde"_a, "muTilde"_a, "sigmaKappa"_a,
                    "hopping"_a=HFAHopping::EXP,
                    "basis"_a=HFABasis::PARTICLE_HOLE,
                    "algorithm"_a= HFAAlgorithm::ML_APPROX_FORCE,
                    "allowShortcut"_a=false,
                    "model_path"_a);

            mod.def("makeHubbardFermiActionMLApprox",
                    [] (const Lattice &lattice, const double beta,
                        const double muTilde, const std::int8_t sigmaKappa,
                        const HFAHopping hopping, const HFABasis basis,
                        const HFAAlgorithm algorithm, const bool allowShortcut,const std::string model_path) {

                        return makeHubbardFermiActionMLApprox(
                            lattice.hopping()*beta/lattice.nt(),
                            muTilde, sigmaKappa,
                            hopping, basis, algorithm, allowShortcut,model_path);
                    },
                    "lat"_a, "beta"_a, "muTilde"_a, "sigmaKappa"_a,
                    "hopping"_a=HFAHopping::EXP,
                    "basis"_a=HFABasis::PARTICLE_HOLE,
                    "algorithm"_a= HFAAlgorithm::ML_APPROX_FORCE,
                    "allowShortcut"_a=false,
                    "model_path"_a);
        }
    }

    void bindActions(py::module &mod) {
        py::module actmod = mod.def_submodule("action", "Actions");

        auto action = bindBaseAction(actmod);
        bindSumAction(actmod, action);
        bindHubbardGaugeAction(actmod, action);
        bindHubbardFermiAction(actmod, action);
    }
}
