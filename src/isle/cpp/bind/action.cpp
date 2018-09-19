#include "action.hpp"

#include "../action/action.hpp"
#include "../action/hubbardGaugeAction.hpp"
#include "../action/hubbardFermiAction.hpp"
#include "../action/hubbardFermiActionDia.hpp"
#include "../action/hubbardFermiActionExp.hpp"

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

        auto hfad = py::class_<HubbardFermiActionDia>{actmod, "HubbardFermiActionDia", action};

        py::enum_<HubbardFermiActionDia::Variant>{hfad, "Variant"}
             .value("ONE", HubbardFermiActionDia::Variant::ONE)
             .value("TWO", HubbardFermiActionDia::Variant::TWO);

        hfad.def(py::init<HubbardFermiMatrix, std::int8_t, HubbardFermiActionDia::Variant>(),
                 "hfm"_a, "alpha"_a=1,
                 "variant"_a=HubbardFermiActionDia::Variant::ONE)
            .def(py::init<SparseMatrix<double>, double, std::int8_t, std::int8_t,
                 HubbardFermiActionDia::Variant>(),
                 "kappa"_a, "mu"_a, "sigmaKappa"_a,
                 "alpha"_a=1,
                 "variant"_a= HubbardFermiActionDia::Variant::ONE)
            .def(py::init<Lattice, double, double, std::int8_t, std::int8_t,
                 HubbardFermiActionDia::Variant>(),
                 "lat"_a, "beta"_a, "mu"_a, "sigmaKappa"_a,
                 "alpha"_a=1,
                 "variant"_a= HubbardFermiActionDia::Variant::ONE)
            .def("eval", &HubbardFermiActionDia::eval)
            .def("force", &HubbardFermiActionDia::force)
            ;

        py::class_<HubbardFermiActionExp>{actmod, "HubbardFermiActionExp", action}
            .def(py::init<HubbardFermiMatrix, std::int8_t>(),
                 "hfm"_a, "alpha"_a=1)
            .def(py::init<SparseMatrix<double>, double, std::int8_t, std::int8_t>(),
                 "kappa"_a, "mu"_a, "sigmaKappa"_a,
                 "alpha"_a=1)
            .def(py::init<Lattice, double, double, std::int8_t, std::int8_t>(),
                 "lat"_a, "beta"_a, "mu"_a, "sigmaKappa"_a,
                 "alpha"_a=1)
            .def("eval", &HubbardFermiActionExp::eval)
            .def("force", &HubbardFermiActionExp::force)
            ;

        py::enum_<Hopping>(actmod, "Hopping")
            .value("DIAG", Hopping::DIAG)
            .value("EXP", Hopping::EXP);

        actmod.def("makeHubbardFermiAction",
                   py::overload_cast<const HubbardFermiMatrix &, std::int8_t,
                   Hopping, HubbardFermiActionDia::Variant>(makeHubbardFermiAction),
                   "hfm"_a, "alpha"_a=1, "hopping"_a=Hopping::DIAG,
                   "variant"_a=HubbardFermiActionDia::Variant::ONE);
        actmod.def("makeHubbardFermiAction",
                   py::overload_cast<const SparseMatrix<double> &, double,
                   std::int8_t, std::int8_t, Hopping, HubbardFermiActionDia::Variant>(makeHubbardFermiAction),
                   "kappa"_a, "mu"_a, "sigmaKappa"_a,
                   "alpha"_a=1, "hopping"_a=Hopping::DIAG,
                   "variant"_a= HubbardFermiActionDia::Variant::ONE);
        actmod.def("makeHubbardFermiAction",
                   py::overload_cast<const Lattice &, double, double, std::int8_t,
                   std::int8_t, Hopping, HubbardFermiActionDia::Variant>(makeHubbardFermiAction),
                   "lat"_a, "beta"_a, "mu"_a, "sigmaKappa"_a,
                   "alpha"_a=1, "hopping"_a=Hopping::DIAG,
                   "variant"_a= HubbardFermiActionDia::Variant::ONE);
    }
}
