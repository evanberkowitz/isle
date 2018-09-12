#include "action.hpp"

#include "../action/action.hpp"
#include "../action/hubbardGaugeAction.hpp"
#include "../action/hubbardFermiAction.hpp"

using namespace pybind11::literals;
using namespace cnxx;

namespace bind {
    namespace {
        /// Trampoline class for cnxx::Action to allow Python classes to
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

        py::class_<HubbardFermiAction>{actmod, "HubbardFermiAction", action}
            .def(py::init<HubbardFermiMatrix, bool>(), "hfm"_a, "variant2"_a=false)
            .def(py::init<SparseMatrix<double>, double, std::int8_t, bool>(),
                      "kappa"_a, "mu"_a, "sigmaKappa"_a, "variant2"_a=false)
            .def(py::init<Lattice, double, double, std::int8_t, bool>(),
                      "lat"_a, "beta"_a, "mu"_a, "sigmaKappa"_a, "variant2"_a=false)
            .def("eval", &HubbardFermiAction::eval)
            .def("force", &HubbardFermiAction::force)
            ;
    }
}
