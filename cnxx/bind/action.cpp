#include "action.hpp"

#include "../action.hpp"
#include "../hubbardGaugeAction.hpp"
#include "../hubbardFermiAction.hpp"

using namespace pybind11::literals;
using namespace cnxx;

namespace bind {
    namespace {
        /// Trampoline class for cnxx::Action to allow Python classes to
        /// override its virtual members.
        struct ActionTramp : Action {
            std::complex<double> eval(const Vector<std::complex<double>> &phi) override {
                PYBIND11_OVERLOAD_PURE(
                    std::complex<double>,
                    Action,
                    eval,
                    phi
                );
            }

            Vector<std::complex<double>> force(
                const Vector<std::complex<double>> &phi) override {

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
        py::class_<Action, ActionTramp> action{mod, "Action"};
        action
            .def(py::init<>())
            .def("eval", &Action::eval)
            .def("force", &Action::force)
            ;

        py::class_<HubbardGaugeAction>{mod, "HubbardGaugeAction", action}
            .def(py::init<double>())
            .def("eval", &HubbardGaugeAction::eval)
            .def("force", &HubbardGaugeAction::force)
            ;

        py::class_<HubbardFermiAction>{mod, "HubbardFermiAction", action}
            .def(py::init<HubbardFermiMatrix, bool>(), "hfm"_a, "variant2"_a=false)
            .def(py::init<SparseMatrix<double>, double, std::int8_t, bool>(),
                      "kappa"_a, "mu"_a, "sigmaKappa"_a, "variant2"_a=false)
            .def("eval", &HubbardFermiAction::eval)
            .def("force", &HubbardFermiAction::force)
            ;
    }
}
