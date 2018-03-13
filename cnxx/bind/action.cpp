#include "action.hpp"

#include "../action.hpp"
#include "../hubbardGaugeAction.hpp"
#include "../hubbardFermiAction.hpp"

using namespace cnxx;

namespace bind {
    namespace {
        using ScalVecPair = std::pair<std::complex<double>, Vector<std::complex<double>>>;
        
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

            ScalVecPair valForce(const Vector<std::complex<double>> &phi) override {
                PYBIND11_OVERLOAD_PURE(
                    ScalVecPair,
                    Action,
                    valForce,
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
            .def("valForce", &Action::valForce)
            ;

        py::class_<HubbardGaugeAction>{mod, "HubbardGaugeAction", action}
            .def(py::init<double>())
            .def("eval", &HubbardGaugeAction::eval)
            .def("force", &HubbardGaugeAction::force)
            .def("valForce", &HubbardGaugeAction::valForce)
            ;

        py::class_<HubbardFermiAction>{mod, "HubbardFermiAction", action}
            .def(py::init<HubbardFermiMatrix>())
            .def(py::init<SparseMatrix<double>, double, std::int8_t, std::int8_t>())
            .def("eval", &HubbardFermiAction::eval)
            .def("force", &HubbardFermiAction::force)
            .def("valForce", &HubbardFermiAction::valForce)
            ;
    }
}
