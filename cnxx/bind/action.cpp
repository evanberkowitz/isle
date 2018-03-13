#include "action.hpp"

#include "../action.hpp"

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

    void bindAction(py::module &mod) {
        py::class_<Action, ActionTramp> action(mod, "Action");
        action
            .def(py::init<>())
            .def("eval", &Action::eval)
            .def("force", &Action::force)
            .def("valForce", &Action::valForce)
            ;
    }
}
