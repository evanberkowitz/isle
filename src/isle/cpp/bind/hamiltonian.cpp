#include "hamiltonian.hpp"

#include "../action/hamiltonian.hpp"

using namespace cnxx;

namespace bind {
    void bindHamiltonian(py::module &mod) {
        py::class_<Hamiltonian>(mod, "Hamiltonian")
            .def(py::init<>())
            .def(py::init([](py::args args) {
                        Hamiltonian ham;
                        for (auto arg : args)
                            ham.add(arg.cast<Action*>(), false);  // Python owns the action
                        return ham;
                    }),
                py::keep_alive<1, 2>())
            .def("add", [](Hamiltonian &self, Action *const action) {
                    self.add(action, false);  // Python owns the action
                },
                py::keep_alive<1, 2>())
            .def("__getitem__", py::overload_cast<std::size_t>(&Hamiltonian::operator[]),
                 py::return_value_policy::reference_internal)
            .def("__len__", &Hamiltonian::size)
            .def("clear", &Hamiltonian::clear)
            .def("eval", &Hamiltonian::eval)
            .def("force", &Hamiltonian::force)
            .def("addMomentum", &Hamiltonian::addMomentum)
            .def("stripMomentum", &Hamiltonian::stripMomentum)
            ;
    }
}
