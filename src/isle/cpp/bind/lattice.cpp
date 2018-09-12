#include "lattice.hpp"

#include "../lattice.hpp"

namespace bind {

    void bindLattice(py::module &mod) {
        using namespace isle;

        py::class_<Lattice>{mod, "Lattice"}
            .def(py::init<std::size_t, std::size_t>())
            .def("hopping", py::overload_cast<>(&Lattice::hopping),
                 py::return_value_policy::reference_internal)
            .def("areNeighbors", &Lattice::areNeighbors)
            .def("setNeighbor", &Lattice::setNeighbor)
            .def("getNeighbor", [](const Lattice &self,
                                   const std::size_t i, const std::size_t j) {
                     if (self.hopping().find(i, j) != self.hopping().end(i))
                         return self.hopping()(i, j);
                     else
                         throw std::invalid_argument("No matrix element at given indices");
                 })
            .def("getNeighbors", [](const Lattice &self, const std::size_t i) {
                    return makeIndexValueIterator(self.hopping().begin(i), self.hopping().end(i));
                }, py::keep_alive<0, 1>())
            .def("distance", static_cast<
                 double (Lattice::*)(std::size_t, std::size_t) const>(&Lattice::distance))
            .def("distance", static_cast<
                 void (Lattice::*)(std::size_t, std::size_t, double)>(&Lattice::distance))
            .def("nt", py::overload_cast<>(&Lattice::nt))
            .def("nt", [](Lattice &lat, const std::size_t nt) {
                    lat.nt() = nt;
                })
            .def("nx", &Lattice::nx)
            .def("lattSize", &Lattice::lattSize)
            ;
    }
}
