#include "bind_lattice.hpp"

#include "../lattice.hpp"

using namespace pybind11::literals;

namespace bind {

    void bindLattice(py::module &mod) {
        using namespace isle;

        py::class_<Lattice>{mod, "Lattice"}
            .def(py::init<std::size_t, std::size_t>())
            .def(py::init<std::size_t, std::size_t, const std::string&, const std::string&>())
            .def(py::init<const Lattice&>())
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
            .def("distance", &Lattice::distance)
            .def("position", [](Lattice &lat, const std::size_t i,
                                const double x, const double y, const double z) {
                                 lat.position(i, Vec3<double>{x, y, z});
                             })
            .def("position", [](const Lattice &lat, const std::size_t i) {
                                 const Vec3<double> &pos = lat.position(i);
                                 return std::make_tuple(pos[0], pos[1], pos[2]);
                             })
            .def("nt", py::overload_cast<>(&Lattice::nt))
            .def("nt", [](Lattice &lat, const std::size_t nt) {
                    lat.nt() = nt;
                })
            .def("nx", &Lattice::nx)
            .def("lattSize", &Lattice::lattSize)

            .def_readonly("name", &Lattice::name)
            .def_readonly("comment", &Lattice::comment)
            ;

        mod.def("isBipartite", py::overload_cast<const SparseMatrix<double> &>(isBipartite));
        mod.def("isBipartite", py::overload_cast<const Lattice &>(isBipartite));
    }
}
