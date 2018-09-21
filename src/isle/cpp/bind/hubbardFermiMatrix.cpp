#include "hubbardFermiMatrix.hpp"

#include "../hubbardFermiMatrix.hpp"

namespace bind {
    void bindHubbardFermiMatrix(py::module &mod) {
        using namespace isle;
        using HFM = HubbardFermiMatrix;

        py::enum_<Species>{mod, "Species"}
            .value("PARTICLE", Species::PARTICLE)
            .value("HOLE", Species::HOLE);

        py::class_<HFM> hfm{mod, "HubbardFermiMatrix"};
        hfm.def(py::init<DSparseMatrix, double, std::int8_t>())
            .def(py::init<Lattice, double, double, std::int8_t>())
            .def("K", py::overload_cast<Species>(&HFM::K, py::const_))
            .def("F", py::overload_cast<std::size_t, const CDVector&,
                 Species, bool>(&HFM::F, py::const_))
            .def("M", py::overload_cast<const CDVector&, Species>(&HFM::M, py::const_))
            .def("MExp", py::overload_cast<const CDVector&, Species>(&HFM::MExp, py::const_))
            .def("P", py::overload_cast<>(&HFM::P, py::const_))
            .def("Tplus", py::overload_cast<std::size_t, const CDVector&>(&HFM::Tplus, py::const_))
            .def("Tminus", py::overload_cast<std::size_t, const CDVector&>(&HFM::Tminus, py::const_))
            .def("Q", py::overload_cast<const CDVector&>(&HFM::Q, py::const_))
            .def("updateKappa", py::overload_cast<const DSparseMatrix&>(&HFM::updateKappa))
            .def("nx", &HFM::nx)
            .def("kappa", &HFM::kappa)
            ;

        py::class_<HFM::QLU>{hfm, "QLU"}
            .def(py::init<std::size_t>())
            .def("reconstruct", &HFM::QLU::reconstruct)
            .def_readonly("dinv", &HFM::QLU::dinv)
            .def_readonly("l", &HFM::QLU::l)
            .def_readonly("u", &HFM::QLU::u)
            .def_readonly("v", &HFM::QLU::v)
            .def_readonly("h", &HFM::QLU::h)
            ;

        mod.def("getQLU",
                static_cast<HubbardFermiMatrix::QLU(*)(const HubbardFermiMatrix&,
                                                       const CDVector&)>(getQLU));
        mod.def("solveQ", static_cast<CDVector(*)(
                    const HubbardFermiMatrix::QLU &,
                    const CDVector &rhs)>(solveQ));
        mod.def("solveQ", static_cast<CDVector(*)(
                    const HubbardFermiMatrix &,
                    const CDVector&,
                    const CDVector&)>(solveQ));
        mod.def("logdetQ",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix&,
                                                    const CDVector&)>(logdetQ));
        mod.def("logdetQ",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix::QLU&)>(logdetQ));
        mod.def("logdetM", logdetM);
        mod.def("logdetMExp", logdetMExp);
        mod.def("solveM", solveM);
        mod.def("solveMExp", solveMExp);
    }
}
