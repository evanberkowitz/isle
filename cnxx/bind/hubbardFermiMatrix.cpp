#include "hubbardFermiMatrix.hpp"

#include "../hubbardFermiMatrix.hpp"

namespace bind {
    void bindHubbardFermiMatrix(py::module &mod) {
        using HFM = HubbardFermiMatrix;

        py::class_<HFM> hfm{mod, "HubbardFermiMatrix"};
        hfm.def(py::init<SparseMatrix<double>, Vector<std::complex<double>>,
                double, std::int8_t, std::int8_t>())
            .def("P", py::overload_cast<>(&HFM::P, py::const_))
            .def("Q", py::overload_cast<std::size_t>(&HFM::Q, py::const_))
            .def("Qdag", py::overload_cast<std::size_t>(&HFM::Qdag, py::const_))
            .def("MMdag", py::overload_cast<>(&HFM::MMdag, py::const_))
            .def("updateKappa", py::overload_cast<const SparseMatrix<double>&>(&HFM::updateKappa))
            .def("updatePhi", py::overload_cast<const Vector<std::complex<double>>&>(&HFM::updatePhi))
            .def("nx", &HFM::nx)
            .def("nt", &HFM::nt)
            ;

        py::class_<HFM::LU>{hfm, "LU"}
            .def(py::init<std::size_t>())
            .def("reconstruct", &HFM::LU::reconstruct)
            ;

        mod.def("getLU",
                static_cast<HubbardFermiMatrix::LU(*)(const HubbardFermiMatrix&)>(getLU));
        mod.def("logdet",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix&)>(logdet));
        mod.def("logdet",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix::LU&)>(logdet));
    }
}
