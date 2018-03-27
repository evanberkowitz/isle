#include "hubbardFermiMatrix.hpp"

#include "../hubbardFermiMatrix.hpp"

namespace bind {
    void bindHubbardFermiMatrix(py::module &mod) {
        using namespace cnxx;
        using HFM = HubbardFermiMatrix;

        py::class_<HFM> hfm{mod, "HubbardFermiMatrix"};
        hfm.def(py::init<SparseMatrix<double>, Vector<std::complex<double>>,
                double, std::int8_t>())
            .def("K", py::overload_cast<bool>(&HFM::K, py::const_))
            .def("F", py::overload_cast<std::size_t, bool>(&HFM::F, py::const_))
            .def("M", py::overload_cast<bool>(&HFM::M, py::const_))
            .def("P", py::overload_cast<>(&HFM::P, py::const_))
            .def("Tplus", py::overload_cast<std::size_t>(&HFM::Tplus, py::const_))
            .def("Tminus", py::overload_cast<std::size_t>(&HFM::Tminus, py::const_))
            .def("Q", py::overload_cast<>(&HFM::Q, py::const_))
            .def("updateKappa", py::overload_cast<const SparseMatrix<double>&>(&HFM::updateKappa))
            .def("updatePhi", py::overload_cast<const Vector<std::complex<double>>&>(&HFM::updatePhi))
            .def("nx", &HFM::nx)
            .def("nt", &HFM::nt)
            ;

        py::class_<HFM::LU>{hfm, "LU"}
            .def(py::init<std::size_t>())
            .def("reconstruct", &HFM::LU::reconstruct)
            .def_readonly("dinv", &HFM::LU::dinv)
            .def_readonly("l", &HFM::LU::l)
            .def_readonly("u", &HFM::LU::u)
            .def_readonly("v", &HFM::LU::v)
            .def_readonly("h", &HFM::LU::h)
            ;

        mod.def("getLU",
                static_cast<HubbardFermiMatrix::LU(*)(const HubbardFermiMatrix&)>(getLU));
        mod.def("solve", static_cast<Vector<std::complex<double>>(*)(
                    const HubbardFermiMatrix::LU &,
                    const Vector<std::complex<double>> &rhs)>(solve));
        mod.def("solve", static_cast<Vector<std::complex<double>>(*)(
                    const HubbardFermiMatrix &,
                    const Vector<std::complex<double>> &rhs)>(solve));
        mod.def("logdet",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix&)>(logdet));
        mod.def("logdet",
                static_cast<std::complex<double>(*)(const HubbardFermiMatrix::LU&)>(logdet));
    }
}
