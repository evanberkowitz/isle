#include "bind_hubbardFermiMatrix.hpp"

#include <array>

#include "../species.hpp"
#include "../hubbardFermiMatrixDia.hpp"
#include "../hubbardFermiMatrixExp.hpp"

using namespace isle;

namespace bind {
    static constexpr std::array<Species, 2> SPECIES_VALUES{Species::PARTICLE, Species::HOLE};

    namespace {
        void bindHoppingSpecific(py::module &, py::class_<HubbardFermiMatrixDia> &) {
            // nothing special in this case
        }

        void bindHoppingSpecific(py::module &, py::class_<HubbardFermiMatrixExp> &hfmd) {
            hfmd.def("expKappa", &HubbardFermiMatrixExp::expKappa);
        }

        template <typename HFM>
        void bindHFM(py::module &mod, const char *const name) {
            py::class_<HFM> hfmd{mod, name};
            hfmd.def(py::init<DSparseMatrix, double, std::int8_t>())
                .def(py::init<Lattice, double, double, std::int8_t>())
                .def("K", py::overload_cast<Species>(&HFM::K, py::const_))
                .def("F", py::overload_cast<std::size_t, const CDVector&,
                     Species, bool>(&HFM::F, py::const_))
                .def("M", py::overload_cast<const CDVector&, Species>(&HFM::M, py::const_))
                .def("P", py::overload_cast<>(&HFM::P, py::const_))
                .def("Tplus", py::overload_cast<std::size_t, const CDVector&>(
                         &HFM::Tplus, py::const_))
                .def("Tminus", py::overload_cast<std::size_t, const CDVector&>(
                         &HFM::Tminus, py::const_))
                .def("Q", py::overload_cast<const CDVector&>(&HFM::Q, py::const_))
                .def("nx", &HFM::nx)
                .def("kappaTilde", &HFM::kappaTilde)
                .def("muTilde", &HFM::muTilde)
                .def("sigmaKappa", &HFM::sigmaKappa)
                ;

            mod.def("logdetQ",
                    static_cast<std::complex<double>(*)(const HFM&,
                                                        const CDVector&)>(logdetQ));
            mod.def("solveQ", static_cast<CDVector(*)(const HFM&,
                                                      const CDVector&,
                                                      const CDVector&)>(solveQ));

            mod.def("logdetM", py::overload_cast<
                    const HFM&, const CDVector &, Species>(logdetM));
            mod.def("solveM", py::overload_cast<
                    const HFM&, const CDVector&, Species, const CDMatrix&>(
                        solveM));

            bindHoppingSpecific(mod, hfmd);
        }
    }

    void bindHubbardFermiMatrix(py::module &mod) {
        py::enum_<Species>(mod, "Species")
            .value("PARTICLE", Species::PARTICLE)
            .value("HOLE", Species::HOLE)
            .def("values",
                 [](){
                     return py::make_iterator(SPECIES_VALUES.cbegin(), SPECIES_VALUES.cend());
                 });

        bindHFM<HubbardFermiMatrixDia>(mod, "HubbardFermiMatrixDia");
        bindHFM<HubbardFermiMatrixExp>(mod, "HubbardFermiMatrixExp");
    }
}
