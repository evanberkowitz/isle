#include "pardiso.hpp"

#if defined(PARDISO_STANDALONE) || defined(PARDISO_MKL)

#include "../pardiso.hpp"

using namespace py::literals;
using namespace isle::Pardiso;

namespace {
    /// Bind a isle::Pardiso::State for given elemental type.
    template <typename ET>
    void bindPState(py::module &mod, const char * const name) {
        using ST = State<ET>;

        py::class_<ST>(mod, name)
            .def(py::init<MType, Solver, int>(),
                 "mtype"_a=MType::NON_SYM,
                 "solver"_a=Solver::DIRECT,
                 "messageLevel"_a=0)
            .def("clear", &ST::clear)
            .def("__call__", [](ST &self,
                                const isle::SparseMatrix<ET> &mat,
                                isle::Vector<ET> &b,
                                const Phase startPhase=Phase::ANALYSIS,
                                const Phase endPhase=Phase::SOLVE) {
                     return self(mat, b, startPhase, endPhase);
                 },
                 "mat"_a, "b"_a,
                 "startPhase"_a=Phase::ANALYSIS, "endPhase"_a=Phase::SOLVE)
            .def("__call__", [](ST &self,
                                const isle::SparseMatrix<ET> &mat,
                                isle::Matrix<ET> &b,
                                const Phase startPhase=Phase::ANALYSIS,
                                const Phase endPhase=Phase::SOLVE) {
                     return self(mat, b, startPhase, endPhase);
                 },
                 "mat"_a, "b"_a,
                 "startPhase"_a=Phase::ANALYSIS, "endPhase"_a=Phase::SOLVE)
            ;
    }
}

namespace bind {
    void bindPARDISO(py::module &mod) {
        using namespace isle::Pardiso;

        // create a submodule for pardiso
        auto pmod = mod.def_submodule("pardiso", "PARDISO sparse solver");

        py::enum_<Solver>{pmod, "Solver"}
            .value("DIRECT", Solver::DIRECT)
            .value("ITERATIVE", Solver::ITERATIVE);

        py::enum_<MType>{pmod, "MType"}
            .value("STRUCT_SYM", MType::STRUCT_SYM)
            .value("SYM_HERM_POS_DEF", MType::SYM_HERM_POS_DEF)
            .value("SYM_HERM_INDEF", MType::SYM_HERM_INDEF)
            .value("DIAGONAL", MType::DIAGONAL)
            .value("BUNCH_KAUF", MType::BUNCH_KAUF)
            .value("COMPL_SYM", MType::COMPL_SYM)
            .value("NON_SYM", MType::NON_SYM);

        py::enum_<Phase>{pmod, "Phase"}
            .value("ANALYSIS", Phase::ANALYSIS)
            .value("FACTORIZATION", Phase::FACTORIZATION)
            .value("SOLVE", Phase::SOLVE);

        bindPState<double>(pmod, "DState");
        bindPState<std::complex<double>>(pmod, "CDState");
    }
}

#else  // PARDISO is not enabled

#include "../core.hpp"

namespace bind {
    void bindPARDISO(py::module &UNUSED(mod)) {
        // don't bind anything
    }
}
#endif
