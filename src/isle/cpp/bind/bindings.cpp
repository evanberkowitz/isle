#include "core.hpp"

#include "bind_version.hpp"
#include "math.hpp"
#include "lattice.hpp"
#include "hubbardFermiMatrix.hpp"
#include "action.hpp"
#include "hamiltonian.hpp"
#include "integrator.hpp"
#include "pardiso.hpp"

#include "../math.hpp"

using namespace isle;

namespace bind {
    namespace {
        /// Bind STL containers to dedicated Python types.
        void bindSTLContainers(py::module &mod) {
            py::bind_vector<std::vector<CDVector>>(mod, "VectorCDVector");
            py::implicitly_convertible<py::list, std::vector<CDVector>>();
        }
    }
}

PYBIND11_MODULE(ISLE_LIBNAME, mod) {
    mod.doc() = "Python bindings for isle";

    bind::storeVersions(mod);

    bind::bindTensors(mod);
    bind::bindLattice(mod);
    bind::bindHubbardFermiMatrix(mod);
    bind::bindActions(mod);
    bind::bindHamiltonian(mod);
    bind::bindIntegrators(mod);
    bind::bindPARDISO(mod);

    bind::bindSTLContainers(mod);
}
