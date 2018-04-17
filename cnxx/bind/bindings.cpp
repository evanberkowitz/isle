#include "core.hpp"

#include "math.hpp"
#include "lattice.hpp"
#include "hubbardFermiMatrix.hpp"
#include "action.hpp"
#include "hamiltonian.hpp"
#include "integrator.hpp"
#include "pardiso.hpp"

#include "../math.hpp"

using namespace cnxx;

namespace bind {
    namespace {
        /// Bind STL containers to dedicated Python types.
        void bindSTLContainers(py::module &mod) {
            py::bind_vector<std::vector<CDVector>>(mod, "VectorCDVector");
            py::implicitly_convertible<py::list, std::vector<CDVector>>();
        }
    }
}

PYBIND11_MODULE(cnxx, mod) {
    mod.doc() = "Python bindings for cnxx";

    bind::bindTensors(mod);
    bind::bindLattice(mod);
    bind::bindHubbardFermiMatrix(mod);
    bind::bindActions(mod);
    bind::bindHamiltonian(mod);
    bind::bindIntegrators(mod);
    bind::bindPARDISO(mod);

    bind::bindSTLContainers(mod);
}
