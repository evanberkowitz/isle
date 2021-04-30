#include "bind_core.hpp"

#include "bind_action.hpp"
#include "bind_hubbardFermiMatrix.hpp"
#include "bind_integrator.hpp"
#include "bind_lattice.hpp"
#include "bind_math.hpp"
#include "bind_version.hpp"

#include "../math.hpp"

using namespace isle;

PYBIND11_MODULE(ISLE_LIBNAME, mod) {
    mod.doc() = "Python bindings for isle";

    bind::storeVersions(mod);

    bind::bindTensors(mod);
    bind::bindLattice(mod);
    bind::bindHubbardFermiMatrix(mod);
    bind::bindActions(mod);
    bind::bindIntegrators(mod);
}
