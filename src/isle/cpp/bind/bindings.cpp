#include "core.hpp"

#include "bind_version.hpp"
#include "math.hpp"
#include "lattice.hpp"
#include "hubbardFermiMatrix.hpp"
#include "action.hpp"
#include "integrator.hpp"

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
