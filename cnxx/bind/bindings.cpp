#include "core.hpp"
#include "math.hpp"
#include "lattice.hpp"

PYBIND11_MODULE(cnxx, mod) {
    bind::bindTensors(mod);
    bind::bindLattice(mod);
}
