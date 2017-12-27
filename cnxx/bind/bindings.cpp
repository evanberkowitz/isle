#include "core.hpp"
#include "math.hpp"

PYBIND11_MODULE(cnxx, mod) {
    bind::bindTensors(mod);
}
