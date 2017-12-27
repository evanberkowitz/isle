#include "core.hpp"
#include "math.hpp"

PYBIND11_MODULE(cns, mod) {
    bindTensors(mod);
}
