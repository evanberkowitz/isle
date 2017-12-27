#include "core.hpp"
#include "math.hpp"

PYBIND11_MODULE(cnxx, mod) {
    bindTensors(mod);
}
