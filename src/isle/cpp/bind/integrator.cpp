#include "integrator.hpp"

#include "../integrator.hpp"

using namespace pybind11::literals;
using namespace isle;

namespace bind {
    void bindIntegrators(py::module &mod) {
        mod.def("leapfrog", leapfrog, "phi"_a, "pi"_a, "action"_a,
                "length"_a, "nsteps"_a, "direction"_a=+1);
        mod.def("rungeKutta4", rungeKutta4, "phi"_a, "action"_a,
                "length"_a, "nsteps"_a, "actVal"_a=std::complex<double>(std::nan(""), std::nan("")),
                "n"_a=0, "direction"_a=+1, "attempts"_a=10, "imActTolerance"_a=0.001);
    }
}
