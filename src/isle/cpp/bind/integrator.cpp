#include "integrator.hpp"

#include "../integrator.hpp"

using namespace pybind11::literals;
using namespace isle;

namespace bind {
    void bindIntegrators(py::module &mod) {
        mod.def("leapfrog", leapfrog, "phi"_a, "pi"_a, "action"_a,
                "length"_a, "nsteps"_a, "direction"_a=+1);
        mod.def("rungeKutta4Flow", rungeKutta4Flow,
                "phi"_a,
                "action"_a,
                "length"_a,
                "stepSize"_a,
                "actVal"_a=std::complex<double>(std::nan(""), std::nan("")),
                "n"_a=0,
                "direction"_a=+1,
                "adaptAttenuation"_a=0.9,
                "adaptThreshold"_a=1.0e-12,
                "minStepSize"_a=std::nan(""),
                "imActTolerance"_a=0.001);
    }
}
