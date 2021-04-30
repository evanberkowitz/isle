#include "bind_version.hpp"

#include "version.hpp"

#include <blaze/system/Version.h>

#define STRINGIFY(s) STRINGIFY_IMPL(s)
#define STRINGIFY_IMPL(s) #s

namespace bind {

    /**
     * Store a version in the form of strings.
     * Versions are fomattted as `major.minor-extra`,
     * where `extra` is optional.
     */
    struct Version {
        /// Major version numer.
        const char * const major;
        /// Minor version numer.
        const char * const minor;
        /// Extra information on version.
        const char * const extra = nullptr;

        /// Return a string representation of the version.
        std::string toString() const {
            std::ostringstream oss;
            oss << major << '.' << minor;
            if (extra)
                oss << '-' << extra;
            return oss.str();
        }
    };

    /// Version of isle; is inserted by the build system.
    const Version isleVersion{ISLE_VERSION_MAJOR,
                              ISLE_VERSION_MINOR,
                              ISLE_VERSION_EXTRA};

    /// Version of Python that this library is compiled with.
    const Version pythonVersion{STRINGIFY(PY_MAJOR_VERSION),
                                STRINGIFY(PY_MINOR_VERSION)};

    /// Version of Pybind11 that this library is compiled with.
    const Version pybind11Version{STRINGIFY(PYBIND11_VERSION_MAJOR),
                                  STRINGIFY(PYBIND11_VERSION_MINOR),
                                  STRINGIFY(PYBIND11_VERSION_PATCH)};

    /// Version of blaze that this library is compiled with.    
    const Version blazeVersion{STRINGIFY(BLAZE_MAJOR_VERSION),
                               STRINGIFY(BLAZE_MINOR_VERSION)};

    void storeVersions(py::module &mod) {
        py::class_<Version>(mod, "Version")
            .def(py::init<const char *, const char *>())
            .def_readonly("major", &Version::major)
            .def_readonly("minor", &Version::minor)
            .def_readonly("extra", &Version::extra)
            .def("__str__", &Version::toString)
            ;

        mod.attr("isleVersion") = isleVersion;
        mod.attr("pythonVersion") = pythonVersion;
        mod.attr("pybind11Version") = pybind11Version;
        mod.attr("blazeVersion") = blazeVersion;
    }
}
