#ifndef BIND_PARDISO_HPP
#define BIND_PARDISO_HPP

#include "core.hpp"

namespace bind {
    /// Bind PARDISO interface.
    void bindPARDISO(py::module &mod);
}

#endif  // ndef BIND_PARDISO_HPP
