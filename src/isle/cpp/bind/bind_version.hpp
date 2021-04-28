/** \file
 * \brief Version information.
 */

#ifndef BIND_VERSION_HPP
#define BIND_VERSION_HPP

#include "bind_core.hpp"

namespace bind {
    /// Store versions as module attributes.
    void storeVersions(py::module &mod);
}

#endif  // ndef BIND_VERSION_HPP
