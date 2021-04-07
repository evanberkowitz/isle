/** \file
 * \brief Bindings for Action.
 */

#ifndef BIND_ACTION_HPP
#define BIND_ACTION_HPP

#include "bind_core.hpp"

namespace bind {
    /// Define action submodule and bind action classes.
    void bindActions(py::module &mod);
}

#endif  // ndef BIND_ACTION_HPP
