/** \file
 * \brief Bindings for HubbardFermiMatrix.
 * \ingroup bind
 */

#ifndef BIND_HUBBARD_FERMI_MATRIX_HPP
#define BIND_HUBBARD_FERMI_MATRIX_HPP

#include "core.hpp"

namespace bind {
    /// Bind class HubbardFermiMatrix, related classes, and free functions.
    void bindHubbardFermiMatrix(py::module &mod);
}

#endif  // ndef BIND_HUBBARD_FERMI_MATRIX_HPP
