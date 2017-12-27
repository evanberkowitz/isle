/** \file
 * \brief Bindings for math routines and classes.
 * \ingroup bind
 */

#ifndef BIND_MATH_HPP
#define BIND_MATH_HPP

#include "core.hpp"

/// Bind Vector, Matrix, and SparseMatrix.
void bindTensors(py::module &mod);

#endif  // ndef BIND_MATH_HPP
