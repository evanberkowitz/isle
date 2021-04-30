/** \file
 * \brief Bindings for math routines and classes.
 */

#ifndef BIND_MATH_HPP
#define BIND_MATH_HPP

#include "bind_core.hpp"

namespace bind {
    /// Bind Vector, Matrix, and SparseMatrix.
    /**
     * \bug Buffer construction from numpy array of ints not possible.
     *      Format descriptor for int does not match between Python and C++.
     *      Looks like a bug in Pybind11. Descriptors are
     *      - numpy array of int: `'l'`
     *      - C++ int: `'i'`
     *      - C++ long: `'q'`
     */
    void bindTensors(py::module &mod);
}

#endif  // ndef BIND_MATH_HPP
