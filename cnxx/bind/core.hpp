/** \file
 * \brief Essential includes and definitions for bindings.
 */

/**
 * \defgroup bind Bindings
 *   Python bindings for C++.
 */

#ifndef BIND_CORE_HPP
#define BIND_CORE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
// #include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

// remove macro defined in termios.h on Mac to avoid clash in blaze
#ifdef VT1
  #undef VT1
#endif

namespace py = pybind11;

#endif  // ndef BIND_CORE_HPP
