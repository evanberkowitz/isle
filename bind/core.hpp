/** \file
 * \brief Essential includes and definitions for bindings.
 *
 * \attention Always include last because Pybind11 and blaze can have name clashes on Macs.
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

namespace py = pybind11;


#endif  // ndef BIND_CORE_HPP
