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

/**
 *  Protect from stupid VT1 macro being defined by Mac headers.
 */
#ifdef VT1  
#undef VT1
#endif

namespace py = pybind11;


#endif  // ndef BIND_CORE_HPP
