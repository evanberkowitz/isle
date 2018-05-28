/** \file
 * \brief Essential includes and definitions for bindings.
 */

#ifndef BIND_CORE_HPP
#define BIND_CORE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl_bind.h>

// remove macro defined in termios.h on Mac to avoid clash in blaze
#ifdef VT1
  #undef VT1
#endif

#include "../math.hpp"

namespace py = pybind11;

// use a custom type
PYBIND11_MAKE_OPAQUE(std::vector<cnxx::CDVector>)

/// Bindings between `cnxx` and Python.
namespace bind {
    /// Represents the state of an iterator (adapted from pybind11 internals).
    template <typename Iterator, typename Sentinel,
              bool KeyIterator, py::return_value_policy Policy>
    struct IteratorState {
        Iterator it;  ///< C++ iterator to represent the state of.
        Sentinel end;  ///< C++ iterator to 'element one past end'.
        bool firstOrDone;  ///< `true` if on first element or done iterating, `false` otherwise.
    };

    /// Make an interator over indices and values.
    /**
     * Type `Iterator` must have member functions `index()` and `value()`.
     */
    template <py::return_value_policy Policy = py::return_value_policy::reference_internal,
              typename Iterator, typename Sentinel, typename... Extra>
    py::iterator makeIndexValueIterator(Iterator first, Sentinel last, Extra && ... extra) {
        using state = IteratorState<Iterator, Sentinel, false, Policy>;

        if (!py::detail::get_type_info(typeid(state), false)) {
            py::class_<state>{py::handle(), "iterator", py::module_local()}
                .def("__iter__", [](state &s) -> state& { return s; })
                .def("__next__", [](state &s) {
                        if (!s.firstOrDone)
                            ++s.it;
                        else
                            s.firstOrDone = false;
                        if (s.it == s.end) {
                            s.firstOrDone = true;
                            throw py::stop_iteration();
                        }
                        return std::make_pair(s.it->index(), s.it->value());
                    }, std::forward<Extra>(extra)..., Policy);
        }

        return py::cast(state{first, last, true});
    }
}

#endif  // ndef BIND_CORE_HPP
