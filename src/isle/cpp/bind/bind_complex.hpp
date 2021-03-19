#ifndef BIND_COMPLEX_HPP
#define BIND_COMPLEX_HPP

#ifdef USE_CUDA
#include <pybind11/pybind11.h>
#include "../complex.hpp"

/// glibc defines I as a macro which breaks things, e.g., boost template names
#ifdef I
#  undef I
#endif

namespace pybind11 {

template <typename T>
struct format_descriptor<
    isle::complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>> {
  static constexpr const char c = format_descriptor<T>::c;
  static constexpr const char value[3] = {'Z', c, '\0'};
  static std::string format() { return std::string(value); }
};

#ifndef PYBIND11_CPP17

template <typename T>
constexpr const char format_descriptor<
    isle::complex<T>,
    detail::enable_if_t<std::is_floating_point<T>::value>>::value[3];

#endif

namespace detail {

template <typename T>
struct is_fmt_numeric<isle::complex<T>,
                      detail::enable_if_t<std::is_floating_point<T>::value>> {
  static constexpr bool value = true;
  static constexpr int index = is_fmt_numeric<T>::index + 3;
};

template <typename T> class type_caster<isle::complex<T>> {
public:
  bool load(handle src, bool convert) {
    if (!src)
      return false;
    if (!convert && !PyComplex_Check(src.ptr()))
      return false;
    Py_complex result = PyComplex_AsCComplex(src.ptr());
    if (result.real == -1.0 && PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    value = isle::complex<T>((T)result.real, (T)result.imag);
    return true;
  }

  static handle cast(const isle::complex<T> &src,
                     return_value_policy /* policy */, handle /* parent */) {
    return PyComplex_FromDoubles((double)src.real(), (double)src.imag());
  }

  PYBIND11_TYPE_CASTER(isle::complex<T>, _("complex"));
};
}
}

#else
#include <pybind11/complex.h>

#endif  // def USE_CUDA
#endif  // ndef BIND_COMPLEX_HPP