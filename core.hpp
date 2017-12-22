/** \file
 * \brief Core definitions and functions.
 */

#ifndef CORE_HPP
#define CORE_HPP

/// `true` iff NDEBUG macro is set.
#ifdef NDEBUG
constexpr bool ndebug = true;
#else
constexpr bool ndebug = false;
#endif

// Suppress compiler warnings about unused function parameters.
#ifndef UNUSED
#if defined(__GNUC__)
#define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#elif defined(__LCLINT__)
#define UNUSED(x) /*@unused@*/ x
#else
#define UNUSED(x) x
#endif
#endif

#endif  // ndef CORE_HPP
