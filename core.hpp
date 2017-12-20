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

#endif  // ndef CORE_HPP
