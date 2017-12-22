/** \file
 * \brief Template meta programming.
 *
 * Throughout this program, the term meta function refers to a struct of the form
 * \code{.cpp}
 template<...>
 struct Foo {
     static void f() { ... }
 };
 \endcode
 * where `f` is allowed to have (template) parameters.
 */

#ifndef TMP_HPP
#define TMP_HPP

#include <utility>


/// Holds a list of types in a head, tail structure.
template <typename T, typename... Args>
struct Types {
    using Head = T; ///< Single type at the top of the list.
    using Tail = Types<Args...>; ///< Another list of the remaining types; `void` when there are none.
};

/// Base case for Types template.
template <typename T>
struct Types<T> {
    using Head = T;
    using Tail = void;
};

/// Internals of template meta programming.
namespace _tmp_internal {

    /// Perform 'foreach' iteration by recursing on the iterator.
    template <typename Cur, typename Rem, template <typename...> class F,
              typename... TParam>
    struct foreach_impl {
        template <typename... Args>
        static void f(Args && ...args) {
            // call function for current type
            F<Cur, TParam...>::f(std::forward<Args>(args)...);
            // recurse
            foreach_impl<typename Rem::Head, typename Rem::Tail,
                         F, TParam...>::f(std::forward<Args>(args)...);
        }
    };

    /// base case of 'foreach' iteration, calls F one last time.
    template <typename Cur, template <typename...> class F,
              typename... TParam>
    struct foreach_impl<Cur, void, F, TParam...> {
        template <typename... Args>
        static void f(Args && ...args) {
            F<Cur, TParam...>::f(std::forward<Args>(args)...);
        }
    };
}

/// Execute a meta function for each argument given in an template iterator.
/**
 * The function is executed like `F<T, TParam...>(args...)` but with perfect forwarding
 * for `args`. `T` is the template parameter from the current iteration.
 *
 * \tparam It Template iterator with a head, tail structure, see e.g. Types.
 * \tparam F Meta function t oexecute on each element of the iterator.
 * \tparam TParam Template parameters to pass to F.
 * \param args Runtime parameters to pass to F.
 */
template <typename It, template <typename...> class F,
          typename... TParam>
struct foreach {
    template <typename... Args>
    static void f(Args && ...args) {
        _tmp_internal::foreach_impl<typename It::Head, typename It::Tail,
                                    F, TParam...>::f(std::forward<Args>(args)...);
    }
};

/// Check whether type is specialization of template
/**
 * Is `true_type` iff T is specialization of template C.
 * \tparam C Template to check against.
 * \tparam T Type to check.
 */
template <template <typename ...> class C, typename T>
struct is_specialization_of : public std::false_type { };
/// Matching version of check.
template <template <typename ...> class C, typename ...Args>
struct is_specialization_of<C, C<Args...>> : public std::true_type { };

#endif  // ndef TMP_HPP
