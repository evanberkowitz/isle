/** \file
 * \brief Template meta programming.
 */

#ifndef TMP_HPP
#define TMP_HPP

#include <utility>

namespace isle {

    /// Template meta programming.
    /**
     * Throughout this program, the term meta function refers to a struct of the form
     * \code{.cpp}
     template<...>
     struct Foo {
     static void f() { ... }
     };
     \endcode
     * where `f` is allowed to have (template) parameters.
     */
    namespace tmp {

        /// Check whether type is specialization of template
        /**
         * Is `true_type` iff T is specialization of template C.
         * \tparam C Template to check against.
         * \tparam T Type to check.
         */
        template <template <typename ...> class C, typename T>
        struct IsSpecialization : public std::false_type { };

        /// Matching version of check.
        template <template <typename ...> class C, typename ...Args>
        struct IsSpecialization<C, C<Args...>> : public std::true_type { };


        /// Map sequence of type to void, useful to detect valid expressions.
        template <typename...>
        using void_t = void;


        /// Contains a value that is always false, useful to defer evaluation of static_assert.
        template <typename T>
        struct AlwaysFalse {
            static constexpr bool value = false;  ///< False.
        };
        /// Convenience alias for AlwaysFalse.
        template <typename T>
        constexpr bool AlwaysFalse_v = AlwaysFalse<T>::value;


        /// Select a new type based on an old one and a new elemental type.
        /**
         * \tparam Orig Original type, can be fundamental or a linear algebra type.
         * \tparam Other New elemental type.
         */
        template <typename Orig, typename Other, typename = void>
        struct Rebind {
            using type = Other;  ///< Rebound type. Can be `Other` or `Orig<Other>` if Orig is a linear algebra type.
        };

        /// Specialization for types supporting rebinding themselves.
        template <typename Orig, typename Other>
        struct Rebind<Orig, Other,
                      void_t<typename Orig::template Rebind<Other>::Other>> {
            using type = typename Orig::template Rebind<Other>::Other;
        };

        /// Convenience alias for Rebind.
        template <typename Orig, typename Other>
        using Rebind_t = typename Rebind<Orig, Other>::type;


        /// Holds a list of types in a head, tail structure.
        template <typename T, typename... Args>
        struct Types {
            using Head = T; ///< Single type at the top of the list.
            using Tail = Types<Args...>; ///< Another list of the remaining types; `void` when there are none.
        };

        /// Base case for Types template.
        template <typename T>
        struct Types<T> {
            using Head = T;  ///< Single type, the only member of the list.
            using Tail = void; ///< And the base-case void type.
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

            /// Base case of 'foreach' iteration, calls F one last time.
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

    }  // namespace tmp
}  // namespace isle

#endif  // ndef TMP_HPP
