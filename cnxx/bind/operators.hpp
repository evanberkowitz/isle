#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <utility>

namespace bind {
    /// Metafunctions and names for arithmetic operators.
    namespace op {

        /// Addition operator.
        struct add {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) + std::forward<RHS>(rhs)) {
                return lhs + rhs;
            }
        };

        /// Subtraction oeprator.
        struct sub {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) - std::forward<RHS>(rhs)) {
                return lhs - rhs;
            }
        };

        /// Multiplication operator.
        struct mul {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) * std::forward<RHS>(rhs)) {
                return lhs * rhs;
            }
        };

        /// Reverse multiplication operator.
        struct rmul {
            template <typename RHS, typename LHS>
            static auto f(RHS &&rhs, LHS &&lhs)
                -> decltype(std::forward<RHS>(rhs) * std::forward<LHS>(lhs)) {
                return lhs * rhs;
            }
        };

        /// In place addition operator.
        struct iadd {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) += std::forward<RHS>(rhs)) {
                return lhs += rhs;
            }
        };
        
        /// In place subtraction operator.
        struct isub {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) -= std::forward<RHS>(rhs)) {
                return lhs -= rhs;
            }
        };

        /// In place multiplication operator.
        struct imul {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) *= std::forward<RHS>(rhs)) {
                return lhs *= rhs;
            }
        };
    }
}

#endif  // ndef OPERATORS_HPP
