#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <utility>

#include "../math.hpp"

namespace bind {
    /// Metafunctions and names for arithmetic operators.
    namespace op {

        /// Addition operator.
        struct add {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) + std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) + std::forward<RHS>(rhs);
            }
        };

        /// Subtraction oeprator.
        struct sub {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) - std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) - std::forward<RHS>(rhs);
            }
        };

        /// Multiplication operator.
        struct mul {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) * std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) * std::forward<RHS>(rhs);
            }
        };

        /// Reverse multiplication operator.
        struct rmul {
            template <typename RHS, typename LHS>
            static auto f(RHS &&rhs, LHS &&lhs)
                -> decltype(std::forward<RHS>(rhs) * std::forward<LHS>(lhs)) {
                return std::forward<RHS>(rhs) * std::forward<LHS>(lhs);
            }
        };

        /// In place addition operator.
        struct iadd {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) += std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) += std::forward<RHS>(rhs);
            }
        };
        
        /// In place subtraction operator.
        struct isub {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) -= std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) -= std::forward<RHS>(rhs);
            }
        };

        /// In place multiplication operator.
        struct imul {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(std::forward<LHS>(lhs) *= std::forward<RHS>(rhs)) {
                return std::forward<LHS>(lhs) *= std::forward<RHS>(rhs);
            }
        };
        
        /// Dot product operator.
        struct dot {
            template <typename LHS, typename RHS>
            static auto f(LHS &&lhs, RHS &&rhs)
                -> decltype(blaze::dot(std::forward<LHS>(lhs), std::forward<RHS>(rhs))) {
                return blaze::dot(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
            }
        };
    }
}

#endif  // ndef OPERATORS_HPP
