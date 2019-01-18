/** \file
 * \brief Abstract base action.
 */

#ifndef ACTION_ACTION_HPP
#define ACTION_ACTION_HPP

#include "../math.hpp"

namespace isle {
    /// Contains all actions implemented in C++.
    namespace action {
        /// Abstract base for Actions.
        struct Action {
            virtual ~Action() noexcept = default;

            /// Evaluate the %Action for given auxilliary field phi.
            virtual std::complex<double> eval(const Vector<std::complex<double>> &phi) const = 0;

            /// Calculate force for given auxilliary field phi.
            virtual Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) const = 0;
        };
    }  // namespace action
}  // namespace isle

#endif  // ndef ACTION_ACTION_HPP
