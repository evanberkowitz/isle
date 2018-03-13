/** \file
 * \brief Abstract base action.
 */

#ifndef ACTION_HPP
#define ACTION_HPP

#include "math.hpp"

namespace cnxx {
    /// Abstract base for Actions.
    struct Action {
        virtual ~Action() = default;

        /// Evaluate the %Action for given auxilliary field phi.
        virtual std::complex<double> eval(const Vector<std::complex<double>> &phi) = 0;

        /// Calculate force for given auxilliary field phi.
        virtual Vector<std::complex<double>> force(const Vector<std::complex<double>> &phi) = 0;

        /// Evaluate %Action and compute force for given auxilliary field phi.
        virtual std::pair<std::complex<double>, Vector<std::complex<double>>> valForce(
            const Vector<std::complex<double>> &phi) = 0;
    };
}  // namespace cnxx

#endif  // ndef ACTION_HPP
