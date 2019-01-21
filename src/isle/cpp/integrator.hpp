/** \file
 * \brief Molecular dynamics integrators.
 */

#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <tuple>

#include "math.hpp"
#include "action/action.hpp"

namespace isle {
    /// Perform leapfrog integration.
    /**
     * \param phi Starting configuration.
     * \param pi Starting momentum.
     * \param action Action to integrate over.
     * \param length Length of the trajectory. The size of each step is `length/nsteps`.
     * \param nsteps Number of integration steps.
     * \param direction Direction of integration, should be `+1` or `-1`.
     *
     * \returns Tuple of (in order)
     *           - final configuration phi
     *           - final momentum pi
     *           - value of action at final phi and pi
     */
    std::tuple<CDVector, CDVector, std::complex<double>>
    leapfrog(const CDVector &phi,
             const CDVector &pi,
             const action::Action *action,
             double length,
             std::size_t nsteps,
             double direction=+1);
}  // namespace isle

#endif  // ndef INTEGRATOR_HPP
