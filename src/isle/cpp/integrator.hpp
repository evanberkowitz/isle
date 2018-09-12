/** \file
 * \brief Molecular dynamics integrators.
 */

#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <tuple>

#include "math.hpp"
#include "action/hamiltonian.hpp"

namespace isle {
    /// Perform leapfrog integration.
    /**
     * \param phi Starting configuration.
     * \param pi Starting momentum.
     * \param ham Hamiltonian, is used for forces and evaluation of action.
     * \param length Length of the trajectory. The size of each step is `length/nsteps`.
     * \param nsteps Number of integration steps.
     * \param direction Direction of integration, should be `+1` or `-1`.
     *
     * \returns Tuple of (in order)
     *           - final configuration phi
     *           - final momentum pi
     */
    std::tuple<CDVector, CDVector> leapfrog(const CDVector &phi,
                                            const CDVector &pi,
                                            action::Hamiltonian &ham,
                                            double length,
                                            std::size_t nsteps,
                                            double direction=+1);
}  // namespace isle

#endif  // ndef INTEGRATOR_HPP
