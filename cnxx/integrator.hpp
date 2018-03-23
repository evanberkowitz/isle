/** \file
 * \brief Molecular dynamics integrators.
 */

#ifndef INTEGRATOR_HPP
#define INTEGRATOR_HPP

#include <tuple>

#include "math.hpp"
#include "hamiltonian.hpp"

namespace cnxx {
    /// Perform leapfrog integration.
    /**
     * \todo Optimize final evaluation and make use of Hamiltonian::valForce.
     *       Be careful about using the correct pi!
     *
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
     *           - 'energy' evaluated with final phi and pi
     */
    std::tuple<CDVector, CDVector, std::complex<double>> leapfrog(const CDVector &phi,
                                                                  const CDVector &pi,
                                                                  Hamiltonian &ham,
                                                                  double length,
                                                                  std::size_t nsteps,
                                                                  double direction=+1);
}

#endif  // ndef INTEGRATOR_HPP
