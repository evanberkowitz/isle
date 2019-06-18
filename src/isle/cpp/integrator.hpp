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
     *           - value of action at final phi
     */
    std::tuple<CDVector, CDVector, std::complex<double>>
    leapfrog(const CDVector &phi,
             const CDVector &pi,
             const action::Action *action,
             double length,
             std::size_t nsteps,
             double direction=+1);

    /// Perform Runge Kutta (RK4) integration for holomorphic flow.
    /**
     * Flow a configuration using the holomorphic flow equation
     * \f[\dot{\phi} = {(\nabla_{\phi} S[\phi])}^{\ast}\f]
     *
     * \param phi Starting configuration.
     * \param action Action to integrate over.
     * \param length Length of the trajectory. The size of each step is `length/nsteps`.
     * \param nsteps Number of integration steps.
     * \param actVal Value of the action at initial field (can be left out).
     * \param n one of (`0`, `1`), type of RK integrator.
     * \param direction Direction of integration, should be `+1` or `-1`.
     * \param attempts Number of attempts to make the integrator finer.
     * \param imActTolerance Tolerance for the deviation of the action from the initial value.
     *
     * \returns Tuple of (in order)
     *           - final configuration phi
     *           - value of action at final phi
     */
    std::tuple<CDVector, std::complex<double>>
    rungeKutta4Flow(const CDVector &phi,
                    const action::Action *action,
                    double length,
                    std::size_t nsteps,
                    std::complex<double> actVal=std::complex<double>(std::nan(""), std::nan("")),
                    int n=0,
                    double direction=+1,
                    int attempts=10,
                    double imActTolerance=0.001);
}  // namespace isle

#endif  // ndef INTEGRATOR_HPP
