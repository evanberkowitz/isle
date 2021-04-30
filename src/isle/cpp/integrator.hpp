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
     * The step size is adjusted to keep the intergation error
     * below the given tolerance.
     * The error is estiamted using the imaginary part of the action.
     * If the resulting step size is too small, integration stops
     * and returns the last point that could be reached within error bounds.
     *
     * Let \f$\Delta = |e^{i \text{Im} S_{i+1}} - e^{i \text{Im} S_i}|\f$
     * be the error of the imaginary part of the action and
     * \f$\Delta_0 = \texttt{imActTolerance}\f$.
     * The step size \f$h\f$ is reduced if \f$\Delta > \Delta_0\f$ to
     * \f[h \rightarrow h \beta {|\Delta_0 / \Delta|}^{1/5}\f]
     * It is increased if \f$\Delta < \gamma \Delta_0\f$ to
     * \f[h \rightarrow \min(2, \max(1, h \beta {|\Delta_0 / \Delta|}^{1/4}))\f]
     *
     * \param phi Starting configuration.
     * \param action Action to integrate over.
     * \param flowTime Length of the trajectory / total flow time.
     * \param stepSize Initial size of integration steps \f$h\f$.
     * \param actVal Value of the action at initial field (can be left out).
     * \param n one of (`0`, `1`), type of RK integrator.
     * \param direction Direction of integration, should be `+1` or `-1`.
     * \param adaptAttenuation \f$\beta\f$.
     * \param adaptThreshold \f$\gamma\f$.
     * \param minStepSize Minimum step size. Computed from `stepSize` by default.
     * \param imActTolerance Tolerance for the deviation of the imaginary part of
     *                       the action from the initial value.
     *
     * \returns Tuple of (in order)
     *           - final configuration phi
     *           - value of action at final phi
     *           - reached flow time in [0, `flowTime`]
     */
    std::tuple<CDVector, std::complex<double>, double>
    rungeKutta4Flow(CDVector phi,
                    const action::Action *action,
                    double flowTime,
                    double stepSize,
                    std::complex<double> actVal=std::complex<double>(std::nan(""), std::nan("")),
                    int n=0,
                    double direction=+1,
                    double adaptAttenuation=0.9,
                    double adaptThreshold=1.0e-8,
                    double minStepSize=std::nan(""),
                    double imActTolerance=0.001);
}  // namespace isle

#endif  // ndef INTEGRATOR_HPP
