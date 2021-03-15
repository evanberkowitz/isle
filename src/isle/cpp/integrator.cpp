#include "integrator.hpp"

#include <cmath>

#include "profile.hpp"


namespace isle {
    std::tuple<CDVector, CDVector, std::complex<double>>
    leapfrog(const CDVector &phi,
             const CDVector &pi,
             const action::Action *const action,
             const double length,
             const std::size_t nsteps,
             const double direction) {
        ISLE_PROFILE_NVTX_RANGE("leapfrog");
        const double eps = direction*length/static_cast<double>(nsteps);

        // initial half step
        CDVector piOut = pi + blaze::real(action->force(phi))*(eps/2);

        // first step in phi explicit in order to init new variable
        CDVector phiOut = phi + piOut*eps;

        // bunch of full steps
        for (std::size_t i = 0; i < nsteps-1; ++i) {
            piOut += blaze::real(action->force(phiOut))*eps;
            phiOut += piOut*eps;
        }

        // last half step
        piOut += blaze::real(action->force(phiOut))*(eps/2);

        const std::complex<double> actVal = action->eval(phiOut);
        return std::make_tuple(std::move(phiOut), std::move(piOut), actVal);
    }


    namespace {
        template <int N>
        struct RK4Params { };

        template <>
        struct RK4Params<0> {
            constexpr static double omega1 = 1.0/6.0;
            constexpr static double omega2 = 1.0/3.0;
            constexpr static double omega3 = 1.0/3.0;
            constexpr static double omega4 = 1.0/6.0;
            constexpr static double beta21 = 0.5;
            constexpr static double beta31 = 0.0;
            constexpr static double beta32 = 0.5;
            constexpr static double beta41 = 0.0;
            constexpr static double beta42 = 0.0;
            constexpr static double beta43 = 1.0;
        };

        template <>
        struct RK4Params<1> {
            constexpr static double omega1 = 0.125;
            constexpr static double omega2 = 0.375;
            constexpr static double omega3 = 0.375;
            constexpr static double omega4 = 0.125;
            constexpr static double beta21 = 1.0/30.;
            constexpr static double beta31 = -1.0/3;
            constexpr static double beta32 = 1.0;
            constexpr static double beta41 = 1.0;
            constexpr static double beta42 = -1.0;
            constexpr static double beta43 = 1.0;
        };

        template <int N>
        CDVector rk4Step(const CDVector &phi,
                         const action::Action *action,
                         const double epsilon,
                         const double direction) {

            using p = RK4Params<N>;

            const double edir = epsilon*direction;

            const CDVector k1 = -edir * conj(action->force(phi));
            const CDVector k2 = -edir * conj(action->force(phi
                                                           + p::beta21*k1));
            const CDVector k3 = -edir * conj(action->force(phi
                                                           + p::beta31*k1
                                                           + p::beta32*k2));
            const CDVector k4 = -epsilon * conj(action->force(phi
                                                              + p::beta41*k1
                                                              + p::beta42*k2
                                                              + p::beta43*k3));

            return phi + p::omega1*k1 + p::omega2*k2 + p::omega3*k3 + p::omega4*k4;
        }

        std::pair<CDVector, std::complex<double>>
        rk4Step(const CDVector &phi,
                const action::Action *action,
                const double epsilon,
                const double direction,
                const int n) {

            const auto phiOut = n == 0
                ? rk4Step<0>(phi, action, epsilon, direction)
                : rk4Step<1>(phi, action, epsilon, direction);
            const auto actValOut = action->eval(phiOut);
            return {std::move(phiOut), actValOut};
        }

        double reduceStepSize(const double stepSize,
                              const double adaptAttenuation,
                              const double minStepSize,
                              const double error,
                              const double imActTolerance) {
            return std::max(
                stepSize*adaptAttenuation*std::pow(imActTolerance/error, 1.0/5.0),
                minStepSize);
        }

        double increaseStepSize(const double stepSize,
                                const double adaptAttenuation,
                                const double error,
                                const double imActTolerance) {
            // max(x, 1) makes sure the step size never decreases
            // min(x, 2) makes sure the step size does not grow too large
            return stepSize * std::min(
                std::max(
                    adaptAttenuation*std::pow(imActTolerance/error, 1.0/4.0),
                    1.0),
                2.0);
        }
    }

    std::tuple<CDVector, std::complex<double>, double>
    rungeKutta4Flow(CDVector phi,
                    const action::Action *action,
                    const double flowTime,
                    double stepSize,
                    std::complex<double> actVal,
                    const int n,
                    const double direction,
                    const double adaptAttenuation,
                    const double adaptThreshold,
                    double minStepSize,
                    const double imActTolerance) {

        if (n != 0 && n != 1) {
            throw std::invalid_argument("n must be 0 or 1");
        }

        if (std::isnan(real(actVal)) || std::isnan(imag(actVal))) {
            actVal = action->eval(phi);
        }

        if (std::isnan(minStepSize)) {
            minStepSize = std::max(stepSize / 1000.0, 1e-12);
        }

        double currentFlowTime;
        for (currentFlowTime = 0.0; currentFlowTime < flowTime;) {
            // make sure we don't integrate for longer than flowTime
            if (currentFlowTime + stepSize > flowTime) {
                stepSize = flowTime - currentFlowTime;
                if (stepSize < minStepSize) {
                    // really short step left to go -> just skip it
                    break;
                }
            }

            const auto attempt = rk4Step(phi, action, stepSize, direction, n);
            const auto error = abs(exp(std::complex<double>{0.0, imag(actVal)-imag(attempt.second)}) - 1.0);

            if (error > imActTolerance) {
                if (stepSize == minStepSize) {
                    break;
                }
                stepSize = reduceStepSize(stepSize, adaptAttenuation,
                                          minStepSize, error, imActTolerance);
                // repeat current step
            }

            else {
                // attempt was successful -> advance
                currentFlowTime += stepSize;
                phi = std::move(attempt.first);
                actVal = attempt.second;

                if (error < adaptThreshold*imActTolerance) {
                    stepSize = increaseStepSize(stepSize, adaptAttenuation,
                                                error, imActTolerance);
                }
            }
        }

        return std::make_tuple(phi, actVal, currentFlowTime);
    }

}  // namespace isle
