#include "integrator.hpp"

#include <cmath>
#include <sstream>

#include "bind/logging.hpp"

using namespace std::complex_literals;


namespace isle {
    std::tuple<CDVector, CDVector, std::complex<double>>
    leapfrog(const CDVector &phi,
             const CDVector &pi,
             const action::Action *const action,
             const double length,
             const std::size_t nsteps,
             const double direction) {

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
    }

    std::tuple<CDVector, std::complex<double>>
    rungeKutta4(const CDVector &phi,
                const action::Action *action,
                const double length,
                const std::size_t nsteps,
                std::complex<double> actVal,
                const int n,
                const double direction,
                int attempts,
                const double imActTolerance) {

        if (n != 0 && n != 1) {
            throw std::invalid_argument("n must be 0 or 1");
        }

        double epsilon = length / static_cast<double>(nsteps);

        if (std::isnan(real(actVal)) || std::isnan(imag(actVal))) {
            actVal = action->eval(phi);
        }

        // results
        CDVector phiOut = phi;
        auto actValOut = actVal;

        // need to keep two copies in order to revert if integration becomes unstable
        CDVector phiAux = phi;

        for (std::size_t i = 0; i < nsteps; ++i) {
            // do one step
            if (n == 0) {
                phiAux = rk4Step<0>(phiOut, action, epsilon, direction);
            }
            else {
                phiAux = rk4Step<1>(phiOut, action, epsilon, direction);
            }
            auto actValAux = action->eval(phiAux);

            auto const actDiff = abs(exp(1.0i*(imag(actVal)-imag(actValAux))) - 1.0);
            if (actDiff > imActTolerance) {
                // Imag part of action deviates too much, do not advance integrator.
                if (attempts > 0) {
                    // There are attempts left, make integration finer.
                    attempts--;
                    epsilon /= 10.0;
                } else {
                    std::ostringstream oss;
                    oss << "Imaginary part of the action deviates by " << actDiff
                        << ", (tolerance = " << imActTolerance
                        << ") at step " <<  i;
                    getLogger("rungeKutta4").error(oss.str());
                    throw std::runtime_error("Imaginary part of action deviates too much");
                }
            }

            else {
                // everything is fine => advance state
                // Quickly set phiOut = phiAux and leave phiAux in a bad but fast to assign-to state.
                swap(phiOut, phiAux);
                actValOut = actValAux;
            }
        }

        return std::make_tuple(phiOut, actValOut);
    }

}  // namespace isle
