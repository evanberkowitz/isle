#include "integrator.hpp"

namespace isle {
    std::tuple<CDVector, CDVector> leapfrog(const CDVector &phi,
                                            const CDVector &pi,
                                            action::Hamiltonian &ham,
                                            const double length,
                                            const std::size_t nsteps,
                                            const double direction) {

        const double eps = direction*length/nsteps;

        // initial half step
        CDVector piOut = pi + blaze::real(ham.force(phi))*(eps/2);

        // first step in phi explicit in order to init new variable
        CDVector phiOut = phi + piOut*eps;

        // bunch of full steps
        for (std::size_t i = 0; i < nsteps-1; ++i) {
            piOut += blaze::real(ham.force(phiOut))*eps;
            phiOut += piOut*eps;
        }

        // last half step
        piOut += blaze::real(ham.force(phiOut))*(eps/2);

        return std::tuple<CDVector, CDVector>{std::move(phiOut), std::move(piOut)};
    }
}  // namespace isle
