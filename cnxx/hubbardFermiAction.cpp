#include "hubbardFermiAction.hpp"

using namespace std::complex_literals;
using HFA = cnxx::HubbardFermiAction;

#include <iostream>


namespace cnxx {
    namespace {
        /// Loop index around boundary; only works if i is at most one step across boundary.
        constexpr std::size_t loopIdx(const std::size_t i, const std::size_t n) noexcept {
            if (i == n)
                return 0;
            if (i == static_cast<std::size_t>(-1))
                return n-1;
            return i;
        }

        /// Calculate force for either particles or holes.
        CDVector forcePart(const HubbardFermiMatrix &hfm, const CDVector &phi, bool hole) {
            const auto nx = hfm.nx();
            const auto nt = phi.size()/nx;

            auto const k = hfm.K(hole);

            // A^-1
            CDMatrix Ainv = IdMatrix<std::complex<double>>(nx);
            for (std::size_t t = 0; t < nt; ++t)
                Ainv = Ainv * hfm.F(t, phi, hole, true)*k;

            // (1+A^-1)^-1
            CDMatrix invmat = IdMatrix<std::complex<double>>(nx) + Ainv;
            auto ipiv = std::make_unique<int[]>(invmat.rows());
            invert(invmat, ipiv);

            CDVector force(nx*nt);
            // all sites except tau = nt-1
            for (std::size_t tau = 0; tau < nt-1; ++tau) {
                CDMatrix left = IdMatrix<std::complex<double>>(nx);
                CDMatrix right = IdMatrix<std::complex<double>>(nx);

                for (std::size_t t = tau+1; t < nt; ++t)
                    left = left * hfm.F(t, phi, hole, true) * k;
                for (std::size_t t = 0; t < tau+1; ++t)
                    right = right * hfm.F(t, phi, hole, true) * k;

                spacevec(force, tau, nx) = -1.i*blaze::diagonal(left*invmat*right);
            }

            // tau = nt-1
            spacevec(force, nt-1, nx) = -1.i*blaze::diagonal(Ainv*invmat);

            if (hole)
                force *= -1;
            
            return force;
        }
    }


    void HFA::updateHFM(const HubbardFermiMatrix &hfm) {
        _hfm = hfm;
    }
    void HFA::updateHFM(HubbardFermiMatrix &&hfm) {
        _hfm = std::move(hfm);
    }

    std::complex<double> HFA::eval(const CDVector &phi) {
        return -toFirstLogBranch(logdetM(_hfm, phi, false) - logdetM(_hfm, phi, true));
    }

    CDVector HFA::force(const CDVector &phi) {
        return force_part(_hfm, phi, true) + force_part(_hfm, phi, false);
    }

}  // namespace cnxx
