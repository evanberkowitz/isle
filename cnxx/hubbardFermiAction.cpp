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
        const auto nx = _hfm.nx();
        const auto nt = phi.size()/nx;

        // invert Q
        CDMatrix Qinv{_hfm.Q(phi)};
        auto ipiv = std::make_unique<int[]>(Qinv.rows());
        invert(Qinv, ipiv);

        // calculate force
        CDVector force(Qinv.rows());
        CDSparseMatrix T;
        for (std::size_t tau = 0; tau < nt; ++tau) {
            _hfm.Tplus(T, loopIdx(tau+1, nt), phi);
            spacevec(force, tau, nx) = 1.i*blaze::diagonal(T*spacemat(Qinv, tau, loopIdx(tau+1, nt), nx));
            _hfm.Tminus(T, tau, phi);
            spacevec(force, tau, nx) -= 1.i*blaze::diagonal(spacemat(Qinv, loopIdx(tau+1, nt), tau, nx)*T);
        }

        return force;
    }

    // namespace {

    //     CDMatrix invMat(HubbardFermiMatrix &hfm, bool hole) {
    //         const auto nx = hfm.nx();
    //         const auto nt = hfm.nt();

    //         auto const k = hfm.K(hole);

    //         CDMatrix mat = IdMatrix<std::complex<double>>(nx);

    //         for (std::size_t t = 0; t < nt; ++t)
    //             mat = mat * hfm.F(t, hole, true)*k;
            
    //         mat += IdMatrix<std::complex<double>>(nx);

    //         auto ipiv = std::make_unique<int[]>(mat.rows());
    //         invert(mat, ipiv);

    //         return mat;
    //     }

    //     CDVector force2_part(HubbardFermiMatrix &hfm, bool hole) {
    //         const auto nx = hfm.nx();
    //         const auto nt = hfm.nt();

    //         auto const k = hfm.K(hole);
    //         auto const invmat = invMat(hfm, hole);
            
    //         CDVector force(nx*nt);
    //         for (std::size_t tau = 0; tau < nt-1; ++tau) {
    //             CDMatrix left = IdMatrix<std::complex<double>>(nx);
    //             CDMatrix right = IdMatrix<std::complex<double>>(nx);

    //             for (std::size_t t = tau+1; t < nt; ++t)
    //                 left = left * hfm.F(t, hole, true) * k;
    //             for (std::size_t t = 0; t < tau+1; ++t)
    //                 right = right * hfm.F(t, hole, true) * k;

    //             spacevec(force, tau, nx) = -1.i*blaze::diagonal(left*invmat*right);
    //         }

    //         CDMatrix Ainv = IdMatrix<std::complex<double>>(nx);
    //         for (std::size_t t = 0; t < nt; ++t)
    //             Ainv = Ainv * hfm.F(t, hole, true)*k;
    //         spacevec(force, nt-1, nx) = -1.i*blaze::diagonal(Ainv*invmat);

    //         if (hole)
    //             force *= -1;
            
    //         return force;
    //     }
        
    // }

    
    // Vector<std::complex<double>> HFA::force2(const Vector<std::complex<double>> &phi) {
    //     _hfm.updatePhi(phi);

    //     return force2_part(_hfm, true) + force2_part(_hfm, false);
    // }
    
}  // namespace cnxx
