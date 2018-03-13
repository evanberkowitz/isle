#include "hubbardFermiAction.hpp"

using HFA = cnxx::HubbardFermiAction;

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

        /// Evaluate action with given mmdag.
        template <typename MT>
        std::complex<double> doEval(MT &&mmdag) {
            return -logdet(Matrix<std::complex<double>>{std::forward<MT>(mmdag)});
        }

        /// Calculate force with given HFM and mmdag.
        template <typename MT>
        Vector<std::complex<double>> doForce(const HubbardFermiMatrix &hfm, MT &&mmdag) {
            const auto nx = hfm.nx();
            const auto nt = hfm.nt();

            // invert mmdag
            Matrix<std::complex<double>> mmdagInv{std::forward<MT>(mmdag)};
            auto ipiv = std::make_unique<int[]>(mmdag.rows());
            invert(mmdagInv, ipiv);

            // calcualte force
            Vector<std::complex<double>> force(mmdag.rows());
            SparseMatrix<std::complex<double>> q;
            for (std::size_t tau = 0; tau < nt; ++tau) {
                hfm.Q(q, tau+1);
                spacevec(force, tau, nx) = blaze::diagonal(q*spacemat(mmdagInv, tau, loopIdx(tau+1, nt), nx));
                hfm.Qdag(q, tau);
                spacevec(force, tau, nx) -= blaze::diagonal(spacemat(mmdagInv, loopIdx(tau+1, nt), tau, nx)*q);
            }

            return force;
        }
    }


    void HFA::updateHFM(const HubbardFermiMatrix &hfm) {
        _hfm = hfm;
    }
    void HFA::updateHFM(HubbardFermiMatrix &&hfm) {
        _hfm = std::move(hfm);
    }

    void HFA::updatePhi(const Vector<std::complex<double>> &phi) {
        _hfm.updatePhi(phi);
    }

    std::complex<double> HFA::eval(const Vector<std::complex<double>> &phi) {
        _hfm.updatePhi(phi);
        return doEval(_hfm.MMdag());
    }

    Vector<std::complex<double>> HFA::force(const Vector<std::complex<double>> &phi) {
        _hfm.updatePhi(phi);
        return doForce(_hfm, _hfm.MMdag());
    }

    std::pair<std::complex<double>, Vector<std::complex<double>>> HFA::valForce(
        const Vector<std::complex<double>> &phi) {

        _hfm.updatePhi(phi);
        const auto mmdag = _hfm.MMdag();
        return {doEval(mmdag), doForce(_hfm, mmdag)};
    }
}  // namespace cnxx

