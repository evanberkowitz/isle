#include "hubbardFermiAction.hpp"

using namespace std::complex_literals;
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

        /// Evaluate action with given fermion matrix Q.
        template <typename MT>
        std::complex<double> doEval(MT &&Q) {
            return -logdet(Matrix<std::complex<double>>{std::forward<MT>(Q)});
        }

        /// Calculate force with given HFM and Q.
        template <typename MT>
        Vector<std::complex<double>> doForce(const HubbardFermiMatrix &hfm, MT &&Q) {
            const auto nx = hfm.nx();
            const auto nt = hfm.nt();

            // invert Q
            Matrix<std::complex<double>> QInv{std::forward<MT>(Q)};
            auto ipiv = std::make_unique<int[]>(Q.rows());
            invert(QInv, ipiv);

            // calculate force
            Vector<std::complex<double>> force(Q.rows());
            SparseMatrix<std::complex<double>> T;
            for (std::size_t tau = 0; tau < nt; ++tau) {
                hfm.Tplus(T, loopIdx(tau+1, nt));
                spacevec(force, tau, nx) = 1.i*blaze::diagonal(T*spacemat(QInv, tau, loopIdx(tau+1, nt), nx));
                hfm.Tminus(T, tau);
                spacevec(force, tau, nx) -= 1.i*blaze::diagonal(spacemat(QInv, loopIdx(tau+1, nt), tau, nx)*T);
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
        return doEval(_hfm.Q());
    }

    Vector<std::complex<double>> HFA::force(const Vector<std::complex<double>> &phi) {
        _hfm.updatePhi(phi);
        return doForce(_hfm, _hfm.Q());
    }

    std::pair<std::complex<double>, Vector<std::complex<double>>> HFA::valForce(
        const Vector<std::complex<double>> &phi) {

        _hfm.updatePhi(phi);
        const auto Q = _hfm.Q();
        return {doEval(Q), doForce(_hfm, Q)};
    }
}  // namespace cnxx
