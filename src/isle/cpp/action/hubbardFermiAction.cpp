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

        /// Calculate force w/o -i uisng variant 1 algorithm for either particles or holes.
        /*
         * Constructs all partial A^-1 to the left of (1+A^-1)^-1 first ('left').
         * Constructs rest on the fly ('right', contains (1+A^-1)^-1).
         */
        CDVector forceVariant1Part(const HubbardFermiMatrix &hfm, const CDVector &phi,
                                   const CDSparseMatrix &k, const Species species) {

            const auto nx = hfm.nx();
            const auto nt = getNt(phi, nx);

            if (nt < 2)
                throw std::runtime_error("nt < 2 in HubbardFermiAction not supported");

            // build A^-1 and partial products on the left of (1+A^-1)^-1
            std::vector<CDMatrix> lefts;  // in reverse order
            lefts.reserve(nt-1);  // not storing full A^-1 here

            // first term for tau = nt-2
            CDSparseMatrix f = hfm.F(nt-1, phi, species, true);
            lefts.emplace_back(f*k);
            // other terms
            for (std::size_t t = nt-2; t != 0; --t) {
                hfm.F(f, t, phi, species, true);
                lefts.emplace_back(f*k*lefts.back());
            }
            // full A^-1
            hfm.F(f, 0, phi, species, true);
            const CDMatrix Ainv = f * k * lefts.back();

            // start right with (1+A^-1)^-1
            CDMatrix right = IdMatrix<std::complex<double>>(nx) + Ainv;
            auto ipiv = std::make_unique<int[]>(right.rows());
            invert(right, ipiv);

            CDVector force(nx*nt);  // the result

            // first term, tau = nt-1
            spacevec(force, nt-1, nx) = blaze::diagonal(Ainv*right);

            // all sites except tau = nt-1
            for (std::size_t tau = 0; tau < nt-1; ++tau) {
                hfm.F(f, tau, phi, species, true);
                right = right * f * k;
                spacevec(force, tau, nx) = blaze::diagonal(lefts[nt-1-tau-1]*right);
            }

            return force;
        }

        /// Calculate force using variant 2 algorithm.
        CDVector forceVariant2(const HubbardFermiMatrix &hfm,
                               const CDVector &phi) {
            const auto nx = hfm.nx();
            const auto nt = getNt(phi, nx);

            // invert Q
            Matrix<std::complex<double>> QInv{hfm.Q(phi)};
            auto ipiv = std::make_unique<int[]>(QInv.rows());
            invert(QInv, ipiv);

            // calculate force
            Vector<std::complex<double>> force(QInv.rows());
            SparseMatrix<std::complex<double>> T;
            for (std::size_t tau = 0; tau < nt; ++tau) {
                hfm.Tplus(T, loopIdx(tau+1, nt), phi);
                spacevec(force, tau, nx) = 1.i*blaze::diagonal(T*spacemat(QInv, tau, loopIdx(tau+1, nt), nx));
                hfm.Tminus(T, tau, phi);
                spacevec(force, tau, nx) -= 1.i*blaze::diagonal(spacemat(QInv, loopIdx(tau+1, nt), tau, nx)*T);
            }

            return force;
        }
    }

    std::complex<double> HFA::eval(const CDVector &phi) const {
        if (_variant == Variant::ONE)
            return -toFirstLogBranch(logdetM(_hfm, phi, Species::PARTICLE)
                                     + logdetM(_hfm, phi, Species::HOLE));
        else
            return -logdetQ(_hfm, phi);
    }

    CDVector HFA::force(const CDVector &phi) const {
        if (_variant == Variant::ONE)
            return -1.i*(forceVariant1Part(_hfm, phi, _kp, Species::PARTICLE)
                         - forceVariant1Part(_hfm, phi, _kh, Species::HOLE));
        else
            return forceVariant2(_hfm, phi);

    }

}  // namespace cnxx
