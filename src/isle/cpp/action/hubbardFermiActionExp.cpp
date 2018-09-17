#include "hubbardFermiActionExp.hpp"

using namespace std::complex_literals;
using HFA = isle::action::HubbardFermiActionExp;

namespace isle {
    namespace action {
        HFA::HubbardFermiActionExp(const HubbardFermiMatrix &hfm,
                                   const std::int8_t alpha)
            : _hfm(hfm),
              _expKappapInv(hfm.expKappa(Species::PARTICLE)),
              _expKappahInv(hfm.expKappa(Species::HOLE)),
              _alpha(alpha) {

            if (hfm.mu() != 0.)
                throw std::invalid_argument("HubbardFermiActionExp does not support non-zero chemical potential.");

            auto ipiv = std::make_unique<int[]>(_expKappapInv.rows());
            invert(_expKappapInv, ipiv);
            invert(_expKappahInv, ipiv);
        }

        HFA::HubbardFermiActionExp(const SparseMatrix<double> &kappaTilde,
                                   const double muTilde, const std::int8_t sigmaKappa,
                                   const std::int8_t alpha)
            : _hfm(kappaTilde, muTilde, sigmaKappa),
              _expKappapInv(_hfm.expKappa(Species::PARTICLE)),
              _expKappahInv(_hfm.expKappa(Species::HOLE)),
              _alpha(alpha) {

            if (muTilde != 0.)
                throw std::invalid_argument("HubbardFermiActionExp does not support non-zero chemical potential.");

            auto ipiv = std::make_unique<int[]>(_expKappapInv.rows());
            invert(_expKappapInv, ipiv);
            invert(_expKappahInv, ipiv);
        }

        HFA::HubbardFermiActionExp(const Lattice &lat, const double beta,
                                   const double muTilde, const std::int8_t sigmaKappa,
                                   const std::int8_t alpha)
            : _hfm(lat, beta, muTilde, sigmaKappa),
              _expKappapInv(_hfm.expKappa(Species::PARTICLE)),
              _expKappahInv(_hfm.expKappa(Species::HOLE)),
              _alpha(alpha) {

            if (muTilde != 0.)
                throw std::invalid_argument("HubbardFermiActionExp does not support non-zero chemical potential.");

            auto ipiv = std::make_unique<int[]>(_expKappapInv.rows());
            invert(_expKappapInv, ipiv);
            invert(_expKappahInv, ipiv);
        }

        std::complex<double> HFA::eval(const CDVector &phi) const {
            if (_alpha == 1)
                return -toFirstLogBranch(logdetMExp(_hfm, phi, Species::PARTICLE)
                                         + logdetMExp(_hfm, phi, Species::HOLE));
            // _alpha==0
            const CDVector aux = -1.i*phi;
            return -toFirstLogBranch(logdetMExp(_hfm, aux, Species::PARTICLE)
                                     + logdetMExp(_hfm, aux, Species::HOLE));
        }

        namespace {
            /// Calculate force w/o -i for either particles or holes.
            /*
             * Constructs all partial B^-1 to the left of (1+B^-1)^-1 first ('left').
             * Constructs rest on the fly ('right', contains (1+B^-1)^-1).
             */
            CDVector forcePart(const HubbardFermiMatrix &hfm, const CDVector &phi,
                               const CDMatrix &eki, const Species species) {
                const auto nx = hfm.nx();
                const auto nt = getNt(phi, nx);

                if (nt < 2)
                    throw std::runtime_error("nt < 2 in HubbardFermiAction not supported");

                // build B^-1 and partial products on the left of (1+B^-1)^-1
                std::vector<CDMatrix> lefts;  // in reverse order
                lefts.reserve(nt-1);  // not storing full B^-1 here

                // first term for tau = nt-2
                CDSparseMatrix f = hfm.F(nt-1, phi, species, true);
                lefts.emplace_back(f*eki);
                // other terms
                for (std::size_t t = nt-2; t != 0; --t) {
                    hfm.F(f, t, phi, species, true);
                    lefts.emplace_back(f*eki*lefts.back());
                }
                // full B^-1
                hfm.F(f, 0, phi, species, true);
                const CDMatrix Binv = f * eki * lefts.back();

                // start right with (1+B^-1)^-1
                CDMatrix right = IdMatrix<std::complex<double>>(nx) + Binv;
                auto ipiv = std::make_unique<int[]>(right.rows());
                invert(right, ipiv);

                CDVector force(nx*nt);  // the result

                // first term, tau = nt-1
                spacevec(force, nt-1, nx) = blaze::diagonal(Binv*right);

                // all sites except tau = nt-1
                for (std::size_t tau = 0; tau < nt-1; ++tau) {
                    hfm.F(f, tau, phi, species, true);
                    right = right * f * eki;
                    spacevec(force, tau, nx) = blaze::diagonal(lefts[nt-1-tau-1]*right);
                }

                return force;
            }
        }

        CDVector HFA::force(const CDVector &phi) const {
            if (_alpha == 1)
                return -1.i*(forcePart(_hfm, phi, _expKappapInv, Species::PARTICLE)
                             - forcePart(_hfm, phi, _expKappahInv, Species::HOLE));
            // _alpha==0
            const CDVector aux = -1.i*phi;
            return (forcePart(_hfm, aux, _expKappahInv, Species::HOLE)
                    -forcePart(_hfm, aux, _expKappapInv, Species::PARTICLE));
        }
    }  // namespace action
}  // namespace isle
