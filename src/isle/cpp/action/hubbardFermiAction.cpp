#include "hubbardFermiAction.hpp"

using namespace std::complex_literals;

// TODO detect bipartiteness and other params to only evaluate det(M_p) and get det(M_h) from it

namespace isle {
    namespace action {
        namespace {
            /// Calculate force w/o -i using variant 1 algorithm for either particles or holes.
            /*
             * Constructs all partial A^-1 to the left of (1+A^-1)^-1 first ('left').
             * Constructs rest on the fly ('right', contains (1+A^-1)^-1).
             */
            template <typename HFM>
            CDVector forceVariant1Part(const HFM &hfm, const CDVector &phi,
                                       const DSparseMatrix &k, const Species species) {

                const auto nx = hfm.nx();
                const auto nt = getNt(phi, nx);

                if (nt < 2)
                    throw std::runtime_error("nt < 2 in HubbardFermiAction algorithm variant 1 not supported");

                // build A^-1 and partial products on the left of (1+A^-1)^-1
                std::vector<CDMatrix> lefts;  // in reverse order
                lefts.reserve(nt-1);  // not storing full A^-1 here

                // first term for tau = nt-2
                auto f = hfm.F(nt-1, phi, species, true);
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

            /// Calculate force using variant 2 algorithm for DIA discretization.
            CDVector forceVariant2(const HubbardFermiMatrixDia &hfm,
                                   const CDVector &phi) {
                const auto nx = hfm.nx();
                const auto nt = getNt(phi, nx);

                // invert Q
                CDMatrix QInv{hfm.Q(phi)};
                auto ipiv = std::make_unique<int[]>(QInv.rows());
                invert(QInv, ipiv);

                // calculate force
                CDVector force(QInv.rows());
                decltype(hfm.Tplus(0ul, phi)) T;  // sparse or dense matrix
                for (std::size_t tau = 0; tau < nt; ++tau) {
                    hfm.Tplus(T, loopIdx(tau+1, nt), phi);
                    spacevec(force, tau, nx) = 1.i*blaze::diagonal(T*spacemat(QInv, tau, loopIdx(tau+1, nt), nx));
                    hfm.Tminus(T, tau, phi);
                    spacevec(force, tau, nx) -= 1.i*blaze::diagonal(spacemat(QInv, loopIdx(tau+1, nt), tau, nx)*T);
                }

                return force;
            }

            /// Calculate force using variant 2 algorithm for EXP discretization.
            CDVector forceVariant2(const HubbardFermiMatrixExp &hfm,
                                   const CDVector &phi) {
                const auto nx = hfm.nx();
                const auto nt = getNt(phi, nx);

                // invert Q
                CDMatrix QInv{hfm.Q(phi)};
                auto ipiv = std::make_unique<int[]>(QInv.rows());
                invert(QInv, ipiv);

                // calculate force
                CDVector force(QInv.rows());
                decltype(hfm.Tplus(0ul, phi)) T;  // sparse or dense matrix
                for (std::size_t tau = 0; tau < nt; ++tau) {
                    hfm.Tplus(T, loopIdx(tau+1, nt), phi);
                    spacevec(force, tau, nx) = 1.i*blaze::diagonal(spacemat(QInv, tau, loopIdx(tau+1, nt), nx)*T);
                    hfm.Tminus(T, tau, phi);
                    spacevec(force, tau, nx) -= 1.i*blaze::diagonal(T*spacemat(QInv, loopIdx(tau+1, nt), tau, nx));
                }

                return force;
            }

        }  // anonymous namespace


        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -toFirstLogBranch(logdetM(_hfm, phi, Species::PARTICLE)
                                     + logdetM(_hfm, phi, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return -1.i*(forceVariant1Part(_hfm, phi, _kp, Species::PARTICLE)
                         - forceVariant1Part(_hfm, phi, _kh, Species::HOLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            const CDVector aux = -1.i*phi;
            return -toFirstLogBranch(logdetM(_hfm, aux, Species::PARTICLE)
                                     + logdetM(_hfm, aux, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            const CDVector aux = -1.i*phi;
            return (forceVariant1Part(_hfm, aux, _kh, Species::HOLE)
                    - forceVariant1Part(_hfm, aux, _kp, Species::PARTICLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return forceVariant2(_hfm, phi);
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, -1.i*phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            return -1.i*forceVariant2(_hfm, -1.i*phi);
        }


        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -toFirstLogBranch(logdetM(_hfm, phi, Species::PARTICLE)
                                     + logdetM(_hfm, phi, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return -1.i*(forceVariant1Part(_hfm, phi, _kp, Species::PARTICLE)
                         - forceVariant1Part(_hfm, phi, _kh, Species::HOLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            const CDVector aux = -1.i*phi;
            return -toFirstLogBranch(logdetM(_hfm, aux, Species::PARTICLE)
                                     + logdetM(_hfm, aux, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            const CDVector aux = -1.i*phi;
            return (forceVariant1Part(_hfm, aux, _kh, Species::HOLE)
                    - forceVariant1Part(_hfm, aux, _kp, Species::PARTICLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return forceVariant2(_hfm, phi);
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, -1.i*phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            return -1.i*forceVariant2(_hfm, -1.i*phi);
        }

        // instantiate all the templates we need right here
        template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::ONE, HFABasis::SPIN>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAVariant::TWO, HFABasis::SPIN>;

        template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::ONE, HFABasis::SPIN>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAVariant::TWO, HFABasis::SPIN>;

    } // namespace action
}  // namespace isle
