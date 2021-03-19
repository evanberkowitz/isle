#include "hubbardFermiAction.hpp"

#include "../core.hpp"
#include "../profile.hpp"
#include "../logging/logging.hpp"

#include "hubbardForces.hpp"

namespace isle {
    namespace action {
      constexpr std::complex<double> I{0.0, 1.0};

        namespace _internal {
            template <> bool _holeShortcutPossible<HFABasis::PARTICLE_HOLE>(
                const SparseMatrix<double> &hopping,
                const double muTilde,
                const std::int8_t sigmaKappa) {

                auto log = getLogger("HubbardFermiAction");

                if (!isBipartite(hopping)) {
                    log.info("Not using shortcut for hole determinant, "
                             "lattice is not bipartite.");
                    return false;
                }
                else if (muTilde != 0.0) {  // must be exactlz zero
                    log.info("Not using shortcut for hole determinant, "
                             "chemical potential is not zero.");
                    return false;
                }
                else if (sigmaKappa != +1) {
                    log.info("Not using shortcut for hole determinant, "
                             "sigmaKappa is not +1");
                    // If other params do not allow for the shortcut,
                    // we hit an earlier return.
                    log.info("The other parameters allow using the shortcut, "
                             "consider setting sigmaKappa to +1 or explicitly "
                             "forbidding the use of the shortcut.");
                    return false;
                }

                log.info("Using shortcut to calculate hole determinant from "
                         "particle determinant.");
                return true;
            }

            template <> bool _holeShortcutPossible<HFABasis::SPIN>(
                const SparseMatrix<double> &UNUSED(hopping),
                const double UNUSED(muTilde),
                const std::int8_t UNUSED(sigmaKappa)) {

                getLogger("HubbardFermiAction").info(
                    "Not using shortcut for hole determinant, "
                    "spin basis is not supported.");
                return false;
            }
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            if (_shortcutForHoles) {
                const auto ldp = logdetM(_hfm, phi, Species::PARTICLE);
                return -toFirstLogBranch(ldp + std::conj(ldp));
            }
            else {
                return -toFirstLogBranch(logdetM(_hfm, phi, Species::PARTICLE)
                                         + logdetM(_hfm, phi, Species::HOLE));
            }
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            if (_shortcutForHoles) {
                const auto fp = forceDirectSinglePart(_hfm, phi, _kp, Species::PARTICLE);
                return -I*(fp - blaze::conj(fp));
            }
            else {
                return -I*(forceDirectSinglePart(_hfm, phi, _kp, Species::PARTICLE)
                         - forceDirectSinglePart(_hfm, phi, _kh, Species::HOLE));
            }
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            const CDVector aux = -I*phi;
            return -toFirstLogBranch(logdetM(_hfm, aux, Species::PARTICLE)
                                     + logdetM(_hfm, aux, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            const CDVector aux = -I*phi;
            return (forceDirectSinglePart(_hfm, aux, _kh, Species::HOLE)
                    - forceDirectSinglePart(_hfm, aux, _kp, Species::PARTICLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return forceDirectSquare(_hfm, phi);
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, -I*phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            return -I*forceDirectSquare(_hfm, -I*phi);
        }


        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            if (_shortcutForHoles) {
                const auto ldp = logdetM(_hfm, phi, Species::PARTICLE);
                return -toFirstLogBranch(ldp + std::conj(ldp));
            }
            else {
                return -toFirstLogBranch(logdetM(_hfm, phi, Species::PARTICLE)
                                         + logdetM(_hfm, phi, Species::HOLE));
            }
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            if (_shortcutForHoles) {
                const auto fp = forceDirectSinglePart(_hfm, phi, _kp, Species::PARTICLE);
                return -I*(fp - blaze::conj(fp));
            }
            else {
                return -I*(forceDirectSinglePart(_hfm, phi, _kp, Species::PARTICLE)
                         - forceDirectSinglePart(_hfm, phi, _kh, Species::HOLE));
            }
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            const CDVector aux = -I*phi;
            return -toFirstLogBranch(logdetM(_hfm, aux, Species::PARTICLE)
                                     + logdetM(_hfm, aux, Species::HOLE));
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            const CDVector aux = -I*phi;
            return (forceDirectSinglePart(_hfm, aux, _kh, Species::HOLE)
                    - forceDirectSinglePart(_hfm, aux, _kp, Species::PARTICLE));
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>::force(
            const CDVector &phi) const {

            return forceDirectSquare(_hfm, phi);
        }

        template <> std::complex<double>
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::eval(
            const CDVector &phi) const {

            return -logdetQ(_hfm, -I*phi);
        }
        template <> CDVector
        HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>::force(
            const CDVector &phi) const {

            return -I*forceDirectSquare(_hfm, -I*phi);
        }

        // instantiate all the templates we need right here
        template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::DIA, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>;

        template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SINGLE, HFABasis::SPIN>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::PARTICLE_HOLE>;
        template class HubbardFermiAction<HFAHopping::EXP, HFAAlgorithm::DIRECT_SQUARE, HFABasis::SPIN>;

    } // namespace action
}  // namespace isle
