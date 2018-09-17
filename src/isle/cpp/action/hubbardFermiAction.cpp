#include "hubbardFermiAction.hpp"

namespace isle {
    namespace action {
        Action *makeHubbardFermiAction(const HubbardFermiMatrix &hfm,
                                       const std::int8_t alpha,
                                       const Hopping hopping,
                                       const HubbardFermiActionDia::Variant variant) {
            if (hopping == Hopping::DIAG)
                return new HubbardFermiActionDia(hfm, alpha, variant);
            else
                return new HubbardFermiActionExp(hfm, alpha);
        }

        Action *makeHubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                                       const double muTilde,
                                       const std::int8_t sigmaKappa,
                                       const std::int8_t alpha,
                                       const Hopping hopping,
                                       const HubbardFermiActionDia::Variant variant) {
            if (hopping == Hopping::DIAG)
                return new HubbardFermiActionDia(kappaTilde, muTilde, sigmaKappa,
                                                 alpha, variant);
            else
                return new HubbardFermiActionExp(kappaTilde, muTilde, sigmaKappa,
                                                 alpha);
        }

        Action *makeHubbardFermiAction(const Lattice &lat,
                                       const double beta,
                                       const double muTilde,
                                       const std::int8_t sigmaKappa,
                                       const std::int8_t alpha,
                                       const Hopping hopping,
                                       const HubbardFermiActionDia::Variant variant) {
            if (hopping == Hopping::DIAG)
                return new HubbardFermiActionDia(lat, beta, muTilde, sigmaKappa,
                                                 alpha, variant);
            else
                return new HubbardFermiActionExp(lat, beta, muTilde, sigmaKappa,
                                                 alpha);
        }

    }  // namespace action
}  // namespace isle
