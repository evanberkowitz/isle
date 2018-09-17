/** \file
 * \brief Fermionic part of Hubbard action.
 */

#ifndef ACTION_HUBBARD_FERMI_ACTION_HPP
#define ACTION_HUBBARD_FERMI_ACTION_HPP

#include "hubbardFermiActionDia.hpp"
#include "hubbardFermiActionExp.hpp"

namespace isle {
    namespace action {
        /// Indicate kind of hopping term for fermionic actions.
        enum class Hopping { DIAG, EXP };

        /// Construct HubbardFermiAction for given kind of hopping term.
        Action *makeHubbardFermiAction(const HubbardFermiMatrix &hfm,
                                       std::int8_t alpha=1,
                                       Hopping hopping=Hopping::DIAG,
                                       HubbardFermiActionDia::Variant variant=
                                       HubbardFermiActionDia::Variant::ONE);

        /// Construct HubbardFermiAction for given kind of hopping term.
        Action *makeHubbardFermiAction(const SparseMatrix<double> &kappaTilde,
                                       double muTilde,
                                       std::int8_t sigmaKappa,
                                       std::int8_t alpha=1,
                                       Hopping hopping=Hopping::DIAG,
                                       HubbardFermiActionDia::Variant variant=
                                       HubbardFermiActionDia::Variant::ONE);

        /// Construct HubbardFermiAction for given kind of hopping term.
        Action *makeHubbardFermiAction(const Lattice &lat,
                                       double beta,
                                       double muTilde,
                                       std::int8_t sigmaKappa,
                                       std::int8_t alpha=1,
                                       Hopping hopping=Hopping::DIAG,
                                       HubbardFermiActionDia::Variant variant=
                                       HubbardFermiActionDia::Variant::ONE);

    }  // namespace action
}  // namespace isle


#endif  // ndef ACTION_HUBBARD_FERMI_ACTION_HPP
