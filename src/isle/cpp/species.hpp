/** \file
 * \brief Enum Species.
 */

#ifndef SPECIES_HPP
#define SPECIES_HPP

namespace isle {
    /// Mark particles and holes.
    enum class Species {
        PARTICLE,  ///< 'Normal' particles.
        HOLE  ///< Anti-particlces, or holes.
    };
}

#endif  // ndef SPECIES_HPP
