#ifndef __HUBBARD_FORCE__
#define __HUBBARD_FORCE__

#include "hubbardFermiAction.hpp"

#include "../core.hpp"
#include "../profile.hpp"
#include "../logging/logging.hpp"

namespace isle {
    namespace action {

      template <typename HFM, typename KMatrix>
	CDVector forceDirectSinglePart(const HFM &hfm, const CDVector &phi,
	    const KMatrix &k, const Species species) ;
      CDVector forceDirectSquare(const HubbardFermiMatrixDia &hfm,
	  const CDVector &phi) ;
      CDVector forceDirectSquare(const HubbardFermiMatrixExp &hfm,
	  const CDVector &phi) ;
    }
}

#endif
