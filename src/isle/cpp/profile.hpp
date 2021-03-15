/** \file
 * \brief Utilities for profiling.
 */

#ifndef ISLE_PROFILE_HPP
#define ISLE_PROFILE_HPP

#ifdef ENABLE_NVTX_PROFILE
#include <nvToolsExt.h>
#define ISLE_PROFILE_NVTX_PUSH(NAME) nvtxRangePushA(NAME)
#define ISLE_PROFILE_NVTX_POP() nvtxRangePop()
#else
#define ISLE_PROFILE_NVTX_PUSH(NAME) do{} while(false)
#define ISLE_PROFILE_NVTX_POP() do{} while(false)
#endif

#endif  // ndef LATTICE_HPP
