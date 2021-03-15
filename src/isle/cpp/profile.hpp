/** \file
 * \brief Utilities for profiling.
 */

#ifndef ISLE_PROFILE_HPP
#define ISLE_PROFILE_HPP

#ifdef ENABLE_NVTX_PROFILE
#include <nvToolsExt.h>
#define ISLE_PROFILE_NVTX_PUSH(NAME) nvtxRangePushA(NAME)
#define ISLE_PROFILE_NVTX_POP() nvtxRangePop()

namespace isle {
namespace profile {
struct NVTXTracer {
  NVTXTracer(const char *name) {
    ISLE_PROFILE_NVTX_PUSH(name);
  }

  ~NVTXTracer() {
    ISLE_PROFILE_NVTX_POP();
  }
};

#define ISLE_PROFILE_NVTX_RANGE(NAME) isle::profile::NVTXTracer nvtx_tracer_{NAME}
}
}

#else
#define ISLE_PROFILE_NVTX_PUSH(NAME) do{} while(false)
#define ISLE_PROFILE_NVTX_POP() do{} while(false)
#define ISLE_PROFILE_NVTX_RANGE(NAME) do{} while(false)
#endif

#endif  // ndef LATTICE_HPP
