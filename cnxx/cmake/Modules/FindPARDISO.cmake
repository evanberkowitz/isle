# Find PARDISO sparse system solver.
#
#  PARDISO_FOUND         - True if PARDISO was found
#  PARDISO_CXX_FLAGS     - Flags for CXX compiler
#  PARDISO_LIBRARIES     - Libraries to link against
#

set(PARDISO "NONE" CACHE STRING "Implementation of PARDISO")

if ("${PARDISO}" STREQUAL "NONE")
  set(PARDISO_CXX_FLAGS "")
  set(PARDISO_LIBRARIES "")
elseif ("${PARDISO}" STREQUAL "PARDISO")
  set(PARDISO_CXX_FLAGS "-DPARDISO")
  find_library(PARDISO_LIBRARIES pardiso)
# TODO case for MKL pardiso, define MKL_PARDISO CPP macro
else ()
  message(FATAL_ERROR "Unknown PARDISO implementation: ${PARDISO}")
endif ()

if (PARDISO_LIBRARIES)
  set(PARDISO_FOUND TRUE)
  message(STATUS "Found PARDISO: ${PARDISO_LIBRARIES}")
else ()
  set(PARDISO_FOUND FALSE)
endif ()
