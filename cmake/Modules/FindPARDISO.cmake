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

elseif ("${PARDISO}" STREQUAL "STANDALONE")
  set(PARDISO_CXX_FLAGS "-DPARDISO_STANDALONE")
  find_library(PARDISO_LIBRARIES pardiso)
  set(PARDISO_LIBRARIES "${PARDISO_LIBRARIES};-lgfortran")

elseif ("${PARDISO}" STREQUAL "MKL")
  if (NOT "${BLAS_VENDOR}" MATCHES "^Intel.*")
    message(FATAL_ERROR "Cannot use MKL PARDISO when linking against non MKL BLAS")
  endif ()
  set(PARDISO_CXX_FLAGS "-DPARDISO_MKL")

  set(ENV{BLA_VENDOR} "${BLAS_VENDOR}")
  find_package(BLAS REQUIRED)
  set(PARDISO_LIBRARIES ${BLAS_LIBRARIES} -lmkl_avx2 -lmkl_def)

else ()
  message(FATAL_ERROR "Unknown PARDISO implementation: ${PARDISO}")
endif ()

if (PARDISO_LIBRARIES)
  set(PARDISO_FOUND TRUE)
  message(STATUS "Found PARDISO: ${PARDISO_LIBRARIES}")
else ()
  set(PARDISO_FOUND FALSE)
endif ()
