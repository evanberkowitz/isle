# Find blaze and BLAS/LAPACK of specified vendor
#
#  blaze_FOUND         - True if blaze was found
#  blaze_INCLUDE_DIRS  - Include directories for blaze
#  blaze_CXX_FLAGS     - Flags for CXX compiler
#  blaze_LIBRARIES     - Libraries to link against
#  blaze_LINKER_FLAGS  - Flags for linker
#

### search for blaze itself ###
set(BLAZE "" CACHE STRING "Path to folder containing blaze headers. (Expects to find \$BLAZE/blaze/Blaze.h)")
find_path(BLAZE_INCLUDE_DIR blaze/Blaze.h PATHS ${BLAZE} NO_DEFAULT_PATH)
find_path(BLAZE_INCLUDE_DIR blaze/Blaze.h)


### search for BLAS and LAPACK ###
set(BLAS_VENDOR "Generic" CACHE STRING "Implementation of BLAS library to use")
set(PARALLEL_BLAS "FALSE" CACHE STRING "Set to TRUE if BLAS library is parallelized")

if ("${BLAS_VENDOR}" STREQUAL "All")
  message(FATAL_ERROR "BLAS vendor (BLAS_VENDOR) is set to 'All'. An unknown implementation cannot be used because its parallelization might interfere with blaze.")

elseif ("${BLAS_VENDOR}" STREQUAL "Generic")
  message(STATUS "Using generic BLAS/LAPACK. Might be slow!")
  if (NOT ${PARALLEL_BLAS})
    message(WARNING "Assuming BLAS is not parallel. This cannot be verified for the generic library.")
  endif ()

elseif ("${BLAS_VENDOR}" MATCHES "^Intel.*")
  set(PARALLEL_BLAS "TRUE")
  message(STATUS "Using Intel MKL for BLAS/LAPACK (${BLAS_VENDOR})")

else ()
  message(FATAL_ERROR "BLAS vendor (BLAS_VENDOR) is not set. An unknown implementation cannot be used because its parallelization might interfere with blaze.")

endif ()

if (${PARALLEL_BLAS})
  set(BLAZE_CXX_FLAGS "-DBLAZE_BLAS_MODE=1;-DBLAS_IS_PARALLEL=1")
else ()
  set(BLAZE_CXX_FLAGS "-DBLAZE_BLAS_MODE=1;-DBLAS_IS_PARALLEL=0")
endif ()

# link against BLAS and LAPACK and tell blaze about it
set(ENV{BLA_VENDOR} "${BLAS_VENDOR}")
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(blaze_LIBRARIES ${LAPACK_LIBRARIES})
set(blaze_LINKER_FLAGS ${LAPACK_LINKER_FLAGS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blaze DEFAULT_MSG BLAZE_INCLUDE_DIR BLAZE_CXX_FLAGS)
mark_as_advanced(BLAZE_INCLUDE_DIR)

set(blaze_INCLUDE_DIRS ${BLAZE_INCLUDE_DIR})
