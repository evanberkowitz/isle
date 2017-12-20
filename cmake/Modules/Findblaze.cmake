# Find blaze and BLAS/LAPACK of specified vendor
#
#  blaze_FOUND         - True if blaze was found
#  blaze_INCLUDE_DIRS  - Include directories for blaze
#  blaze_CXX_FLAGS     - Flags for CXX compiler
#  blaze_LIBRARIES     - Libraries to link against
#  blaze_LINKER_FLAGS  - Flags for linker
#

# search for blaze itself
set(BLAZE "" CACHE STRING "Path to folder containing blaze headers. (Expects to find \$BLAZE/blaze/Blaze.h)")
find_path(BLAZE_INCLUDE_DIR blaze/Blaze.h PATHS ${BLAZE} NO_DEFAULT_PATH)
find_path(BLAZE_INCLUDE_DIR blaze/Blaze.h)


# search for BLAS and LAPACK
set(BLAS_VENDOR "None" CACHE STRING "Implementation of BLAS library to use")

set(BLAS_OK "FALSE")
if ("${BLAS_VENDOR}" STREQUAL "None")
  set(blaze_CXX_FLAGS "-DBLAZE_BLAS_MODE=0")
  set(blaze_LIBRARIES "")
  set(blaze_LINKER_FLAGS "")
  set(BLAS_OK "TRUE")
  message(STATUS "Using blaze without BLAS/LAPACK")

elseif ("${BLAS_VENDOR}" STREQUAL "All")
  set(blaze_CXX_FLAGS "-DBLAZE_BLAS_MODE=0")
  set(blaze_LIBRARIES "")
  set(blaze_LINKER_FLAGS "")
  set(BLAS_OK "TRUE")
  message(WARNING "BLAS vendor (BLA_VENDOR) is set to 'All'. An unknown implementation cannot be used because its parallelization might interfere with blaze. Fall back to non BLAS mode.")

else ()
  # check that BLAS / LAPACK vendor is ok
  if ("${BLAS_VENDOR}" STREQUAL "Generic")
    set(blaze_CXX_FLAGS "-DBLAZE_BLAS_MODE=1 -DBLAZE_BLAS_IS_PARALLEL=0")
    set(BLAS_OK "TRUE")
    message(STATUS "Using generic BLAS/LAPACK. Might be slow!")
  endif ()

  string(REGEX MATCH "^Intel.*" IS_INTEL ${BLAS_VENDOR})
  if (IS_INTEL)
    set(blaze_CXX_FLAGS "-DBLAZE_BLAS_MODE=1 -DBLAZE_BLAS_IS_PARALLEL=1")
    set(BLAS_OK "TRUE")
    message(STATUS "Using Intel MKL for BLAS/LAPACK (${BLAS_VENDOR})")
  endif ()
  unset(IS_INTEL)
  
  if (NOT ${BLAS_OK})
    message(SEND_ERROR "BLAS vendor not recognized: ${BLAS_VENDOR}. Supported values are None, All, Generic, Intel10_32, Intel10_64lp, Intel10_64lp_seq, Intel\nSee CMake documentation on FindBLAS for details.")
  endif ()

  # link against BLAS and LAPACK and tell blaze about it
  set(ENV{BLA_VENDOR} "${BLAS_VENDOR}")
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  set(blaze_LIBRARIES ${LAPACK_LIBRARIES})
  set(blaze_LINKER_FLAGS ${LAPACK_LINKER_FLAGS})

endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(blaze DEFAULT_MSG BLAZE_INCLUDE_DIR BLAS_OK)
mark_as_advanced(BLAZE_INCLUDE_DIR)
unset(BLAS_OK)

set(blaze_INCLUDE_DIRS ${BLAZE_INCLUDE_DIR})
