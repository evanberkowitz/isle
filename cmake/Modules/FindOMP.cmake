# Find OpenMP. Wrapper around FindOpenMP that also picks a library to link against.
#
#  OMP_FOUND         - True if OpenMP was found
#  OMP_CXX_FLAGS     - Flags for CXX compiler
#  OMP_LIBRARIES     - Libraries to link against
#  OMP_LINKER_FLAGS  - Flags for linker
#

find_package(OpenMP REQUIRED)
set(OMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
set(OMP_LINKER_FLAGS "${OpenMP_LINKER_FLAGS}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(omp_lib omp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(omp_lib gomp)
else ()
  message(FATAL_ERROR "Unable to link against OpenMP for compiler ${CMAKE_CXX_COMPILER_ID}")
endif ()

string(REPLACE ":" ";" LD_LIBRARY_PATH "$ENV{LD_LIBRARY_PATH}")
find_library(OMP_LIBRARIES ${omp_lib} PATHS ${LD_LIBRARY_PATH})
unset(LD_LIBRARY_PATH)
unset(omp_lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OMP DEFAULT_MSG OMP_CXX_FLAGS OMP_LIBRARIES)
