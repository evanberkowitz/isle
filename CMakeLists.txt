cmake_minimum_required(VERSION 3.9.4)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(cns CXX)

set(SOURCE
  bindings.cpp)

add_library(cns SHARED ${SOURCE})
set_target_properties(cns PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

target_compile_options(cns PUBLIC -Wall -Wextra -pedantic)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(cns PUBLIC -Wno-c++98-compat)
endif ()

# Pybind11
find_package(Pybind11 REQUIRED)
set_target_properties(cns PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")
include_directories(SYSTEM "${PYBIND11_INCLUDE_DIRS}")

# blaze
find_package(blaze REQUIRED)
target_link_libraries(cns INTERFACE blaze::blaze)

set(BLAS_VENDOR "None" CACHE STRING "Implementation of BLAS library to use")

if ("${BLAS_VENDOR}" STREQUAL "None")
  add_definitions(-DBLAZE_BLAS_MODE=0)
  message(STATUS "Using blaze without BLAS/LAPACK")

elseif ("${BLAS_VENDOR}" STREQUAL "All")
  add_definitions(-DBLAZE_BLAS_MODE=0)
  message(WARNING "BLAS vendor (BLA_VENDOR) is set to 'All'. An unknown implementation cannot be used because its parallelization might interfere with blaze. Fall back to non BLAS mode.")

else ()
  set(BLAS_VENDOR_OK "FALSE")

  if ("${BLAS_VENDOR}" STREQUAL "Generic")
    message(STATUS "Using generic BLAS/LAPACK. Might be slow!")
    add_definitions(-DBLAZE_BLAS_MODE=1 -DBLAZE_BLAS_IS_PARALLEL=0)
    set(BLAS_VENDOR_OK "TRUE")
  endif ()

  string(REGEX MATCH "^Intel.*" IS_INTEL ${BLAS_VENDOR})
  if (IS_INTEL)
    message(STATUS "Using Intel MKL for BLAS/LAPACK (${BLAS_VENDOR})")
    add_definitions(-DBLAZE_BLAS_MODE=1 -DBLAZE_BLAS_IS_PARALLEL=1)
    set(BLAS_VENDOR_OK "TRUE")
  endif ()
  unset(IS_INTEL)
  
  if (NOT ${BLAS_VENDOR_OK})
    message(FATAL_ERROR "BLAS vendor not recognized: ${BLAS_VENDOR}. Supported values are None, All, Generic, Intel10_32, Intel10_64lp, Intel10_64lp_seq, Intel\nSee CMake documentation on FindBLAS for details.")
  endif ()

  # link against BLAS and LAPACK and tell blaze about it
  set(ENV{BLA_VENDOR} "${BLAS_VENDOR}")
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  target_link_libraries(cns PUBLIC ${LAPACK_LIBRARIES})
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")

endif ()
