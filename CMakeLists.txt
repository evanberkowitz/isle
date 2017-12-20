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

# link against BLAS and LAPACK and tell blaze about it
find_package(BLAS REQUIRED)
find_package(LAPACK)
target_link_libraries(cns PUBLIC ${LAPACK_LIBRARIES})
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
add_definitions(-DBLAZE_BLAS_MODE=1)

# check which BLAS version we have
string(COMPARE EQUAL ${BLA_VENDOR} "All" IS_ALL)
if (IS_ALL)
  message(FATAL_ERROR "BLAS vendor (BLA_VENDOR) is not set or set to 'All'. An unknown implementation cannot be used because its parallelization might interfere with blaze.")
endif()
unset(IS_ALL)

if ("${BLA_VENDOR}" STREQUAL "Generic")
  message(STATUS "Using generic BLAS/LAPACK. Might be slow!")
  add_definitions(-DBLAZE_BLAS_IS_PARALLEL=0)
endif()

string(REGEX MATCH "^Intel.*" IS_INTEL ${BLA_VENDOR})
if (IS_INTEL)
  message(STATUS "Using Intel MKL for BLAS/LAPACK")
  add_definitions(-DBLAZE_BLAS_IS_PARALLEL=1)
endif()
unset(IS_INTEL)
