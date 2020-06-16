# C++ module isle_cpp
#
#  ISLE_CPP_SOURCE - List of source files for isle_cpp library
#

# sources of the base library
set(SOURCE
  hubbardFermiMatrixDia.cpp
  hubbardFermiMatrixExp.cpp
  integrator.cpp
  lattice.cpp
  # unusual but logging must be linked against by the unit tests as well
  bind/logging.cpp)

# add submodules
add_subdirectory(action)

# store sources (w/o bindings) for other modules
set(libsrc)
foreach (src IN LISTS SOURCE)
  list(APPEND libsrc "${CMAKE_CURRENT_LIST_DIR}/${src}")
endforeach ()
set(ISLE_CPP_SOURCE ${libsrc} PARENT_SCOPE)
unset(libsrc)

# add bindings code
add_subdirectory(bind)

# define the library
set(LIBNAME "isle_cpp")
add_library(${LIBNAME} MODULE ${SOURCE})
target_compile_definitions(${LIBNAME} PUBLIC -DISLE_LIBNAME=${LIBNAME})

# set standard
set_target_properties(${LIBNAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result)
  if (${result})
    set_target_properties(${LIBNAME} PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION True)
    message(STATUS "Enabling link time optimization")
  else()
    message(STATUS "Cannot enable link time optimization")
  endif ()
  unset(result)
endif ()

# warnings
set(GCC_CLANG_WARNINGS -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic
 -Wold-style-cast -Wcast-align -Wunused -Woverloaded-virtual
 -Wconversion -Wnull-dereference
 -Wdouble-promotion -Wformat=2)

set(GCC_EXTRA_WARNINGS -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches
 -Wlogical-op -Wuseless-cast)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${LIBNAME} PUBLIC ${GCC_CLANG_WARNINGS} -Wno-c++98-compat)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(${LIBNAME} PUBLIC ${GCC_CLANG_WARNINGS} ${GCC_EXTRA_WARNINGS})
endif ()

if (APPLE)
  # Allow for undefined symbols in the library.
  # These should all be filled in by the Python executable.
  target_link_options(${LIBNAME} PUBLIC "-flat_namespace" "-undefined" "suppress")
endif ()

# versioning - create version.hpp file
set(ISLE_VERSION_MAJOR "0" CACHE STRING "Major version number of Isle")
set(ISLE_VERSION_MINOR "0" CACHE STRING "Minor version number of Isle")
set(ISLE_VERSION_EXTRA "nullptr" CACHE STRING "Extra version info of Isle")
configure_file(version.hpp.in version.hpp)
target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Pybind11
set_target_properties(${LIBNAME} PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")
target_include_directories(${LIBNAME} SYSTEM PUBLIC "${PYBIND11_INCLUDE_DIRS}")
target_compile_options(${LIBNAME} PUBLIC ${PYBIND11_CXX_FLAGS})
target_link_libraries(${LIBNAME} ${PYBIND11_LIBRARIES})
target_linker_flags(${LIBNAME} "${PYBIND11_LINKER_FLAGS}")

# blaze
target_include_directories(${LIBNAME} SYSTEM PUBLIC ${blaze_INCLUDE_DIRS})
target_compile_options(${LIBNAME} PUBLIC "${blaze_CXX_FLAGS}")
target_link_libraries(${LIBNAME} ${blaze_LIBRARIES})
target_linker_flags(${LIBNAME} "${blaze_LINKER_FLAGS}")

# OpenMP if blaze does not use it
if (NOT "${BLAZE_PARALLELISM}" STREQUAL "OMP")
  target_compile_options(${LIBNAME} PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(${LIBNAME} "${OpenMP_CXX_LIBRARIES}")
  target_linker_flags(${LIBNAME} "${OpenMP_LINKER_FLAGS}")
endif ()
