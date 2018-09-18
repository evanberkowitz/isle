# C++ module isle_cpp
#
#  ISLE_CPP_SOURCE - List of source files for isle_cpp library
#

# sources of the base library
set(SOURCE
  hubbardFermiMatrix.cpp
  integrator.cpp
  lattice.cpp)

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
set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

# warnings
target_compile_options(${LIBNAME} PUBLIC -Wall -Wextra -pedantic)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${LIBNAME} PUBLIC -Wno-c++98-compat)
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
  target_compile_options(${LIBNAME} PUBLIC "${OMP_CXX_FLAGS}")
  target_link_libraries(${LIBNAME} ${OMP_LIBRARIES})
  target_linker_flags(${LIBNAME} "${OMP_LINKER_FLAGS}")
endif ()

# PARDISO
if (${PARDISO_FOUND})
  target_compile_options(${LIBNAME} PUBLIC ${PARDISO_CXX_FLAGS})
  target_link_libraries(${LIBNAME} ${PARDISO_LIBRARIES})
endif ()
