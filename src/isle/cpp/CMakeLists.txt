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
  logging/logging.cpp)

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

target_link_libraries(
    ${LIBNAME}
    PRIVATE project_options
    project_warnings)

# Set the correct file extension for a Python extension library.
set_target_properties(${LIBNAME} PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")

target_link_libraries(${LIBNAME} PUBLIC pybind11_options blaze_options openmp_options)

# versioning - create version.hpp file
set(ISLE_VERSION_MAJOR "0" CACHE STRING "Major version number of Isle")
set(ISLE_VERSION_MINOR "0" CACHE STRING "Minor version number of Isle")
set(ISLE_VERSION_EXTRA "nullptr" CACHE STRING "Extra version info of Isle")
configure_file(version.hpp.in version.hpp)
target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
