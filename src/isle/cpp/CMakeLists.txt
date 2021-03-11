# C++ module isle_cpp
#
# ISLE_CPP_SOURCE - List of source files for isle_cpp library
#

# sources of the base library
set(SOURCE
    hubbardFermiMatrixDia.hpp
    hubbardFermiMatrixDia.cpp
    hubbardFermiMatrixExp.hpp
    hubbardFermiMatrixExp.cpp
    integrator.hpp
    integrator.cpp
    lattice.hpp
    lattice.cpp
    action/sumAction.hpp
    action/sumAction.cpp
    action/hubbardGaugeAction.hpp
    action/hubbardGaugeAction.cpp
    action/hubbardFermiAction.hpp
    action/hubbardFermiAction.cpp)

# store sources (w/o bindings) for other modules
set(libsrc)
foreach(src IN LISTS SOURCE)
  list(APPEND libsrc "${CMAKE_CURRENT_LIST_DIR}/${src}")
endforeach()
set(ISLE_CPP_SOURCE
    ${libsrc}
    PARENT_SCOPE)
unset(libsrc)

# define the library
set(LIBNAME "isle_cpp_internal")
add_library(${LIBNAME} STATIC ${SOURCE})
set_target_properties(${LIBNAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_link_libraries(${LIBNAME} PRIVATE project_options project_warnings)
target_link_libraries(${LIBNAME} PUBLIC blaze_options openmp_options)

add_subdirectory(logging)
target_link_libraries(${LIBNAME} PUBLIC isle_logging)

if (APPLE)
  # Allow for undefined symbols in the library.
  # These should all be filled in by the Python executable.
  target_link_options(${LIBNAME} PUBLIC "-flat_namespace" "-undefined" "suppress")
endif ()

# versioning - create version.hpp file
set(ISLE_VERSION_MAJOR
    "0"
    CACHE STRING "Major version number of Isle")
set(ISLE_VERSION_MINOR
    "0"
    CACHE STRING "Minor version number of Isle")
set(ISLE_VERSION_EXTRA
    "nullptr"
    CACHE STRING "Extra version info of Isle")
configure_file(version.hpp.in version.hpp)
target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# add bindings code
add_subdirectory(bind)
