# Python bindings for C++ module isle_cpp
#

# define the library
set(LIBNAME "isle_cpp")
pybind11_add_module(
  ${LIBNAME}
  bindings.cpp
  bind_version.hpp
  bind_version.cpp
  bind_math.hpp
  bind_math.cpp
  bind_lattice.hpp
  bind_lattice.cpp
  bind_hubbardFermiMatrix.cpp
  bind_hubbardFermiMatrix.hpp
  bind_action.cpp
  bind_action.hpp
  bind_integrator.hpp
  bind_integrator.cpp)

target_compile_definitions(${LIBNAME} PRIVATE -DISLE_LIBNAME=${LIBNAME})

target_link_libraries(${LIBNAME} PRIVATE project_options project_warnings)
target_link_libraries(${LIBNAME} LINK_PRIVATE isle_cpp_internal)

# Set symbol visibility to hidden to reduce binary size, as recommended in
# pybind11 FAQ.
set_target_properties(${LIBNAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(UNIX AND NOT APPLE)
  set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
elseif(APPLE)
  set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH
                                              "@loader_path/../lib")
endif()
