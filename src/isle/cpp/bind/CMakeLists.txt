# Python bindings for C++ module isle_cpp
#

# define the library
set(LIBNAME "isle_cpp")
add_library(${LIBNAME} MODULE  bindings.cpp
            bind_version.cpp
            math.cpp
            lattice.cpp
            hubbardFermiMatrix.cpp
            action.cpp
            integrator.cpp)

target_compile_definitions(${LIBNAME} PRIVATE -DISLE_LIBNAME=${LIBNAME})

target_link_libraries(
    ${LIBNAME}
    PRIVATE project_options
    project_warnings)

# Set the correct file extension for a Python extension library.
set_target_properties(${LIBNAME} PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")

target_link_libraries(${LIBNAME} PUBLIC pybind11_options)
target_link_libraries(${LIBNAME} LINK_PRIVATE isle_cpp_internal)

# Set symbol visibility to hidden to reduce binary size, as recommended in
# pybind11 FAQ.
set_target_properties(${LIBNAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(UNIX AND NOT APPLE)
  set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
elseif(APPLE)
  set_target_properties(${LIBNAME} PROPERTIES INSTALL_RPATH "@loader_path/../lib")
endif()
