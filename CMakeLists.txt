cmake_minimum_required(VERSION 3.15)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# save rpath in installed library
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

function (target_linker_flags target flags)
  if (flags)
    string(REPLACE ";" " " str_flags "${flags}")
    get_target_property(old ${target} LINK_FLAGS)
    if ("${old}" MATCHES "^.*-NOTFOUND")
      set_target_properties(${target} PROPERTIES LINK_FLAGS "${str_flags}")
    else ()
      set_target_properties(${target} PROPERTIES LINK_FLAGS "${old} ${str_flags}")
    endif ()
  endif ()
endfunction ()

# define the project
project(isle CXX)
include(cmake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_14)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

option(ENABLE_TESTING "Enable Test Builds" ON)

# look for 3rd party packages
find_package(OpenMP REQUIRED)
find_package(Pybind11 REQUIRED)
find_package(blaze 3.6 REQUIRED)

get_filename_component(SOURCE_DIR "src/isle/cpp"
  REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

if (ENABLE_TESTING)
  message(STATUS "Building tests")
add_subdirectory(tests/cpp)
endif()