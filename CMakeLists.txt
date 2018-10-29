cmake_minimum_required(VERSION 3.9.2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# do not allow any compiler specific c++ extensions
set(CMAKE_CXX_EXTENSIONS OFF)

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

# verify required arguments are there
set(TEST_DIR "" CACHE STRING "Directory for unit tests")
if ("${TEST_DIR}" STREQUAL "")
  message(FATAL_ERROR "No test directory given")
endif ()

# define the project
project(isle CXX)

# look for 3rd party packages
find_package(Pybind11 REQUIRED)
find_package(blaze REQUIRED)
find_package(OMP REQUIRED)

get_filename_component(SOURCE_DIR "src/isle/cpp"
  REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

get_filename_component(TEST_DIR "tests/cpp"
  REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(${TEST_DIR})
