cmake_minimum_required(VERSION 3.19)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# save rpath in installed library
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# define the project
project(isle CXX)
include(cmake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_14)
set_target_properties(project_options
                      PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
  set(CMAKE_INSTALL_RPATH @loader_path)
endif()

option(ENABLE_TESTING "Enable Test Builds" ON)

# look for 3rd party packages
include(cmake/pybind11.cmake)

find_package(blaze 3.6 REQUIRED)
add_library(blaze_options INTERFACE)
target_include_directories(blaze_options SYSTEM INTERFACE ${blaze_INCLUDE_DIRS})
target_compile_options(blaze_options INTERFACE "${blaze_CXX_FLAGS}")
target_link_libraries(blaze_options INTERFACE ${blaze_LIBRARIES})
target_link_options(blaze_options INTERFACE "${blaze_LINKER_FLAGS}")

include(cmake/cuda.cmake)

get_filename_component(SOURCE_DIR "src/isle/cpp" REALPATH BASE_DIR
                       "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

if(ENABLE_TESTING)
  message(STATUS "Building tests")
  add_subdirectory(tests/cpp)
endif()
