cmake_minimum_required(VERSION 3.15)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# save rpath in installed library
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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

if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
  set(CMAKE_INSTALL_RPATH @loader_path)
endif()

include(GenerateExportHeader)

option(ENABLE_TESTING "Enable Test Builds" ON)

# look for 3rd party packages
find_package(OpenMP REQUIRED)
add_library(openmp_options INTERFACE)
# OpenMP if blaze does not use it
if (NOT "${BLAZE_PARALLELISM}" STREQUAL "OMP")
  target_compile_options(openmp_options INTERFACE "${OpenMP_CXX_FLAGS}")
  target_link_libraries(openmp_options INTERFACE "${OpenMP_CXX_LIBRARIES}")
  target_link_options(openmp_options INTERFACE "${OpenMP_LINKER_FLAGS}")
endif ()

find_package(Pybind11 REQUIRED)
add_library(pybind11_options INTERFACE)
target_include_directories(pybind11_options SYSTEM INTERFACE "${PYBIND11_INCLUDE_DIRS}")
target_compile_options(pybind11_options INTERFACE ${PYBIND11_CXX_FLAGS})
target_link_libraries(pybind11_options INTERFACE ${PYBIND11_LIBRARIES})
target_link_options(pybind11_options INTERFACE "${PYBIND11_LINKER_FLAGS}")

find_package(blaze 3.6 REQUIRED)
add_library(blaze_options INTERFACE)
target_include_directories(blaze_options SYSTEM INTERFACE ${blaze_INCLUDE_DIRS})
target_compile_options(blaze_options INTERFACE "${blaze_CXX_FLAGS}")
target_link_libraries(blaze_options INTERFACE ${blaze_LIBRARIES})
target_link_options(blaze_options INTERFACE "${blaze_LINKER_FLAGS}")

get_filename_component(SOURCE_DIR "src/isle/cpp"
  REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

if (ENABLE_TESTING)
  message(STATUS "Building tests")
add_subdirectory(tests/cpp)
endif()