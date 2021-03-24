cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# save rpath in installed library
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# define the project
project(isle CXX C)
include(ExternalProject)
include(cmake/StandardProjectSettings.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)
set_target_properties(project_options
                      PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# enable cache system
include(cmake/Cache.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# include cuda if wanted
include(cmake/cuda.cmake)

if(NOT APPLE)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
else()
  set(CMAKE_INSTALL_RPATH @loader_path)
endif()

option(ENABLE_TESTING "Enable Test Builds" ON)

# look for 3rd party packages
include(cmake/pybind11.cmake)

find_package(blaze 3.8 REQUIRED)
add_library(blaze_options INTERFACE)
target_include_directories(blaze_options SYSTEM INTERFACE ${blaze_INCLUDE_DIRS})
target_compile_options(blaze_options INTERFACE "${blaze_CXX_FLAGS}")
target_link_libraries(blaze_options INTERFACE ${blaze_LIBRARIES})
target_link_options(blaze_options INTERFACE "${blaze_LINKER_FLAGS}")
if(USE_CUDA)
    target_compile_definitions(
        blaze_options INTERFACE
            "-DBLAZE_USE_VECTORIZATION=0"
            "-DBLAZE_DEFAULT_PADDING_FLAG=blaze::unpadded"
            "-DBLAZE_DEFAULT_ALIGNMENT_FLAG=blaze::unaligned"
    )
endif()

find_package(Threads REQUIRED)
target_link_libraries(project_options INTERFACE Threads::Threads)

get_filename_component(SOURCE_DIR "src/isle/cpp" REALPATH BASE_DIR
                       "${CMAKE_SOURCE_DIR}")
add_subdirectory(${SOURCE_DIR})

if(ENABLE_TESTING)
  message(STATUS "Building tests")
  add_subdirectory(tests/cpp)
endif()
