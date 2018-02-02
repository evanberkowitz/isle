cmake_minimum_required(VERSION 3.9.4)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(cnxx CXX)

set(SOURCE hubbardFermiMatrix.cpp)
add_subdirectory(bind)

add_library(cnxx SHARED ${SOURCE})

# set standard
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # set it explicitly for GNU and Clang, gnu++14 does not work with complex literals...
  target_compile_options(cnxx PUBLIC --std=c++14)
else ()
  set_target_properties(cnxx PROPERTIES CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
endif ()

target_compile_options(cnxx PUBLIC -Wall -Wextra -pedantic)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(cnxx PUBLIC -Wno-c++98-compat)
endif ()

# Pybind11
find_package(Pybind11 REQUIRED)
set_target_properties(cnxx PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")
include_directories(SYSTEM "${PYBIND11_INCLUDE_DIRS}")
target_compile_options(cnxx PUBLIC ${PYBIND11_CXX_FLAGS})
target_link_libraries(cnxx ${PYBIND11_LIBRARIES})
set_target_properties(cnxx PROPERTIES LINK_FLAGS "${PYBIND11_LINKER_FLAGS}")

# blaze
find_package(blaze REQUIRED)
include_directories(SYSTEM ${blaze_INCLUDE_DIRS})
target_compile_options(cnxx PUBLIC ${blaze_CXX_FLAGS})
target_link_libraries(cnxx ${blaze_LIBRARIES})
set_target_properties(cnxx PROPERTIES LINK_FLAGS "${blaze_LINKER_FLAGS}")

# install library under ../modules
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/../modules" CACHE PATH
    "Directory to install the library" FORCE)
endif ()
install(TARGETS cnxx DESTINATION ".")
