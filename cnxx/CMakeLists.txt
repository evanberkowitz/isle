cmake_minimum_required(VERSION 3.9.4)

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

project(cnxx CXX)

set(SOURCE
  hubbardFermiMatrix.cpp
  hubbardGaugeAction.cpp)
add_subdirectory(bind)

add_library(cnxx SHARED ${SOURCE})

# set standard
set_target_properties(cnxx PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

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
target_linker_flags(cnxx "${PYBIND11_LINKER_FLAGS}")

# blaze
find_package(blaze REQUIRED)
include_directories(SYSTEM ${blaze_INCLUDE_DIRS})
target_compile_options(cnxx PUBLIC "${blaze_CXX_FLAGS}")
target_link_libraries(cnxx ${blaze_LIBRARIES})
target_linker_flags(cnxx "${blaze_LINKER_FLAGS}")

# OpenMP if blaze does not use it
if (NOT "${BLAZE_PARALLELISM}" STREQUAL "OMP")
  find_package(OMP REQUIRED)
  target_compile_options(cnxx PUBLIC "${OMP_CXX_FLAGS}")
  target_link_libraries(cnxx ${OMP_LIBRARIES})
  target_linker_flags(cnxx "${OMP_LINKER_FLAGS}")
endif ()

# PARDISO
find_package(PARDISO)
if (${PARDISO_FOUND})
  target_compile_options(cnxx PUBLIC ${PARDISO_CXX_FLAGS})
  target_link_libraries(cnxx ${PARDISO_LIBRARIES})
endif ()

# install library under ../modules
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/../modules" CACHE PATH
    "Directory to install the library" FORCE)
endif ()
install(TARGETS cnxx DESTINATION ".")
