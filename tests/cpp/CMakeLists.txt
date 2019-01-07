# C++ unit tests for isle
#

set(TEST_SOURCE
  ${ISLE_CPP_SOURCE}
  main.cpp
  )

set(TEST_EXE "isle_cpp_test")
add_executable(${TEST_EXE} ${TEST_SOURCE})

# set standard
set_target_properties(${TEST_EXE} PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)

# warnings
target_compile_options(${TEST_EXE} PUBLIC -Wall -Wextra -pedantic)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(${TEST_EXE} PUBLIC -Wno-c++98-compat)
endif ()

# use a system include because we get all the warnings from the library compilation already
target_include_directories(${TEST_EXE} SYSTEM PUBLIC ${SOURCE_DIR})

# Pybind11
set_target_properties(${TEST_EXE} PROPERTIES PREFIX "" SUFFIX "${PYBIND11_LIB_SUFFIX}")
target_include_directories(${TEST_EXE} SYSTEM PUBLIC "${PYBIND11_INCLUDE_DIRS}")
target_compile_options(${TEST_EXE} PUBLIC ${PYBIND11_CXX_FLAGS})
target_link_libraries(${TEST_EXE} ${PYBIND11_LIBRARIES})
target_linker_flags(${TEST_EXE} "${PYBIND11_LINKER_FLAGS}")

# blaze
target_include_directories(${TEST_EXE} SYSTEM PUBLIC ${blaze_INCLUDE_DIRS})
target_compile_options(${TEST_EXE} PUBLIC "${blaze_CXX_FLAGS}")
target_link_libraries(${TEST_EXE} ${blaze_LIBRARIES})
target_linker_flags(${TEST_EXE} "${blaze_LINKER_FLAGS}")

# OpenMP if blaze does not use it
if (NOT "${BLAZE_PARALLELISM}" STREQUAL "OMP")
  target_compile_options(${LIBNAME} PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(${LIBNAME} "${OpenMP_CXX_LIBRARIES}")
  target_linker_flags(${LIBNAME} "${OpenMP_LINKER_FLAGS}")
endif ()

# install into test directory
install(TARGETS ${TEST_EXE} DESTINATION "${TEST_DIR}/../bin")
