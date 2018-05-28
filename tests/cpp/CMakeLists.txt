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

# catch
# set(CATCH_INCLUDE "" CACHE STRING "Path to catch")
# if (NOT "${CATCH_INCLUDE}" STREQUAL "")
#   message(STATUS "Using catch in ${CATCH_INCLUDE}")
#   get_filename_component(catch_dir "${CATCH_INCLUDE}"
#     REALPATH BASE_DIR "${CMAKE_SOURCE_DIR}")
#   target_include_directories(${TEST_EXE} SYSTEM PUBLIC "${catch_dir}")
#   unset(catch_dir)
# else ()
#   message(STATUS "Using catch in default path")
# endif ()

# blaze
find_package(blaze REQUIRED)
target_include_directories(${TEST_EXE} SYSTEM PUBLIC ${blaze_INCLUDE_DIRS})
target_compile_options(${TEST_EXE} PUBLIC "${blaze_CXX_FLAGS}")
target_link_libraries(${TEST_EXE} ${blaze_LIBRARIES})
target_linker_flags(${TEST_EXE} "${blaze_LINKER_FLAGS}")

# install into test directory
install(TARGETS ${TEST_EXE} DESTINATION "${TEST_DIR}/../bin")
