# C++ unit tests for isle
#

set(TEST_SOURCE
  # TODO activate
  # ${ISLE_CPP_SOURCE}
  main.cpp
  )

set(TEST_EXE "isle_cpp_test")
add_executable(${TEST_EXE} ${TEST_SOURCE})

target_link_libraries(${TEST_EXE} PRIVATE
                      project_options
                      project_warnings)

# install into test directory
install(TARGETS ${TEST_EXE} DESTINATION "${TEST_DIR}/../bin")
