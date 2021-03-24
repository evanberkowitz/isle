# C++ unit tests for isle
#

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.1)

FetchContent_MakeAvailable(Catch2)

set(TEST_EXE "isle_cpp_test")
add_executable(${TEST_EXE} test_main.cpp test_math.cpp
                           HubbardFermiMatrixExp/test_HubbardFermiMatrixExp.cpp)

target_link_libraries(${TEST_EXE} PRIVATE project_options project_warnings
                                          Catch2::Catch2 pybind11::embed)
target_link_libraries(${TEST_EXE} LINK_PRIVATE isle_cpp_internal)

set_target_properties(${TEST_EXE} PROPERTIES CXX_VISIBILITY_PRESET hidden)

set_target_properties(${TEST_EXE} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
