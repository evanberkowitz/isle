set(PYTHON_EXECUTABLE
    "python3"
    CACHE STRING "Python 3 executable")

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-m" "pybind11" "--cmakedir"
  RESULT_VARIABLE RC
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT "${RC}" STREQUAL "0")
  message(
    FATAL_ERROR
      "Could not execute python to get pybind11 configuration. Return code: ${RC}"
  )
endif()

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
