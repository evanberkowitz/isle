# Find Pybind11 and related parameter
#
#  PYBIND11_FOUND         - True if Pybind11 was found
#  PYBIND11_INCLUDE_DIRS  - Pybind11 include directories
#  PYBIND11_LIB_SUFFIX    - Library extension suffix for this version of python
#

set(PYTHON "python" CACHE STRING "Python 3 executable")

function (get_pybind11_includes includes)
  execute_process(COMMAND ${PYTHON} "-m" "pybind11" "--includes"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE raw_includes
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT "${rc}" STREQUAL "0")
    message(FATAL_ERROR "Could not execute python to get pybind11 modules. Return code: ${rc}")
  endif ()

  string(REPLACE " " ";" list_includes ${raw_includes})
  
  # remove prefix from paths list
  set(incs)
  foreach (inc IN LISTS list_includes)
    string(REGEX REPLACE "^-I" "" aux ${inc})
    list(APPEND incs ${aux})
  endforeach ()

  # store output in parent scope
  set(${includes} ${incs} PARENT_SCOPE)
endfunction ()

get_pybind11_includes(PYBIND11_INCLUDE_DIRS)
set(PYBIND11_LIBRARIES " ")

execute_process(COMMAND "${PYTHON}-config" "--extension-suffix"
  RESULT_VARIABLE RC
  OUTPUT_VARIABLE PYBIND11_LIB_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT "${RC}" STREQUAL "0")
  message(FATAL_ERROR "Unable to execute python-config to get library extension suffix")
endif ()

# handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pybind11 DEFAULT_MSG
  PYBIND11_INCLUDE_DIRS)
