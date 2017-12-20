# Find Pybind11 and related parameters
#
#  PYBIND11_FOUND         - True if Pybind11 was found
#  PYBIND11_CXX_FLAGS     - Pybind11 compiler flags
#  PYBIND11_INCLUDE_DIRS  - Pybind11 include directories
#  PYBIND11_LINKER_FLAGS  - Pybind11 linker flags
#  PYBIND11_LIBRARIES     - Pybind11 libraries
#  PYBIND11_LIB_SUFFIX    - Library extension suffix for this version of python
#

set(PYTHON "python" CACHE STRING "Python 3 executable")


# Execute python-config with given arguments.
function (python_config args output)
  execute_process(COMMAND "${PYTHON}-config" ${args}
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT "${rc}" STREQUAL "0")
    message(FATAL_ERROR "Command '${PYTHON}-config ${args}' could not be executed. Return code: ${rc}")
  endif (NOT "${rc}" STREQUAL "0")
  set(${output} ${result} PARENT_SCOPE)
endfunction ()

# Split input into lists of flags and paths where all paths are
# have a certain prefix in the input (stripped from output).
function (extract_flags_paths in_list prefix out_flags out_paths)
  # make copies of input
  set(flags ${${in_list}})
  set(paths_in ${${in_list}})

  # remove / retain only needed elements
  list(FILTER flags EXCLUDE REGEX "^${prefix}")
  list(FILTER paths_in INCLUDE REGEX "^${prefix}")

  # remove prefix from paths list
  set(paths)
  foreach (path IN LISTS paths_in)
    string(REGEX REPLACE "^${prefix}" "" aux ${path})
    list(APPEND paths ${aux})
  endforeach ()

  # store output in parent scope
  set(${out_flags} ${flags} PARENT_SCOPE)
  set(${out_paths} ${paths} PARENT_SCOPE)
endfunction ()


### get includes ###
execute_process(COMMAND ${PYTHON} "-m" "pybind11" "--includes"
  RESULT_VARIABLE RC
  OUTPUT_VARIABLE RAW_INCLUDES
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT "${RC}" STREQUAL "0")
  message(FATAL_ERROR "Could not execute python to get pybind11 modules. Return code: ${RC}")
endif ()
# turn it into a list
string(REPLACE " " ";" AUX_LIST ${RAW_INCLUDES})
# split
extract_flags_paths(AUX_LIST "-I" AUX_FLAGS PYBIND11_INCLUDE_DIR)
# back to string
string(REPLACE ";" " " PYBIND11_CXX_FLAGS "${AUX_FLAGS}")
unset(AUX_FLAGS)
unset(AUX_LIST)
unset(RAW_INCLUDES)


### get linker flags and lirbaries ###
python_config("--ldflags" AUX)
# turn it into a list
string(REPLACE " " ";" AUX_LIST ${AUX})
# split
extract_flags_paths(AUX_LIST "-L" AUX_FLAGS PYBIND11_LIB_PATHS)
unset(AUX_LIST)
unset(AUX)

# hijack extract function to separate actual libraries from flags
extract_flags_paths(AUX_FLAGS "-l" AUX_LINKER_FLAGS AUX_LIBS)
string(REPLACE ";" " " PYBIND11_LINKER_FLAGS "${AUX_LINKER_FLAGS}")
unset(AUX_LINKER_FLAGS)
unset(AUX_FLAGS)

# search for all libs
set(PYBIND11_LIBRARIES)
foreach (lib IN LISTS AUX_LIBS)
  # search twice to prefer custom paths over system paths
  find_library(aux_path ${lib} PATHS ${PYBIND11_LIB_PATHS} NO_DEFAULT_PATH)
  find_library(aux_path ${lib} PATHS ${PYBIND11_LIB_PATHS})
  if (NOT aux_path)
    message(FATAL_ERROR "Did not find library ${lib}. Try to adjust PYBIND11_LIB_PATHS")
  endif ()

  list(APPEND PYBIND11_LIBRARIES ${aux_path})
  unset(aux_path CACHE)
endforeach ()
unset(AUX_LIBS)


### get library extension suffix ###
python_config("--extension-suffix" PYBIND11_LIB_SUFFIX)


# handle find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pybind11 DEFAULT_MSG
  PYBIND11_INCLUDE_DIR)
mark_as_advanced(PYBIND11_INCLUDE_DIR PYBIND11_LIBRARY)

set(PYBIND11_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR})
set(PYBIND11_INCLUDE_LIBRARIES ${PYBIND11_INCLUDE_LIBRARY})
