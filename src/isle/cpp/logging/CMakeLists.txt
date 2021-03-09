set(LIBNAME isle_logging)

add_library(${LIBNAME} STATIC logging.hpp logging.cpp)

set_target_properties(${LIBNAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${LIBNAME} PUBLIC pybind11_options)
target_link_libraries(${LIBNAME} PRIVATE project_options project_warnings)
