set(LIBNAME isle_logging)

add_library(${LIBNAME} SHARED
            logging.hpp
            logging.cpp)

generate_export_header(${LIBNAME})
set_target_properties(${LIBNAME} PROPERTIES EXPORT_NAME logging)

target_include_directories(
    ${LIBNAME}
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(${LIBNAME} PUBLIC pybind11_options)
