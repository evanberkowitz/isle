set(LIBNAME isle_logging)

add_library(${LIBNAME} SHARED
            logging.hpp
            logging.cpp)

target_link_libraries(${LIBNAME} PUBLIC pybind11_options)
