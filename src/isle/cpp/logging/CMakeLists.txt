set(LIBNAME isle_logging)

add_library(${LIBNAME} SHARED
            logging.hpp
            logging.cpp)

target_link_libraries(${LIBNAME} PRIVATE pybind11_options)