cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
option(USE_NN "Enable gardient calculation with NN" ON)
list(APPEND CMAKE_PREFIX_PATH "/p/project/cjjsc37/john/isle_ML_Approx_env/libtorch")
if (USE_NN)
    message(STATUS "Using NNgHMC")
    find_package(Torch REQUIRED)
    target_link_libraries(project_options INTERFACE "${TORCH_LIBRARIES}")
    
endif ()