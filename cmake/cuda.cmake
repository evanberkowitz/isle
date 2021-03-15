option(USE_CUDA "Enable CUDA GPU kernels" OFF)
option(CUDA_PROFILE "Enable CUDA profiling support" OFF)

if (USE_CUDA)
  message(STATUS "Compiling with CUDA")

  enable_language(CUDA)

  target_compile_features(project_options INTERFACE cuda_std_14)

  find_package(CUDAToolkit REQUIRED)

  add_library(cuda_toolkit INTERFACE)

  if (CUDA_PROFILE)
    target_link_libraries(cuda_toolkit INTERFACE CUDA::nvToolsExt)
    target_compile_definitions(cuda_toolkit INTERFACE -DENABLE_NVTX_PROFILE)
  endif()
endif ()