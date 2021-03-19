option(USE_CUDA "Enable CUDA GPU kernels" OFF)
option(CUDA_PROFILE "Enable CUDA profiling support" OFF)

if (USE_CUDA)
  message(STATUS "Compiling with CUDA")

  enable_language(CUDA)

  target_compile_features(project_options INTERFACE cuda_std_14)
  find_package(CUDAToolkit REQUIRED)
  add_library(cuda_toolkit INTERFACE)

  add_library(cudaAllocation STATIC src/isle/cpp/allocation_overload.cpp)
  target_compile_features(cudaAllocation PRIVATE cxx_std_14)
  target_link_libraries(cudaAllocation PRIVATE CUDA::cudart)
  target_link_libraries(cudaAllocation PRIVATE project_options project_warnings)

  message("-- Downloading external project - libcudacxx")

  ExternalProject_Add( libcudacxx
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    DOWNLOAD_COMMAND git clone --depth 1 https://github.com/NVIDIA/libcudacxx.git
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )

  ExternalProject_Get_Property(libcudacxx source_dir)
  set(libcudacxx_INCLUDE_DIR "${source_dir}/include")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${libcudacxx_INCLUDE_DIR}")

  if (CUDA_PROFILE)
    target_link_libraries(cuda_toolkit INTERFACE CUDA::nvToolsExt)
    target_compile_definitions(cuda_toolkit INTERFACE -DENABLE_NVTX_PROFILE)
  endif()
endif ()
