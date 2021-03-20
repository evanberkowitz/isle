option(USE_CUDA "Enable CUDA GPU kernels" OFF)
option(CUDA_PROFILE "Enable CUDA profiling support" OFF)

if (USE_CUDA)
  message(STATUS "Compiling with CUDA")

  enable_language(CUDA)

  target_compile_features(project_options INTERFACE cuda_std_14)
  target_compile_definitions(project_options INTERFACE -DUSE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(cuda_toolkit INTERFACE)

  add_library(cudaAllocation STATIC src/isle/cpp/allocation_overload.cpp)
  target_compile_features(cudaAllocation PRIVATE cxx_std_14)
  target_link_libraries(cudaAllocation PRIVATE CUDA::cudart)
  target_link_libraries(cudaAllocation PRIVATE project_options project_warnings)

  # TODO: works only on linux like systems!
  find_path(libcudacxx_INCLUDE_DIR "complex"
      PATHS "/opt/nvidia" "/opt/cuda"
            "/usr/include"
            "/usr/local" "/usr/local/include" "/usr/local/share"
            "/usr/share"
            "$ENV{HOME}"
            "$ENV{HOME}/.local/" "$ENV{HOME}/.local/include" "$ENV{HOME}/.local/share"
	    "../"
      PATH_SUFFIXES "libcudacxx" "libcudacxx/include" "libcudacxx/include/cuda/std"
      REQUIRED
  )
  message("-- Found libcudacxx - ${libcudacxx_INCLUDE_DIR}")
  target_link_directories(project_options INTERFACE SYSTEM "${libcudacxx_INCLUDE_DIR}")

  if (CUDA_PROFILE)
    target_link_libraries(cuda_toolkit INTERFACE CUDA::nvToolsExt)
    target_compile_definitions(cuda_toolkit INTERFACE -DENABLE_NVTX_PROFILE)
  endif()
endif ()
