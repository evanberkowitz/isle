#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH
#include<cstdio>

#include <cuda_runtime_api.h>

#define CHECK_CU_ERR(t_err) \
    do { \
        auto err = (t_err); \
        if (err != cudaSuccess){ \
            std::puts(cudaGetErrorString(err)); \
            std::puts(cudaGetErrorName(err)); \
            std::abort(); \
        } \
    } while(false)

template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

#endif // CUDA_HELPER_CUH
