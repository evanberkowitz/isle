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


#endif // CUDA_HELPER_CUH
