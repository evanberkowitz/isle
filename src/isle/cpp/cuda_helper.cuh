#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH
#include<cstdio>

#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<cublas_v2.h>

#define CHECK_CU_ERR(t_err) \
    do { \
        auto err = (t_err); \
        if (err != cudaSuccess){ \
            std::puts(cudaGetErrorString(err)); \
            std::puts(cudaGetErrorName(err)); \
            std::abort(); \
        } \
    } while(false)

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#define CHECK_CUBLAS_ERR(t_err) \
    do { \
        auto err = (t_err); \
        if (err != CUBLAS_STATUS_SUCCESS){ \
            std::puts(_cudaGetErrorEnum(err)); \
            std::abort(); \
        } \
    } while(false)

template <typename IntType>
constexpr __host__ __device__ IntType ceildiv(IntType a, IntType b) {
  return (a + b - 1) / b;
}

#endif // CUDA_HELPER_CUH
