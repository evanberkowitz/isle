#include "math.cuh"

namespace isle{

    CDMatrix mult_CDMatrix_wrapper(const CDMatrix &a, const CDMatrix &b, const std::size_t dim) {
        cublasHandle_t handle;
        CHECK_CUBLAS_ERR(cublasCreate(&handle));

        const cuDoubleComplex alpha = make_cuDoubleComplex(1,0), beta = make_cuDoubleComplex(0,0);
        const int N = static_cast<int>(dim);

        cuDoubleComplex * A;
        cuDoubleComplex * B;
        cuDoubleComplex * C;

        CHECK_CU_ERR(cudaMalloc(&A, dim*dim*sizeof(cuDoubleComplex)));
        CHECK_CU_ERR(cudaMalloc(&B, dim*dim*sizeof(cuDoubleComplex)));
        CHECK_CU_ERR(cudaMalloc(&C, dim*dim*sizeof(cuDoubleComplex)));

        CHECK_CU_ERR(cudaMemcpy(A, cast_cmpl(a.data()), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CU_ERR(cudaMemcpy(B, cast_cmpl(b.data()), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        // 2nd and 3rd arguments: CUBLAS_OP_T means transposition which is needed for compatibility of blaze and cublas
        // 4-6,9,11,14-th arguments: matrix dimensions (all equal because matrices are square)
        // calculates C = a * A*B + b * C, with a=1 and b=0 in our case
        CHECK_CUBLAS_ERR(cublasZgemm3m(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, A, N, B, N, &beta, C, N));

        CDMatrix res = a; // WARNING: This is very dirty and has to be removed later.
        CHECK_CU_ERR(cudaMemcpy(res.data(), cast_cmpl(C), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        cudaFree(A); cudaFree(B); cudaFree(C);

        CHECK_CUBLAS_ERR(cublasDestroy(handle));

    return res;
  }
} // namespace isle
