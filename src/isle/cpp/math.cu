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

        CDMatrix res(a.rows(),a.columns()); 
        CHECK_CU_ERR(cudaMemcpy(res.data(), cast_cmpl(C), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        cudaFree(A); cudaFree(B); cudaFree(C);

        CHECK_CUBLAS_ERR(cublasDestroy(handle));

        return res;
    }

    void lu_CDMatrix_wrapper(CDMatrix &a, std::unique_ptr<int[]> &ipiv, const std::size_t dim) {
        cusolverDnHandle_t handle;
        CHECK_CUSOLVER_ERR(cusolverDnCreate(&handle));

        const int N = static_cast<int>(dim);
	int * d_ipiv;

        cuDoubleComplex * A;
        cuDoubleComplex * Workspace;

        CHECK_CU_ERR(cudaMalloc(&A, dim*dim*sizeof(cuDoubleComplex)));
        CHECK_CU_ERR(cudaMalloc(&d_ipiv, dim*sizeof(int)));

        CHECK_CU_ERR(cudaMemcpy(A, cast_cmpl(a.data()), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

	int Lwork;
        CHECK_CUSOLVER_ERR(cusolverDnZgetrf_bufferSize(handle, N, N, A, N, &Lwork));
        CHECK_CU_ERR(cudaMalloc(&Workspace, static_cast<std::size_t>(Lwork)*sizeof(cuDoubleComplex)));

        int * devInfo;
        CHECK_CU_ERR(cudaMallocManaged(&devInfo, sizeof(int)));
        // 2nd argument: CUBLAS_OP_T(N) means (no) transposition, but one additional is needed for transfer from blaze to cublas
        // 4-6,9,11,14-th arguments: matrix dimensions (all equal because matrices are square)
        // calculates C = a * A*B + b * C, with a=1 and b=0 in our case
        CHECK_CUSOLVER_ERR(cusolverDnZgetrf(handle, N, N, A, N, Workspace, d_ipiv, devInfo));
        cudaDeviceSynchronize();
        assert(*devInfo == 0);

        CHECK_CU_ERR(cudaMemcpy(a.data(), cast_cmpl(A), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        CHECK_CU_ERR(cudaMemcpy(ipiv.get(), d_ipiv, dim*sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(A); cudaFree(Workspace); cudaFree(d_ipiv); cudaFree(devInfo);

        CHECK_CUSOLVER_ERR(cusolverDnDestroy(handle));
    }

    void inv_CDMatrix_wrapper(const CDMatrix &a, CDMatrix &b, const std::unique_ptr<int[]> &ipiv, const std::size_t dim, const bool transpose) {
        cusolverDnHandle_t handle;
        CHECK_CUSOLVER_ERR(cusolverDnCreate(&handle));

        const int N = static_cast<int>(dim);
	int * d_ipiv;

        cuDoubleComplex * A;
        cuDoubleComplex * B;

        CHECK_CU_ERR(cudaMalloc(&A, dim*dim*sizeof(cuDoubleComplex)));
        CHECK_CU_ERR(cudaMalloc(&B, dim*dim*sizeof(cuDoubleComplex)));
        CHECK_CU_ERR(cudaMalloc(&d_ipiv, dim*sizeof(int)));

	blaze::transpose(b);
        CHECK_CU_ERR(cudaMemcpy(A, cast_cmpl(a.data()), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CU_ERR(cudaMemcpy(B, cast_cmpl(b.data()), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CU_ERR(cudaMemcpy(d_ipiv, ipiv.get(), dim*sizeof(int), cudaMemcpyHostToDevice));

        cublasOperation_t trans = transpose? CUBLAS_OP_N : CUBLAS_OP_T;
        int * devInfo;
        CHECK_CU_ERR(cudaMallocManaged(&devInfo, sizeof(int)));
        // 2nd argument: CUBLAS_OP_T(N) means (no) transposition, but one additional is needed for transfer from blaze to cublas
        // 4-6,9,11,14-th arguments: matrix dimensions (all equal because matrices are square)
        // calculates C = a * A*B + b * C, with a=1 and b=0 in our case
        CHECK_CUSOLVER_ERR(cusolverDnZgetrs(handle, trans, N, N, A, N, d_ipiv, B, N, devInfo));
        cudaDeviceSynchronize();
        assert(*devInfo == 0);

        CHECK_CU_ERR(cudaMemcpy(b.data(), cast_cmpl(B), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        cudaFree(A); cudaFree(B); cudaFree(d_ipiv); cudaFree(devInfo);

	blaze::transpose(b);

        CHECK_CUSOLVER_ERR(cusolverDnDestroy(handle));
    }
} // namespace isle
