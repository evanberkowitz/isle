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

        blaze::transpose(res);
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
        // 2-3,5-th arguments: matrix dimensions (all equal because matrices are square)
        // calculates LU-decomposition of A inplace: ipiv * A = L * U 
        CHECK_CUSOLVER_ERR(cusolverDnZgetrf(handle, N, N, A, N, Workspace, d_ipiv, devInfo));
        CHECK_CU_ERR(cudaStreamSynchronize(0));
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
        // 2nd argument: CUBLAS_OP_T(N) means (no) transposition of A, but one additional is needed for transfer from blaze to cublas
        // 3-4,6,9-th arguments: matrix dimensions (all equal because matrices are square)
        // solves A * x = B for x, where A has been LU-decomposed before
        CHECK_CUSOLVER_ERR(cusolverDnZgetrs(handle, trans, N, N, A, N, d_ipiv, B, N, devInfo));
        CHECK_CU_ERR(cudaStreamSynchronize(0));
        assert(*devInfo == 0);

        CHECK_CU_ERR(cudaMemcpy(b.data(), cast_cmpl(B), dim*dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        cudaFree(A); cudaFree(B); cudaFree(d_ipiv); cudaFree(devInfo);

	blaze::transpose(b);

        CHECK_CUSOLVER_ERR(cusolverDnDestroy(handle));
    }

    __global__ void ilogdet_kernel(cuDoubleComplex * d_matrix, cuDoubleComplex * out){
        // adapted from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

        extern __shared__ double sdata[];

        auto id = threadIdx.x;
        auto i = blockIdx.x*blockDim.x + threadIdx.x;

        sdata[id] = d_matrix[i];

        __synchthreads();

        for(unsigned int s = blockDim.x/2; s > 0; s>>=2){
            if (id < s){
                sdata[id] += log(sdata[id + s]);
            }
            __synchthreads();
        }

        if(id == 0){
            *out = sdata[0];
        }
    }

    std::complex<double> ilogdet_wrapper(std::complex<double> * matrix, std::size_t NX, bool & negDetP){
        cublasHandle_t handle;
        CHECK_CUBLAS_ERR(cublasCreate(&handle));

        int * ipiv;
        int * info;
        cuDoubleComplex * d_matrix;
        cuDoubleComplex * d_res;
        std::complex<double> res;
        CHECK_CU_ERR(cudaMallocManaged(&ipiv, NX*sizeof(int)));
        CHECK_CU_ERR(cudaMallocManaged(&d_res, sizeof(cuDoubleComplex)))
        CHECK_CU_ERR(cudaMallocManaged(&info, sizeof(int)));
        CHECK_CU_ERR(cudaMallocManaged(&d_matrix,NX*NX*sizeof(cuDoubleComplex)));

        CHECK_CU_ERR(cudaMemcpy(d_matrix,matrix,NX*NX*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));


        CHECK_CUBLAS_ERR(
            cublasZgetrfBatched(
                handle,
                NX,
                d_matrix,
                1,
                ipiv,
                info,
                1
            )
        );

        auto num_blocks = ceildiv((int) NX,1024);

        ilogdet_kernel<<<dim3(num_blocks,1,1),dim3(1024,1,1)>>>(d_matrix,res);

        negDetP = false;  // if true det(P) == -1, else det(P) == +1
        for(int i = 0; i < NX; ++i){
            if (ipiv[i]-1 != i) {
                negDetP = !negDetP;
            }
        }

        auto res = cast_cmpl(*d_res);

        CHECK_CU_ERR(cudaFree(ipiv));
        CHECK_CU_ERR(cudaFree(d_matrix));

        CHECK_CUBLAS_ERR(cublasDestroy(handle));

        return res;
    }
} // namespace isle
