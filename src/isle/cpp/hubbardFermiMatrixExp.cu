#include <complex>
#include <iostream>
#include <cuComplex.h>

#include "array_lookup.cuh"
#include "cuda_cast.cuh"
#include "cuda_helper.cuh"
#include "math.cuh"

#include "species.hpp"
#include "profile.hpp"

namespace isle{

__global__ void F_kernel(cuDoubleComplex * f_begin,
                const std::size_t tp,
                const cuDoubleComplex * phi_begin,
                const double * expKappa_begin,
                const cuDoubleComplex sign,
                const std::size_t Nx, const std::size_t Nt, const bool inv) {

    auto tm1 = tp == 0 ? Nt-1 : tp-1; // apply periodic boundary or the used t-1
    auto x_row = threadIdx.y + blockIdx.y * blockDim.y;
    auto x_col = threadIdx.x + blockIdx.x * blockDim.x;

    if (x_row >= Nx || x_col >= Nx){return;}
    assert(tp < Nt);

    if(inv){ // if the inverse has to be computed phi is expanded as column vector
        f_begin[ lookup_matrix(x_row,x_col,Nx) ] = expKappa_begin[ lookup_matrix(x_row,x_col,Nx) ] * cexp(sign * phi_begin[ lookup_vector(tm1,x_row,Nx) ] );
    } else { // if non inverse has to ve computed phi is expanded as row vector
        f_begin[lookup_matrix(x_row,x_col,Nx)] = expKappa_begin[lookup_matrix(x_row,x_col,Nx)] * cexp(sign * phi_begin[lookup_vector(tm1,x_col,Nx)]);
    }
}

// HubbardFermiMatrixExp::F wrapper
void F_wrapper(std::complex<double> * f,
        const std::size_t tp,
        const std::complex<double> * phi,
        const double * expKappa,
        const std::size_t NX, const std::size_t NT,
        const Species species, const bool inv) {

    ISLE_PROFILE_NVTX_RANGE(species == Species::PARTICLE
        ? "F_wrapper(particle)"
        : "F_wrapper(hole)");

    // create the device pointer
    cuDoubleComplex * d_f;
    cuDoubleComplex * d_phi;
    double * d_expKappa;

    CHECK_CU_ERR(cudaMallocManaged(&d_f, NX*NX*sizeof(cuDoubleComplex)));
    CHECK_CU_ERR(cudaMallocManaged(&d_phi, NT*NX*sizeof(cuDoubleComplex)));
    CHECK_CU_ERR(cudaMallocManaged(&d_expKappa,NX*NX*sizeof(double)));

    CHECK_CU_ERR(cudaMemcpy(d_phi, cast_cmpl(phi), NT*NX*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CU_ERR(cudaMemcpy(d_expKappa, expKappa, NX*NX*sizeof(double), cudaMemcpyHostToDevice));

    // the sign in the exponential of phi
    auto const sign = ((species == Species::PARTICLE && !inv)
                        || (species == Species::HOLE && inv))
        ? cuDoubleComplex{0.0, +1.0}
        : cuDoubleComplex{0.0, -1.0};

    auto num_blocks = ceildiv((int) NX,32);

    //ToDo: warning: conversion to ‘unsigned int’ from ‘int’ may change the sign of the result [-Wsign-conversion]
    F_kernel<<<dim3(num_blocks,num_blocks,1),dim3(32,32,1)>>>(d_f,tp,d_phi,d_expKappa,sign,NX,NT,inv);

    CHECK_CU_ERR(cudaMemcpy(f, cast_cmpl(d_f), NX*NX*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    CHECK_CU_ERR(cudaFree(d_f));CHECK_CU_ERR(cudaFree(d_phi));CHECK_CU_ERR(cudaFree(d_expKappa));
}

} // namespace isle
