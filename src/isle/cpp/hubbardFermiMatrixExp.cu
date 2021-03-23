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
    //       cuDoubleComplex * f_begin  : Output matrix of size Nt x (Nx x Nx) (e^{kappaTilde - muTilde} @ e*{+/- i * Phi})
    // const cuDoubleComplex * phi_begin: Input vector of size Nt*Nx (configuration)
    // const double                * expKappa_begin: Input matrix of size (Nx x Nx) (e^{kappaTilde - muTilde})
    // const cuDoubleComplex sign       : Input +/- i (phase) depending on species,inverse
    // const std::size_t nx                        : Input spatial sites (number of atoms)
    // const std::size_t nt                        : Input temporal sites (number of time slices)
    // const bool inv                              : Input Flag to calculate inverse
    // Note: It is assumed that expKappa_begin refers to the inverse if inv == True.
    // This is handled at accessing time and must be passed correctly!

    // each block operates on one time slize
    auto tm1 = tp == 0 ? Nt-1 : tp-1; // apply periodic boundary or the used t-1
    // each 1D thread (x direction) on a given block acts as the row coordinate of the N_x by N_x sub matrix
    auto x_row = threadIdx.y + blockIdx.y * blockDim.y;//threadIdx.y;
    // each 1D thread (y direction) on a given block acts as the col coordinate of the N_x by N_x sub matrix
    auto x_col = threadIdx.x + blockIdx.x * blockDim.x;
    // Note: The handling of row and col of the indices is done at referencing below
    //       * Each time slices has an nx^2 (matrix) or nx (vector) offset
    //       * Each col has an nx (matrix) offset
    // example: accessing Nt x (Nx x Nx): t*nx^2 + x_row + nx*x_col
    // example: accessing Nt x  Nx      : t*nx + x_row
    // example: accessing Nx x  Nx      : x_col*nx + x_row

    // ensure that only threads participate which do not access memory out of bounds

    if (x_row > Nx || x_col > Nx || tp > Nt){return;}

    if(inv){ // if the inverse has to be computed phi is expanded as column vector
        f_begin[ lookup_matrix(x_row,x_col,Nx) ] = expKappa_begin[ lookup_matrix(x_row,x_col,Nx) ] * cexp( sign * phi_begin[ lookup_vector(tm1,x_row,Nx) ] );
    } else { // if non inverse has to ve computed phi is expanded as row vector
        f_begin[ lookup_matrix(x_row,x_col,Nx) ] = expKappa_begin[ lookup_matrix(x_row,x_col,Nx) ] * cexp( sign*phi_begin[ lookup_vector(tm1,x_col,Nx) ] );
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
    CHECK_CU_ERR(cudaMallocManaged(&d_expKappa,NX*NX*sizeof(cuDoubleComplex)));

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

    cudaMemcpy(f, cast_cmpl(d_f), NX*NX*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_f);cudaFree(d_phi);cudaFree(d_expKappa);
}

} // namespace isle
