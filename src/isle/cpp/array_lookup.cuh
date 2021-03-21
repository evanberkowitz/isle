#ifndef ARRAY_LOOKUP_CUH
#define ARRAY_LOOKUP_CUH
// On the GPU the arrays are refered to some look up scheme which are defined here
// This schemes shall provide the way how we store tensors/matrices/vectors on the

__host__ __device__ std::size_t lookup_3Tensor(
    std::size_t t, std::size_t x_row, std::size_t x_col,
    const std::size_t Nx){

    return t * Nx^2 + x_col * Nx + x_row;
}

__host__ __device__ std::size_t lookup_matrix(
    std::size_t x_row, std::size_t x_col,
    const std::size_t Nx) {

    return x_col * Nx + x_row;
}

__host__ __device__ std::size_t lookup_vector(
    std::size_t t, std::size_t x,
    const std::size_t Nx){

    return t * Nx + x;
}

#endif // ARRAY_LOOKUP_CUH
