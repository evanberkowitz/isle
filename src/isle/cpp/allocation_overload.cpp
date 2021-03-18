#include <cstdlib>
#include <new>

#include "cuda_helper.cuh"

void * operator new(std::size_t size){
    void * ptr = nullptr;

    CHECK_CU_ERR(cudaMallocManaged(&ptr,size));

    return ptr;
}

void * operator new(std::size_t size, const std::nothrow_t&) noexcept{
    void * ptr = nullptr;

    CHECK_CU_ERR(cudaMallocManaged(&ptr,size));

    return ptr;
}

void * operator new[](std::size_t size){
    void * ptr = nullptr;

    CHECK_CU_ERR(cudaMallocManaged(&ptr,size));

    return ptr;
}

void * operator new[](std::size_t size, const std::nothrow_t& ) noexcept{
    void * ptr = nullptr;

    CHECK_CU_ERR(cudaMallocManaged(&ptr,size));

    return ptr;
}

// =============================================================================

void operator delete( void* ptr ) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}

void operator delete( void* ptr, std::size_t) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}

void operator delete( void* ptr, const std::nothrow_t& ) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}

void operator delete[]( void * ptr) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}

void operator delete[]( void* ptr, std::size_t) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}

void operator delete[]( void* ptr, const std::nothrow_t& ) noexcept {
    CHECK_CU_ERR(cudaFree(ptr));
}
