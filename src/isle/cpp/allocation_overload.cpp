#include <cuda_runtime_api.h>
#include <cstdlib>
#include <new>

void * operator new(std::size_t size){
    void * ptr = nullptr;

    if (cudaMallocManaged(&ptr,size)){
        throw std::bad_alloc{};
    }

    return ptr;
}

void * operator new(std::size_t size, const std::nothrow_t&) {
    void * ptr = nullptr;

    cudaMallocManaged(&ptr,size);

    return ptr;
}

void * operator new[](std::size_t size){
    void * ptr = nullptr;

    if(cudaMallocManaged(&ptr,size)){
        return ptr;
    }

    throw std::bad_alloc{};
}


void * operator new[](std::size_t size, const std::nothrow_t& ){
    void * ptr = nullptr;

    cudaMallocManaged(&ptr,size);

    return ptr;
}

// =============================================================================

void operator delete( void* ptr ) noexcept {
    cudaFree(ptr);
}

void operator delete( void* ptr, std::size_t) noexcept {
    cudaFree(ptr);
}

void operator delete( void* ptr, const std::nothrow_t& ) noexcept {
    cudaFree(ptr);
}

void operator delete[]( void * ptr) noexcept {
    cudaFree(ptr);
}

void operator delete[]( void* ptr, std::size_t) noexcept {
    cudaFree(ptr);
}

void operator delete[]( void* ptr, const std::nothrow_t& ) noexcept {
    cudaFree(ptr);
}
