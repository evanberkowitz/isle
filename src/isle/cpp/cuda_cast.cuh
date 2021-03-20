#ifndef CUDA_CAST_CUH
#define CUDA_CAST_CUH

#include<cuda/std/complex>
#include <complex>

template<typename T>
cuda::std::complex<T> * cast_cptr(std::complex<T> * cptr){
    return reinterpret_cast<cuda::std::complex<double> *>(cptr);
}

template<typename T>
cuda::std::complex<T> cast_cptr(std::complex<T> cptr){
    return *reinterpret_cast<cuda::std::complex<double> * >(&cptr);
}

#endif // CUDA_CAST_CUH
