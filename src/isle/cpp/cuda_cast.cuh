#ifndef CUDA_CAST_CUH
#define CUDA_CAST_CUH

#ifdef USE_CUDA
#include <complex>
#include <cuComplex.h>

//template<typename T>
//cuda::std::complex<T> * cast_cmpl(std::complex<T> * cptr){
//    return reinterpret_cast<cuda::std::complex<T> *>(cptr);
//}
//
//template<typename T>
//const cuda::std::complex<T> * cast_cmpl(const std::complex<T> * cptr){
//    return reinterpret_cast<const cuda::std::complex<T> *>(cptr);
//}
//
//template<typename T>
//cuda::std::complex<T> cast_cmpl(std::complex<T> cptr){
//    return *reinterpret_cast<cuda::std::complex<T> *>(&cptr);
//}

__device__ __host__ __forceinline__ cuDoubleComplex cexp(cuDoubleComplex const z){
    double t = exp(z.x);
    double real, imag;
    sincos(z.y, &imag, &real);
    real *= t;
    imag *= t;
    return make_cuDoubleComplex(real, imag);
}

__device__ __host__ __forceinline__ cuDoubleComplex  operator*(double a, cuDoubleComplex b) {
    double real = a*b.x;
    double imag = a*b.y;
    return make_cuDoubleComplex(real, imag);
}
__device__ __host__ __forceinline__ cuDoubleComplex  operator*(cuDoubleComplex a, double b) {
    double real = a.x*b;
    double imag = a.y*b;
    return make_cuDoubleComplex(real, imag);
}
__device__ __host__ __forceinline__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a,b);
}
//__device__ __host__ __forceinline__ const cuDoubleComplex operator*(const cuDoubleComplex a, const cuDoubleComplex b) { return cuCmul(a,b); }


cuDoubleComplex * cast_cmpl(std::complex<double> * cptr){
    return reinterpret_cast<cuDoubleComplex * > (cptr);
}
const cuDoubleComplex * cast_cmpl(const std::complex<double> * cptr){
    return reinterpret_cast<const cuDoubleComplex * > (cptr);
}
cuDoubleComplex cast_cmpl(std::complex<double> c_num){
    return make_cuDoubleComplex(c_num.real(),c_num.imag());
}

std::complex<double> * cast_cmpl(cuDoubleComplex * cptr){
    return reinterpret_cast<std::complex<double> * > (cptr);
}
const std::complex<double> * cast_cmpl(const cuDoubleComplex * cptr){
    return reinterpret_cast<const std::complex<double> * > (cptr);
}
std::complex<double> cast_cmpl(cuDoubleComplex c_num){
    return std::complex<double>{c_num.x,c_num.y};
}

#endif // __CUDA_ARCH__

#endif // CUDA_CAST_CUH
