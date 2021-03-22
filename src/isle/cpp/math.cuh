#include<cuComplex.h>

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
