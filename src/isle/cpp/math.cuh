#ifndef __CUDA_MATH__
#define __CUDA_MATH__

#include<complex>
#include<cuComplex.h>

#include "math.hpp"
#include "cuda_cast.cuh"
#include "cuda_helper.cuh"

namespace isle{

  void mult_CDMatrix_wrapper(const CDMatrix &a, const CDMatrix &b, CDMatrix &res, const std::size_t N);
}

#endif
