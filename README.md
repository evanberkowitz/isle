Some CNS stuff

# Requirements
- C++14 compiler
- Python 3
- blaze
- Pybind11
- BLAS/LAPACK

# Build
```
cd cnxx
mkdir build
cd build
cmake ..
make
```

CMake supports the following options: (use `-DOPT=VAL` in call to cmake)
- PYTHON: select Python 3 interpreter
- BLAZE: point to blaze installation (has to contain `blaze/Blaze.h`)
- BLAS_VENDOR: select vendor of BLAS/LAPACK library. Supported values are:
    - `Generic` Reference implementation
    - `Intel10_32`, `Intel10_64lp`, `Intel10_64lp_seq`, `Intel` for MKL, see https://cmake.org/cmake/help/v3.0/module/FindBLAS.html
- PARALLEL_BLAS: Set whether the BLAS implementation is parallelized or not.

You can set the compiler via `-DCMAKE_CXX_COMPILER` when calling CMake.
Remember to configure in release mode (`-DCMAKE_BUILD_TYPE=RELEASE`) when compiling for production.
