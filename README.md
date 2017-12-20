Some CNS stuff (bat nano?) (name pending...)

# Requirements
- blaze
- Pybind11
- BLAS/LAPACK

# Build
```
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
