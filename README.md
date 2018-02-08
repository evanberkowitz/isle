Some CNS stuff


# Requirements

- C++14 compiler
- Python 3
- blaze
- Pybind11
- BLAS/LAPACK


# Build and install cnxx

```
cd cnxx
mkdir build
cd build
cmake ..
make
make install
```
This will compile and install the `cnxx` Python module under `modules/cnxx.suffix` with
a system specific suffix.

CMake supports the following options: (use `-DOPT=VAL` in call to cmake)
- PYTHON: select Python 3 interpreter
- BLAZE: point to blaze installation (has to contain `blaze/Blaze.h`)
- BLAS_VENDOR: select vendor of BLAS/LAPACK library. Supported values are:
    - `Generic` Reference implementation
    - `Intel10_32`, `Intel10_64lp`, `Intel10_64lp_seq`, `Intel` for MKL, see https://cmake.org/cmake/help/v3.0/module/FindBLAS.html
- PARALLEL_BLAS: Set whether the BLAS implementation is parallelized or not.

You can set the compiler via `-DCMAKE_CXX_COMPILER` when calling CMake.
Remember to configure in release mode (`-DCMAKE_BUILD_TYPE=RELEASE`) when compiling for production.

## Notes for JURECA
Example that works as of 2018-02-08:
- Install blaze and pybind11
- `ml Intel/2018.1.163-GCC-5.4.0 ParaStationMPI/5.2.0-1 imkl/2018.1.163 Python/3.6.3 SciPy-Stack/2017b-Python-3.6.3 CMake`
- run cmake: `cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=RELEASE -DBLAZE=/work/cascade/casc0003/logdet/blaze-3.2 -DBLAS_VENDOR=Intel10_64lp`

Note that `icc` has a bug which prevents it from compiling `hubbardFermimatrix.cpp`.
Something related to pushing back a blaze matrix into a std vector.
Couldn't reproduce, something really nasty apparently.


# Usage

After installing `cnxx`, the Python package `cns` can be imported from `modules/cns`.
It provides a high level interface for `cnxx` and extra routines.


# Documentation

Go into `docs` and run `make` to generate source code documentation for `cnxx` and some
Python scipts using Doxygen. Note that the `C++` interface is not identical to the
`Python` interface. Documentation of the `Python` side is incomplete as of now.

If you don't want to look at Doxygens ugly default style, initialize the git submodule
`that_style` under `docs`.
