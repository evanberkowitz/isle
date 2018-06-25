Simulating carbon nanostructures.

# Short Summary

[The Hubbard Model][hubbard] is an approximate model that describes fermions on a lattice that have a contact interaction.  The Hamiltonian  includes a nearest-neighbor "hopping", in which the fermions can go from one site to another, and an interaction that two fermions feel if they're on the same site.  In solid-state and condensed matter physics, sites can represent atoms in some material, while the fermions are the electrons in the outer-most shells.

It's thought that the Hubbard model approximates carbon nanostructures, such as graphene, nanotubes, and fullerenes, very well.  This code is intended to allow us to study the Hubbard model, focusing on carbon nanostructures.  However, it has enough generality to allow the study of the Hubbard model on arbitrary graphs.  Some graphs have special features---the hexagonal lattice of graphene is [*bipartite*](https://en.wikipedia.org/wiki/Bipartite_graph), for example.  In some circumstances, these special features may unlock numerical speedup.

[hubbard]:  https://doi.org/10.1098%2Frspa.1963.0204


# Requirements

## C++
- C++14 compiler (tested with recent versions of gcc and clang)
- [Python 3](https://www.python.org/), [numpy](http://www.numpy.org/)
- [blaze](https://bitbucket.org/blaze-lib/blaze)
- BLAS/LAPACK

## Python
- [Pybind11](https://github.com/pybind/pybind11)
- [numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)

# Setup

## TL;DR
See other sections for an explanation and details.
```
python3 setup.py configure --build-type=RELEASE --blaze=<path-to-blaze>
pip3 install . [--user]
```

## Configuring
This project is installable via `pip`. However since the C++ extension is rather complex, you need to configure the build first by running
```
python3 setup.py configure
```
with suitable arguments. The supported arguments are (run `python3 setup.py configure --help`):
- *compiler*: Choose a C++ compiler. isle is tested with current versions of gcc and clang. The intel compiler is not supported because of a bug in the compiler.
- *build-type*: Set the CMake build type. Can be DEVEL (default), RELEASE, DEBUG
- *blaze*: Point to blaze installation (has to contain `blaze/Blaze.h`)
- *blaze-parallelism*: Select parallelism used by blaze. Allowed values are NONE (default), OMP (OpenMP), CPP (C++ threads)
- *blas-vendor*: Select vendor of BLAS/LAPACK library. Supported values are:
    - `Generic` - Reference implementation (default)
    - `Intel10_32`, `Intel10_64lp`, `Intel10_64lp_seq`, `Intel` - MKL, see https://cmake.org/cmake/help/v3.0/module/FindBLAS.html
- *parallel-blas*: Specify this flag if the BLAS/LAPACK inplementation is parallelized. Otherwise blaze might parallelize on top of it which can slow down the program.
- *pardiso*: Select a PARDISO implementation. Can be either `STANDALONE` or `MKL`. PARDISO is currently not used by isle but if this argument is set, a wrapper around it is created in the module `isle.pardiso`.

This does not actually run CMake to configure the C++ build but merely performs some rudimentary checks and saves the parameters.

Note that specifying the compiler of build type when running `python3 setup.py build` does not work; they need to be set when configuring.

## Building
For **users**:
Compile and install the package using
```
pip3 install . [--user]
# or alternatively: python3 setup.py install [--user]
```

For **developers**:
Compile and install in development mode using
```
pip3 install -e . [--user]
# or alternatively: python3 setup.py develop [--user]
```
And just re-run the command after changing the code.

Unfortunately neither pip nor setupt.py's install or develop commands support concurrent builds. So the first time you compile takes some time. If you installed in development mode, you can run
```
python3 setup.py build -j<n-threads>
```
to compile your changes in a parallel fashion.


# Documentation
Run
```
python3 setup.py doc
```
to generate the source code documentation using Doxygen. Then open `docs/html/index.html` in a browser.

Note that the `C++` interface is not identical to the `Python` interface. Documentation of the `Python` side is incomplete as of now.

There is additional documentation available under `docs/algorithm`; run `make` to generate it. This needs LaTeX to compiler PDFs.
