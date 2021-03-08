"""
Build and install isle.
"""

from pathlib import Path
from setuptools import setup, find_packages

from setup import CMakeExtension, configure_command, predicate, \
    get_cmake_builder, version_from_git, doxygen_command

VERSION = version_from_git(plain=True)
PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT/"build"
CONFIG_FILE = BUILD_DIR/"configure.out.pkl"
# relative path to coax doxygen into placing the output where it belongs
DOXY_FILE = "docs/doxyfile.conf"

# allowed values for configure arguments
BLAS_VENDORS = ("Generic", "Intel10_32", "Intel10_64lp", "Intel10_64lp_seq", "Intel")


@configure_command(CONFIG_FILE)
class Configure:
    blaze = dict(help="Path to blaze. Has to contain blaze/Blaze.h",
                 cmake="BLAZE",
                 check=predicate.directory)
    blaze_parallelism = dict(help="Select parallelism used by blaze. Allowed values are NONE (default), OMP, CPP",
                             cmake="BLAZE_PARALLELISM",
                             default="NONE",
                             check=predicate.one_of("NONE", "OMP", "CPP"))
    blas_vendor = dict(help=f"Select vendor of BLAS/LAPACK. Allowed values are {BLAS_VENDORS}."
                       "See documentation of CMake for more information.",
                       cmake="BLAS_VENDOR",
                       check=predicate.one_of(*BLAS_VENDORS))
    parallel_blas = dict(help="Pass flag if the BLAS implementation is parallelized",
                         cmake="PARALLEL_BLAS", bool=True)


setup(
    name="isle",
    version=VERSION,
    author="Jan-Lukas Wynen",
    author_email="j-l.wynen@hotmail.de",
    description="Lattice Monte-Carlo for carbon nano systems",
    long_description="",
    # add all packages under 'src'
    packages=find_packages("src"),
    # don't add any packages outside of 'src' -> hide setup and test
    package_dir={"": "src"},
    # add an extension module named 'isle_cpp' to package 'isle'
    ext_modules=[CMakeExtension("isle/isle_cpp")],
    cmdclass={
        "configure": Configure,
        "build_ext": get_cmake_builder(CONFIG_FILE),
        "doc": doxygen_command(DOXY_FILE, VERSION)
    },
    zip_safe=False,
    python_requires=">=3.6.5",
    entry_points={
        "console_scripts": ["isle=isle.scripts.base_command:main"]
    },
    package_data={"isle": ["resources/lattices/*.yml"]},
    install_requires=["numpy", "PyYAML", "h5py", "pybind11", "scipy"],
)
