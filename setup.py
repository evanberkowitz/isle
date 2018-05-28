"""
Build and install isle.
"""

from pathlib import Path
from setuptools import setup, find_packages

from setup import CMakeExtension, configure_command, predicate, \
    get_cmake_builder, version_from_git, doxygen_command

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT/"build"
TEST_DIR = PROJECT_ROOT/"tests"
CONFIG_FILE = BUILD_DIR/"configure.out.pkl"

@configure_command(CONFIG_FILE)
class Configure:
    pass

setup(
    name="isle",
    version=version_from_git(plain=True),
    author="Jan-Lukas Wynen",
    author_email="j-l.wynen@hotmail.de",
    description="Lattice Monte-Carlo for carbon nano systems",
    long_description="",
    # add all packages under 'src'
    packages=find_packages("src"),
    # don't add any outside of 'src' packages -> hide setup and test
    package_dir={"": "src"},
    # add an extension module named 'isle_cpp' to package 'isle'
    ext_modules=[CMakeExtension("isle/isle_cpp")],
    cmdclass={
        "configure": Configure,
        "build_ext": get_cmake_builder(CONFIG_FILE, TEST_DIR),
        "doc": doxygen_command("docs/doxyfile.conf")
    },
    zip_safe=False,
    python_requires=">=3.6.5",
    entry_points={
    },
)
