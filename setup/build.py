"""
Command to build CMake extensions.
"""

import sys
from pathlib import Path
import shutil
import subprocess
import pickle
from setuptools.command.build_ext import build_ext

from . import predicate
from .cmake_extension import CMakeExtension

def _in_virtualenv():
    "Detect whether Python is running in a virutal environment."
    return sys.prefix == sys.base_prefix

def _python_config():
    "Return name of the python-config script."
    if not _in_virtualenv():
        return f"{sys.executable}-config"
    return "python-config"

def _common_cmake_args(config, test_dir):
    "Format arguments for CMake common to all extensions."
    args = [f"-D{key}={val}" for key, val in config.items() if val is not None] \
        + [f"-DPYTHON_EXECUTABLE={sys.executable}",
           f"-DPYTHON_CONFIG={_python_config()}"]
    if test_dir is not None:
        args += [f"-DTEST_DIR={test_dir}"]
    return args


def get_cmake_builder(config_file, test_dir=None):
    """
    Return a setuptools command class to build CMake extensions.

    Arguments:
       - config_file: File written by configure.Configure command
       - test_dir: Directory which contains unit tests. If None, it is not passed
                   to CMake and the CMake install target is not built.
    """

    class _BuildCMakeExtension(build_ext):
        def run(self):
            # make sure that cmake is installed
            if not predicate.executable("cmake"):
                sys.exit(1)

            # read configuration file and prepare arguments for cmake
            try:
                config = pickle.load(open(str(config_file), "rb"))
            except FileNotFoundError:
                print("error: Configuration file not found. Did you forget "
                      "to run the configure command first?")
                sys.exit(2)
            config_time = config_file.stat().st_mtime
            cmake_args = _common_cmake_args(config, test_dir)

            # build all extensions
            for ext in self.extensions:
                if not isinstance(ext, CMakeExtension):
                    print(f"error: Extension {ext.name} is not a CMakeExtension")
                self._build_extension(ext, cmake_args, config_time)

        def _run_cmake(self, extension, libname, ext_build_dir, cmake_args):
            print("running cmake")

            # where the extension library has to be placed (during build, not installation)
            extdir = Path(self.get_ext_fullpath(extension.name)).parent.resolve()
            # finalize arguments for cmake
            cmake_args = cmake_args \
                         + [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}"]

            try:
                # call cmake from ext_build_dir
                subprocess.check_call(["cmake", extension.sourcedir] + cmake_args,
                                      cwd=ext_build_dir)
            except subprocess.CalledProcessError as err:
                print(f"Calling cmake failed with arguments {err.cmd}")
                sys.exit(1)

        def _build_extension(self, extension, cmake_args, config_time):
            print(f"building extension {extension.name}")

            # name of the output library
            libname = extension.name.rsplit("/", 1)[-1]
            # directory to build the extension in
            ext_build_dir = Path(self.build_temp).resolve()/libname

            # configure was run after making the directory => start over
            if ext_build_dir.exists() and ext_build_dir.stat().st_mtime < config_time:
                shutil.rmtree(str(ext_build_dir))

            # need to run CMake
            if not ext_build_dir.exists():
                ext_build_dir.mkdir(parents=True)
                self._run_cmake(extension, libname, ext_build_dir, cmake_args)

            print("compiling extension")
            # arguments for build tool
            extra_args = []
            if self.parallel:
                extra_args += ["-j", str(self.parallel)]
            # construct CMake command
            build_cmd = ["cmake", "--build", "."]
            if extra_args:
                build_cmd += ["--", *extra_args]
            try:
                subprocess.check_call(build_cmd, cwd=ext_build_dir)
                if test_dir is not None:
                    subprocess.check_call(["cmake", "--build", ".", "--target", "install"],
                                          cwd=ext_build_dir)
            except subprocess.CalledProcessError as err:
                print(f"Calling cmake to build failed, arguments {err.cmd}")
                sys.exit(1)
    return _BuildCMakeExtension
