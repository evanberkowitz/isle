"""
Configure CMake based extensions.
"""

import sys
from pathlib import Path
import re
import pickle
import inspect
import distutils.cmd

from .predicate import executable, one_of

# allowed values of build types
_BUILD_TYPES = ("DEVEL", "RELEASE", "DEBUG")
# default arguments for options
_DEFAULT_OPT_ARGS = dict(short_name=None, help="", bool=False, default=None, check=None)
# options inserted into class by default
_DEFAULT_OPTIONS = dict(compiler={**_DEFAULT_OPT_ARGS,
                                  "help": "C++ compiler",
                                  "cmake": "CMAKE_CXX_COMPILER",
                                  "long_name": "compiler=",
                                  "check": executable},
                        build_type={**_DEFAULT_OPT_ARGS,
                                    "help": f"CMake build type, allowed values are {_BUILD_TYPES}",
                                    "cmake": "CMAKE_BUILD_TYPE",
                                    "long_name": "build_type=",
                                    "default": "DEVEL",
                                    "check": one_of(*_BUILD_TYPES)})
# protected attribute names that may not be used as options
_FORBIDDEN_OPTIONS = ("description", "user_options")


def _parse_option(name, args):
    "Parse a single user option."
    # make sure everything is ok
    if name in _FORBIDDEN_OPTIONS:
        print(f"error: Illegal option name for configure command: '{name}'")
        sys.exit(1)
    if name in _DEFAULT_OPTIONS:
        print(f"warning: Option to configure command will be overwritten by default: '{name}'")
    if "cmake" not in args:
        print(f"error: No key 'cmake' in arguments for configure option '{name}'")
        sys.exit(1)

    # incorporate default arguments
    args = {**_DEFAULT_OPT_ARGS, **args}
    if "long_name" not in args:  # add long_name if not given
        args["long_name"] = name + ("=" if not args["bool"] else "")
    # default for bools must be False, otherwise False could not be specified
    if args["bool"]:
        args["default"] = False
    return args

def _get_options(cls):
    "Extract user options from a class."
    underscore_re = re.compile("^_.*_$")
    options = {name: _parse_option(name, args)
               for name, args in inspect.getmembers(cls, lambda x: not inspect.isroutine(x))
               if not underscore_re.match(name)}
    return {**options, **_DEFAULT_OPTIONS}

def _format_option(name, val):
    "Format name, value pair of user options."
    if val is None:
        val = "<unspecified>"
    return "{} = {}".format(name.split("=", 1)[0], val)

def configure_command(outfile):
    """
    Decorator for a distutils configure command class.
    Arguments:
       - outfile: Path to the output file.
    """

    def _wrap(cls):
        class _Configure(distutils.cmd.Command):
            description = "configure build of C++ extensions"
            # store full metadata on options
            options = _get_options(cls)
            # handled by distutils
            user_options = [(args["long_name"].replace("_", "-"), args["short_name"],
                             args["help"])
                            for _, args in options.items()]

            def initialize_options(self):
                "Set defaults for all user options."
                for name, args in self.options.items():
                    setattr(self, name, args["default"])

            def finalize_options(self):
                "Post process and verify user options."
                for name, args in self.options.items():
                    if args["bool"]:
                        setattr(self, name, str(bool(getattr(self, name))).upper())

                    check = args["check"]
                    val = getattr(self, name)
                    if check is not None and val is not None and not check(val):
                        print(f"Invalid argument to option {name}: '{getattr(self, name)}'")
                        sys.exit(1)

            def run(self):
                "Execute the command, writes configure file."
                print("-- " +
                      "\n-- ".join(_format_option(args["long_name"], getattr(self, name))
                                   for name, args in self.options.items()))
                print(f"writing configuration to file {outfile}")
                self.mkpath(str(Path(outfile).parent))
                pickle.dump({args["cmake"]: getattr(self, name)
                             for name, args in self.options.items()},
                            open(str(outfile), "wb"))

        return _Configure
    return _wrap
