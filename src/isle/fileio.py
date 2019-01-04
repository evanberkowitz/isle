"""!
General file input / output.
"""

from enum import Enum, auto
import inspect
from logging import getLogger
from pathlib import Path

# import those to make them available outside of the package
from . import yamlio as yaml  # (unused import) pylint: disable=W0611
from . import h5io as h5  # (unused import) pylint: disable=W0611


class FileType(Enum):
    """!Identify file types."""
    YAML = auto()
    HDF5 = auto()
    PY = auto()
    INDETERMINATE = auto()

## Map file extensions to types.
EXTENSION_MAP = {"yml": FileType.YAML,
                 "yaml": FileType.YAML,
                 "h5": FileType.HDF5,
                 "hdf5": FileType.HDF5,
                 "py": FileType.PY,
                }

def fileType(path):
    """!Return type of given file."""
    ext = Path(path).suffix[1:]
    try:
        return EXTENSION_MAP[ext]
    except KeyError:
        return FileType.INDETERMINATE

def pathAndType(fname):
    """!Return path and type of file."""
    return Path(fname), fileType(fname)

def sourceOfFunction(func):
    r"""!
    Return the source code of a function.
    Works only on free functions, not methods or lambdas.

    \see functionFromSource for the inverse.
    """

    if not inspect.isfunction(func) or inspect.ismethod(func):
        raise RuntimeError("Not a function, methods not allowed")
    # missing a check for class methods!

    src = inspect.getsource(func)

    # lambdas are weird, need do make sure this is a normal function definition
    if not src.lstrip().startswith("def"):
        raise RuntimeError("Not a proper function, did you pass a lambda?")

    return src

def functionFromSource(src):
    r"""!
    Return a function defined by a piece of source code.

    \see sourceOfFunction for the inverse.
    """

    # strip indentation of function definition
    src = src.lstrip()

    # an empty local scope
    scope = dict()
    # no access to globals so it can't fuck around
    exec(src, {}, scope)

    if not scope:
        raise RuntimeError("Nothing defined by source code")
    if len(scope) > 1:
        raise RuntimeError("More than one thing defined")

    # the only thing in scope is the one we want
    obj = list(scope.values())[0]
    if not inspect.isfunction(obj):
        raise RuntimeError("Source code defined something else than a function")

    return obj

def callFunctionFromSource(src, *args, **kwargs):
    r"""!
    Extract a function from source code and call it.

    `args` and `kwargs` are passed on to the function and its result is returned.

    \see soruceOfFunction and functionFromSource.
    """

    func = functionFromSource(src)
    try:
        return func(*args, **kwargs)
    except NameError as e:
        getLogger(__name__).error("Undefined symbol in function constructed from source: %s",
                                  str(e))
        raise
    except:
        # must re-raise it so things like keyboard interrupts get processed properly
        raise
