"""!
General file input / output.
"""

from enum import Enum, auto
from pathlib import Path
import argparse

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
