"""!
Base Python package of isle.

Contains Python modules and imports everything from C++ extension into the isle namespace.
"""

from .cpp_wrappers import *
from . import checks
from . import fileio
from . import proposers
from . import random
from . import util
from . import cli
import isle.action

__version__ = str(isleVersion)


def initialize(*args, **kwargs):
    args = cli.init(*args, **kwargs)

    from sys import argv
    from logging import getLogger
    log = getLogger("isle")

    log.info("This is Isle v%s", str(isleVersion))
    log.info("Command line arguments: %s", " ".join(argv))
    log.info("Initialized command line interface")

    return args
