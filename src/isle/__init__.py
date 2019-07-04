"""!
Base Python package of isle.

Contains Python modules and imports everything from C++ extension into the isle namespace.
"""

from .cpp_wrappers import *  # (unused import) pylint: disable=wildcard-import, unused-wildcard-import
from . import checks  # (unused import) pylint: disable=unused-import
from . import cli  # (unused import) pylint: disable=unused-import
from . import collection  # (unused import) pylint: disable=unused-import
from . import fileio  # (unused import) pylint: disable=unused-import
from . import meta  # (unused import) pylint: disable=unused-import
from . import evolver  # (unused import) pylint: disable=unused-import
from . import random  # (unused import) pylint: disable=unused-import
from . import util  # (unused import) pylint: disable=unused-import
import isle.action  # (unused import) pylint: disable=unused-import

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
