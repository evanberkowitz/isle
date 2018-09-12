"""!
Base Python package of isle.

Contains Python modules and imports everything from C++ extension into the isle namespace.
"""

from .cpp_wrappers import *
from . import checks
from . import ensemble
from . import fileio
from . import hmc
from . import random
from . import util
import isle.action

__version__ = str(isleVersion)

## Environment variables for CNS.
# TODO remove!
env = {}
