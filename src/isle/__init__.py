"""!
Base Python package of isle.

Contains Python modules and imports everything from C++ extension into the isle namespace.
"""

from logging import getLogger

from .cpp_wrappers import *  # (unused import) pylint: disable=wildcard-import, unused-wildcard-import
from . import checks  # (unused import) pylint: disable=unused-import
from . import cli  # (unused import) pylint: disable=unused-import
from . import collection  # (unused import) pylint: disable=unused-import
from . import fileio  # (unused import) pylint: disable=unused-import
from . import memoize  # (unused import) pylint: disable=unused-import
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


class _LatticeHandle:
    """!
    Load built in lattices.
    """

    import pkg_resources as pkgr

    LATPATH = "resources/lattices"

    def __init__(self):
        if not self.pkgr.resource_exists(__name__, self.LATPATH):
            getLogger(__name__).error("Lattice files are not installed as part of Isle. "
                                      "Check your installation.")
            raise RuntimeError("Lattice files are not installed as part of Isle.")

    def keys(self):
        """!Return a list of the names of all built in lattices."""
        return [fname.rsplit(".", 1)[0]
                for fname in self.pkgr.resource_listdir(__name__, self.LATPATH)]

    def __getitem__(self, name):
        """!Load and return a built in lattice."""
        try:
            with self.pkgr.resource_stream(__name__, self._fname(name)) as latfile:
                return fileio.yaml.loadLattice(latfile)
        except FileNotFoundError:
            getLogger(__name__).error("Unknown lattice: '%s' Installed lattices:\n %s\n"
                                      "Use isle.LATTICES.loadExternal(fname) to load "
                                      "from a custom file.",
                                      name, self.keys())
            raise ValueError(f"Unknown lattice: {name}") from None

    def loadExternal(self, fname):
        """!Load and return a lattice from an external file."""
        return fileio.yaml.loadLattice(fname)

    def _fname(self, name):
        """!Format a resource file name for built in lattices."""
        return f"{self.LATPATH}/{name}.yml"


## Global Handler for built in lattices.
LATTICES = _LatticeHandle()
