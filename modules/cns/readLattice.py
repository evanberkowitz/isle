"""!
Read a lattice from a file.
"""

import yaml

from . import *

def readLattice(fileName):
    r"""!
    Read a lattice file from cns.env["latticeDirectory"].
    \param fileName The name of the lattice to read.
    """
    try:
        with open(cns.env["latticeDirectory"]+"/"+fileName) as yamlf:
            return yaml.safe_load(yamlf)
    except KeyError:
        raise KeyError("Need to set cns.env[\"latticeDirectory\"] before reading a lattice.") from None
