"""!
Routines for YAML IO.

Registers representable types with YAML allowing automatic loading and dumping.
"""

from logging import getLogger

import yaml
import numpy as np

from . import Lattice
from .util import parameters
from .action import HFAHopping, HFABasis, HFAAlgorithm

def _parseLattice(adjacency, hopping, positions, nt=0, name="", comment=""):
    """!
    Parse a `!lattice` YAML node.
    """
    log = getLogger(__name__)

    # turn hopping into a list if it isn't already
    if not isinstance(hopping, list):
        hopping = [hopping]*len(adjacency)
    elif len(adjacency) != len(hopping):
        log.error("Lengths of adjacency matrix and list of hopping strengths do not match")
        raise RuntimeError("Lengths of adjacency matrix and list of hopping strengths do not match")

    # construct lattice
    lat = Lattice(nt, len(positions), name, comment)

    # set neighbors and hopping strengths
    for (i, j), hop in zip(adjacency, hopping):
        lat.setNeighbor(i, j, hop)

    # set positions
    for i, pos in enumerate(positions):
        if len(pos) == 3:
            lat.position(i, *pos)
        elif len(pos) == 2:
            lat.position(i, *pos, 0)
        else:
            raise RuntimeError(f"Lattice site positions given with {len(pos)} coorddinates."
                               +"only supports 2D and 3D positions.")

    log.info("Read lattice '%s': %s", name, comment)
    return lat

yaml.add_constructor("!lattice",
                     lambda loader, node: \
                     _parseLattice(**loader.construct_mapping(node, deep=True)),
                     Loader=yaml.SafeLoader)


def _representLattice(dumper, lat):
    """!
    Create a YAML representation of a Lattice using a `!lattice` node.
    """

    if lat.nx() > 1:
        adj, hopping = zip(*[([i, neigh[0]], neigh[1]) for i in range(lat.nx())
                             for neigh in lat.getNeighbors(i)
                             if neigh[0] > i])
    else:
        adj, hopping = [], []

    positions = [list(lat.position(i)) for i in range(lat.nx())]
    return dumper.represent_mapping("!lattice",
                                    {"name": lat.name,
                                     "comment": lat.comment,
                                     "nt": lat.nt(),
                                     "adjacency": list(adj),
                                     "hopping": list(hopping),
                                     "positions": positions},
                                    flow_style=False)

yaml.add_representer(Lattice, _representLattice)

def loadLattice(fname):
    """!Load a Lattice from a YAML file."""
    if hasattr(fname, "read"):
        string = fname.read()
    else:
        with open(fname, "r") as yamlf:
            string = yamlf.read()
    return yaml.safe_load(string)


# register parameters function
yaml.add_constructor("!parameters",
                     lambda loader, node: \
                     parameters(**loader.construct_mapping(node, deep=True)),
                     Loader=yaml.SafeLoader)

# register isle.action.HFAHopping
yaml.add_representer(HFAHopping,
                     lambda dumper, hop: \
                     dumper.represent_scalar("!HFAHopping", str(hop).rsplit(".")[-1]))
yaml.add_constructor("!HFAHopping",
                     lambda loader, node: \
                     HFAHopping.DIA if loader.construct_scalar(node) == "DIA"
                     else HFAHopping.EXP,
                     Loader=yaml.SafeLoader)

# register isle.action.HFABasis
yaml.add_representer(HFABasis,
                     lambda dumper, basis: \
                     dumper.represent_scalar("!HFABasis", str(basis).rsplit(".")[-1]))
yaml.add_constructor("!HFABasis",
                     lambda loader, node: \
                     HFABasis.PARTICLE_HOLE if loader.construct_scalar(node) == "PARTICLE_HOLE"
                     else HFABasis.SPIN,
                     Loader=yaml.SafeLoader)

# register isle.action.HFAAlgorithm
yaml.add_representer(HFAAlgorithm,
                     lambda dumper, var: \
                     dumper.represent_scalar("!HFAAlgorithm", str(var).rsplit(".")[-1]))
yaml.add_constructor("!HFAAlgorithm",
                     lambda loader, node: \
                     {"DIRECT_SINGLE": HFAAlgorithm.DIRECT_SINGLE,
                      "DIRECT_SQUARE": HFAAlgorithm.DIRECT_SQUARE,
                      "ML_APPROX_FORCE":HFAAlgorithm.ML_APPROX_FORCE}[loader.construct_scalar(node)],
                     Loader=yaml.SafeLoader)
