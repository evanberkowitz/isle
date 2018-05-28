"""!
Routines for YAML IO.

Registers isle.Lattice with YAML allowing automatic loading and dumping.
"""

import yaml
import numpy as np

from . import Lattice

def _parse_lattice(adjacency, hopping, positions, nt=0, name="", comment=""):
    """!
    Parse a `!lattice` YAML node.
    """

    # turn hopping into a list if it isn't already
    if not isinstance(hopping, list):
        hopping = [hopping]*len(adjacency)
    elif len(adjacency) != len(hopping):
        raise RuntimeError("Lengths of adjacency matrix and list of hopping strengths do not match")

    # construct lattice
    lat = Lattice(nt, len(positions))

    # set neighbors and hopping strengths
    for (i, j), hop in zip(adjacency, hopping):
        lat.setNeighbor(i, j, hop)

    # set positions
    for i, pos1 in enumerate(positions):
        lat.distance(i, i, 0.)
        for j, pos2 in enumerate(positions[i+1:]):
            lat.distance(i, j+i+1, np.linalg.norm(np.array(pos1)-np.array(pos2)))

    print("Read lattice '{}': {}".format(name, comment))
    return lat

yaml.add_constructor("!lattice",
                     lambda loader, node: \
                     _parse_lattice(**loader.construct_mapping(node, deep=True)),
                     Loader=yaml.SafeLoader)


def _represent_lattice(dumper, lat):
    """!
    Create a YAML representation of a Lattice using a `!lattice` node.
    """

    adj, hopping = zip(*[([i, neigh[0]], neigh[1]) for i in range(lat.nx())
                         for neigh in lat.getNeighbors(i)
                         if neigh[0] > i])
    pos = [0]*lat.nx()
    return dumper.represent_mapping("!lattice",
                                    {"name": "",
                                     "comment": "",
                                     "nt": lat.nt(),
                                     "adjacency": list(adj),
                                     "hopping": list(hopping),
                                     "positions": pos},
                                    flow_style=False)

yaml.add_representer(Lattice, _represent_lattice)
