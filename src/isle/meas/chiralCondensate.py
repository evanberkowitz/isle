"""!
Measurement of the chiral condensate.
"""

import numpy as np

import isle
from ..h5io import createH5Group


class ChiralCondensate:
    r"""!
    \ingroup meas
    Tabulate the chiral condensate.
    """

    def __init__(self, seed, nsamples, hfm, species):
        self.nsamples = nsamples
        self.rng = isle.random.NumpyRNG(seed)
        self.hfm = hfm
        self.species = species
        self.chiCon = []

    def __call__(self, phi, action, itr):
        """!Record the chiral condensate."""

        nx = self.hfm.nx()
        nt = int(len(phi) / nx)

        # Create a large set of sources:
        rhss = [isle.Vector(self.rng.normal(0, 1, nt*nx)+0j)
                for _ in range(self.nsamples)]

        # Solve M*x = b for different right-hand sides,
        # Normalize by spacetime volume
        res = np.array(isle.solveM(self.hfm, phi, self.species, rhss), copy=False) / (nx*nt)

        self.chiCon.append(np.mean([np.dot(rhs, r) for rhs, r in zip(rhss, res)]))

    def save(self, base, name):
        r"""!
        Write the chiral condensate
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["chiralCondensate"] = self.chiCon

    def read(self, group):
        r"""!
        Read the chiral condensate from HDF5.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.chiCon = group["chiralCondensate"][()]
