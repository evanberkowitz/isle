r"""!\file
\ingroup meas
Measurement of the chiral condensate.
"""

import numpy as np

import isle
from .measurement import Measurement

class ChiralCondensate(Measurement):
    r"""!
    \ingroup meas
    Tabulate the chiral condensate.
    """

    def __init__(self, seed, nsamples, hfm, species,
                 savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath, configSlice)

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

    def save(self, h5group):
        r"""!
        Write the chiral condensate
        \param base HDF5 group in which to store data.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = isle.h5io.createH5Group(h5group, self.savePath)
        subGroup["chiralCondensate"] = self.chiCon


def read(h5group):
    r"""!
    Read the chiral condensate from HDF5.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["chiralCondensate"][()]
