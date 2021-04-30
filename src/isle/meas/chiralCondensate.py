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
        super().__init__(savePath,
                         ("chiCon", (), complex, "chiCon"),
                         configSlice)

        self.nsamples = nsamples
        self.rng = isle.random.NumpyRNG(seed)
        self.hfm = hfm
        self.species = species

        # need to know Nt to set those, do it in _getRHSs
        self._rhss = None

    def __call__(self, stage, itr):
        """!Record the chiral condensate."""

        nx = self.hfm.nx()
        nt = int(len(stage.phi) / nx)
        rhss = self._getRHSs(nt)

        # Solve M*x = b for different right-hand sides,
        # Normalize by spacetime volume
        res = np.array(isle.solveM(self.hfm, stage.phi, self.species, rhss), copy=False) / (nx*nt)

        self.nextItem("chiCon")[...] = np.mean([np.dot(rhs, r) for rhs, r in zip(rhss, res)])

    def _getRHSs(self, nt):
        """!
        Get all right hand side vectors as a matrix.
        For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        In other words, time is the faster running index.
        """

        if self._rhss is None or self._rhss.rows() != nt*self.hfm.nx():
            # Create a large set of sources:
            self._rhss = isle.Matrix(np.array([isle.Vector(self.rng.normal(0, 1, nt*self.hfm.nx())+0j)
                                               for _ in range(self.nsamples)]))
        return self._rhss
