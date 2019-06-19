r"""!\file
\ingroup meas
Measurement of single-particle correlator.
"""

import numpy as np

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, temporalRoller
from ..h5io import createH5Group
from .propagator import AllToAll

class SingleParticleCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    def __init__(self, allToAll, savePath, configSlice=slice(None, None, None), projector=None):
        super().__init__(savePath, configSlice)

        self.correlators = []
        self._inverter = allToAll
        self._path = None
        self._indices = "idf,bx,xfyi,ya->bad"
        self._roll = None

        self.nx = self._inverter.nx
        self.nt = None

        if projector is None:
            _, self.irreps = np.linalg.eigh(isle.Matrix(allToAll.hfm.kappaTilde()))
            self.irreps = self.irreps.T
        else:
            self.irreps = projector.T

        self.irreps = np.matrix(self.irreps)

    def __call__(self, stage, itr):
        """!Record the single-particle correlators."""

        S = self._inverter(stage.phi, stage.actVal, itr)

        if self.nt is None:
            self.nt = len(stage.phi) // self.nx
        if self._roll is None:
            self._roll = np.array([temporalRoller(self.nt, -t) for t in range(self.nt)])

        self._tensors = (self._roll, self.irreps, S, self.irreps.H)

        if self._path is None:
            self._path, _ = np.einsum_path(self._indices, *self._tensors, optimize='optimal')

        # The temporal roller sums over time, but does not *average* over time.  So, divide by nt:
        correlator = np.einsum(self._indices, *self._tensors, optimize=self._path) / self.nt

        # Done!
        self.correlators.append(correlator)

    def save(self, h5group):
        r"""!
        Write the irreps and their correlators to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["correlators"] = self.correlators
        subGroup["irreps"] = self.irreps
        subGroup["einsum_path"] = self._path

def read(h5group):
    r"""!
    Read the irreps and their correlators from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["correlators"][()], h5group["irreps"][()]
