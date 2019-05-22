r"""!\file
\ingroup meas
Measurement of single-particle correlator.
"""

from logging import getLogger

import numpy as np
# import scipy.linalg as la   # scipy is needed because np.linalg.eig and np.linalg.eigh do not guarantee an orthonormal eigenbasis.

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, rollTemporally
from ..h5io import createH5Group

class SingleParticleCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    def __init__(self, hfm, species, savePath, configSlice=slice(None, None, None), alpha=1, projector=None):
        super().__init__(savePath, configSlice)

        self.hfm = hfm
        self.corr = []
        self.species = species
        self._alpha = alpha

        # self.irreps = np.transpose(la.orth(isle.Matrix(hfm.kappaTilde())))
        if projector is None:
            _, self.irreps = np.linalg.eigh(isle.Matrix(hfm.kappaTilde()))
            self.irreps = self.irreps.T
        else:
            self.irreps = projector.T
        # self.irreps = np.identity(self.hfm.nx())

        # need to know Nt to set those, do it in _getRHSs
        self._rhss = None

        if species == isle.Species.PARTICLE:
            getLogger(__name__).info("Initialized for particles, irreps:\n %s", self.irreps)
        elif species == isle.Species.HOLE:
            getLogger(__name__).info("Initialized for holes, irreps:\n %s", self.irreps)
        else:
            getLogger(__name__).error("Unknown species: %s", species)
            raise ValueError(f"Unknown species: {species}")

    def __call__(self, phi, action, itr):
        """!Record the single-particle correlators."""

        nt = int(len(phi) / self.hfm.nx())
        rhss = self._getRHSs(nt)

        # Solve M*x = b for all right-hand sides:
        if self._alpha == 1:
            res = np.array(isle.solveM(self.hfm, phi, self.species, rhss), copy=False)
        else:
            res = np.linalg.solve(isle.Matrix(self.hfm.M(-1j*phi, self.species)), np.array(rhss).T).T

        propagators = res.reshape([len(self.irreps), nt, nt, len(self.irreps)])
        # The logic for the one-liner goes as:
        # For each source irrep we need to apply a sink for every irrep.
        #     This produces a big cross-correlator.
        # For each source time we need to roll the vector such that the
        #     source lives on timeslice 0.
        # Finally, we need to average over all the source correlators with
        #     the same source irrep but different timeslices.
        self.corr.append(np.mean(
            np.array([
                [
                    [rollTemporally(np.dot(propagators[i, src], np.conj(irrepj)), -src)
                     for src in range(nt)]
                    for i in range(len(self.irreps))]
                for irrepj in self.irreps]),
            axis=2))

    def save(self, h5group):
        r"""!
        Write the irreps and their correlators to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["correlators"] = self.corr
        subGroup["irreps"] = self.irreps

    def _getRHSs(self, nt):
        """!
        Get all right hand side vectors as a matrix.
        For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        In other words, time is the faster running index.
        """

        if self._rhss is None or self._rhss.rows() != nt*self.hfm.nx():
            # Create a large set of sources:
            self._rhss = isle.Matrix(np.array([isle.Vector(spaceToSpacetime(irrep, time, nt))
                                               for irrep in self.irreps for time in range(nt)]))
        return self._rhss


def read(h5group):
    r"""!
    Read the irreps and their correlators from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["correlators"][()], h5group["irreps"][()]
