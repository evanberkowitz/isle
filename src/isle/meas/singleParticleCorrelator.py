r"""!\file
\ingroup meas
Measurement of single-particle correlator.
"""

import numpy as np
import scipy.linalg as la   # scipy is needed because np.linalg.eig and np.linalg.eigh do not guarantee an orthonormal eigenbasis.

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, rollTemporally
from ..h5io import createH5Group

class SingleParticleCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    def __init__(self, hfm, species, savePath, configSlice=slice(None, None, None), alpha=1):
        super().__init__(savePath, configSlice)

        self.hfm = hfm
        self.corr = []
        self.irreps = np.transpose(la.orth(isle.Matrix(hfm.kappaTilde())))
        self.species = species
        self._alpha = alpha

    def __call__(self, phi, action, itr):
        """!Record the single-particle correlators."""

        nt = int(len(phi) / self.hfm.nx())

        # Create a large set of sources:
        rhss = [isle.Vector(spaceToSpacetime(irrep, time, nt))
                for irrep in self.irreps for time in range(nt)]
        # For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        # In other words, time is the faster running index.

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

def read(h5group):
    r"""!
    Read the irreps and their correlators from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["correlators"][()], h5group["irreps"][()]
