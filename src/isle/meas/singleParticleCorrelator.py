"""!
Measurement of single-particle correlator.
"""

import numpy as np

import isle
from ..util import spaceToSpacetime, rollTemporally
from ..h5io import createH5Group

class SingleParticleCorrelator:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    def __init__(self, hfm, species, alpha):
        self.hfm = hfm
        self.corr = []
        self.irreps = np.transpose(np.linalg.eig(isle.Matrix(hfm.kappa()))[1])
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

    def save(self, base, name):
        r"""!
        Write the irreps and their correlators to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["correlators"] = self.corr
        group["irreps"] = self.irreps

    def read(self, group):
        r"""!
        Read the irreps and their correlators from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.corr = group["correlators"][()]
        self.irreps = group["irreps"][()]
