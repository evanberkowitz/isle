"""!
Measurement of single-particle correlator.
"""

import numpy as np

import cns
from .common import newAxes
from ..util import spaceToSpacetime, rotateTemporally
from ..h5io import createH5Group

class SingleParticleCorrelator:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, nt, kappaTilde, mu, SIGMA_KAPPA, species=cns.Species.PARTICLE):
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.nt = nt           # number of timeslices of the problem
        self.corr = []
        self.irreps = np.transpose(np.linalg.eig(cns.Matrix(kappaTilde))[1])
        self.species = species

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""

        # Create a large set of sources:
        rhss = [cns.Vector(spaceToSpacetime(irrep, time, self.nt)) for irrep in self.irreps for time in range(self.nt)]
        # For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        # In other words, time is faster.

        # Solve M.x = b for different right-hand sides:
        res = np.array(cns.solveM(self.hfm, phi, self.species, rhss), copy=False)

        propagators = res.reshape([len(self.irreps), self.nt, self.nt, len(self.irreps)])
        # The logic for the one-liner goes as:
        # For each source irrep we need to apply a sink for every irrep.  This produces a big cross-correlator.
        # For each source time we need to rotate so that the source lives on timeslice 0.
        # Finally, we need to average over all the source correlators with the same source irrep but different timeslices.
        self.corr.append(np.mean(
            np.array([[[
                rotateTemporally(np.dot(propagators[i, src], np.conj(self.irreps[j])), -src)
                for src in range(self.nt)]
                for i in range(len(self.irreps))]
                for j in range(len(self.irreps))]), axis=2))

    def report(self, ax=None):
        r"""!
        Produce a log-scale plot of the correlation functions.
        """

        if ax is None:
            fig, ax = newAxes(str(self.species)+" Correlator", r"t", r"C")
            doTightLayout = True

        correlator = np.array(self.corr)

        timeSlice = range(self.nt)
        avg = np.mean(np.real(correlator), axis=0)
        err = np.std(np.real(correlator), axis=0)

        ax.set_yscale("log")

        for i in range(len(self.irreps)):
            ax.errorbar(timeSlice, avg[i], yerr=err[i])

        if doTightLayout:
            fig.tight_layout()

        return ax

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
