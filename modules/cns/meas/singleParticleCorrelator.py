"""!
Measurement of single-particle correlator.
"""

import numpy as np
import cns

from .common import newAxes, ensureH5GroupExists
from ..util import spaceToSpacetime, rotateTemporally

class SingleParticleCorrelator:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, nt, kappaTilde, mu, SIGMA_KAPPA, species=cns.Species.PARTICLE):
        
        noninteracting_energies, irreps = np.linalg.eig(cns.Matrix(kappaTilde))
        irreps = np.transpose(irreps)
        
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.numIrreps = len(irreps)
        self.nt = nt           # number of timeslices of the problem
        self.corr = [] #[[[] for t in range(nt)] for i in range(len(irreps))]  # array where correlators will be stored
        self.irreps = irreps
        self.species = species

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""
        
        # Create a large set of sources:
        rhss = [cns.Vector(spaceToSpacetime(irrep, time, self.nt)) for irrep in self.irreps for time in range(self.nt)]
        # For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        # In other words, time is faster.
        
        # Solve M.x = b for different right-hand sides:
        res = np.array(cns.solveM(self.hfm, phi, self.species, rhss))#.reshape([self.numIrreps, self.nt, self.numIrreps, self.nt])
        
        propagators = res.reshape([self.numIrreps, self.nt, self.nt, self.numIrreps])
        # The logic for the one-liner goes as:
        # sunk = np.array([ [ np.dot(propagators[i,src],np.conj(self.irreps[i])) for src in range(self.nt) ] for i in range(len(self.irreps)) ])
        # rotated = np.array([ [ rotateTemporally(sunk[i,src], -src) for src in range(self.nt) ] for i in range(self.numIrreps) ])
        # sourceAveraged = np.mean(rotated, axis=1)
        self.corr.append(np.mean(
            np.array([[ 
                rotateTemporally(np.dot(propagators[i, src], np.conj(self.irreps[i])), -src)
                for src in range(self.nt)] 
                for i in range(self.numIrreps)]), axis=1))

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

        for i in range(self.numIrreps):
            ax.errorbar(timeSlice, avg[i], yerr=err[i])

        if doTightLayout:
            fig.tight_layout()

        return ax

    def save(self, theFile, path):
        r"""!
        Write the irreps and their correlators to a file.
        \param theFile An open HDF5 file.
        \param path Where to write to in theFile
        """
        ensureH5GroupExists(theFile, path)
        theFile.create_array(path, "correlators", np.array(self.corr))
        theFile.create_array(path, "irreps", self.irreps)

    def read(self, theFile, path):
        r"""!
        Read the irreps and their correlators from a file.
        \param theFile An open HDF5 file.
        \param path Where to read from in theFile.
        """
        self.corr = theFile.get_node(path+"/correlators").read()
        self.irreps = theFile.get_node(path, "irreps").read()
