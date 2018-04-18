"""!
Measurement of single-particle correlator.
"""

import numpy as np
import cns

from .common import newAxes, ensureH5GroupExists
from ..util import spaceToSpacetime

class SingleParticleCorrelator:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, irreps, nt, kappaTilde, mu, SIGMA_KAPPA, species=cns.Species.PARTICLE):
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.numIrreps = len(irreps)
        self.nt = nt           # number of timeslices of the problem
        self.corr = [[[] for t in range(nt)] for i in range(len(irreps))]  # array where correlators will be stored
        self.irreps = irreps
        self.species=species

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""
        nx = len(self.irreps[0])  # this should give the number of ions
        
        # Create a large set of sources:
        rhss = [ cns.Vector(spaceToSpacetime(irrep, time, self.nt)) for irrep in self.irreps for time in range(self.nt) ]
        # For the j^th spacetime vector of the i^th state, go to self.rhss[i * nt + j]
        # In other words, time is faster.

        # Solve M.x = b for different right-hand sides:
        res = np.array(cns.solveM(self.hfm, phi, self.species, rhss))#.reshape([self.numIrreps, self.nt, self.numIrreps, self.nt])
        
        print(res.shape)
        r = res.reshape([self.numIrreps, self.nt, self.numIrreps, self.nt])
        
        [ ]

        print(r[0,0].shape)

        evancorr = [ np.vdot(self.irreps[i], r[i,time]) for i in range(self.numIrreps) for time in range(self.nt) ]
        print(evancorr.shape)
        exit()
        
        for i in range(self.numIrreps):
            corr = [0 for time in range(self.nt)]
            for t0 in range(self.nt):
                for time in range(t0, t0+self.nt):
                    if time < self.nt:
                        corr[time-t0] += np.vdot(rhss[i*self.nt + time], res[i*self.nt + t0])
                    elif time >= self.nt:  # this takes into account the anti-periodic boundary conditions
                        corr[time-t0] -= np.vdot(rhss[i*self.nt + time-self.nt], res[i*self.nt + t0])
            for time in range(self.nt):  # now average over initial timeslice and append to correlator
                corr[time] /= self.nt
                self.corr[i][time].append(corr[time])


    def report(self, ax=None):
        r"""!
        Produce a log-scale plot of the correlation functions.
        """

        if ax is None:
            fig, ax = newAxes(str(self.species)+" Correlator", r"t", r"C")
            doTightLayout = True

        correlator = np.array(self.corr)

        timeSlice = range(self.nt)
        avg = np.mean(np.real(correlator), axis=2)
        err = np.std(np.real(correlator), axis=2)

        ax.set_yscale("log")

        for mean, stdDev in zip(avg, err):
            ax.errorbar(timeSlice, mean, yerr=stdDev)

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
