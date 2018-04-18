"""!
Measurement of single-particle correlator.
"""

import numpy as np
import cns

from .common import newAxes, ensureH5GroupExists

class Corr_1p:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, irreps, nt, kappaTilde, mu, SIGMA_KAPPA):
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.numIrreps = len(irreps)
        self.nt = nt           # number of timeslices of the problem
        self.corr = [[[] for t in range(nt)] for i in range(len(irreps))]  # array where correlators will be stored
        self.irreps = irreps

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""
        nx = len(self.irreps[0])  # this should give the number of ions
        rhss = []
        for eigenstate in self.irreps:
            for time in range(self.nt):
                vec = np.zeros(nx*self.nt, dtype=complex)
                vec[time*nx:(time+1)*nx] = eigenstate
                rhss.append(cns.Vector(vec))
        # for the t^th spacetime vector of the i^th state, go to self.rhss[i * nt + t]

        res = cns.solveM(self.hfm, phi, cns.Species.PARTICLE, rhss)  # this solves M.x = b for the different timeslices and eigenstates
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
            fig, ax = newAxes("Correlator", r"t", r"C")
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
