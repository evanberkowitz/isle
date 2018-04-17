"""!
Measurement of single-particle correlator.
"""

import numpy as np
import cns

from .common import newAxes
from ..util import binnedArray

class Corr_1p:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, eigenstates, nt, kappaTilde, mu, SIGMA_KAPPA):
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.numEigenstates = len(eigenstates)
        nx = len(eigenstates[0])  # this should give the number of ions
        self.nt = nt           # number of timeslices of the problem
        self.corr = [[[] for t in range(nt)] for i in range(len(eigenstates))]  # array where correlators will be stored
        self.rhss = []  # make right-hand side(s)
        for eigenstate in eigenstates:
            for t in range(nt):
                vec = np.zeros(nx*self.nt, dtype=complex)
                vec[t*nx:(t+1)*nx] = eigenstate
                self.rhss.append(cns.Vector(vec))
        # for the t^th spacetime vector of the i^th state, go to self.rhss[i * nt + t]


    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""
        res = cns.solveM(self.hfm, phi, cns.Species.PARTICLE, self.rhss)  # this solves M.x = b for the different timeslices and eigenstates
        for i in range(self.numEigenstates):
            corr = [0 for t in range(self.nt)]
            for t0 in range(self.nt):
                for t in range(t0, t0+self.nt):
                    if(t < self.nt):
                        corr[t-t0] += np.vdot(self.rhss[i*self.nt + t], res[i*self.nt + t0])
                    elif(t >= self.nt):  # this takes into account the anti-periodic boundary conditions
                        corr[t-t0] -= np.vdot(self.rhss[i*self.nt + t-self.nt], res[i*self.nt + t0])
            for t in range(self.nt):  # now average over initial timeslice and append to correlator
                corr[t] /= self.nt
                self.corr[i][t].append(corr[t])
            
                

