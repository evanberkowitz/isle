"""!
Measurement of single-particle correlator.
"""

import numpy as np
import cns

from .common import newAxes, ensureH5GroupExists
from ..util import binnedArray

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
            for t in range(self.nt):
                vec = np.zeros(nx*self.nt, dtype=complex)
                vec[t*nx:(t+1)*nx] = eigenstate
                rhss.append(cns.Vector(vec))
        # for the t^th spacetime vector of the i^th state, go to self.rhss[i * nt + t]

        res = cns.solveM(self.hfm, phi, cns.Species.PARTICLE, rhss)  # this solves M.x = b for the different timeslices and eigenstates
        for i in range(self.numIrreps):
            corr = [0 for t in range(self.nt)]
            for t0 in range(self.nt):
                for t in range(t0, t0+self.nt):
                    if(t < self.nt):
                        corr[t-t0] += np.vdot(rhss[i*self.nt + t], res[i*self.nt + t0])
                    elif(t >= self.nt):  # this takes into account the anti-periodic boundary conditions
                        corr[t-t0] -= np.vdot(rhss[i*self.nt + t-self.nt], res[i*self.nt + t0])
            for t in range(self.nt):  # now average over initial timeslice and append to correlator
                corr[t] /= self.nt
                self.corr[i][t].append(corr[t])
            
                
    def report(self, ax=None):
        if ax is None:
            fig, ax = newAxes("Correlator", r"t", r"C")
            doTightLayout = True
        
        c = np.array(self.corr)
        
        x = range(self.nt)
        # x=[ range(self.nt) for i in range(self.numEigenstates) ]
        avg = np.mean(c, axis=2)
        err = np.std(c, axis=2)
        
        ax.set_yscale("log")
        
        for m,s in zip(avg,err):
            ax.errorbar(x, m, yerr=s)
            
    def save(self, the_file, path):
        ensureH5GroupExists(the_file, path)
        the_file.create_array(path, "correlators", np.array(self.corr))
        the_file.create_array(path, "irreps", self.irreps)
    
    def read(self, the_file, path):
        self.corr = the_file.get_node(path+"/correlators").read()
        self.irreps = the_file.get_node(path, "irreps").read()