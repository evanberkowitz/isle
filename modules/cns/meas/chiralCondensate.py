"""!
Measurement of single-particle correlator.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import cns
from .common import newAxes
from ..util import spaceToSpacetime, rotateTemporally
from ..h5io import createH5Group


class ChiralCondensate:
    r"""!
    \ingroup meas
    Tabulate single-particle correlator
    """

    def __init__(self, seed, samples, nt, kappaTilde, mu, SIGMA_KAPPA, species=cns.Species.PARTICLE):
        self.samples = samples
        self.seed = seed
        self.rng = cns.random.NumpyRNG(seed)
        self.nt = nt           # number of timeslices of the problem
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.nx = len(np.array(cns.Matrix(kappaTilde)))
        self.species = species
        self.cc = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the single-particle correlators."""

        # Create a large set of sources:
        rhss = [cns.Vector(self.rng.normal(0,1,self.nt*self.nx)+0j) for s in range(self.samples)]

        # Solve M.x = b for different right-hand sides,
        # Normalize by spacetime volume
        res = np.array(cns.solveM(self.hfm, phi, self.species, rhss), copy=False) / (self.nx * self.nt)

        self.cc.append(np.mean([ np.dot(rhss[s], res[s]) for s in range(self.samples)]))

    def report(self, ax=None):
        spacer = 0.05
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.8
        
        fig = plt.figure()
        nullfmt = NullFormatter()
        
        history = plt.axes([left, bottom, width, height])
        dist    = plt.axes([left+width+spacer, bottom, 1-left-width-2*spacer, height])
        
        history.set_title(r"Monte Carlo History of Chiral Condensate")
        history.set_xlabel(r"$N_{\mathrm{tr}}$")
        history.set_ylabel(r"$\Phi$")
        history.plot(np.arange(len(self.cc)), np.real(self.cc), color='magenta', alpha=0.75)
        
        ylimits = history.get_ylim()
        
        dist.set_title(r"PDF")
        dist.set_xlabel(r"Freq.")
        dist.hist(np.real(self.cc), 50, normed=1, facecolor='magenta', alpha=0.75, orientation="horizontal")
        dist.yaxis.set_major_formatter(nullfmt)
        dist.set_ylim(ylimits)

        return fig

    def save(self, base, name):
        r"""!
        Write the chiral condensate
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["chiralCondensate"] = self.cc

    def read(self, group):
        r"""!
        Read the irreps and their correlators from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.cc = group["chiralCondensate"][()]
