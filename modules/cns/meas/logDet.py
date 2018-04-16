"""!
Measurement of logDet.
"""

import numpy as np
import cns
from matplotlib.colors import LogNorm

from .common import newAxes

class LogDet:
    r"""!
    \ingroup meas
    Measure the log determinant of particles and holes.
    """

    def __init__(self, kappaTilde, mu, SIGMA_KAPPA):
        self.hfm = cns.HubbardFermiMatrix(kappaTilde, mu, SIGMA_KAPPA)
        self.logDet = {cns.PH.PARTICLE: [], cns.PH.HOLE: []}

    def __call__(self, phi, inline=False, **kwargs):
        """!Record logDet."""
        for species in self.logDet:
            self.logDet[species].append(cns.logdetM(self.hfm, phi, species))

    def report(self, species=cns.PH.PARTICLE, binsize=41, ax=None):
        r"""!
        Plot the determinant in the complex plane
        \param species a `cns.PH` species that identifies the determinant of interest.
        \param binzise Number of bins in each direction.
                       An odd number highlights the avoidance of 0 determinant.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \returns The Axes with the plot.
        """

        if ax is None:
            fig, ax = newAxes(str(species)+" Determinant", r"Re($det$)", r"Im($det$)")
            doTightLayout = True

        dets = np.exp(self.logDet[species])

        ax.hist2d(np.real(dets), np.imag(dets), bins=binsize, norm=LogNorm())
        ax.set_aspect('equal')

        if doTightLayout:
            fig.tight_layout()
        return ax
