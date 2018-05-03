"""!
Measurement of action.
"""

import numpy as np

from .common import newAxes
from ..util import binnedArray
from ..h5io import createH5Group

class Phase:
    r"""!
    \ingroup meas
    Measure the complex phase from the action.
    """

    def __init__(self):
        self.theta = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record phase."""
        self.theta.append(np.imag(kwargs["act"]))

    def report(self, binsize=1, ax=None, fmtre="", fmtim=""):
        r"""!
        Plot the phase against Monte Carlo time.
        \param binsize The phase is averaged over `binsize` trajectories.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmtre Plot format passed to matplotlib.
                     Can encode color, marker and line styles of real part.
        \param fmtim Plot format passed to matplotlib.
                     Can encode color, marker and line styles of imaginary part.
        \returns The Axes with the plot.
        """

        doTightLayout = False
        binned = binnedArray(self.theta, binsize)
        if ax is None:
            fig, ax = newAxes(r"Phase: global mean of <$\theta$> = {:3.5f} +/- {:3.5f}".format(np.mean(binned),
                                                                                               np.std(binned)),
                              r"$N_{\mathrm{tr}}$", r"$S$")
            doTightLayout = True

        mcTime = np.arange(0, len(self.theta), binsize)
        ax.plot(mcTime, np.real(binned), fmtre, label=r"$\theta$")
        ax.legend()
        if doTightLayout:
            fig.tight_layout()
        return ax

    def save(self, base, name):
        r"""!
        Write the phase to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["theta"] = self.theta

    def read(self, group):
        r"""!
        Read the phase from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.theta = group["theta"][()]
