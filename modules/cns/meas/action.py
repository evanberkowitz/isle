"""!
Measurement of action.
"""

import numpy as np

from .common import newAxes
from ..util import binnedArray

class Action:
    r"""!
    \ingroup meas
    Measure the action.
    """

    def __init__(self):
        self.act = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record action."""
        self.act.append(kwargs["act"])

    def report(self, binsize, ax=None, fmtre="", fmtim=""):
        r"""!
        Plot the action against Monte Carlo time.
        \param binsize The action is averaged over `binsize` trajectories.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmtre Plot format passed to matplotlib.
                     Can encode color, marker and line styles of real part.
        \param fmtim Plot format passed to matplotlib.
                     Can encode color, marker and line styles of imaginary part.
        \returns The Axes with the plot.
        """

        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("Action", r"$N_{\mathrm{tr}}$", r"$S$")
            doTightLayout = True

        binned = binnedArray(self.act, binsize)
        mcTime = np.arange(0, len(self.act), binsize)
        ax.plot(mcTime, np.real(binned), fmtre, label="Re($S$)")
        ax.plot(mcTime, np.imag(binned), fmtim, label="Im($S$)")
        ax.legend()
        if doTightLayout:
            fig.tight_layout()
        return ax
