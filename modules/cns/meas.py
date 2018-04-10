"""!
Various basic measurements.
"""

## \defgroup meas Measurements
# Perform measurements on configurations.

import numpy as np
import matplotlib.pyplot as plt

from .util import binnedArray

def _newAxes(title, xlabel, ylabel):
    """Make a new axes with given title and axis labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


class AcceptanceRate:
    r"""!
    \ingroup meas
    Measure the acceptance rate.
    """

    def __init__(self):
        self.accRate = []

    def __call__(self, itr, phi, act, acc):
        """!Record acceptance rate."""
        self.accRate.append(acc)

    def report(self, binsize, ax=None, fmt=""):
        r"""!
        Plot the acceptance rate against Monte Carlo time.
        \param binsize The acceptance rate is averaged over `binsize` trajectories.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmt Plot format passed to matplotlib. Can encode color, marker and line styles.
        \returns The Axes with the plot.
        """

        binned = binnedArray(self.accRate, binsize)

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = _newAxes("Acceptance Rate, total = {:3.2f}%".format(np.mean(binned)*100),
                               r"$N_{\mathrm{tr}}$", "Acceptance Rate")
            doTightLayout = True

        # plot acceptance rate
        ax.plot(np.arange(0, len(self.accRate), binsize), binned,
                fmt, label="Acceptance Rate")
        if doTightLayout:
            fig.tight_layout()

        return ax


class Action:
    r"""!
    \ingroup meas
    Measure the action.
    """

    def __init__(self):
        self.act = []

    def __call__(self, itr, phi, act, acc):
        """!Record action."""
        self.act.append(act)

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
            fig, ax = _newAxes("Action", r"$N_{\mathrm{tr}}$", r"$S$")
            doTightLayout = True

        binned = binnedArray(self.act, binsize)
        ax.plot(np.arange(0, len(self.act), binsize), np.real(binned), fmtre, label="Re($S$)")
        ax.plot(np.arange(0, len(self.act), binsize), np.imag(binned), fmtim, label="Im($S$)")
        ax.legend()
        if doTightLayout:
            fig.tight_layout()
        return ax
