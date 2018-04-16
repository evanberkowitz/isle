"""!
Measurement of total phi and norm of phi.
"""

import numpy as np

from .common import newAxes
from ..util import binnedArray

class TotalPhi:
    r"""!
    \ingroup meas
    Tabulate phi and mean value of phi^2.
    """

    def __init__(self):
        self.Phi = []
        self.phiSq = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record the total phi and mean value of phi^2."""
        self.Phi.append(np.sum(phi))
        self.phiSq.append(np.linalg.norm(phi)**2 / len(phi))

    def reportPhiSq(self, binsize, ax=None, fmt=""):
        r"""!
        Plot the <phi^2> against Monte Carlo time.
        \param binsize The acceptance rate is averaged over `binsize` trajectories.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmt Plot format passed to matplotlib. Can encode color, marker and line styles.
        \returns The Axes with the plot.
        """

        binned = binnedArray(self.phiSq, binsize)

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes(r"global mean of <$\phi^2$> = {:3.5f}+/-{:3.5f}".format(np.mean(binned),
            np.std(binned)),
                              r"$N_{\mathrm{tr}}$", r"<$\phi^2$>($N_{\mathrm{tr}})$")
            doTightLayout = True

        # plot <phi^2>
        ax.plot(np.arange(0, len(self.phiSq), binsize), binned,
                fmt, label=r"$\langle\phi^2\rangle$($N_{\mathrm{tr}})$")
        ax.set_ylim(ymin=0)
        if doTightLayout:
            fig.tight_layout()

        return ax

    def reportPhiHistogram(self, ax=None):
        r"""!
        Plot histogram of summed Phi.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmt Plot format passed to matplotlib. Can encode color, marker and line styles.
        \returns The Axes with the plot.
        """

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("", r"$\Phi$", r"PDF")
            doTightLayout = True

        # the histogram of the data
        ax.hist(np.real(self.Phi), 50, normed=1, facecolor='green', alpha=0.75)

        ax.grid(True)
        if doTightLayout:
            fig.tight_layout()

        return ax

    def reportPhi(self, ax=None, fmt=""):
        r"""!
        Plot monte carlo history of summed Phi.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param fmt Plot format passed to matplotlib. Can encode color, marker and line styles.
        \returns The Axes with the plot.
        """

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("", r"$N_{\mathrm{tr}}$", r"$\Phi$")
            doTightLayout = True

        ax.plot(np.arange(len(self.Phi)), np.real(self.Phi), fmt,
                label=r"\Phi($i_{\mathrm{tr}})$")

        ax.grid(True)
        if doTightLayout:
            fig.tight_layout()

        return ax
