"""!
Measurement of total phi and norm of phi.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from .common import newAxes, oneDimKDE
from ..util import binnedArray
from ..h5io import createH5Group

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

    def report(self):
        """!Calls TotalPhi.reportPhISq()."""
        return self.reportTotalPhi()

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

    def reportTotalPhi(self, axes=None):
        r"""!
        Plot \f$\Phi = \sum \phi\f$.
        \param axes Tuple of two axes to draw history and distribution of \f$\Phi\f$ in.
                    If `None`, a new figure is created.
        \returns Tuple of axes objects for history and distribution plots.
        """

        if axes is None:  # need to make new Axes
            spacer = 0.05
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.8

            fig = plt.figure()
            histax = plt.axes([left, bottom, width, height])
            distax = plt.axes([left+width+spacer, bottom, 1-left-width-2*spacer, height])
            doTightLayout = False
        else:  # use Axes that were passed in
            fig = plt.gcf()
            histax, distax = axes
            doTightLayout = True

        # history
        self.plotPhi(histax, alpha=0.75)
        histax.set_title(r"Monte Carlo History of $\Phi$")
        histax.set_xlabel(r"$N_{\mathrm{tr}}$")
        histax.set_ylabel(r"$\Phi$")
        histax.tick_params(right=True)

        # density
        self.plotPhiDistribution(distax, orientation="horizontal", alpha=0.75)
        distax.set_title(r"PDF")
        distax.set_xlabel(r"Freq.")
        distax.yaxis.set_major_formatter(NullFormatter())
        distax.set_ylim(histax.get_ylim())

        if doTightLayout:
            fig.tight_layout()
        return (histax, distax)

    def plotPhiDistribution(self, ax=None, hist=True, kde=True, orientation="vertical",
                            facecolor="green", color="black", **kwargs):
        r"""!
        Plot histogram of summed Phi.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param hist Choose whether to show histogram.
        \param kde Choose whether to show kernel densitiy estimation.
        \param orientation Set to `"horizontal"` to rotate plot 90°.
        \param facecolor Color of the histogram.
        \param color Line color of kernel density estimate.
        \param kwargs Passed to `matplotlib.Axes.hist` <B>and</B> `matplotlib.Axes.plot`.
        \returns The Axes with the plot.
        """

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("", r"$\Phi$", r"PDF")
            doTightLayout = True

        totalPhi = np.real(self.Phi)  # get real

        if hist:  # history
            ax.hist(totalPhi, 50, normed=1, facecolor=facecolor,
                    orientation=orientation, label="histogram", **kwargs)

        if kde:  # kernel density estimation
            samplePts, dens = oneDimKDE(totalPhi, bandwidth=3/5)
            if orientation == "vertical":
                ax.plot(samplePts, dens, color=color, label="kde", **kwargs)
            elif orientation == "horizontal":
                ax.plot(dens, samplePts, color=color, label="kde", **kwargs)
            else:
                raise RuntimeError(f"Unknown orientation: {orientation}")
            ax.set_xlim((0, np.max(dens)*1.1))

        if doTightLayout:
            fig.tight_layout()
        return ax

    def plotPhi(self, ax=None, color="green", **kwargs):
        r"""!
        Plot monte carlo history of total Phi.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param color Line color of the plot.
        \param kwargs Passed to `matplotlib.Axes.plot`.
        \returns The Axes with the plot.
        """

        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("", r"$N_{\mathrm{tr}}$", r"$\Phi$")
            doTightLayout = True

        if "label" in kwargs:
            ax.plot(np.arange(len(self.Phi)), np.real(self.Phi), color=color, **kwargs)
        else:
            ax.plot(np.arange(len(self.Phi)), np.real(self.Phi),
                    label=r"\Phi($i_{\mathrm{tr}})$", color=color, **kwargs)

        if doTightLayout:
            fig.tight_layout()
        return ax

    def save(self, base, name):
        r"""!
        Write both Phi and phiSquared.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["Phi"] = self.Phi
        group["phiSquared"] = self.phiSq

    def read(self, group):
        r"""!
        Read Phi and phiSquared from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.Phi = group["Phi"][()]
        self.phiSq = group["phiSquared"][()]
