"""!
Measurement of action.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


from .common import newAxes, oneDimKDE
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

    def report(self, axes=None, **kwargs):
        if axes is None:
            spacer = 0.05
            left, width = 0.1, 0.5
            bottom, height = 0.1, 0.8

            fig = plt.figure()
            histax = plt.axes([left, bottom, width, height])
            distax = plt.axes([left+width+spacer, bottom, 1-left-width-2*spacer, height])
            doTightLayout = False
        
        else:  # use Axes that were passed in
            fig = plt.gcf()
            histax, distax = axes
            doTightLayout = False
        
        # history
        self.plotMCHistory(histax, alpha=0.75)
        histax.set_title(r"Monte Carlo History of $\theta$")
        histax.set_xlabel(r"$N_{\mathrm{tr}}$")
        histax.set_ylabel(r"$\theta$")
        histax.tick_params(right=True)

        # 2D histogram
        self.plotHist2D(distax)
        distax.set_title(r"Histogram of $e^{-i\theta}$")

        if doTightLayout:
            fig.tight_layout()
        return (histax, distax)
        

    def plotMCHistory(self, ax=None, binsize=1, color="orange", **kwargs):
        r"""!
        Plot monte carlo history of the phase.
        \param binsize The phase is averaged over `binsize` trajectories.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param color Line color of the plot.
        \param kwargs Passed to `matplotlib.Axes.plot`.
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
        if "label" in kwargs:
            ax.plot(np.arange(len(self.theta)), np.real(self.theta), color=color, **kwargs)
        else:
            ax.plot(np.arange(len(self.theta)), np.real(self.theta),
                    label=r"$\theta", color=color, **kwargs)

        if doTightLayout:
            fig.tight_layout()
        return ax
    
    def plotDistribution(self, ax=None, hist=True, kde=True, orientation="vertical", facecolor="orange", color="black", **kwargs):
        
        r"""!
        Plot histogram of complex part of the action.
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
        
        theta = self.theta
        
        if hist:    # MC history
            ax.hist(theta, 50, normed=1, facecolor=facecolor, orientation=orientation, label="histogram", **kwargs)
        
        if kde:     # kernel density estimation
            samplePts, dens = oneDimKDE(theta, bandwidth=3/5)
            if orientation == "vertical":
                ax.plot(samplePts, dens, color=color, label="kde", **kwargs)
            elif orientation == "horizontal":
                ax.plot(dens, samplePts, color=color, label="kde", **kwargs)
            else:
                raise RuntimeError(f"Uknown orientation: {orientation}")
            ax.set_xlim((0, np.max(dens)*1.1))
        
        if doTightLayout:
            fig.tight_layout()
        
        return ax

    def plotHist2D(self, ax=None, **kwargs):
        
        r"""!
        Plot histogram of complex part of the action.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \param kwargs Passed to `matplotlib.Axes.hist2d` <B>and</B> `matplotlib.Axes.plot`.
        \returns The Axes with the plot.
        """
        
        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("Weight Histogram", "", "", figsize=(10,10))
            doTightLayout = True
        
        theta = np.array(self.theta)
        weight = np.exp(-theta*1j)
        x = np.real(weight)
        y = np.imag(weight)
        
        view = [-1.05, 1.05]
        h = ax.hist2d(x, y, bins=51, range=[view, view], normed=1, label="histogram", **kwargs)
        ax.set_xlim(view)
        ax.set_ylim(view)
        ax.set_aspect(1)
        plt.colorbar(h[3])
        
        # Indicate the average weight
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ax.scatter([x_mean], [y_mean], c="w", )
        
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
