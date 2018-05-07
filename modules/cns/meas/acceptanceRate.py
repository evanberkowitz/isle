"""!
Measurement of acceptance rate.
"""

import numpy as np

from .common import newAxes
from ..util import binnedArray
from ..h5io import createH5Group

class AcceptanceRate:
    r"""!
    \ingroup meas
    Measure the acceptance rate.
    """

    def __init__(self):
        self.accRate = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record acceptance rate."""
        if not inline:
            raise RuntimeError("Cannot call AcceptanceRate measurement out of line")
        self.accRate.append(kwargs["acc"])

    def report(self, binsize=1, ax=None, fmt=""):
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
            fig, ax = newAxes("Acceptance Rate, total = {:3.2f}%".format(np.mean(binned)*100),
                              r"$N_{\mathrm{tr}}$", "Acceptance Rate")
            doTightLayout = True

        # plot acceptance rate
        t = np.arange(0, len(self.accRate), binsize)
        ax.plot(t, binned,
                fmt, label="Acceptance Rate")
                
        # Plot cumulative average
        cumulative = np.cumsum(binned) / (1 + np.arange(len(binned)))
        ax.plot(t, cumulative)
                
        ax.set_ylim((-0.05, 1.05))
        if doTightLayout:
            fig.tight_layout()

        return ax

    def save(self, base, name):
        r"""!
        Write the acceptance rate to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["acceptance"] = self.accRate

    def read(self, group, fromCfgFile=False):
        r"""!
        Read the acceptance rate from a file.
        \param group HDF5 group which contains the data of this measurement.
        \param fromCfgFile
               - `True`: Read data from a configuration file as written by
                          cns.meas.WriteConfiguration.
               - `False`: Read data from a single dataset as written by this measurement.
        """
        try:
            if fromCfgFile:
                self.accRate = [group[cfg]["acceptance"][()] for cfg in group]
            else:
                self.accRate = group["acceptance"][()]
        except KeyError:
            raise RuntimeError("No dataset 'acceptance' found in group {} in file {}"\
                               .format(group.name, group.file)) from None
