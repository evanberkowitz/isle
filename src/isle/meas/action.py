"""!
Measurement of action.
"""

import numpy as np

from .common import newAxes
from ..util import binnedArray
from ..h5io import createH5Group

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

    def report(self, binsize=1, ax=None, fmtre="", fmtim=""):
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
        binned = binnedArray(self.act, binsize)
        if ax is None:
            fig, ax = newAxes(r"Action: global mean of <$S$> = {:3.5f}+{:3.5f}*I+/-{:3.5f}+{:3.5f}*I".format(np.mean(np.real(binned)),
                                                                                                             np.mean(np.imag(binned)),
                                                                                                             np.std(np.real(binned)),
                                                                                                             np.std(np.imag(binned))),
                              r"$N_{\mathrm{tr}}$", r"$S$")
            doTightLayout = True

        mcTime = np.arange(0, len(self.act), binsize)
        ax.plot(mcTime, np.real(binned), fmtre, label="Re($S$)")
        ax.plot(mcTime, np.imag(binned), fmtim, label="Im($S$)")
        ax.legend()
        if doTightLayout:
            fig.tight_layout()
        return ax

    def save(self, base, name):
        r"""!
        Write the action to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["action"] = self.act

    def read(self, group, fromCfgFile=False):
        r"""!
        Read the action from a file.
        \param group HDF5 group which contains the data of this measurement.
        \param fromCfgFile
               - `True`: Read data from a configuration file as written by
                          isle.meas.WriteConfiguration.
               - `False`: Read data from a single dataset as written by this measurement.
        """
        try:
            if fromCfgFile:
                self.act = [group[cfg]["action"][()] for cfg in group]
            else:
                self.act = group["action"][()]
        except KeyError:
            raise RuntimeError("No dataset 'action' found in group {} in file {}"\
                               .format(group.name, group.file)) from None
