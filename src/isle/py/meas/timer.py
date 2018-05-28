"""!
A crude timer
"""

import time
import numpy as np

import cns
from .common import newAxes
from ..util import binnedArray
from ..h5io import createH5Group

class Timer:
    r"""!
    \ingroup meas
    Capture the time between measurements.
    """

    def __init__(self):
        self.prev = None
        self.times = []

    def __call__(self, phi, inline=False, **kwargs):
        """!Record time."""
        if self.prev == None:
            self.prev = time.perf_counter()
        else:
            now = time.perf_counter()
            self.times.append(now - self.prev)
            self.prev = now

    def report(self, binsize=20, ax=None, fmt=""):
        r"""!
        Plot the time between calls against Monte Carlo time.
        \param binzise Number of bins in each direction.
                       An odd number highlights the avoidance of 0 determinant.
        \param ax Matplotlib Axes to plot in. If `None`, a new one is created in a new figure.
        \returns The Axes with the plot.
        """

        if ax is None:
            fig, ax = newAxes("Time Per Iteration", r"$N_{\mathrm{tr}}$", r"Time / seconds")
            doTightLayout = True

        binned = binnedArray(self.times, binsize)

        ax.plot(np.arange(0, len(self.times), binsize), binned, 
                fmt, color="magenta")

        if doTightLayout:
            fig.tight_layout()
        return ax
        
    def histogram(self, bins=50, ax=None):
        r"""!
        Plot a histogram of the iteration times.
        \param bins Number of bins.
        \param ax Matplotlib axes to plot in.  If `None`, a new one is created in a new figure.
        """
        
        # make a new axes is needed
        doTightLayout = False
        if ax is None:
            fig, ax = newAxes("Time Per Iteration", r"time / s", r"PDF")
            doTightLayout = True
        
        ax.hist(self.times, bins, normed=1, facecolor="magenta", alpha=0.75)
        
        return ax

    def save(self, base, name):
        r"""!
        Write the time difference between calls to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        group = createH5Group(base, name)
        group["time"] =  self.times

    def read(self, group):
        r"""!
        Read the time differences from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.times = group["time"]
        self.prev = None
