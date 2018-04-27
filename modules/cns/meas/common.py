"""!
Functions common to all measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def newAxes(title, xlabel, ylabel):
    """!Make a new axes with given title and axis labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def oneDimKDE(dat, bandwidth=0.2, nsamples=1024, kernel="gaussian"):
    """!
    Perform a 1D kenrel density estimation on some data.
    \returns Tuple of sampling points and densities.
    """

    # make 2D array shape (len(totalPhi), 1)
    twoDDat = np.array(dat)[:, np.newaxis]
    # make 2D set of sampling points
    samplePts = np.linspace(np.min(dat)*1.1, np.max(dat)*1.1, nsamples)[:, np.newaxis]
    # estimate density
    dens = np.exp(KernelDensity(kernel=kernel, bandwidth=bandwidth)
                  .fit(twoDDat)
                  .score_samples(samplePts))
    return samplePts[:, 0], dens
