"""!
Functions common to all measurements.
"""

import matplotlib.pyplot as plt

def newAxes(title, xlabel, ylabel):
    """!Make a new axes with given title and axis labels."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax
