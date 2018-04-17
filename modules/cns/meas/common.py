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

def ensureH5GroupExists(the_file, group):
    if group[0] != "/":
        group = "/" + group
    
    gs = group.split("/")
    paths = [ "/" + "/".join(gs[1:i]) for i in range(1,len(gs)) ]

    for prev,new in zip(paths,gs[1:]):
        if prev+"/"+new not in the_file:
            the_file.create_group(prev,new)
