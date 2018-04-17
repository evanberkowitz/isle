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

def ensureH5GroupExists(theFile, group):
    r"""!
    Ensure a path in an HDF5 file exists.
    The `tables` module lets you `create_group` but only if the parent group already exists.
    Sometimes you want to create a group deep down in a hierarchy that may only partially exist.
    \param theFile An open HDF5 file where you wish `/a/group/like/this` to exist.
    \param group A string of `/slash/delimited/groups/with/no/trailing/slash`.
    """
    if group[0] != "/":
        group = "/" + group

    gs = group.split("/")
    paths = ["/" + "/".join(gs[1:i]) for i in range(1, len(gs))]

    for prev, new in zip(paths, gs[1:]):
        if prev+"/"+new not in theFile:
            theFile.create_group(prev, new)
