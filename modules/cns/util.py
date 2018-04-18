"""!
Some general utilities.
"""

import numpy as np
import h5py as h5

def hingeRange(start, end, stepSize):
    r"""!
    A generator that behaves similarly to the builtin range with two differences:
    `%hingeRange`
      - keeps going after reaching the end value and yields `end` forever and
      - accepts floats as parameters.

    \param start Start value (inclusive).
    \param end End value (exculsive if called only `int((end-start)/stepSize)` times).
    \param stepSize Size of the steps while iterating from start to end.
                    Must be negative if `end < start`.
    """

    cur = start  # current value
    # iterate from start to end
    while (stepSize > 0 and cur < end) or (stepSize < 0 and cur > end):
        yield cur
        cur += stepSize
    # stay at end value forever
    while True:
        yield end

def binnedArray(data, binsize):
    r"""!
    Return a binned array by averaging over subarrays of the input.

    \param data Input iterable. Needs to be convertible to a 1D numpy array.
    \param binsize Width of the bins. The length of `data` must be divisible by binsize
                   without remainder.
    """
    nbins = len(data) // binsize
    if len(data) % binsize != 0:
        raise RuntimeError("Size of data is not a multiple of the bin size")
    return np.reshape(data, (nbins, binsize)).mean(1)

def createH5Group(base, name):
    r"""!
    Create a new HDF5 group if it does not yet exist.
    \param base H5 group in which to create the new group.
    \param name Name of the new group relative to base.
    \returns The (potentially newly created) group.
    """

    if name in base:
        if isinstance(base[name], h5.Group):
            return base[name] # there is already a group with that name
        # something else than a group with that name
        raise ValueError(("Cannot create group '{}', another object with the same"\
                         +" name already exists in '{}/{}'").format(name, base.filename, base.name))
    # does not exists yet
    return base.create_group(name)

def spaceToSpacetime(vector, time, nt):
    r"""!
    Take a spatial vector and a timeslice and return a spacetime vector.
    \param vector The spatial vector.
    \param time The timeslice on which the wall vector should live.
    \param nt The number of total timeslices.
    """

    nx=len(vector)
    spacetimeVector = np.zeros(nx*nt, dtype=complex)
    spacetimeVector[time*nx:(time+1)*nx] = vector
    return spacetimeVector

def rotateTemporally(spacetimeVector, space, time, fermionic=True):
    result = np.roll(spacetimeVector.reshape([space,time]),time)
    mask = np.array([ [-1 for x in range(space)] if t < 3 else [+1 for x in range(space)] for t in range(time) ])
    result *= mask
    result = result.reshape([space*time])
    return result
