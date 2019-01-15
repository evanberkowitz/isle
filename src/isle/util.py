"""! \file
Some general utilities.
"""

from dataclasses import make_dataclass, field

import yaml
import numpy as np

from . import Lattice

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

def _splitVersionStr(ver):
    """!Split a version string into its components major, minor, and extra."""
    major, rest = ver.split(".", 1)
    try:
        minor, extra = rest.split("-", 1)
    except ValueError:
        minor, extra = rest, ""
    return int(major), int(minor), extra

def compareVersions(version0, version1):
    """!
    Compare to versions.

    \returns - "newer" if version0 is newer than version1
             - "older" if version0 is older than version1
             - "equal" if both are exactly equal
             - "none" if both major and minor versions are equal but the extra fields are different
    """

    v0major, v0minor, v0extra = _splitVersionStr(str(version0))
    v1major, v1minor, v1extra = _splitVersionStr(str(version1))

    diffMajor = v0major - v1major
    if diffMajor > 0:
        return "newer"
    if diffMajor < 0:
        return "older"
    # diffMajor == 0 here

    diffMinor = v0minor - v1minor
    if diffMinor > 0:
        return "newer"
    if diffMinor < 0:
        return "older"
    # diffMinor == 0 here

    if v0extra == v1extra:
        return "equal"
    return "none"  # no further comparison possible

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

def parameters(**kwargs):
    r"""!
    Return a dataclass instance holding the parameters passed to this function.

    Can take any arbitrary keyword arguments and makes a new dataclass
    to store them. Every time this function gets called, it creates a new
    class and immediately constructs an instance. The returned object is read only.

    The returned object has additional methods on top of the usual
    dataclass members:
     - <B>asdict</B>():
           Returns a dict mapping attribute names to values.
     - <B>tilde</B>(value, nt, beta):
           Return `value*beta/nt`.
           Parameters:
            - `value`: If `str`, read attribute with that name from dataclass,
                       else use argument directly.
            - `nt`: Number of time slices or isle.Lattice.
            - `beta`: Inverse temperature. Read from dataclass attribute 'beta'
                      if argument set to `None` (default).

    The new classes are automatically registered with YAML so they can
    be dumped. A loader is registered if yamlio is imported.
    """

    def _tilde(self, value, nt, beta=None):
        if beta is None:
            if not hasattr(self, "beta"):
                raise RuntimeError("No parameter 'beta' stored. Cannot compute tilde parameter.")
            beta = self.beta

        if isinstance(value, str):
            value = getattr(self, value)
        if isinstance(nt, Lattice):
            nt = nt.nt()

        return value*beta/nt

    cls = make_dataclass("Parameters",
                         ((key, type(value), field(default=value))
                          for key, value in kwargs.items()),
                         namespace={"asdict": lambda self:
                                              {key: getattr(self, key)
                                               for key in self.__dataclass_fields__.keys()},
                                    "tilde": _tilde},
                         frozen=True)

    yaml.add_representer(cls, lambda dumper, params:
                         dumper.represent_mapping("!parameters",
                                                  params.asdict(),
                                                  flow_style=False))
    return cls()

def spaceToSpacetime(vector, time, nt):
    r"""!
    Take a spatial vector and a timeslice and return a spacetime vector.
    \param vector The spatial vector.
    \param time The timeslice on which the wall vector should live.
    \param nt The number of total timeslices.
    """
    nx = len(vector)
    spacetimeVector = np.zeros(nx*nt, dtype=complex)
    spacetimeVector[time*nx:(time+1)*nx] = vector
    return spacetimeVector

def rollTemporally(timeVector, time, fermionic=True):
    r"""!
    Roll a time vector, accounting for anti/periodic boundary conditions.
    \param timeVector The vector to rotate.
    \param time The zero entry of `timeVector` will wind up as as time entry of the result.
    \param fermionic Account for antisymmetry in time.
    """
    result = np.roll(timeVector, time, axis=0)
    if fermionic:
        if time > 0:
            result[:time, ...] *= -1
        elif time < 0:
            nt = len(timeVector)
            result[nt+time:] *= -1
    return result
