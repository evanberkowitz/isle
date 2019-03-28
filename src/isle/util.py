"""! \file
Some general utilities.
"""

from logging import getLogger

try:
    # use dataclasses if available (Python3.7 or later)
    from dataclasses import make_dataclass, field, asdict
    _HAVE_DATACLASSES = True
except ImportError:
    # construct poor man's workaround
    _HAVE_DATACLASSES = False
    getLogger(__name__).info("Could not import dataclasses, using a workaround "
                             "for storing parameters")

import yaml
import numpy as np

from . import Lattice, isleVersion, pythonVersion, blazeVersion, pybind11Version, fileio

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

def _verifyVersion(current, other, name, fname, permissive):
    comp = compareVersions(current, other)
    if comp == "none":
        getLogger(__name__).info("Extra version string of %s (%s) different "
                                 "from version in file %s (%s)",
                                 name, current, fname, other)
    if comp != "equal":
        message = f"Version of {name} ({current}) is {comp} than in file {fname} ({other})."
        if permissive:
            getLogger(__name__).warning(message)
        else:
            getLogger(__name__).error(message)
            raise RuntimeError(f"Version mismatch for {name}")

def verifyVersionsByException(versions, fname, permissive=False):
    r"""!
    Compare versions of Isle and Python with current versions.
    Raise RuntimeError if the versions of Isle do not match.
    \param versions Dict of versions. Must contain 'isle' and 'python'.
    \param fname Name of the file `versions` have been read from.
    \param permissve If `True`, a mismatch only triggers a warning not an error.
    """
    _verifyVersion(isleVersion, versions["isle"], "isle", fname, permissive=permissive)
    _verifyVersion(pythonVersion, versions["python"], "Python", fname, permissive=True)

def verifyMetadataByException(fname, lattice, params):
    """!
    Make sure that metadata in file agrees with function parameters.
    Raises RuntimeError if there is a mismatch.
    """

    storedLattice, storedParams, _, versions = fileio.h5.readMetadata(fname)

    if storedLattice.name != lattice.name:
        getLogger(__name__).error("Name of lattice in output file is %s but new lattice has name %s. "
                                  "Cannot write into existing output file.",
                                  storedLattice.name, lattice.name)
        raise RuntimeError("Lattice name inconsistent")

    if storedParams.asdict() != params.asdict():
        getLogger(__name__).error("Stored parameters do not match new parameters. "
                                  "Cannot write into existing output file.")
        raise RuntimeError("Parameters inconsistent")

    verifyVersionsByException(versions, fname)


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


def _makeParametersClass(fields):
    r"""!
    Construct a new class for parameters. Uses either a dataclass or custom workaround.
    """
    def _tilde(self, value, nt, beta=None):
        if beta is None:
            # try to used stored beta
            if not hasattr(self, "beta"):
                raise RuntimeError("No parameter 'beta' stored. "
                                   "Pass it as parameter to tilde().")
            beta = self.beta

        # normalize parameters
        if isinstance(value, str):
            value = getattr(self, value)
        if isinstance(nt, Lattice):
            nt = nt.nt()

        return value*beta/nt

    if _HAVE_DATACLASSES:
        # use nifty built in dataclass
        return make_dataclass("Parameters",
                              ((key, type(value), field(default=value))
                               for key, value in fields.items()),
                              namespace={"asdict": asdict,
                                         "tilde": _tilde},
                              frozen=True)

    # Use a workaround that lacks almost all of dataclasses features.
    # Just stores a bunch of variables in a class.
    class Parameters:
        # store a list of all field names for asdict
        _FIELDS = list(fields.keys())

        def asdict(self):
            return {key: getattr(self, key) for key in self._FIELDS}

    # store all fields in class
    for key, value in fields.items():
        setattr(Parameters, key, value)

    # add tilde method
    setattr(Parameters, "tilde", _tilde)

    return Parameters

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

    \warning Dataclasses were added in Python3.7.
             This function uses a workaround in earlier versions.
             That workaround has greatly reduced features compared to dataclasses.
             Be careful when you need portability.
    """

    cls = _makeParametersClass(kwargs)
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
