"""! \file
Some general utilities.
"""

from logging import getLogger
import dataclasses

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
    elif comp != "equal":
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
    Construct a new dataclass for parameters.
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

    return dataclasses.make_dataclass("Parameters",
                                      ((key, type(value), dataclasses.field(default=value))
                                       for key, value in fields.items()),
                                      namespace={"asdict": dataclasses.asdict,
                                                 "tilde": _tilde,
                                                 "__getitem__": lambda obj,key: getattr(obj, key),
                                                 "replace": dataclasses.replace},
                                      frozen=True)

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
    - <B>replace</B>(**kwargs):
           Works like `dataclasses.replace` but is provided as an instance member
           for compatibility with the pre Python3.7 workaround.

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

def temporalRoller(nt, dt, fermionic=True):
    r"""!
    A matrix that, when applied to a temporal vector of length nt, rotates by t.
    \param nt The number of time slices.
    \param dt How many time slices to rotate by.
    \param fermionic If true, use antiperiodic boundary conditions.
    \returns An `nt * nt` matrix `R`, such that `R@(v[t]) = v[t+dt]` with correct boundary conditions.
    """
    shift = dt % nt
    result = np.ones(nt)
    if fermionic:
        completeOrbits = dt // nt # integer division does the right thing, even if dt is negative.
        if 1 == completeOrbits % 2:
            result *= -1
        result[nt-shift:]*=-1
    return np.roll(np.diag(result), shift, axis=0)

def signAlternator(nx, sigmaKappa=-1):
    """
    Return a unit matrix of size `nx` by `nx` if `sigmaKappa==+1`.
    If `sigmaKappa==-1`, return a diagonal matrix with +1 in even rows
    and -1 on odd rows.

    This matrix corresponds to (-sigmaKappa)^x iff the lattice is bipartite and partition
    A has even indices, and B odd indices.
    """

    if sigmaKappa == -1:
        return np.eye(nx)
    if sigmaKappa == +1:
        return np.diag([+1 if np.mod(x, 2) == 0 else -1 for x in range(nx)])
    raise ValueError(f"sigmaKappa must be +1 or -1; is {sigmaKappa}")
