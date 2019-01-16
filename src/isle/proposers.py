"""!
Proposers for HMC evolutions.
"""

## \defgroup proposers Proposers
# Propose new configurations for HMC.

from abc import ABCMeta, abstractmethod
from inspect import getmodule
from logging import getLogger

import numpy as np

from . import Vector, leapfrog
from .util import hingeRange
from .meta import classFromSource, sourceOfClass


class Proposer(metaclass=ABCMeta):
    r"""! \ingroup proposers
    Abstract base class for proposers.
    """

    @abstractmethod
    def propose(self, phi, pi, energy, trajPoint):
        r"""!
        Propose a new configuration and momentum.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param energy Energy computed on phi and pi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - New energy
        """

    @abstractmethod
    def save(self, h5group):
        r"""!
        Save the proposer to HDF5.
        Has to be the inverse of Proposer.fromH5().
        \param h5group HDF5 group to save to.
        """

    @classmethod
    @abstractmethod
    def fromH5(cls, h5group, action, lattice):
        r"""!
        Construct a proposer from HDF5.
        Create and initialize a new instance from parameters stored via Proposer.save().
        \param h5group HDF5 group to load parameters from.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \returns A newly constructed proposer.
        """

class ConstStepLeapfrog(Proposer):
    r"""! \ingroup proposers
    A leapfrog proposer with constant parameters.
    """

    def __init__(self, action, length, nstep):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param length Length of the MD trajectory.
        \param nstep Number of MD steps per trajectory.
        """
        self.action = action
        self.length = length
        self.nstep = nstep

    def propose(self, phi, pi, _energy, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param _energy \e ignored.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - New energy
        """
        return leapfrog(phi, pi, self.action, self.length, self.nstep)

    def save(self, h5group):
        r"""!
        Save the proposer to HDF5.
        \param h5group HDF5 group to save to.
        """
        h5group["length"] = self.length
        h5group["nstep"] = self.nstep

    @classmethod
    def fromH5(cls, h5group, action, _lattice):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param action Action to use.
        \param _lattice \e ignored.
        \returns A newly constructed leapfrog proposer.
        """
        return cls(action, h5group["length"][()], h5group["nstep"][()])


class LinearStepLeapfrog(Proposer):
    r"""! \ingroup proposers
    A leapfrog proposer with linearly changing parameters.

    Both trajectory length and number of MD steps are interpolated between
    starting and final values. If the number of interpolating steps is lower than the number
    of trajectories computed using the proposer, the parameters stay at their final
    values.
    """

    def __init__(self, action, lengthRange, nstepRange, ninterp, startPoint=0):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param lengthRange Tuple of initial and final trajectory lengths.
        \param nstepRange Tuple of initial and final number of steps.
        \param ninterp Number of interpolating steps.
        \param startPoint Iteration number to start at.
        """

        self.action = action
        self.lengthRange = lengthRange
        self.nstepRange = nstepRange
        self.ninterp = ninterp

        self._lengthIter = hingeRange(*lengthRange, (lengthRange[1]-lengthRange[0])/ninterp)
        self._nstepIter = hingeRange(*nstepRange, (nstepRange[1]-nstepRange[0])/ninterp)
        # Keep track of where the interpolation currently is at.
        # Allows it to be resumed when using fromH5.
        self._current = 0

        self._advanceTo(startPoint)

    def _advanceTo(self, idx):
        for _ in range(idx):
            next(self._lengthIter)
            next(self._nstepIter)

    def propose(self, phi, pi, _energy, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param _energy \e ignored.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - New energy
        """
        self._current += 1
        return leapfrog(phi, pi, self.action,
                        next(self._lengthIter), int(next(self._nstepIter)))

    def save(self, h5group):
        r"""!
        Save the proposer to HDF5.
        \param h5group HDF5 group to save to.
        """
        h5group["minLength"] = self.lengthRange[0]
        h5group["maxLength"] = self.lengthRange[1]
        h5group["minNstep"] = self.nstepRange[0]
        h5group["maxNstep"] = self.nstepRange[1]
        h5group["ninterp"] = self.ninterp
        h5group["current"] = self._current

    @classmethod
    def fromH5(cls, h5group, action, _lattice):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param action Action to use.
        \param _lattice \e ignored.
        \returns A newly constructed leapfrog proposer.
        """
        return cls(action,
                   (h5group["minLength"][()], h5group["maxLength"][()]),
                   (h5group["minLength"][()], h5group["maxLength"][()]),
                   h5group["ninterp"][()],
                   h5group["current"][()])


def saveProposer(proposer, h5group, definitions={}):
    r"""! \ingroup proposers
    Save a proposer to HDF5.

    Calls `proposer.save(h5group)` and stores additional data such that
    loadProposer() can reconstruct the object.
    There are three possible scenarios:
     - The proposer is built into Isle: Only its name is stored.
       It can be reconstructed automatically.
     - The proposer is custom defined and included in parameter `definitions`:
       Only its name is stored. It needs to be passed to loadProposer() when
       reconstructing.
     - The proposer is custom defined and not included in `definitions`:
       The full source code of the proposer's definition is stored.
       This requries that the proposer is fully self-contained, i.e. not use
       any symbols from outside its own definition (except for `isle`).
    """

    if not isinstance(proposer, Proposer):
        getLogger(__name__).error("Can only save instances of subclasses of "
                                  "isle.proposers.Proposer, given %s",
                                  type(proposer))
        raise ValueError("Not a proposer")

    # let the proposer go first to make sure it doesn't mess up any special datasets
    proposer.save(h5group)
    for special in ("__name__", "__source__"):
        if special in h5group:
            getLogger(__name__).error("Proposer wrote to the special dataset %s. "
                                      "This is reserved for internal use.",
                                      special)
            raise RuntimeError(f"Proposer wrote to {special}")

    # get the name of the proposer's class
    name = type(proposer).__name__
    if name == "__as_source__":
        getLogger(__name__).error("Proposers must be called __as_source__. "
                                  "That name is required for internal use.")
        raise ValueError("Proposer must not be called __as_source__")

    if proposer.__module__ == "isle.proposers" or type(proposer).__name__ in definitions:
        # builtin or custom
        h5group["__name__"] = name

    else:
        # store source
        h5group["__name__"] = "__as_source__"
        src = sourceOfClass(type(proposer))
        _ = classFromSource(src)  # attempt to recosntruct it to check for errors early
        h5group["__source__"] = src

def loadProposer(h5group, action, lattice, definitions={}):
    r"""! \ingroup proposers
    Load a proposer from HDF5.

    Retrieves the class of a proposer from HDF5 and constructs an instance.

    \param h5group HDF5 group containing name, (source), parameters of a proposer.
    \param action Action to use the proposer with. Passed to Proposer.fromH5().
    \param lattice Lattice passed to Proposer.fromH5().
    \param definitions Dict containing custom definitions. If it contains an entry
                       with the name of the proposer, it is loaded based on that
                       entry instead of from source code.
    \return Newly constructed proposer.

    \see saveProposer() to save proposers in a supported format.
    """

    name = h5group["__name__"][()]

    if name == "__as_source__": # from source
        import isle  # get it here so it is not imported unless needed
        try:
            cls = classFromSource(h5group["__source__"][()],
                                  {"isle": isle,
                                   "proposers": isle.proposers,
                                   "Proposer": Proposer})
        except ValueError:
            getLogger(__name__).error("Source code for proposer does not define a class."
                                      "Cannot load proposer.")
            raise RuntimeError("Cannot load proposer from source") from None

    else:  # from name + known definition
        try:
            # builtin, in the module that Proposer is defined in (i.e. this one)
            cls = getmodule(Proposer).__dict__[name]
        except KeyError:
            try:
                # provided by user
                cls = definitions[name]
            except KeyError:
                getLogger(__name__).error(
                    "Unable to load proposer of type '%s' from source."
                    "The type is neither built in nor provided through argument 'definitions'.",
                    name)
                raise RuntimeError("Cannot load proposer from source") from None

    if not issubclass(cls, Proposer):
        getLogger(__name__).error("Loaded type is not a proposer: %s", name)

    return cls.fromH5(h5group, action, lattice)
