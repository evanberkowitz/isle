"""!
Proposers for HMC evolutions.

\todo descript file layout
"""

## \defgroup proposers Proposers
# Propose new configurations for HMC.

from abc import ABCMeta, abstractmethod
from inspect import getmodule
from logging import getLogger
from pathlib import Path

import h5py as h5
import numpy as np

import isle
import isle.action
from . import Vector, leapfrog
from .util import hingeRange
from .meta import classFromSource, sourceOfClass


class Proposer(metaclass=ABCMeta):
    r"""! \ingroup proposers
    Abstract base class for proposers.
    """

    @abstractmethod
    def propose(self, phi, pi, actVal, trajPoint):
        r"""!
        Propose a new configuration and momentum.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
        """

    @abstractmethod
    def save(self, h5group, manager):
        r"""!
        Save the proposer to HDF5.
        Has to be the inverse of Proposer.fromH5().
        \param h5group HDF5 group to save to.
        \param manager ProposerManager whose purview to save the proposer in.
        """

    @classmethod
    @abstractmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct a proposer from HDF5.
        Create and initialize a new instance from parameters stored via Proposer.save().
        \param h5group HDF5 group to load parameters from.
        \param manager ProposerManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
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

    def propose(self, phi, pi, _actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param _actVal \e ignored.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
        """
        return leapfrog(phi, pi, self.action, self.length, self.nstep)

    def save(self, h5group, manager):
        r"""!
        Save the proposer to HDF5.
        \param h5group HDF5 group to save to.
        \param manager ProposerManager whose purview to save the proposer in.
        """
        h5group["length"] = self.length
        h5group["nstep"] = self.nstep

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, _rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param _lattice \e ignored.
        \param _rng \e ignored.
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

    def propose(self, phi, pi, _actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param _actVal \e ignored.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
        """
        self._current += 1
        return leapfrog(phi, pi, self.action,
                        next(self._lengthIter), int(next(self._nstepIter)))

    def save(self, h5group, manager):
        r"""!
        Save the proposer to HDF5.
        \param h5group HDF5 group to save to.
        \param manager ProposerManager whose purview to save the proposer in.
        """
        h5group["minLength"] = self.lengthRange[0]
        h5group["maxLength"] = self.lengthRange[1]
        h5group["minNstep"] = self.nstepRange[0]
        h5group["maxNstep"] = self.nstepRange[1]
        h5group["ninterp"] = self.ninterp
        h5group["current"] = self._current

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, _rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param _lattice \e ignored.
        \param _rng \e ignored.
        \returns A newly constructed leapfrog proposer.
        """
        return cls(action,
                   (h5group["minLength"][()], h5group["maxLength"][()]),
                   (h5group["minNstep"][()], h5group["maxNstep"][()]),
                   h5group["ninterp"][()],
                   h5group["current"][()])


class TwoPiJumps(Proposer):
    def __init__(self, selectionProbability, action, lattice, rng):
        self.rng = rng
        self.latSize = lattice.lattSize()
        self.selectionProb = selectionProbability

        # probabilities to shift by [+2pi, -2pi, no shift]
        self._shiftProb = [self.selectionProb/2, self.selectionProb/2, 1-self.selectionProb]

        # _sumInvUtildes is either sum(1/Utilde)/2 from all HubbardGaugeActions
        #                or None
        # _action is either None or action that was passed in (same order as _sumInvUtildes)
        utildes = self._findUtildes(action)
        if utildes is None:
            self._action = action
            self._sumHalfInvUtilde = None
            getLogger(__name__).info("Action does not allow for evaluation shortcut under "
                                     "jumps by 2pi, using normal (slow) evaluation")
        else:
            self._action = None
            self._sumHalfInvUtilde = sum(1 / utilde for utilde in utildes) / 2
            getLogger(__name__).info("Action allows for evaluation shortcut under jumps "
                                     "by 2pi, found utilde = %s", utildes)

        self.act = action

    def _findUtildes(self, action):
        r"""!
        Find all HubbardGaugeActions and extract Utilde.
        \returns Either a list of all Utildes found or None if there is any action
                 which does not allow for the shortcut calculation.
        """

        # all sub actions must allow for the shortcut
        if isinstance(action, isle.action.SumAction):
            utildes = []
            for i in range(len(action)):
                aux = self._findUtildes(action[i])
                if aux is None:
                    return None
                utildes.extend(aux)
            return utildes

        # single action
        if isinstance(action, isle.action.HubbardGaugeAction):
            return [action.utilde]

        # action is invariant under 2pi jump
        if isinstance(action, (isle.action.HubbardFermiActionDiaOneOne,
                               isle.action.HubbardFermiActionDiaTwoOne,
                               isle.action.HubbardFermiActionExpOneOne,
                               isle.action.HubbardFermiActionExpTwoOne)):
            return []

        return None

    def propose(self, phi, pi, actVal, trajPoint):
        r"""!
        Propose a new configuration, momentum remains unchanged.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
        """

        # site i is shifted by 2*pi*shifts[i]
        shifts = np.random.choice([1, -1, 0],
                                  p=self._shiftProb,
                                  size=(self.latSize,))

        newPhi = phi + isle.Vector(2*np.pi*shifts)

        if self._action is None:
            # calculate difference in gauge action
            actVal = actVal + (np.linalg.norm(newPhi)**2 - np.linalg.norm(phi)**2) \
                * self._sumHalfInvUtilde

        else:
            # evaluate the full action on new field
            actVal = self._action.eval(newPhi)

        return newPhi, pi, actVal

    def save(self, h5group, _manager):
        r"""!
        Save the proposer to HDF5.
        Has to be the inverse of Proposer.fromH5().
        \param h5group HDF5 group to save to.
        \param _manager \e ignored.
        """
        h5group["selectionProbability"] = self.selectionProb

    @classmethod
    def fromH5(cls, h5group, _manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed TwoPiJumps proposer.
        """
        return cls(h5group["selectionProbability"][()], action, lattice, rng)

class Alternator(Proposer):
    r"""! \ingroup proposers
    Alternate between different proposers.

    Each sub proposer is run a given number of times.
    Then Alternator advances to the next proposer and runs it for its
    number of iterations.
    After the last sub proposer, Alternator cycles round to the first one and repeats.
    """

    def __init__(self, proposers=None, startIndex=0):
        r"""!
        Store given sub proposers.
        \param proposers List of lists of the form `[[c0, p0], [c1, p1], ...]`,
                         where `pi` are the sub proposers to cycle through.
                         `ci` is the number of iterations to run `pi` for.
        \param startIndex Start iteration at this point. Exists mostly for internal use.
        """
        if proposers:
            self._counts, self._subProposers = map(list, zip(*proposers))
        else:
            self._subProposers = []
            self._counts = []

        self._current = startIndex

    def add(self, count, proposer):
        r"""!
        Add a sub proposer to the Alternator.
        \param count Number of iterations the proposer shall be run for.
        \param proposer Sub proposer to add.
        """

        if count < 1:
            getLogger(__name__).warning(
                "Iteration count for proposer %s is less than one: %d",
                proposer, count)

        self._counts.append(count)
        self._subProposers.append(proposer)

    def _advance(self):
        """!Advance internal proposer index."""
        self._current = (self._current + 1) % sum(self._counts)

    def _pickCurrentProposer(self):
        """!Pick a sub proposer based on the current proposer index."""
        currentTotalCount = 0
        for count, proposer in zip(self._counts, self._subProposers):
            currentTotalCount += count
            if currentTotalCount > self._current:
                return proposer
        return None

    def propose(self, phi, pi, actVal, trajPoint):
        r"""!
        Delegate to a sub proposer next in line.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated on new configuration
        """

        subProposer = self._pickCurrentProposer()
        self._advance()

        return subProposer.propose(phi, pi, actVal, trajPoint)

    def save(self, h5group, manager):
        r"""!
        Save the proposer to HDF5.
        \param h5group HDF5 group to save to.
        \param manager ProposerManager whose purview to save the proposer in.
        """
        h5group["counts"] = self._counts
        h5group["current"] = self._current
        for idx, proposer in enumerate(self._subProposers):
            grp = h5group.create_group(f"sub{idx}")
            manager.save(proposer, grp)

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param manager ProposerManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed alternating proposer.
        """
        alternator = cls(startIndex=h5group["current"][()])
        counts = h5group["counts"][()]
        for idx, count in enumerate(counts):
            subProposer = manager.load(h5group[f"sub{idx}"], action, lattice, rng)
            alternator.add(count, subProposer)
        return alternator



class ProposerManager:
    r"""!
    Manages proposers in a file.

    Handles saving and loading of types and parameters of proposers in a centralized way.
    Operates with the file structure descriped in
    \todo link to file doc
    """

    def __init__(self, fname, typeLocation="/meta/proposers", definitions={}):
        r"""!
        Initialize for a given file.
        \param fname Path to the file to manage.
        \param typeLocation Path inside the file where proposer types are to be stored.
        \param definitions Dictionary of extra definitions to take into account when
               saving/loading proposers. See saveProposerType() and loadProposerType().
        """

        self.typeLocation = typeLocation
        self.extraDefinitions = definitions
        # list of all currently existing proposers in the file
        # index in this list is index in file
        self._proposers = self._loadTypes(Path(fname))

    def _loadTypes(self, fname):
        r"""!
        Load all types of proposers in file `fname`.
        """

        if not fname.exists():
            getLogger(__name__).info("File to load proposers from does not exist: %s", fname)
            return []

        with h5.File(fname, "r") as h5f:
            try:
                grp = h5f[self.typeLocation]
            except KeyError:
                getLogger(__name__).info("No proposers found in file %s", fname)
                return []

            proposers = [loadProposerType(g, self.extraDefinitions)
                         for _, g in sorted(grp.items(), key=lambda p: int(p[0]))]
        getLogger(__name__).info("Loaded proposer types from file %s:\n    %s",
                                 fname,
                                 "\n    ".join(f"{i}: {p}" for i, p in enumerate(proposers)))
        return proposers

    def saveType(self, proposer, h5file):
        r"""!
        Save the type of a proposer if it is not already stored.
        \param proposer Proposer object (<I>not</I> type!) to save.
        \param h5file File to save to.
        \returns Index of the proposer type in the file.
        """

        typ = type(proposer)

        # check if it is already stored
        for index, stored in enumerate(self._proposers):
            if stored.__name__ == typ.__name__:
                return index

        # else: is not stored yet
        index = len(self._proposers)
        self._proposers.append(typ)

        grp = h5file.create_group(self.typeLocation+f"/{index}")
        saveProposerType(proposer, grp, self.extraDefinitions)

        getLogger(__name__).info("Saved proposer number %d: %s", index, typ)

        return index

    def save(self, proposer, h5group):
        r"""!
        Save a proposer including its type.
        \param proposer Proposer object to save.
        \param h5group Group in the HDF5 file to save the proposer's parameters to.
                       Stores the index of the proposer's type in the attribute `__index__`
                       of the group.
        """

        # save the type
        index = self.saveType(proposer, h5group.file)

        # save the parameters
        proposer.save(h5group, self)

        # link to the type
        h5group.attrs["__index__"] = index

    def loadType(self, index, h5file):
        r"""!
        Load a proposer type from file.
        \param index Index of the type to load. Corresponds to group name in the type location.
        \param h5file HDF5 file to load the type from.
        """
        return loadProposerType(h5file[self.typeLocation+f"/{index}"])

    def load(self, h5group, action, lattice, rng):
        r"""!
        Load a proposer's type and construct an instance from given group.
        The type has to be stored in the 'type location' (see `__init__`) in the same file
        as `h5group`.
        \param h5group Group in the file where proposer parameters are stored.
        \param action Passed to proposer's constructor.
        \param lattice Passed to proposer's constructor.
        \param rng Passed to proposer's constructor.
        """
        return self.loadType(h5group.attrs["__index__"][()], h5group.file) \
                   .fromH5(h5group, self, action, lattice, rng)


def saveProposerType(proposer, h5group, definitions={}):
    r"""! \ingroup proposers
    Save a proposer's type to HDF5.

    There are three possible scenarios:
     - The proposer is built into Isle: Only its name is stored.
       It can be reconstructed automatically.
     - The proposer is custom defined and included in parameter `definitions`:
       Only its name is stored. It needs to be passed to loadProposerType() when
       reconstructing.
     - The proposer is custom defined and not included in `definitions`:
       The full source code of the proposer's definition is stored.
       This requries that the proposer is fully self-contained, i.e. not use
       any symbols from outside its own definition (except for `isle`).

    \see loadProposerType to load proposers saved with this function.
    """

    if not isinstance(proposer, Proposer):
        getLogger(__name__).error("Can only save instances of subclasses of "
                                  "isle.proposers.Proposer, given %s",
                                  type(proposer))
        raise ValueError("Not a proposer")

    # get the name of the proposer's class
    name = type(proposer).__name__
    if name == "__as_source__":
        getLogger(__name__).error("Proposers must not be called __as_source__. "
                                  "That name is required for internal use.")
        raise ValueError("Proposer must not be called __as_source__")

    if proposer.__module__ == "isle.proposers" or type(proposer).__name__ in definitions:
        # builtin or custom
        h5group["__name__"] = name
        getLogger(__name__).info("Saved type of proposer %s via its name", name)

    else:
        # store source
        h5group["__name__"] = "__as_source__"
        src = sourceOfClass(type(proposer))
        # attempt to reconstruct it to check for errors early
        import isle
        classFromSource(src, {"isle": isle, "proposers": isle.proposers, "Proposer": Proposer})
        h5group["__source__"] = src
        getLogger(__name__).info("Saved type of proposer %s as source", name)

def loadProposerType(h5group, definitions={}):
    r"""! \ingroup proposers
    Retrieves the class of a proposer from HDF5.

    \param h5group HDF5 group containing name, (source), parameters of a proposer.
    \param definitions Dict containing custom definitions. If it contains an entry
                       with the name of the proposer, it is loaded based on that
                       entry instead of from source code.
    \return Class loaded from file.

    \see saveProposerType() to save proposers in a supported format.
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

    return cls
