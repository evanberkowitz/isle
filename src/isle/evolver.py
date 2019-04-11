r"""!\file
Evolvers for HMC evolutions.
"""

## \defgroup evolvers Evolvers
# Evolve configurations for HMC.

from abc import ABCMeta, abstractmethod
from inspect import getmodule
from logging import getLogger
from pathlib import Path

import h5py as h5
import numpy as np

import isle
import isle.action
from . import Vector, leapfrog
from .collection import hingeRange
from .meta import classFromSource, sourceOfClass


class Evolver(metaclass=ABCMeta):
    r"""! \ingroup evolvers
    Abstract base class for evolvers.
    """

    @abstractmethod
    def evolve(self, phi, pi, actVal, trajPoint):
        r"""!
        Evolve a configuration and momentum.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - Point along trajectory that was selected
        """

    @abstractmethod
    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        Has to be the inverse of Evolver.fromH5().
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """

    @classmethod
    @abstractmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct an evolver from HDF5.
        Create and initialize a new instance from parameters stored via Evolver.save().
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed evolver.
        """


class ConstStepLeapfrog(Evolver):
    r"""! \ingroup evolvers
    A leapfrog evolver with constant parameters.
    """

    def __init__(self, action, length, nstep, rng):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param length Length of the MD trajectory.
        \param nstep Number of MD steps per trajectory.
        \param rng Central random number generator for the run. Used for accept/reject.
        """
        self.action = action
        self.length = length
        self.nstep = nstep
        self.selector = BinarySelector(rng)

    def evolve(self, phi, pi, actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - Point along trajectory that was selected
        """

        phi1, pi1, actVal1 = leapfrog(phi, pi, self.action, self.length, self.nstep)
        trajPoint = self.selector.selectTrajPoint(actVal+np.linalg.norm(pi)**2/2,
                                                  actVal1+np.linalg.norm(pi1)**2/2)
        return (phi1, pi1, actVal1, trajPoint) if trajPoint == 1 \
            else (phi, pi, actVal, trajPoint)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["length"] = self.length
        h5group["nstep"] = self.nstep

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param _lattice \e ignored.
        \param rng Central random number generator for the run.
        \returns A newly constructed leapfrog evolver.
        """
        return cls(action, h5group["length"][()], h5group["nstep"][()], rng)


class LinearStepLeapfrog(Evolver):
    r"""! \ingroup evolvers
    A leapfrog evolver with linearly changing parameters.

    Both trajectory length and number of MD steps are interpolated between
    starting and final values. If the number of interpolating steps is lower than the number
    of trajectories computed using the evolver, the parameters stay at their final
    values.
    """

    def __init__(self, action, lengthRange, nstepRange, ninterp, rng, startPoint=0):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param lengthRange Tuple of initial and final trajectory lengths.
        \param nstepRange Tuple of initial and final number of steps.
        \param ninterp Number of interpolating steps.
        \param rng Central random number generator for the run. Used for accept/reject.
        \param startPoint Iteration number to start at.
        """

        self.action = action
        self.lengthRange = lengthRange
        self.nstepRange = nstepRange
        self.ninterp = ninterp
        self.selector = BinarySelector(rng)

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

    def evolve(self, phi, pi, actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - Point along trajectory that was selected
        """
        self._current += 1

        phi1, pi1, actVal1 = leapfrog(phi, pi, self.action,
                                      next(self._lengthIter), int(next(self._nstepIter)))
        trajPoint = self.selector.selectTrajPoint(actVal+np.linalg.norm(pi)**2/2,
                                                  actVal1+np.linalg.norm(pi1)**2/2)
        return (phi1, pi1, actVal1, trajPoint) if trajPoint == 1 \
            else (phi, pi, actVal, trajPoint)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["minLength"] = self.lengthRange[0]
        h5group["maxLength"] = self.lengthRange[1]
        h5group["minNstep"] = self.nstepRange[0]
        h5group["maxNstep"] = self.nstepRange[1]
        h5group["ninterp"] = self.ninterp
        h5group["current"] = self._current

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param _lattice \e ignored.
        \param rng Central random number generator for the run.
        \returns A newly constructed leapfrog evolver.
        """
        return cls(action,
                   (h5group["minLength"][()], h5group["maxLength"][()]),
                   (h5group["minNstep"][()], h5group["maxNstep"][()]),
                   h5group["ninterp"][()],
                   rng,
                   h5group["current"][()])


class TwoPiJumps(Evolver):
    """!
    Shift configuration by 2*pi at random lattice sites.

    Evolver that selects lattice sites and shifts the configuration by
    `+2*pi` or `-2*pi` at those sites where the sign is chosen randomly per site.
    The conjugate momentum is not changed.
    Can modify multiple sites at the same time (one "batch") and do multiple batches.
    Each batch is accepted or rejected on its own.

    If the action consists only of isle.action.HubbardGaugeAction and any variant
    of isle.action.HubbardFermiAction, a shortcurt can be used to compute the change
    in action which allows for very fast execution.
    """

    def __init__(self, nBatches, batchSize, action, lattice, rng):
        r"""!
        \param nBatches Number of batches of sites.
        \param batchSize Number of sites in each batch.
        \param action Instance of isle.Action to use for molecular dynamics.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        """

        self.nBatches = nBatches
        self.batchSize = batchSize
        self.rng = rng
        self.latSize = lattice.lattSize()
        self.selector = BinarySelector(rng)

        self._action = _HubbardActionShortcut(action)

    def evolve(self, phi, pi, actVal, trajPoint):
        r"""!
        Evolve a configuration, momentum remains unchanged.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint \e ignored.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - 0 if no jump was accepted, 1 if at least one was accepted (across all batches).
        """

        # nothing accepted yet
        trajPoint = 0

        for _ in range(self.nBatches):
            # lattice sites at which to jump
            sites = self.rng.choice(self.latSize, self.batchSize, replace=False)
            # shifts for the above sites
            shifts = self.rng.choice((-2*np.pi, +2*np.pi), self.batchSize, replace=True)

            # perform jumps
            newPhi = np.array(phi, copy=True)
            newPhi[sites] += shifts
            newActVal = self._action.eval(newPhi, sites, shifts, actVal)

            # no need to put pi into check, it never changes in here
            if self.selector.selectTrajPoint(actVal, newActVal) == 1:
                phi, actVal = newPhi, newActVal
                trajPoint = 1  # something has been accepted
            # else:
            #     nothing to do, variables are already set up

        return phi, pi, actVal, trajPoint

    def save(self, h5group, _manager):
        r"""!
        Save the evolver to HDF5.
        Has to be the inverse of Evolver.fromH5().
        \param h5group HDF5 group to save to.
        \param _manager \e ignored.
        """
        h5group["nBatches"] = self.nBatches
        h5group["batchSize"] = self.batchSize

    @classmethod
    def fromH5(cls, h5group, _manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param _manager \e ignored.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed TwoPiJumps evolver.
        """
        return cls(h5group["nBatches"][()], h5group["batchSize"][()], action, lattice, rng)


class Alternator(Evolver):
    r"""! \ingroup evolvers
    Alternate between different evolvers.

    Each sub evolver is run a given number of times.
    Then Alternator advances to the next evolver and runs it for its
    number of iterations.
    After the last sub evolver, Alternator cycles round to the first one and repeats.

    <B>Example</B><br>
    This example defines an %Alternator which uses ConstStepLeapfrog for 100 trajectories,
    then TwoPiJumps once, and repeats.
    ```{.py}
    # Set up parameters
    # hmcState, length, nstep
    # ...

    evolver = isle.evolver.Alternator((
        (100, isle.evolver.ConstStepLeapfrog(hmcState.action, length, nstep, hmcState.rng)),
        (1, isle.evolver.TwoPiJumps(hmcState.lattice.lattSize(), 1, hmcState.action,
                                    hmcState.lattice, hmcState.rng))
    ))
    ```
    """

    def __init__(self, evolvers=None, startIndex=0):
        r"""!
        Store given sub evolvers.
        \param evolvers List of lists of the form `[[c0, e0], [c1, e1], ...]`,
                        where `ei` are the sub evolvers to cycle through.
                        `ci` is the number of iterations to run `ei` for.
        \param startIndex Start iteration at this point. Exists mostly for internal use.
        """
        if evolvers:
            self._counts, self._subEvolvers = map(list, zip(*evolvers))
        else:
            self._subEvolvers = []
            self._counts = []

        self._current = startIndex

    def add(self, count, evolver):
        r"""!
        Add a sub evolver to the Alternator.
        \param count Number of iterations the evolver shall be run for.
        \param evolver Sub evolver to add.
        """

        if count < 1:
            getLogger(__name__).warning(
                "Iteration count for evolver %s is less than one: %d",
                evolver, count)

        self._counts.append(count)
        self._subEvolvers.append(evolver)

    def _advance(self):
        """!Advance internal evolver index."""
        self._current = (self._current + 1) % sum(self._counts)

    def _pickCurrentEvolver(self):
        """!Pick a sub evolver based on the current evolver index."""
        currentTotalCount = 0
        for count, evolver in zip(self._counts, self._subEvolvers):
            currentTotalCount += count
            if currentTotalCount > self._current:
                return evolver
        return None

    def evolve(self, phi, pi, actVal, trajPoint):
        r"""!
        Delegate to a sub evolver next in line.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated on new configuration
          - Point along trajectory that was selected
        """

        subEvolver = self._pickCurrentEvolver()
        self._advance()

        return subEvolver.evolve(phi, pi, actVal, trajPoint)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["counts"] = self._counts
        h5group["current"] = self._current
        for idx, evolver in enumerate(self._subEvolvers):
            grp = h5group.create_group(f"sub{idx}")
            manager.save(evolver, grp)

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed alternating evolver.
        """
        alternator = cls(startIndex=h5group["current"][()])
        counts = h5group["counts"][()]
        for idx, count in enumerate(counts):
            subEvolver = manager.load(h5group[f"sub{idx}"], action, lattice, rng)
            alternator.add(count, subEvolver)
        return alternator



class _HubbardActionShortcut:
    """!
    Evaulates actions if a shortcut can be taken.

    \todo write!!
    """


    def __init__(self, action):
        # _sumHalfInvUtildes is either sum(1/Utilde)/2 from all HubbardGaugeActions
        #                    or None
        # _action is either None or action that was passed in (same order as _sumInvUtildes)
        utildes = self._findUtildes(action)
        if utildes is None:
            self._action = action
            self._sumHalfInvUtilde = None
            getLogger(__name__).info("Action does not allow for evaluation shortcut by "
                                     "invariance of the fermion determinant, "
                                     "using normal (slow) evaluation")
        else:
            self._action = None
            self._sumHalfInvUtilde = sum(1 / utilde for utilde in utildes) / 2
            getLogger(__name__).info("Action allows for evaluation shortcut by "
                                     "invariance of the fermion determinant, "
                                     "found utilde = %s", utildes)

    def _findUtildes(self, action):
        r"""!
        Find all HubbardGaugeActions and extract Utilde.
        \returns Either a list of all Utildes found or None if there is any action
                 which does not allow for the shortcut calculation.
        """

        # all sub actions must allow for the shortcut
        if isinstance(action, isle.action.SumAction):
            utildes = []
            for act in action:
                aux = self._findUtildes(act)
                if aux is None:  # propagate 'non-regognized action'
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

        # not recognized, can't use shortcut
        return None

    def eval(self, newPhi, sites, shifts, actVal):
        """!
        Compute the value of the action at a given phi after a jump.
        """

        if self._action is None:
            # shortcut: Computes (newPhi**2 - oldPhi**2) / (2*Utilde), the difference in
            #           gauge action. The fermion action is invariant under jumps by 2*pi
            #           if this branch is taken.
            return actVal + np.dot(2*np.array(newPhi, copy=False)[sites]-shifts, shifts) \
                * self._sumHalfInvUtilde

        # Some part of the action does not allow for the shortcut.
        return self._action.eval(newPhi)



class EvolverManager:
    r"""!
    Manages evolvers in a file.

    Handles saving and loading of types and parameters of evolvers in a centralized way.
    Operates with the file structure descriped in
    \todo link to file doc
    """

    def __init__(self, fname, typeLocation="/meta/evolvers", definitions={}):
        r"""!
        Initialize for a given file.
        \param fname Path to the file to manage.
        \param typeLocation Path inside the file where evolver types are to be stored.
        \param definitions Dictionary of extra definitions to take into account when
               saving/loading evolvers. See saveEvolverType() and loadEvolverType().
        """

        self.typeLocation = typeLocation
        self.extraDefinitions = definitions
        # list of all currently existing evolvers in the file
        # index in this list is index in file
        self._evolvers = self._loadTypes(Path(fname))

    def _loadTypes(self, fname):
        r"""!
        Load all types of evolvers in file `fname`.
        """

        if not fname.exists():
            getLogger(__name__).info("File to load evolvers from does not exist: %s", fname)
            return []

        with h5.File(fname, "r") as h5f:
            try:
                grp = h5f[self.typeLocation]
            except KeyError:
                getLogger(__name__).info("No evolvers found in file %s", fname)
                return []

            evolvers = [loadEvolverType(g, self.extraDefinitions)
                         for _, g in sorted(grp.items(), key=lambda p: int(p[0]))]
        getLogger(__name__).info("Loaded evolver types from file %s:\n    %s",
                                 fname,
                                 "\n    ".join(f"{i}: {p}" for i, p in enumerate(evolvers)))
        return evolvers

    def saveType(self, evolver, h5file):
        r"""!
        Save the type of an evolver if it is not already stored.
        \param evolver Evolver object (<I>not</I> type!) to save.
        \param h5file File to save to.
        \returns Index of the evolver type in the file.
        """

        typ = type(evolver)

        # check if it is already stored
        for index, stored in enumerate(self._evolvers):
            if stored.__name__ == typ.__name__:
                return index

        # else: is not stored yet
        index = len(self._evolvers)
        self._evolvers.append(typ)

        grp = h5file.create_group(self.typeLocation+f"/{index}")
        saveEvolverType(evolver, grp, self.extraDefinitions)

        getLogger(__name__).info("Saved evolver number %d: %s", index, typ)

        return index

    def save(self, evolver, h5group):
        r"""!
        Save an evolver including its type.
        \param evolver Evolver object to save.
        \param h5group Group in the HDF5 file to save the evolver's parameters to.
                       Stores the index of the evolver's type in the attribute `__index__`
                       of the group.
        """

        # save the type
        index = self.saveType(evolver, h5group.file)

        # save the parameters
        evolver.save(h5group, self)

        # link to the type
        h5group.attrs["__index__"] = index

    def loadType(self, index, h5file):
        r"""!
        Load an evolver type from file.
        \param index Index of the type to load. Corresponds to group name in the type location.
        \param h5file HDF5 file to load the type from.
        """
        return loadEvolverType(h5file[self.typeLocation+f"/{index}"])

    def load(self, h5group, action, lattice, rng):
        r"""!
        Load an evolver's type and construct an instance from given group.
        The type has to be stored in the 'type location' (see `__init__`) in the same file
        as `h5group`.
        \param h5group Group in the file where evolver parameters are stored.
        \param action Passed to evolver's constructor.
        \param lattice Passed to evolver's constructor.
        \param rng Passed to evolver's constructor.
        """
        return self.loadType(h5group.attrs["__index__"][()], h5group.file) \
                   .fromH5(h5group, self, action, lattice, rng)


def saveEvolverType(evolver, h5group, definitions={}):
    r"""! \ingroup evolvers
    Save an evolver's type to HDF5.

    There are three possible scenarios:
     - The evolver is built into Isle: Only its name is stored.
       It can be reconstructed automatically.
     - The evolver is custom defined and included in parameter `definitions`:
       Only its name is stored. It needs to be passed to loadEvolverType() when
       reconstructing.
     - The evolver is custom defined and not included in `definitions`:
       The full source code of the evolver's definition is stored.
       This requries that the evolver is fully self-contained, i.e. not use
       any symbols from outside its own definition (except for `isle`).

    \see loadEvolverType to load evolvers saved with this function.
    """

    if not isinstance(evolver, Evolver):
        getLogger(__name__).error("Can only save instances of subclasses of "
                                  "isle.evolver.Evolver, given %s",
                                  type(evolver))
        raise ValueError("Not an evolver")

    # get the name of the evolver's class
    name = type(evolver).__name__
    if name == "__as_source__":
        getLogger(__name__).error("Evolvers must not be called __as_source__. "
                                  "That name is required for internal use.")
        raise ValueError("Evolver must not be called __as_source__")

    if evolver.__module__ == "isle.evolver" or type(evolver).__name__ in definitions:
        # builtin or custom
        h5group["__name__"] = name
        getLogger(__name__).info("Saved type of evolver %s via its name", name)

    else:
        # store source
        h5group["__name__"] = "__as_source__"
        src = sourceOfClass(type(evolver))
        # attempt to reconstruct it to check for errors early
        import isle
        classFromSource(src, {"isle": isle, "evolvers": isle.evolver, "Evolver": Evolver})
        h5group["__source__"] = src
        getLogger(__name__).info("Saved type of evolver %s as source", name)

def loadEvolverType(h5group, definitions={}):
    r"""! \ingroup evolvers
    Retrieves the class of an evolver from HDF5.

    \param h5group HDF5 group containing name, (source), parameters of an evolver.
    \param definitions Dict containing custom definitions. If it contains an entry
                       with the name of the evolver, it is loaded based on that
                       entry instead of from source code.
    \return Class loaded from file.

    \see saveEvolverType() to save evolvers in a supported format.
    """

    name = h5group["__name__"][()]

    if name == "__as_source__": # from source
        import isle  # get it here so it is not imported unless needed
        try:
            cls = classFromSource(h5group["__source__"][()],
                                  {"isle": isle,
                                   "evolvers": isle.evolver,
                                   "Evolver": Evolver})
        except ValueError:
            getLogger(__name__).error("Source code for evolver does not define a class."
                                      "Cannot load evolver.")
            raise RuntimeError("Cannot load evolver from source") from None

    else:  # from name + known definition
        try:
            # builtin, in the module that Evolver is defined in (i.e. this one)
            cls = getmodule(Evolver).__dict__[name]
        except KeyError:
            try:
                # provided by user
                cls = definitions[name]
            except KeyError:
                getLogger(__name__).error(
                    "Unable to load evolver of type '%s' from source."
                    "The type is neither built in nor provided through argument 'definitions'.",
                    name)
                raise RuntimeError("Cannot load evolver from source") from None

    if not issubclass(cls, Evolver):
        getLogger(__name__).error("Loaded type is not an evolver: %s", name)

    return cls


class BinarySelector:
    """!
    Select one of two trajectories based on Metropolis accept/reject.
    """

    def __init__(self, rng):
        r"""!
        \param rng Central random number generator of the run used for accept/reject.
        """
        self.rng = rng

    def selectTrajPoint(self, energy0, energy1):
        r"""!
        Select a trajectory point using Metropolis accept/reject.
        \param energy0 Energy at point 0 including the artificial kinetic term .
        \param energy1 Energy at point 1 including the artificial kinetic term.
        \return `0` if `energy0` was selected, `1` otherwise.
        """

        deltaE = np.real(energy1 - energy0)
        return 1 if deltaE < 0 or np.exp(-deltaE) > self.rng.uniform(0, 1) \
            else 0

    def selectTrajectory(self, energy0, data0, energy1, data1):
        r"""!
        Select a trajectory point and pass along extra data.
        \param energy0 Energy at point 0 including the artificial kinetic term .
        \param data0 Arbitrary data assiciated with point 0.
        \param energy1 Energy at point 1 including the artificial kinetic term.
        \param data1 Arbitrary data assiciated with point 1.
        \return `(energy0, data0, 0)` if `energy0` was selected, otherwise
                `(energy1, data1, 1)`.
        """

        return (energy1, data1, 1) if self.selectTrajPoint(energy0, energy1) == 1 \
            else (energy0, data0, 0)
