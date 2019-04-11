r"""!\file
\ingroup evolvers
Evolvers that use symmetries of the Hubbard action to perform large jumps in configuration space.
"""

from logging import getLogger

import numpy as np

import isle.action
from .evolver import Evolver
from .selector import BinarySelector


class TwoPiJumps(Evolver):
    r"""! \ingroup evolvers
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
            newActVal = self._action.evalLocal(newPhi, sites, shifts, actVal)

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


class UniformJump(Evolver):
    r"""! \ingroup evolvers
    Shift configuration by 2*pi/Nt at all lattice sites.

    If the action consists only of isle.action.HubbardGaugeAction and any variants
    of isle.action.HubbardFermiAction, a shortcurt can be used to compute the change
    in action which allows for very fast execution.
    """

    def __init__(self, action, lattice, rng):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        """

        self.rng = rng
        self.selector = BinarySelector(rng)

        # use this to evaluate action
        self._action = _HubbardActionShortcut(action)
        # absolute value of shift
        self._absShift = 2*np.pi/lattice.nt()

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

        shift = self.rng.choice((-1, +1)) * self._absShift
        newPhi = np.array(phi, copy=False) + shift
        newActVal = self._action.evalGlobal(newPhi, shift, actVal)

        if self.selector.selectTrajPoint(actVal, newActVal) == 1:
            phi, actVal = isle.Vector(newPhi), newActVal
            trajPoint = 1
        else:
            trajPoint = 0

        return phi, pi, actVal, trajPoint

    def save(self, _h5group, _manager):
        r"""!
        Save the evolver to HDF5.
        Has to be the inverse of Evolver.fromH5().
        \param _h5group \e ignored.
        \param _manager \e ignored.
        """

    @classmethod
    def fromH5(cls, _h5group, _manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param _h5group \e ignored.
        \param _manager \e ignored.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed TwoPiJumps evolver.
        """
        return cls(action, lattice, rng)


class _HubbardActionShortcut:
    """!
    Evaulates actions if a shortcut can be taken.

    Some evolvers change the configuration in such a way that the fermion action
    of the Hubbard model is invariant.
    If this is the case, this class can be used to evaluate the action after such
    a change in a very fast way.
    If the action consists only of (potentially multiple instances of) HubbardGaugeAction
    and any of HubbardFermiAction*, `evalGlobal` and `evalLocal` compute the action
    from the change in gauge action, assuming that the fermion action is invariant.
    If other actions are present, the full action is evaluated and no shortcut is taken.
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

    def evalLocal(self, newPhi, sites, shifts, actVal):
        r"""!
        Compute the value of the action at a given phi after a jump at given sites.
        \param newPhi Configuration after the jump.
        \param sites Array of indices of sites that were shifted.
        \param shifts Amount by which newPhi was shifted per site in parameter `sites`.
        \param actVal Original value of the action (at newPhi - shifts).
        \returns Value of the action at newPhi.
        """

        if self._action is None:
            # shortcut: Computes (newPhi**2 - oldPhi**2) / (2*Utilde), the difference in
            #           gauge action. The fermion action is invariant under jumps by 2*pi
            #           if this branch is taken.
            return actVal + np.dot(2*np.array(newPhi, copy=False)[sites]-shifts, shifts) \
                * self._sumHalfInvUtilde

        # Some part of the action does not allow for the shortcut.
        return self._action.eval(newPhi)

    def evalGlobal(self, newPhi, shift, actVal):
        r"""!
        Compute the value of the action after a uniform jump at all sites.
        \param newPhi Configuration after the jump.
        \param shift Amount by which newPhi was shifted.
        \param actVal Original value of the action (at newPhi - shift).
        \returns Value of the action at newPhi.
        """

        if self._action is None:
            # shortcut: Computes (newPhi**2 - oldPhi**2) / (2*Utilde), the difference in
            #           gauge action. The fermion action is invariant under jumps by 2*pi
            #           if this branch is taken.
            return actVal + shift*(2*np.sum(newPhi) - shift*len(newPhi)) / self._sumHalfInvUtilde

        # Some part of the action does not allow for the shortcut.
        return self._action.eval(newPhi)
