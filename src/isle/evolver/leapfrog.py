r"""!\file
\ingroup evolvers
Evolvers that perform molecular dynamics integration of configurations using leapfrog.
"""

import numpy as np

from .evolver import Evolver
from .selector import BinarySelector
from .. import Vector, leapfrog
from ..collection import hingeRange


class ConstStepLeapfrog(Evolver):
    r"""! \ingroup evolvers
    A leapfrog evolver with constant parameters.
    """

    def __init__(self, action, length, nstep, rng, transform=None):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param length Length of the MD trajectory.
        \param nstep Number of MD steps per trajectory.
        \param rng Central random number generator for the run.
        \param transform (Instance of isle.evolver.transform.Transform)
                         Used this to transform a configuration after MD integration
                         but before Metropolis accept/reject.
        """
        self.action = action
        self.length = length
        self.nstep = nstep
        self.rng = rng
        self.selector = BinarySelector(rng)
        self.transform = transform

    def evolve(self, phi, actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param actVal Value of the action at phi.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - Action evaluated at new configuration
          - Point along trajectory that was selected
          - Weights for re-weighting for new configuration, not including the action.
            `dict` or `None`.
        """

        # get start phi for MD integration
        (phiMD, logdetJ) = (phi, 0) if self.transform is None else self.transform.backward(phi)

        # do MD integration
        pi = Vector(self.rng.normal(0, 1, len(phi))+0j)
        phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action, self.length, self.nstep)

        # transform to MC manifold
        (phi1, actVal1, logdetJ1) = (phiMD1, actValMD1, 0) if self.transform is None \
            else self.transform.forward(phiMD1, actValMD1)

        # accept/reject on MC manifold
        trajPoint = self.selector.selectTrajPoint(actVal+np.linalg.norm(pi)**2/2+logdetJ,
                                                  actVal1+np.linalg.norm(pi1)**2/2+logdetJ1)
        extraWeights = None if self.transform is None\
            else {"logdetJ": (logdetJ, logdetJ1)[trajPoint]}
        return (phi1, actVal1, trajPoint, extraWeights) if trajPoint == 1 \
            else (phi, actVal, trajPoint, extraWeights)

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["length"] = self.length
        h5group["nstep"] = self.nstep
        if self.transform is not None:
            # TODO dispatch to the manager as well
            self.transform.save(h5group.create_group("transform"), manager)

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
        if "transform" in h5group:
            # TODO get type from manager
            transform = None
        else:
            transform = None
        return cls(action, h5group["length"][()], h5group["nstep"][()], rng, transform)


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
        self.rng = rng
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

    def evolve(self, phi, actVal, _trajPoint):
        r"""!
        Run leapfrog integrator.
        \param phi Input configuration.
        \param actVal Value of the action at phi.
        \param _trajPoint \e ignored.
        \returns In order:
          - New configuration
          - Action evaluated at new configuration
          - Point along trajectory that was selected
          - Weights for re-weighting for new configuration, not including the action.
            `dict` or `None`.
        """
        self._current += 1

        pi = Vector(self.rng.normal(0, 1, len(phi))+0j)
        phi1, pi1, actVal1 = leapfrog(phi, pi, self.action,
                                      next(self._lengthIter), int(next(self._nstepIter)))
        trajPoint = self.selector.selectTrajPoint(actVal+np.linalg.norm(pi)**2/2,
                                                  actVal1+np.linalg.norm(pi1)**2/2)
        return (phi1, actVal1, trajPoint, None) if trajPoint == 1 \
            else (phi, actVal, trajPoint, None)

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
