r"""!\file
\ingroup evolvers
Evolvers that perform molecular dynamics integration of configurations using leapfrog.
"""

import numpy as np

from .evolver import Evolver
from .transform import backwardTransform, forwardTransform
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
        self.trajPoints = []

    def evolve(self, stage):
        r"""!
        Run leapfrog integrator.
        \param stage EvolutionStage at the beginning of this evolution step.
        \returns EvolutionStage at the end of this evolution step.
        """

        # get start phi for MD integration
        phiMD, logdetJ = backwardTransform(self.transform, stage)
        if self.transform is not None and "logdetJ" not in stage.logWeights:
            stage.logWeights["logdetJ"] = logdetJ

        # do MD integration
        pi = Vector(self.rng.normal(0, 1, len(stage.phi))+0j)
    
        phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action, self.length, self.nstep)

        # transform to MC manifold
        phi1, actVal1, logdetJ1 = forwardTransform(self.transform, phiMD1, actValMD1)
        
        # accept/reject on MC manifold
        energy0 = stage.sumLogWeights()+0.5*np.linalg.norm(pi)**2  
        energy1 = actVal1+logdetJ1+0.5*np.linalg.norm(pi1)**2 

        trajPoint = self.selector.selectTrajPoint(energy0, energy1)
        self.trajPoints.append(trajPoint)

        logWeights = None if self.transform is None \
            else {"logdetJ": (logdetJ, logdetJ1)[trajPoint]}
        return stage.accept(phi1, actVal1, logWeights) if trajPoint == 1 \
            else stage.reject()

    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """
        h5group["length"] = self.length
        h5group["nstep"] = self.nstep
        if self.transform is not None:
            manager.save(self.transform, h5group.create_group("transform"))

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed evolver.
        """
        if "transform" in h5group:
            transform = manager.load(h5group[f"transform"], action, lattice, rng)
        else:
            transform = None
        return cls(action, h5group["length"][()], h5group["nstep"][()], rng, transform)

    def report(self):
        r"""!
        Return a string summarizing the evolution since the evolver
        was constructed including by fromH5.
        """
        return f"""<ConstStepLeapfrog> (0x{id(self):x})
  length = {self.length}, nstep = {self.nstep}
  acceptance rate = {np.mean(self.trajPoints)}"""


class LinearStepLeapfrog(Evolver):
    r"""! \ingroup evolvers
    A leapfrog evolver with linearly changing parameters.

    Both trajectory length and number of MD steps are interpolated between
    starting and final values. If the number of interpolating steps is lower than the number
    of trajectories computed using the evolver, the parameters stay at their final
    values.
    """

    def __init__(self, action, lengthRange, nstepRange, ninterp, rng, startPoint=0, transform=None):
        r"""!
        \param action Instance of isle.Action to use for molecular dynamics.
        \param lengthRange Tuple of initial and final trajectory lengths.
        \param nstepRange Tuple of initial and final number of steps.
        \param ninterp Number of interpolating steps.
        \param rng Central random number generator for the run. Used for accept/reject.
        \param startPoint Iteration number to start at.
        \param transform (Instance of isle.evolver.transform.Transform)
                         Used this to transform a configuration after MD integration
                         but before Metropolis accept/reject.
        """

        self.action = action
        self.lengthRange = lengthRange
        self.nstepRange = nstepRange
        self.ninterp = ninterp
        self.rng = rng
        self.selector = BinarySelector(rng)
        self.transform = transform
        self.trajPoints = []

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

    def evolve(self, stage):
        r"""!
        Run leapfrog integrator.
        \param stage EvolutionStage at the beginning of this evolution step.
        \returns EvolutionStage at the end of this evolution step.
        """
        self._current += 1

        # get start phi for MD integration
        phiMD, logdetJ = backwardTransform(self.transform, stage)
        if self.transform is not None and "logdetJ" not in stage.logWeights:
            stage.logWeights["logdetJ"] = logdetJ
        
        # do MD integration
        pi = Vector(self.rng.normal(0, 1, len(stage.phi))+0j)
        
        phiMD1, pi1, actValMD1 = leapfrog(phiMD, pi, self.action,
                                          next(self._lengthIter), int(next(self._nstepIter)))

        # transform to MC manifold
        phi1, actVal1, logdetJ1 = forwardTransform(self.transform, phiMD1, actValMD1)

        # accept/reject on MC manifold
        energy0 = stage.sumLogWeights()+0.5*np.linalg.norm(pi)**2
        energy1 = actVal1+logdetJ1+np.linalg.norm(pi1)**2/2
        trajPoint = self.selector.selectTrajPoint(energy0, energy1)
        self.trajPoints.append(trajPoint)

        logWeights = None if self.transform is None \
            else {"logdetJ": (logdetJ, logdetJ1)[trajPoint]}
        return stage.accept(phi1, actVal1, logWeights) if trajPoint == 1 \
            else stage.reject()

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
        if self.transform is not None:
            manager.save(self.transform, h5group.create_group("transform"))

    @classmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct from HDF5.
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed evolver.
        """
        if "transform" in h5group:
            transform = manager.load(h5group[f"transform"], action, lattice, rng)
        else:
            transform = None
        return cls(action,
                   (h5group["minLength"][()], h5group["maxLength"][()]),
                   (h5group["minNstep"][()], h5group["maxNstep"][()]),
                   h5group["ninterp"][()],
                   rng,
                   startPoint=h5group["current"][()],
                   transform=transform)

    def report(self):
        r"""!
        Return a string summarizing the evolution since the evolver
        was constructed including by fromH5.
        """
        return f"""<LinearStepLeapfrog> (0x{id(self):x})
  lengthRange = {self.lengthRange}, nstepRange = {self.nstepRange}, ninterp = {self.ninterp}
  acceptance rate = {np.mean(self.trajPoints)}"""
