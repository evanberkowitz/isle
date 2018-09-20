"""!
Proposers for HMC evolutions.
"""

## \defgroup proposers Proposers
# Propose new configurations for HMC.


import numpy as np

from . import Vector, leapfrog
from .util import hingeRange


class ConstStepLeapfrog:
    r"""!
    \ingroup proposers
    A leapfrog proposer with constant parameters.
    """

    def __init__(self, hamiltonian, length, nstep):
        r"""!
        \param hamiltonian Instance of isle.Hamiltonian to use for molecular dynamics.
        \param length Length of the MD trajectory.
        \param nstep Number of MD steps per trajectory.
        """
        self.hamiltonian = hamiltonian
        self.length = length
        self.nstep = nstep

    def __call__(self, phi, pi, acc):
        r"""!
        Run leapfrog integrator.
        \param phi Starting configuration.
        \param pi Starting momentum.
        \param acc \e Ignored
        """
        return leapfrog(phi, pi, self.hamiltonian, self.length, self.nstep)


class LinearStepLeapfrog:
    r"""!
    \ingroup proposers
    A leapfrog proposer with linearly changing parameters.

    Both trajectory length and number of MD steps are interpolated between
    starting and final values. If the number of interpolating steps is lower than the number
    of trajectories computed using the proposer, the parameters stay at their final
    values.
    """

    def __init__(self, hamiltonian, lengthRange, nstepRange, ninterp):
        r"""!
        \param hamiltonian Instance of isle.Hamiltonian to use for molecular dynamics.
        \param lengthRange Tuple of initial and final trajectory lengths.
        \param nstepRange Tuple of initial and final number of steps.
        \param ninterp Number of interpolating steps.
        """
        self.hamiltonian = hamiltonian
        self._lengthIter = hingeRange(*lengthRange, (lengthRange[1]-lengthRange[0])/ninterp)
        self._nstepIter = hingeRange(*nstepRange, (nstepRange[1]-nstepRange[0])/ninterp)

    def __call__(self, phi, pi, acc):
        r"""!
        Run leapfrog integrator.
        \param phi Starting configuration.
        \param pi Starting momentum.
        \param acc \e Ignored
        """
        return leapfrog(phi, pi, self.hamiltonian,
                        next(self._lengthIter), int(next(self._nstepIter)))
