"""!
Hybrid Monte-Carlo.
"""

## \defgroup proposers Proposers
# Propose new configurations for HMC.


import numpy as np

from . import Vector, leapfrog
from .util import hingeRange


def _initialConditions(ham, oldPhi, oldAct):
    r"""!
    Construct initial conditions for proposer.

    \param ham Hamiltonian.
    \param oldPhi Old configuration, result of previous run or some new phi.
    \param oldAct Old action, result of previous run or `None` if first run.

    \returns Tuple `(phi, pi, energy)`.
    """

    # new random pi
    pi = Vector(np.random.normal(0, 1, len(oldPhi))+0j)
    if oldAct is None:
        # need to compute energy from scratch
        energy = ham.eval(oldPhi, pi)
    else:
        # use old action for energy
        energy = ham.addMomentum(pi, oldAct)
    return oldPhi, pi, energy

def hmc(phi, ham, proposer, ntr, measurements=[], checks=[]):
    r"""!
    Compute Hybrid Monte-Carlo trajectories.

    Evolves a configuration using a proposer and an accept-reject step.
    Optionally performs measurements and consistency checks on the fly.
    Note that no results are saved, use measurements to store configurations on disk.

    \param phi Initial configuration.

    \param ham Hamiltonian describing the model.

    \param proposer Callable that proposes a new configuration which can be accepted
                    or rejected. The proposer shall return a new phi and a new pi.
                    It is called with arguments `startPhi, startPi, acc`, where:
                      - `startPhi`: Initial configuration.
                      - `startPi`: Initial momentum.
                      - `acc`: `True` if previous trajectory was accepted, `False` otherwise.

    \param ntr Number of trajectories to compute.

    \param measurements List of tuples `(freq, meas)`, where `freq` is the measurement
                        frequency: 1 means measure every trajectory, 2 means
                        measure every second trajectory, etc. The requirements on meas
                        are listed under \ref measdoc.

     \param checks List of tuples `(freq, check)`, where `freq` is the check
                   frequency: 1 means check every trajectory, 2 means check every
                   second trajectory, etc.<BR>
                   `check` is a callable with arguments `startPhi`, `startPi`, `startEnergy`,
                   `endPhi`, `endPi`, `endEnergy` which shall not return a value but
                   raise an exception in case of failure. Arguments are:
                     - `startPhi`/`endPhi`: Configuration before and after the proposer.
                     - `startPi`/`endPi`: Momentum before and after the proposer.
                     - `startEnergy`/`endEnergy`: Energy before and after the proposer,
                                                  includes the momentum.

    \returns Result configuration after all trajectories.
    """

    acc = True  # was last trajectory accepted?
    act = None  # running action (without pi)
    for itr in range(ntr):
        # get initial conditions for proposer
        startPhi, startPi, startEnergy = _initialConditions(ham, phi, act)

        # evolve fields using proposer
        endPhi, endPi = proposer(startPhi, startPi, acc)
        # get new energy
        endEnergy = ham.eval(endPhi, endPi)

        # perform consistency checks
        for (freq, check) in checks:
            if freq > 0 and itr % freq == 0:
                check(startPhi, startPi, startEnergy,
                      endPhi, endPi, endEnergy)

        # accept-reject
        deltaE = np.real(endEnergy - startEnergy)
        if deltaE < 0 or np.exp(-deltaE) > np.random.uniform(0, 1):
            acc = True
            phi = endPhi
            act = ham.stripMomentum(endPi, endEnergy)
        else:
            acc = False
            phi = startPhi
            act = ham.stripMomentum(startPi, startEnergy)

        # perform measurements
        for (freq, meas) in measurements:
            if freq > 0 and itr % freq == 0:
                meas(phi, True, itr=itr, act=act, acc=acc)

    return phi


class ConstStepLeapfrog:
    r"""!
    \ingroup proposers
    A leapfrog proposer with constant parameters.
    """

    def __init__(self, hamiltonian, length, nstep):
        r"""!
        \param hamiltonian Instance of cnxx.Hamiltonian to use for molecular dynamics.
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
        \param acc `True` if last trajectory was accepted, `False` otherwise.
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
        \param hamiltonian Instance of cnxx.Hamiltonian to use for molecular dynamics.
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
        \param acc `True` if last trajectory was accepted, `False` otherwise.
        """
        return leapfrog(phi, pi, self.hamiltonian,
                            next(self._lengthIter), int(next(self._nstepIter)))
