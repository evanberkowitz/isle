from pathlib import Path

import numpy as np
import h5py as h5

from .. import Vector
from .. import fileio

class HMC:
    def __init__(self, lat, params, rng, action, outfname, startIdx):
        self.lat = lat
        self.params = params
        self.rng = rng
        self.action = action
        self.outfname = outfname

        self._trajIdx = startIdx

def _isValidOutfileByException(outfile, overwrite, startIdx):
    """!
    Check if the output file is a valid parameter and if it is possible to write to it.

    Writing is not possible if the file exists and `overwrite == False` and
    it contains configurations with an index greater than `startIdx`.

    \throws ValueError if output file type is not supported.
    \throws RuntimeError if the file is not valid.
    """

    if outfile is None:
        print("Error: no output file given")
        raise RuntimeError("No output file given to HMC driver.")

    if outfile[1] != fileio.FileType.HDF5:
        raise ValueError(f"Output file type no supported by HMC driver. Output file is '{outfile[0]}'")

    outfname = outfile[0]
    if outfname.exists():
        if overwrite:
            print(f"Output file '{outfname}' exists -- overwriting")
            outfname.unlink()

        else:
            with h5.File(str(outfname), "r") as outf:
                # get greatest index of stored config
                lastStored = max(map(int, outf["configuration"].keys()))
                if lastStored > startIdx:
                    print(f"Error: Output file '{outfname}' exists and has entries with higher index than HMC start index.")
                    raise RuntimeError("Cannot write into output file, contains newer data")

                print(f"Output file '{outfname}' exists -- appending")

def init(lat, params, rng, makeAction, outfile,
         overwrite, startIdx=0):

    _isValidOutfileByException(outfile, overwrite, startIdx)

    makeActionSrc = fileio.sourceOfFunction(makeAction)
    driver = HMC(lat, params, rng,
                 fileio.callFunctionFromSource(makeActionSrc, lat, params), outfile[0], startIdx)
    return driver




def _initialConditions(ham, oldPhi, oldAct, rng):
    r"""!
    Construct initial conditions for proposer.

    \param ham Hamiltonian.
    \param oldPhi Old configuration, result of previous run or some new phi.
    \param oldAct Old action, result of previous run or `None` if first run.
    \param rng Randum number generator that implements isle.random.RNGWrapper.

    \returns Tuple `(phi, pi, energy)`.
    """

    # new random pi
    pi = Vector(rng.normal(0, 1, len(oldPhi))+0j)
    if oldAct is None:
        # need to compute energy from scratch
        energy = ham.eval(oldPhi, pi)
    else:
        # use old action for energy
        energy = ham.addMomentum(pi, oldAct)
    return oldPhi, pi, energy

def run_hmc(phi, ham, proposer, ntr, rng, measurements=[], checks=[], itrOffset=0):
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

    \param rng Randum number generator that implements isle.random.RNGWrapper.

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

    \param itrOffset Is added to the trajectory index when calling measurements.
                     Also affects when the measurement gets called.

    \returns Result configuration after all trajectories.
    """

    acc = True  # was last trajectory accepted?
    act = None  # running action (without pi)
    for itr in range(ntr):
        # get initial conditions for proposer
        startPhi, startPi, startEnergy = _initialConditions(ham, phi, act, rng)

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
        if deltaE < 0 or np.exp(-deltaE) > rng.uniform(0, 1):
            acc = True
            phi = endPhi
            act = ham.stripMomentum(endPi, endEnergy)
        else:
            acc = False
            phi = startPhi
            act = ham.stripMomentum(startPi, startEnergy)

        # perform measurements
        for (freq, meas) in measurements:
            if freq > 0 and (itr+itrOffset) % freq == 0:
                meas(phi, True, itr=itr+itrOffset, act=act, acc=acc, rng=rng)

    return phi
