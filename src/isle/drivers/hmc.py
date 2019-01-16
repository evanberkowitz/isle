from logging import getLogger

import numpy as np
import h5py as h5

from .. import Vector
from .. import fileio
from .. import cli
from ..meta import sourceOfFunction, callFunctionFromSource
from ._util import verifyMetadataByException, prepareOutfile


class HMC:
    def __init__(self, lattice, params, rng, action, outfname, startIdx):
        self.lattice = lattice
        self.params = params
        self.rng = rng
        self.ham = action  # TODO rewrite for pure action
        self.outfname = str(outfname)

        self._trajIdx = startIdx


    def __call__(self, phi, proposer, ntr,  saveFreq, checkpointFreq):
        if checkpointFreq != 0 and checkpointFreq % saveFreq != 0:
            raise ValueError("checkpointFreq must be a multiple of saveFreq."
                             f" Got {checkpointFreq} and {saveFreq}, resp.")

        acc = 1  # was last trajectory accepted? (int so it can be written as trajPoint)
        act = None  # running action (without pi)
        for _ in cli.progressRange(ntr, message="HMC evolution",
                                   updateRate=max(ntr//100, 1)):
            # get initial conditions for proposer
            startPhi, startPi, startEnergy = _initialConditions(self.ham, phi, act, self.rng)

            # evolve fields using proposer
            endPhi, endPi, endEnergy = proposer.propose(startPhi, startPi, startEnergy, acc)

            # TODO consistency checks

            # accept-reject
            deltaE = np.real(endEnergy - startEnergy)
            if deltaE < 0 or np.exp(-deltaE) > self.rng.uniform(0, 1):
                acc = 1
                phi = endPhi
                act = self.ham.stripMomentum(endPi, endEnergy)
            else:
                acc = 0
                phi = startPhi
                act = self.ham.stripMomentum(startPi, startEnergy)

            # TODO inline meas

            if saveFreq != 0 and self._trajIdx % saveFreq == 0:
                if checkpointFreq != 0 and self._trajIdx % checkpointFreq == 0:
                    self.saveFieldAndCheckpoint(phi, act, acc)
                else:
                    self.save(phi, act, acc)
            else:
                self.advance()

        return phi

    def saveFieldAndCheckpoint(self, phi, act, acc):
        "!Write a trajectory (endpoint) and checkpoint to file and advance internal counter."
        with h5.File(self.outfname, "a") as outf:
            cfgGrp = self._writeTrajectory(outf, phi, act, acc)
            self._writeCheckpoint(outf, cfgGrp)
        self._trajIdx += 1

    def save(self, phi, act, acc):
        "!Write a trajectory (endpoint) to file and advance internal counter."
        with h5.File(self.outfname, "a") as outf:
            self._writeTrajectory(outf, phi, act, acc)
        self._trajIdx += 1

    def advance(self, amount=1):
        "!Advance the internal trajectory counter by amount without saving."
        self._trajIdx += amount

    def resetIndex(self, idx=0):
        "!Reset the internal trajectory index to idx."
        self._trajIdx = idx

    def _writeTrajectory(self, h5file, phi, act, trajPoint):
        "!Write a trajectory (endpoint) to a HDF5 group."
        try:
            return fileio.h5.writeTrajectory(h5file["configuration"], self._trajIdx,
                                             phi, act, trajPoint)
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                raise RuntimeError(f"Cannot write trajectory {self._trajIdx} to file '{self.outfname}'."
                                   " A dataset with the same name already exists.") from None
            raise

    def _writeCheckpoint(self, h5file, trajGrp):
        "!Write a checkpoint to a HDF5 group."
        try:
            return fileio.h5.writeCheckpoint(h5file["checkpoint"], self._trajIdx,
                                             self.rng, trajGrp.name)
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                raise RuntimeError(f"Cannot write checkpoint for trajectory {self._trajIdx} to file '{self.outfname}'."
                                   " A dataset with the same name already exists.") from None
            raise


def init(lattice, params, rng, makeAction, outfile, overwrite, startIdx=0):
    if outfile is None:
        getLogger(__name__).error("No output file given for HMC driver.")
        raise ValueError("No output file")

    # convert to (name, type) tuple if necessary
    if not isinstance(outfile, (tuple, list)):
        outfile = fileio.pathAndType(outfile)
    _ensureIsValidOutfile(outfile, overwrite, startIdx, lattice, params)

    makeActionSrc = sourceOfFunction(makeAction)
    if not outfile[0].exists():
        prepareOutfile(outfile[0], lattice, params, makeActionSrc,
                       ["/configuration", "/checkpoint"])

    return HMC(lattice, params, rng, callFunctionFromSource(makeActionSrc, lattice, params),
               outfile[0], startIdx)


def _latestConfig(fname):
    "!Get greatest index of stored configs."
    with h5.File(str(fname), "r") as h5f:
        return max(map(int, h5f["configuration"].keys()), default=0)

def _verifyConfigsByException(outfname, startIdx):
    # TODO what about checkpoints?

    # TODO if there are no configs -> ok but warn

    lastStored = _latestConfig(outfname)
    if lastStored > startIdx:
        getLogger(__name__).error(
            "Error: Output file '%s' exists and has entries with higher index than HMC start index.\n"
            "Greatest index in file: %d, user set start index: %d",
            outfname, lastStored, startIdx)
        raise RuntimeError("Cannot write into output file, contains newer data")

def _ensureIsValidOutfile(outfile, overwrite, startIdx, lattice, params):
    r"""!
    Check if the output file is a valid parameter and if it is possible to write to it.
    Deletes the file if `overwrite == True`.

    Writing is not possible if the file exists and `overwrite == False` and
    it contains configurations with an index greater than `startIdx`.

    \throws ValueError if output file type is not supported.
    \throws RuntimeError if the file is not valid.
    """

    if outfile[1] != fileio.FileType.HDF5:
        getLogger(__name__).error("Output file type not supported by HMC driver: %s", outfile[1])
        raise ValueError(f"Output file type no supported by HMC driver. Output file is '{outfile[0]}'")

    outfname = outfile[0]
    if outfname.exists():
        if overwrite:
            getLogger(__name__).info("Output file '%s' exists -- overwriting", outfname)
            outfname.unlink()

        else:
            verifyMetadataByException(outfname, lattice, params)
            _verifyConfigsByException(outfname, startIdx)
            getLogger(__name__).info("Output file '%s' exists -- appending", outfname)

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
