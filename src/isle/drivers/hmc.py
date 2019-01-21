from logging import getLogger

import numpy as np
import h5py as h5

from .. import Vector, fileio, cli
from ..meta import sourceOfFunction, callFunctionFromSource
from ..util import verifyMetadataByException
from ..proposers import ProposerManager


class HMC:
    def __init__(self, lattice, params, rng, action, outfname, startIdx, definitions={}):
        self.lattice = lattice
        self.params = params
        self.rng = rng
        self.action = action
        self.outfname = str(outfname)

        self._trajIdx = startIdx
        self._propManager = ProposerManager(outfname, definitions=definitions)

    def __call__(self, phi, proposer, ntr,  saveFreq, checkpointFreq):
        if checkpointFreq != 0 and checkpointFreq % saveFreq != 0:
            raise ValueError("checkpointFreq must be a multiple of saveFreq."
                             f" Got {checkpointFreq} and {saveFreq}, resp.")

        trajPoint = 1  # point of last trajectory that was selected
        actVal = self.action.eval(phi)  # running action (without pi)

        for _ in cli.progressRange(ntr, message="HMC evolution",
                                   updateRate=max(ntr//100, 1)):
            # get initial conditions for proposer
            startPhi, startPi, startActVal = phi, Vector(self.rng.normal(0, 1, len(phi))+0j), actVal

            # get new proposed fields
            endPhi, endPi, endActVal = proposer.propose(startPhi, startPi, startActVal, trajPoint)

            # TODO consistency checks

            # accept-reject
            deltaE = np.real((endActVal + np.linalg.norm(endPi)**2/2)
                             - (startActVal + np.linalg.norm(startPi)**2/2))
            if deltaE < 0 or np.exp(-deltaE) > self.rng.uniform(0, 1):
                trajPoint = 1
                phi = endPhi
                actVal = endActVal
            else:
                trajPoint = 0
                phi = startPhi
                actVal = startActVal

            # TODO inline meas

            if saveFreq != 0 and self._trajIdx % saveFreq == 0:
                if checkpointFreq != 0 and self._trajIdx % checkpointFreq == 0:
                    self.saveFieldAndCheckpoint(phi, actVal, trajPoint, proposer)
                else:
                    self.save(phi, actVal, trajPoint)
            else:
                self.advance()

        return phi

    def saveFieldAndCheckpoint(self, phi, actVal, trajPoint, proposer):
        "!Write a trajectory (endpoint) and checkpoint to file and advance internal counter."
        with h5.File(self.outfname, "a") as outf:
            cfgGrp = self._writeTrajectory(outf, phi, actVal, trajPoint)
            self._writeCheckpoint(outf, cfgGrp, proposer)
        self._trajIdx += 1

    def save(self, phi, actVal, trajPoint):
        "!Write a trajectory (endpoint) to file and advance internal counter."
        with h5.File(self.outfname, "a") as outf:
            self._writeTrajectory(outf, phi, actVal, trajPoint)
        self._trajIdx += 1

    def advance(self, amount=1):
        "!Advance the internal trajectory counter by amount without saving."
        self._trajIdx += amount

    def resetIndex(self, idx=0):
        "!Reset the internal trajectory index to idx."
        self._trajIdx = idx

    def _writeTrajectory(self, h5file, phi, actVal, trajPoint):
        "!Write a trajectory (endpoint) to a HDF5 group."
        try:
            return fileio.h5.writeTrajectory(h5file["configuration"], self._trajIdx,
                                             phi, actVal, trajPoint)
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                raise RuntimeError(f"Cannot write trajectory {self._trajIdx} to file '{self.outfname}'."
                                   " A dataset with the same name already exists.") from None
            raise

    def _writeCheckpoint(self, h5file, trajGrp, proposer):
        "!Write a checkpoint to a HDF5 group."
        try:
            return fileio.h5.writeCheckpoint(h5file["checkpoint"], self._trajIdx,
                                             self.rng, trajGrp.name, proposer, self._propManager)
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
        fileio.h5.initializeFile(outfile[0], lattice, params, makeActionSrc,
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
