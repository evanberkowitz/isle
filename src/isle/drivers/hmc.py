r"""!\file
Handle %HMC evolution and file output.
"""

from logging import getLogger

import h5py as h5

import time
import nvtx

from .. import Vector, fileio, cli
from ..meta import sourceOfFunction, callFunctionFromSource
from ..util import verifyVersionsByException
from ..evolver import EvolverManager, EvolutionStage

class HMC:
    r"""!
    Driver to control %HMC evolution.

    Stores metadata and current state of %HMC evolution (apart from configuration and evolver).
    Manages generation of configurations and file output.

    \note Use isle.drivers.hmc.newRun() or isle.drivers.hmc.continueRun()
          factory functions to construct this driver and set up / load files.
    """

    def __init__(self, lattice, params, rng, action, outfname, startIdx,
                 definitions={}, evManager=None):
        """!
        Construct with given parameters.
        """

        self.lattice = lattice
        self.params = params
        self.rng = rng
        self.action = action
        self.outfname = str(outfname)

        self._trajIdx = startIdx
        self._evManager = evManager if evManager \
            else EvolverManager(outfname, definitions=definitions)

    @nvtx.annotate(message="py_call_hmcState", color="orange")
    def __call__(self, stage, evolver, ntr, saveFreq, checkpointFreq, maxNtr=None):
        r"""!
        Evolve configuration using %HMC.

        \param stage Start configuration or evolution stage to evolve.
        \param evolver Used to evolve configurations for Metropolis accept/reject.
        \param ntr Number of trajectories to generate.
                   May be `None` in which case production never stops unless the
                   evolver raises an instance of `StopIteration`.
        \param saveFreq Save configurations every `saveFreq` trajectories.
        \param checkpointFreq Write a checkpoint every `checkpointFreq` trajectories.
                              Must be a multiple of `saveFreq`.
        \param maxNtr If not `None` and `ntr is None`, stop after at most `maxNtr`
                      trajectories. Ignored if `ntr is not None`.

        \returns EvolutionStage of last trajectory.
        """

        if checkpointFreq != 0 and checkpointFreq % saveFreq != 0:
            getLogger(__name__).error("checkpointFreq must be a multiple of saveFreq."
                                      " Got %d and %d, resp.", checkpointFreq, saveFreq)
            raise ValueError("checkpointFreq must be a multiple of saveFreq.")

        if not isinstance(stage, EvolutionStage):
            # make sure it is an EvolutionStage
            stage = EvolutionStage(stage, self.action.eval(stage), 1)

        for _ in _iterTrajectories(ntr, maxNtr):
            try:
                # do evolution
                stage = evolver.evolve(stage)
            except StopIteration:
                # the evolver wants to stop,
                # don't advance or save because no new config was produced
                break

            # TODO inline meas

            # advance before saving because phi is a new configuration (no 0 is handled above)
            self.advance()
            self._saveConditionally(stage, evolver, saveFreq, checkpointFreq)

        _report(evolver)
        return stage

    def saveFieldAndCheckpoint(self, stage, evolver):
        """!
        Write a trajectory (endpoint) and checkpoint to file.
        """
        with h5.File(self.outfname, "a") as outf:
            cfgGrp = self._writeTrajectory(outf, stage)
            self._writeCheckpoint(outf, cfgGrp, evolver)

    def save(self, stage):
        """!
        Write a trajectory (endpoint) to file.
        """
        with h5.File(self.outfname, "a") as outf:
            self._writeTrajectory(outf, stage)

    def advance(self, amount=1):
        """!
        Advance the internal trajectory counter by amount without saving.
        """
        self._trajIdx += amount

    @nvtx.annotate(message="py_resetIndex", color="green")
    def resetIndex(self, idx=0):
        """!
        Reset the internal trajectory index to idx.
        """
        self._trajIdx = idx

    def _saveConditionally(self, stage, evolver, saveFreq, checkpointFreq):
        """!Save the trajectory and checkpoint if frequencies permit."""
        if saveFreq != 0 and self._trajIdx % saveFreq == 0:
            if checkpointFreq != 0 and self._trajIdx % checkpointFreq == 0:
                self.saveFieldAndCheckpoint(stage, evolver)
            else:
                self.save(stage)

    def _writeTrajectory(self, h5file, stage):
        """!
        Write a trajectory (endpoint) to a HDF5 group.
        """
        try:
            return fileio.h5.writeTrajectory(h5file["configuration"], self._trajIdx, stage)
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                getLogger(__name__).error("Cannot write trajectory %d to file %s."
                                          " A dataset with the same name already exists.",
                                          self._trajIdx, self.outfname)
            raise

    def _writeCheckpoint(self, h5file, trajGrp, evolver):
        """!
        Write a checkpoint to a HDF5 group.
        """
        try:
            return fileio.h5.writeCheckpoint(h5file["checkpoint"], self._trajIdx,
                                             self.rng, trajGrp.name, evolver, self._evManager)
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                getLogger(__name__).error("Cannot write checkpoint for trajectory %d to file %s."
                                          " A dataset with the same name already exists.",
                                          self._trajIdx, self.outfname)
            raise


def newRun(lattice, params, rng, makeAction, outfile, overwrite, definitions={}):
    r"""!
    Start a fresh %HMC run.

    Constructs a %HMC driver from given parameters and initializes the output file.
    Most parameters are stored in the HMC object under the same name.

    \param lattice Lattice to run simulation on, passed to `makeAction`.
    \param params Parameters passed to `makeAction`.
    \param rng Random number generator used for all random numbers needed during %HMC evolution.
    \param makeAction Function or source code of a function to construct an action.
                      Must be self-contained!
    \param outfile Name (Path) of the output file. Must not exist unless `overwrite==True`.
    \param overwrite If `False`, nothing in the output file will be erased/overwritten.
                     If `True`, the file is removed and re-initialized, whereby all content is lost.
    \param definitions Dictionary of mapping names to custom types. Used to control how evolvers
                       are stored for checkpoints.

    \returns A new HMC instance to control evolution initialized with given parameters.
    """

    if outfile is None:
        getLogger(__name__).error("No output file given for HMC driver")
        raise ValueError("No output file")

    makeActionSrc = makeAction if isinstance(makeAction, str) else sourceOfFunction(makeAction)
    fileio.h5.initializeNewFile(outfile, overwrite, lattice, params, makeActionSrc,
                                ["/configuration", "/checkpoint"])

    return HMC(lattice, params, rng, callFunctionFromSource(makeActionSrc, lattice, params),
               outfile, 0, definitions)


def continueRun(infile, outfile, startIdx, overwrite, definitions={}):
    r"""!
    Continue a previous %HMC run.

    Loads metadata and a given checkpoint from the input file, constructs a new HMC driver
    object from them, and initializes the output file.

    \param infile Name of the input file. Must contain at least one checkpoint to continue from.
    \param outfile Name of the output file. Can be `None`, which means equal to the input file.
    \param startIdx Index of the checkpoint to start from. See parameter `overwrite`.
                    Can be negative in which case it is counted from the end (-1 is last checkpoint).
                    The number the checkpoint is saved as, not 'the nth checkpoint in the file'.
    \param overwrite If `False`, nothing in the output file will be erased/overwritten.
                     If `True`,
                        - (`infile==outfile`): all configurations and checkpoints newer than `startIdx`
                          are removed and have to be re-computed.
                        - (`infile!=outfile`): outfile is removed and re-initialized,
                          whereby all content is lost.
    \param definitions Dictionary of mapping names to custom types. Used to control how evolvers
                       are stored for checkpoints.

    \returns In order:
        - Instance of HMC constructed from parameters found in `infile`.
        - Configuration loaded from checkpoint.
        - Evolver loaded from checkpoint.
        - Save frequency computed on last two configurations.
          `None` if there is only one configuration.
        - Checkpoint frequency computed on last two checkpoints.
          `None` if there is only one checkpoint.
    """

    if infile is None:
        getLogger(__name__).error("No input file given for HMC driver in continuation run")
        raise ValueError("No input file")
    if outfile is None:
        getLogger(__name__).info("No output file given for HMC driver")
        outfile = infile

    lattice, params, makeActionSrc, versions = fileio.h5.readMetadata(infile)
    verifyVersionsByException(versions, infile)
    action = callFunctionFromSource(makeActionSrc, lattice, params)
    if outfile != infile:
        fileio.h5.initializeNewFile(outfile, overwrite, lattice, params, makeActionSrc,
                                    ["/configuration", "/checkpoint"])

    configurations, checkpoints = _loadIndices(infile)
    evManager = EvolverManager(infile, definitions=definitions)
    checkpointIdx, rng, stage, evolver = _loadCheckpoint(infile, startIdx, checkpoints,
                                                         evManager, action, lattice)

    if outfile == infile:
        _ensureNoNewerConfigs(infile, checkpointIdx, checkpoints, configurations, overwrite)

    return (HMC(lattice, params, rng, action, outfile,
                checkpointIdx,
                definitions,
                evManager if infile == outfile else None),  # need to re-init manager for new outfile
            stage,
            evolver,
            _stride(configurations),
            _stride(checkpoints))

def _stride(values):
    """!
    Calculate difference in values in an array.
    """
    try:
        return values[-1] - values[-2]
    except IndexError:
        return None  # cannot get stride with less than two elements

def _loadIndices(fname):
    """!
    Load all configuration anc checkpoint indices from a file.
    """
    with h5.File(str(fname), "r") as h5f:
        configurations = sorted(map(int, h5f["configuration"].keys()))
        checkpoints = sorted(map(int, h5f["checkpoint"].keys()))
    return configurations, checkpoints

def _ensureNoNewerConfigs(fname, checkpointIdx, checkpoints, configurations, overwrite):
    """!
    Check if there are configurations or checkpoints with indices greater than checkpointIdx.
    If so and `overwrite==True`, erase them.
    """

    latestCheckpoint = checkpoints[-1]
    if latestCheckpoint > checkpointIdx:
        message = f"Output file {fname} contains checkpoints with greater index than HMC starting point.\n" \
            f"    Greatest index is {latestCheckpoint}, start index is {checkpointIdx}."
        if not overwrite:
            getLogger(__name__).error(message)
            raise RuntimeError("HMC start index is not latest")
        else:
            getLogger(__name__).warning(message+"\n    Overwriting")
            _removeGreaterThan(fname, "checkpoints", checkpointIdx)

    latestConfig = configurations[-1]
    if latestConfig > checkpointIdx:
        message = f"Output file {fname} contains configurations with greater index than HMC starting point.\n" \
            f"    Greatest index is {latestConfig}, start index is {checkpointIdx}."
        if not overwrite:
            getLogger(__name__).error(message)
            raise RuntimeError("HMC start index is not latest")
        else:
            getLogger(__name__).warning(message+"\n    Overwriting")
            _removeGreaterThan(fname, "configuration", checkpointIdx)

def _removeGreaterThan(fname, groupPath, maxIdx):
    """!
    Remove all elements under groupPath in file that are greater then maxidx.
    """
    with h5.File(str(fname), "a") as h5f:
        grp = h5f[groupPath]
        for idx in grp.keys():
            if int(idx) > maxIdx:
                del grp[idx]


def _loadCheckpoint(fname, startIdx, checkpoints, evManager, action, lattice):
    """!
    Load a checkpoint from file allowing for negative indices.
    """

    if startIdx < 0:
        startIdx = checkpoints[-1] + (startIdx+1) # +1 so that startIdx=-1 gives last point

    if startIdx < 0 or startIdx > checkpoints[-1]:
        getLogger(__name__).error("Start index for HMC continuation is out of range: %d",
                                  startIdx)
        raise ValueError("Start index out of range")

    if startIdx not in checkpoints:
        getLogger(__name__).error("There is no checkpoint matching the given start index: %d",
                                  startIdx)
        raise ValueError("No checkpoint matching start index")

    with h5.File(str(fname), "r") as h5f:
        rng, cfgGrp, evolver = fileio.h5.loadCheckpoint(h5f["checkpoint"], startIdx,
                                                        evManager, action, lattice)
        stage = EvolutionStage.fromH5(cfgGrp)
        return startIdx, rng, stage, evolver

def _iterTrajectories(ntr, maxNtr):
    """!
    Iterator for production.
    Either iterate ntr times or infinitely long if ntr is None.
    Shows a progressbar in both cases.
    """

    if ntr is not None:
        yield from cli.progressRange(ntr, message="HMC evolution",
                                     updateRate=max(ntr//100, 1))

    else:
        with cli.trackProgress(ntr, message="HMC evolution", updateRate=1) as pbar:
            count = 0
            while True:
                yield count
                count += 1
                if maxNtr is not None and count >= maxNtr:
                    getLogger(__name__).warning("Reached maximum number of trajectories for "
                                                "'unbounded' evolution.")
                    return
                pbar.advance()

def _report(evolver):
    """!Log a report after evolution."""
    getLogger(__name__).info("Evolution stopped, evolver:\n%s", evolver.report())
