r"""!\file
Driver to perform measurements on configurations stored in a file.
"""

from logging import getLogger
from pathlib import Path

import h5py as h5
import psutil

from .. import fileio, cli
from ..collection import inSlice, withStart, withStop, withStep
from ..meta import callFunctionFromSource
from ..util import verifyVersionsByException
from ..evolver import EvolutionStage


class Measure:
    r"""!
    Driver to perform measurements on configurations stored in a file.

    Assumes that configurations are stored in the default format as written by e.g.
    the driver isle.drivers.hmc.HMC.
    Measurement results can be written to a new file or to the input file.
    The driver checks for conflicts should the output file already exist and removes
    conflicting entries iff it is initialized with `overwrite==True`.
    This assumes that the measurements only write to the file under their `savePath`
    attribute and nowhere else.

    \note The driver should not be initialized directly, instead use `isle.drivers.meas.init`.

    \see
        \ref filelayout and \ref measdoc
    """


    def __init__(self, lattice, params, action, infile, outfile,
                 overwrite, maxBufferSize=None, maxTotalMemory=None):
        ## The spatial lattice.
        self.lattice = lattice
        ## Run parameters.
        self.params = params
        ## The action (object).
        self.action = action
        ## Path to the input.
        self.infile = infile
        ## Path to the output.
        self.outfile = outfile
        ## True if existing data may be overwritten.
        self.overwrite = overwrite
        ## Maximum size to use for result buffers of measurements.
        self._maxBufferSize = maxBufferSize
        ## Maximum memory to use for all buffers.
        self._maxTotalMemory = maxTotalMemory

    def __call__(self, measurements, adjustConfigSlices=True):
        r"""!
        Apply measurements to all configurations in the input file and
        save results to the output file.

        Makes sure that the measurements can be saved without conflicts.
        If entries with the same paths as measurement save paths exist in the file, they are
        deleted iff `self.overwrite == True`.
        Otherwise, a `RuntimeError` is raised.

        Reads configurations from the input file in the order of the Markov chain.
        Compares the trajectory index extracted from file to the configSlice attribute of each
        measurement and if the index is in the slice, calls the measurement with that trajectory.

        \warning Calls Measure.mapOverConfigs with `adjustSlices=True`. See doc of that function.

        \param measurements List of instances of isle.meas.measurement.Measurement to be called
                            on each configuration in the input file.
        \param adjustConfigSlices If `True`, the configuration slices in all measurements
                                  will be modified to reflect the actual range of
                                  configurations that the measurement is performed on.
        """

        _ensureCanWriteMeas(self.outfile, measurements, self.overwrite)

        with h5.File(self.infile, ("r+" if self.outfile == self.infile else "r")) as cfgf:
            # get all configuration groups (index, h5group) pairs
            configurations = fileio.h5.loadList(cfgf["/configuration"])

            _setupMeasurements(measurements, configurations, self.outfile, self.lattice,
                               adjustConfigSlices, self._maxBufferSize, self._maxTotalMemory)

            # apply measurements to all configs
            with cli.trackProgress(len(configurations), "Measurements", updateRate=1000) as pbar:
                for itr, grp in configurations:
                    # load trajectory
                    stage = EvolutionStage.fromH5(grp)

                    # measure
                    for imeas, measurement in enumerate(measurements):
                        if inSlice(itr, measurement.configSlice):
                            measurement(stage, itr)
                        elif itr >= measurement.configSlice.stop:
                            # this measurement is done => drop it
                            del measurements[imeas]
                    if not measurements:
                        break  # no measurements left

                    pbar.advance()


def init(infile, outfile, overwrite, maxBufferSize=None, maxTotalMemory=None):
    r"""!
    Initialize a new measurement driver.

    Reads parameters and other metadata from the input file to set up a new Measure driver.
    Ensures that the output file can be written to and stores the meatadata in it if necessary.

    \param infile Path to the input file. Must contain configurations.
    \param outfile Path to the output file. May be the same as `infile`.
                   If this file exists, it must be compatible with the metadata and measurements.
                   Conflicts with existing measurement results are only checked by the driver
                   when the actual measurements are known.
    \param overwrite Indicate whether data in the output file may be overwritten.
    \param maxBufferSize Maximum size that may be used for result buffers in bytes (per buffer).
    \param maxBufferSize Maximum total memory that may be used for result buffers in bytes (all buffers).
    \returns A new isle.drivers.meas.Measure driver.
    """

    if infile is None:
        getLogger(__name__).error("No input file given to meas driver.")
        raise ValueError("No intput file")

    if outfile is None:
        getLogger(__name__).info("No output file given to meas driver. "
                                 "Writing to input file.")
        outfile = infile

    lattice, params, makeActionSrc, versions = fileio.h5.readMetadata(infile)
    verifyVersionsByException(versions, infile)
    _ensureIsValidOutfile(outfile, lattice, params, makeActionSrc)

    return Measure(lattice, params,
                   callFunctionFromSource(makeActionSrc, lattice, params),
                   infile, outfile, overwrite, maxBufferSize, maxTotalMemory)

def _isValidPath(path):
    """!Check if parameter is a valid path to a measurement inside an HDF5 file."""

    nonempty = [component for component in str(path).split("/") if component.strip()]
    if len(nonempty) == 0:
        getLogger(__name__).error(
            "Output path of a measurement is the root HDF5 group. "
            "All measurements must be stored in a subgroup")
        return False
    return True

def _ensureCanWriteMeas(outfile, measurements, overwrite):
    r"""!
    Ensure that measurements can be written to the output file.
    \param outfile The output file, must already exist!
    \param measurements Measurements that want to save at some point.
    \param overwrite If `True`, erase all existing HDF5 objects under the given paths.
                     if `False`, fail if an object exists under any given path.
    """

    paths = [measurement.savePath for measurement in measurements]

    # check if all paths are OK
    for path in paths:
        if not _isValidPath(path):
            raise ValueError(f"Invalid output path for measurement: {path}")

    # make sure that no paths point to existing objects in the output file
    with h5.File(outfile, "a" if overwrite else "r") as h5f:
        for path in paths:
            if path in h5f:
                if overwrite:
                    # preemptively get rid of that HDF5 object
                    getLogger(__name__).warning("Removing object from output file: %s",
                                                path)
                    del h5f[path]
                else:
                    # can't remove old object, needs action from user
                    getLogger(__name__).error("Object exists in output file: %s\n"
                                              "    Not allowed to overwrite",
                                              path)
                    raise RuntimeError("Object exists in output file")

def _ensureIsValidOutfile(outfile, lattice, params, makeActionSrc):
    r"""!
    Check if the output file is a valid parameter.
    If the file does not yet exists, create and initialize it.

    \throws ValueError if output file type is not supported.
    """

    if fileio.fileType(outfile) != fileio.FileType.HDF5:
        getLogger(__name__).error("Output file type not supported by Meas driver: %s", outfile)
        raise ValueError("Output file type no supported by Meas driver. "
                         f"Output file is '{outfile}'")

    outfile = Path(outfile)
    if not outfile.exists():
        # the easy case, just make a new file
        fileio.h5.initializeNewFile(outfile, False, lattice, params, makeActionSrc)
    # else:
    # Check measurement paths when running driver or saving measurements.
    # Can't do this here because the paths are not know at this point.

def _adjustConfigSlices(measurements, configurations):
    r"""!
    Change the configSlices of all measurements to reflect the actual range of configurations.
    \param measurements List of measurement objects. Each element's configSlice member is modified.
    \param configurations List of tuples `(index, ...)`. The indices are used to determine
                          the configSlices for the measurements. All other tuple elements
                          are ignored.
    """

    configStep = configurations[1][0] - configurations[0][0]
    # last index in the list plus the stride to go '1 past the end'
    length = configurations[-1][0] + configStep

    for measurement in measurements:
        try:
            # replace step first (needed by other with* functions)
            aux = withStart(withStep(measurement.configSlice, configStep), configurations[0][0])
            measurement.configSlice = withStop(aux, length) if aux.stop is None else aux
        except ValueError:
            getLogger(__name__).error("Invalid configuration slice %s in measurement %s "
                                      "given the actual configurations",
                                      measurement.configSlice, type(measurement))
            raise

def _availableMemory():
    r"""!
    Return the amount of available memory in bytes.
    """
    svmem = psutil.virtual_memory()
    getLogger(__name__).info(f"""System memory:
    Total:     {svmem.total:,} B
    Available: {svmem.available:,} B""")
    return svmem.available

def _totalMemoryAllowance(lattice, bufferFactor=0.8, maxBufferSize=None,
                          maxTotalMemory=None):
    r"""!
    Return the total amount of memory that may be used for storing measurement results in bytes.
    """
    log = getLogger(__name__)

    available = _availableMemory()
    if maxTotalMemory:
        if available < maxTotalMemory:
            log.info(f"The given maxiumum memory ({maxTotalMemory:,} B) is more "
                     f"than the available memory ({available:,} B).")
        else:
            available = maxTotalMemory

    allowance = int(bufferFactor * (available - 10 * lattice.lattSize() * 16))
    message = f"""Maximum allowed memory usage by measurements: {allowance:,} B
    Based on lattice size {lattice.lattSize()}
    and reserving {100 - bufferFactor*100}% of available memory for other purposes."""
    if maxBufferSize:
        message += f"\n    Restricted to buffers of size {maxBufferSize:,} B."
    log.info(message)
    return allowance

def _setupMeasurements(measurements, configurations, outfile, lattice,
                       adjustConfigSlices, maxBufferSize, maxTotalMemory):
    if adjustConfigSlices:
        _adjustConfigSlices(measurements, configurations)

    usableMemory = _totalMemoryAllowance(lattice, maxBufferSize=maxBufferSize,
                                         maxTotalMemory=maxTotalMemory)
    nremaining = len(measurements)

    with h5.File(outfile, "a") as h5f:
        for measurement in measurements:
            configSlice = measurement.configSlice

            residualMemory = measurement.setup(usableMemory // nremaining,
                                               (configSlice.stop - configSlice.start) // configSlice.step,
                                               h5f,
                                               maxBufferSize=maxBufferSize)
            usableMemory -= usableMemory // nremaining - residualMemory
            nremaining -= 1
