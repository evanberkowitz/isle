r"""!
\todo document

meas freq is relative to config index not loop iteration in mapOverConfigs
"""

from logging import getLogger

import h5py as h5

from .. import fileio, cli
from ..collection import inSlice, withStart, withStop, withStep
from ..meta import callFunctionFromSource
from ..util import verifyVersionsByException


class Measure:
    def __init__(self, lattice, params, action, infile, outfile, overwrite):
        self.lattice = lattice
        self.params = params
        self.action = action
        self.infile = infile
        self.outfile = outfile
        self.overwrite = overwrite

    def __call__(self, measurements):
        _ensureCanWriteMeas(self.outfile, measurements, self.overwrite)

        self.mapOverConfigs(measurements, adjustSlices=True)
        self.save(measurements, checkedBefore=True)

    def mapOverConfigs(self, measurements, adjustSlices=True):
        """!
        Apply measurements to all configurations in the input file
        of this driver.
        """

        # copy so the list can be modified in this function
        measurements = list(measurements)

        with h5.File(self.infile, "r") as cfgf:
            # get all configuration groups (index, h5group) pairs
            configurations = fileio.h5.loadList(cfgf["/configuration"])

            if adjustSlices:
                _adjustConfigSlices(measurements, configurations)

            # apply measurements to all configs
            with cli.trackProgress(len(configurations), "Measurements", updateRate=1000) as pbar:
                for itr, grp in configurations:
                    # read config and action
                    phi = grp["phi"][()]
                    action = grp["action"][()]
                    # measure
                    for imeas, measurement in enumerate(measurements):
                        if inSlice(itr, measurement.configSlice):
                            measurement(phi, action, itr)
                        elif itr >= measurement.configSlice.stop:
                            # this measurement is done => drop it
                            del measurements[imeas]
                    if not measurements:
                        break  # no measurements left

                    pbar.advance()

    def save(self, measurements, checkedBefore=False):
        if not checkedBefore:
            _ensureCanWriteMeas(self.outfile, measurements, self.overwrite)

        with h5.File(self.outfile, "a") as measFile:
            for measurement in measurements:
                measurement.saveAll(measFile)


# TODO allow to set a base path in file
def init(infile, outfile, overwrite):
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
                   infile, outfile, overwrite)

def _isValidPath(path):
    """!Check if parameter is a valid path to a measurement inside an HDF5 file."""
    components = str(path).split("/")
    if components[0] == "":
        getLogger(__name__).warning(
            "Output path of a measurement is specified as absolute path: %s\n"
            "    Use relative paths so the base HDF5 group can be adjusted.",
            path)

    nonempty = [component for component in components if component.strip()]
    if len(nonempty) == 0:
        getLogger(__name__).error(
            "Output path of a measurement is the root HDF5 group. "
            "All measurements must be stored in a subgroup")
        return False

    return True

def _ensureCanWriteMeas(outfile, measurements, overwrite):
    r"""!
    Ensure that measurements can be written to the output file.
    \param outfile The output file, must alreay exist!
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
