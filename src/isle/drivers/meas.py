r"""!
\todo document

meas freq is relative to config index not loop iteration in mapOverConfigs
"""

from logging import getLogger

import h5py as h5

from .. import fileio, cli
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

    def __call__(self, measurements, configRange=slice(None)):
        _ensureCanWriteMeas(self.outfile, [meas[2] for meas in measurements], self.overwrite)

        getLogger(__name__).info("Performing measurements")
        self.mapOverConfigs(measurements, configRange)
        getLogger(__name__).info("Saving measurements")
        self.save(measurements, configRange, checked=True)

    def mapOverConfigs(self, measurements, configRange=slice(None)):
        """!
        Apply measurements to all configurations in the input file
        of this driver.
        """

        _checkStepIsDefaultByException(configRange)

        with h5.File(self.infile, "r") as cfgf:
            # sorted list of configurations
            # each entry is a pair (index: int, config: H5group)
            # take only those in configRange
            configurations = sorted(map(lambda p: (int(p[0]), p[1]),
                                        cfgf["/configuration"].items()),
                                    key=lambda item: item[0])[configRange]

            # apply measurements to all configs
            with cli.trackProgress(len(configurations), "Measurements", updateRate=1000) as pbar:
                for i, grp in configurations:
                    # read config and action
                    phi = grp["phi"][()]
                    action = grp["action"][()]
                    # measure
                    for frequency, measurement, _ in measurements:
                        if i % frequency == 0:
                            measurement(phi, action, i)

                    pbar.advance()

    def save(self, measurements, configRange, checked=False):
        if not checked:
            _ensureCanWriteMeas(self.outfile, [meas[2] for meas in measurements], self.overwrite)

        _checkStepIsDefaultByException(configRange)

        with h5.File(self.outfile, "a") as measFile:
            for frequency, measurement, path in measurements:
                measurement.save(measFile, path)
                measFile[path].attrs["configurations"] = \
                    f"{configRange.start},{configRange.stop},{frequency}"

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

def _checkStepIsDefaultByException(slic):
    """!Check if step size is None or 1 in given slice and raise ValueError if not."""
    if slic.step not in (None, 1):
        getLogger(__name__).error("Step size of configRange must be None (or 1). "
                                  "Given value is %s", slic.step)
        raise ValueError("Step size must be None (or 1)")


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

def _ensureCanWriteMeas(outfile, paths, overwrite):
    r"""!
    Ensure that measurements can be written to the output file.
    \param outfile The output file, must alreay exist!
    \param paths Paths inside `outfile` that the measurements want to write to.
    \param overwrite If `True`, erase all existing HDF5 objects under the given paths.
                     if `False`, fail if an object exists under any given path.
    """

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
