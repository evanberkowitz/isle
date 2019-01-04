r"""!

"""

from logging import getLogger

import h5py as h5

from .. import fileio
from .. import cli
from ._util import verifyMetadataByException, prepareOutfile


class Measure:
    def __init__(self, lattice, params, action, infname, outfname):
        self.lattice = lattice
        self.params = params
        self.action = action
        self.infname = str(infname)
        self.outfname = str(outfname)

    def __call__(self, measurements):
        getLogger(__name__).info("Performing measurements")
        self.mapOverConfigs(measurements)
        getLogger(__name__).info("Saving measurements")
        self.save(measurements)

    def mapOverConfigs(self, measurements):
        """!
        Apply measurements to all configurations in the input file
        of this driver.
        """

        with h5.File(self.infname, "r") as cfgf:
            # sorted list of configurations
            # each entry is a pair (index: int, config: H5group)
            configurations = sorted(map(lambda p: (int(p[0]), p[1]), cfgf["/configuration"].items()),
                                    key=lambda item: item[0])

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

    def save(self, measurements):
        with h5.File(self.outfname, "a") as measFile:
            for _, measurement, path in measurements:
                measurement.save(measFile, path)


def init(infile, outfile, overwrite):
    if infile is None:
        getLogger(__name__).critical("No input file given to meas driver.")
        raise RuntimeError("No input file")

    if outfile is None:
        getLogger(__name__).info("No output file given to meas driver. "
                                 "Writing to input file.")
        outfile = infile

    if not isinstance(infile, (tuple, list)):
        infile = fileio.pathAndType(infile)
    if not isinstance(outfile, (tuple, list)):
        outfile = fileio.pathAndType(outfile)

    lattice, params, makeActionSrc = fileio.h5.readMetadata(infile)

    _ensureIsValidOutfile(outfile, overwrite, lattice, params)

    if not outfile[0].exists():
        prepareOutfile(outfile[0], lattice, params, makeActionSrc)

    return Measure(lattice, params,
                   fileio.callFunctionFromSource(makeActionSrc, lattice, params),
                   infile[0], outfile[0])

def _ensureIsValidOutfile(outfile, overwrite, lattice, params):
    r"""!
    Check if the output file is a valid parameter and if it is possible to write to it.
    Deletes the file if `overwrite == True`.

    \throws ValueError if output file type is not supported.
    \throws RuntimeError if the file is not valid.
    """

    # TODO if outfile exists, check if there is data for all configs and if not,
    #      we can continue (how to check given that every meas has its own format?)

    if outfile[1] != fileio.FileType.HDF5:
        raise ValueError(f"Output file type no supported by Meas driver. Output file is '{outfile[0]}'")

    outfname = outfile[0]
    if outfname.exists():
        if overwrite:
            getLogger(__name__).info("Output file '%s' exists -- erasing", outfname)
            outfname.unlink()

        else:
            verifyMetadataByException(outfname, lattice, params)
            # TODO verify version(s)
            getLogger(__name__).info("Output file '%s' exists -- appending", outfname)
