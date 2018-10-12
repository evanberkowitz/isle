r"""!

"""

import h5py as h5

from .. import fileio
from ._util import verifyMetadataByException, prepareOutfile


class Measure:
    def __init__(self, lattice, params, action, infname, outfname):
        self.lattice = lattice
        self.params = params
        self.action = action
        self.infname = str(infname)
        self.outfname = str(outfname)

    def __call__(self, measurements):
        print("Performing measurements...")
        self.mapOverConfigs(measurements)
        print("Saving measurements...")
        self.save(measurements)

    def mapOverConfigs(self, measurements):
        """!
        Apply measurements to all configurations in the input file
        of this driver.
        """

        with h5.File(self.infname, "r") as cfgf:
            # iterate over all configs sorted by their number
            for i, grp in map(lambda p: (int(p[0]), p[1]),
                              sorted(cfgf["/configuration"].items(),
                                     key=lambda item: int(item[0]))):
                if i % 1000 == 0:
                    print(f"Measurement: Processing configuration {i}")

                # read config and action
                phi = grp["phi"][()]
                action = grp["action"][()]
                # measure
                for frequency, measurement, _ in measurements:
                    if i % frequency == 0:
                        measurement(phi, action, i)

    def save(self, measurements):
        with h5.File(self.outfname, "a") as measFile:
            for _, measurement, path in measurements:
                measurement.save(measFile, path)


def init(infile, outfile, overwrite):
    if infile is None:
        print("Error: no input file given")
        raise RuntimeError("No input file given to Meas driver.")

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

    if outfile is None:
        print("Error: no output file given")
        raise RuntimeError("No output file given to Meas driver.")

    if outfile[1] != fileio.FileType.HDF5:
        raise ValueError(f"Output file type no supported by Meas driver. Output file is '{outfile[0]}'")

    outfname = outfile[0]
    if outfname.exists():
        if overwrite:
            print(f"Output file '{outfname}' exists -- overwriting")
            outfname.unlink()

        else:
            verifyMetadataByException(outfname, lattice, params)
            # TODO verify version(s)
            print(f"Output file '{outfname}' exists -- appending")

