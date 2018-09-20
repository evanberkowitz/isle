from pathlib import Path

import numpy as np
import yaml
import h5py as h5

from .. import fileio
from .. import meas
from ._util import verifyMetadataByException, prepareOutfile


class Measure:
    def __init__(self, lattice, params, action, infname, outfname):
        self.lattice = lattice
        self.params = params
        self.action = action
        self.infname = str(infname)
        self.outfname = str(outfname)


    def __call__(self, measurements):
        # Keep configuration h5 file closed as much as possible during measurements
        # First find find out all the configurations.
        with h5.File(self.infname, "r") as cfgf:
            configNames = sorted(cfgf["/configuration"], key=int)

        print("Performing measurements...")
        for i, configName in enumerate(configNames):
            # read config and action
            with h5.File(self.infname, "r") as cfgf:
                phi = cfgf["configuration"][configName]["phi"][()]
                action = cfgf["configuration"][configName]["action"][()]
                # measure
                for frequency, measurement, _ in measurements \
                    +[(100, meas.Progress("Measurement", len(configNames)), "")]:
                    if i % frequency == 0:
                        measurement(phi, act=action, itr=i)

        print("Saving measurements...")
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
    """!
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

