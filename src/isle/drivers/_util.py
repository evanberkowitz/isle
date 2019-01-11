from logging import getLogger

import yaml
import h5py as h5

from .. import fileio

def verifyMetadataByException(outfname, lattice, params):
    "!Make sure that metadata in file agrees with function parameters."

    storedLattice, storedParams, _ = fileio.h5.readMetadata(outfname)

    if storedLattice.name != lattice.name:
        getLogger(__name__).error("Name of lattice in output file is %s but new lattice has name %s. "
                                  "Cannot write into existing output file.",
                                  storedLattice.name, lattice.name)
        raise RuntimeError("Lattice name inconsistent")

    if storedParams.asdict() != params.asdict():
        getLogger(__name__).error("Stored parameters do not match new parameters. "
                                  "Cannot write into existing output file.")
        raise RuntimeError("Parameters inconsistent")

def prepareOutfile(outfname, lattice, params, makeActionSrc,
                   extraGroups=[]):
    """!
    Prepare the output file by storing program versions, metadata, and creating groups.
    """

    fileio.h5.writeMetadata(outfname, lattice, params, makeActionSrc)

    with h5.File(str(outfname), "w") as outf:
        for group in extraGroups:
            fileio.h5.createH5Group(outf, group)
