from logging import getLogger

import yaml
import h5py as h5

from .. import fileio, isleVersion, pythonVersion, blazeVersion, pybind11Version
from ..util import compareVersions

def _verifyVersion(current, other, name, fname):
    comp = compareVersions(current, other)
    if comp == "none":
        getLogger(__name__).info("Extra version string of %s (%s) different from version in file %s (%s)",
                                 name, current, fname, other)
    if comp != "equal":
        getLogger(__name__).error("Version of %s (%s), is %s than in file %s (%s).",
                                  name, current, comp, fname, other)
        raise RuntimeError(f"Version mismatch for {name}")


def verifyVersionsByException(versions, fname):
    _verifyVersion(isleVersion, versions["isle"], "isle", fname)
    _verifyVersion(pythonVersion, versions["python"], "Python", fname)

def verifyMetadataByException(fname, lattice, params):
    "!Make sure that metadata in file agrees with function parameters."

    storedLattice, storedParams, _, versions = fileio.h5.readMetadata(fname)

    if storedLattice.name != lattice.name:
        getLogger(__name__).error("Name of lattice in output file is %s but new lattice has name %s. "
                                  "Cannot write into existing output file.",
                                  storedLattice.name, lattice.name)
        raise RuntimeError("Lattice name inconsistent")

    if storedParams.asdict() != params.asdict():
        getLogger(__name__).error("Stored parameters do not match new parameters. "
                                  "Cannot write into existing output file.")
        raise RuntimeError("Parameters inconsistent")

    verifyVersionsByException(versions, fname)

def prepareOutfile(outfname, lattice, params, makeActionSrc,
                   extraGroups=[]):
    """!
    Prepare the output file by storing program versions, metadata, and creating groups.
    """

    with h5.File(str(outfname), "w-") as outf:
        for group in extraGroups:
            fileio.h5.createH5Group(outf, group)

    fileio.h5.writeMetadata(outfname, lattice, params, makeActionSrc)
