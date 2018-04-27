"""!
Handle ensembles.
"""

import importlib.util
import tempfile
from pathlib import Path

import yaml
import h5py as h5

import cns

def importEnsemble(ensembleFile):
    r"""!
    Import an ensemble Python module from a given file location.
    \param ensembleFile Path to the ensemble module to load.
    """

    ensembleFile = str(ensembleFile) # so we can deal with pathlib.Path

    # As explained in
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    moduleName = ensembleFile.rsplit(".")[0].replace("/", "_")
    spec = importlib.util.spec_from_file_location(moduleName, ensembleFile)
    ensemble = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble)
    return ensemble

def writeH5(ensembleFile, group, name="ensemble"):
    r"""!
    Write the text of an ensemble module to HDF5.
    \param ensembleFile Path to the ensemble module.
    \param group HDF5 group to write into.
    \param name Name of the dataset written in group.
    """
    with open(str(ensembleFile), "r") as ensf:
        group[name] = ensf.read()

def readH5(group, name="ensemble"):
    r"""!
    Read ensemble from HDF5.

    Needs the ensemble module to be represented as a string in the HDF5.
    \param group HDF5 group that contains the ensemble string.
    \param name Name of the ensemble dataset in group.
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as modf:
        # just write it to a temporary file
        modf.write(group[name][()])
        modf.flush()

        # and read it back in
        moduleName = str(Path(group.filename).stem)
        spec = importlib.util.spec_from_file_location(moduleName, modf.name)
        ensemble = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble)
    return ensemble

def loadEnsemble(infname, inftype):
    r"""!
    Load an ensemble from a file.
    \param infname Name of the input file.
    \param inftype Type of the input file, allowed values are `cns.fileio.FileType.PY`
                   and `cns.fileio.FileType.HDF5`.
    \returns The ensemble.
    """

    if inftype == cns.fileio.FileType.PY:
        # load from Python module
        ensemble = cns.ensemble.importEnsemble(infname)
    elif inftype == cns.fileio.FileType.HDF5:
        # load from HDF5 file
        with h5.File(infname, "r") as h5f:
            ensemble = cns.ensemble.readH5(h5f)
    else:
        # no other file types supported
        raise ValueError("Cannot load ensemble from file type '{}', file = {}" \
                         .format(str(inftype).split(".")[1].lower(), infname))

    return ensemble

def readLattice(filename):
    r"""!
    Read a lattice file from cns.env["latticeDirectory"].
    \param filename The name of the lattice to read.
    """
    try:
        with open(str(cns.env["latticeDirectory"]/filename)) as yamlf:
            return yaml.safe_load(yamlf)
    except KeyError:
        raise KeyError("Need to set cns.env[\"latticeDirectory\"] before reading a lattice.") from None
