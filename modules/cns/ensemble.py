"""!
Handle ensembles.
"""

import importlib.util
import tempfile

import yaml
import h5py as h5

import cns

def importEnsemble(name, ensembleText):
    r"""!
    Import an ensemble Python module from a given text.
    \param name Name of the ensemble. Can be chosen arbitrarily and will be given
                to the Python module.
    \param ensembleText String holding the module as text.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as modf:
        # write it to a temporary file
        modf.write(ensembleText)
        modf.flush()

        # and read it back in
        spec = importlib.util.spec_from_file_location(name, modf.name)
        ensemble = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble)
    return ensemble

def loadH5(modName, group, name="ensemble"):
    r"""!
    Read and import an ensemble from HDF5.

    Needs the ensemble module to be represented as a string in the HDF5.
    \param modName Name of the Python module. Can be chosen arbitrarily and is
                   assigned to the new module.
    \param group HDF5 group that contains the ensemble string.
    \param name Name of the ensemble dataset in group.
    \returns Imported module and plain text of module contents.
    """
    text = group[name][()]
    return importEnsemble(modName, text), text

def saveH5(ensembleText, group, name="ensemble"):
    r"""!
    Write the text of an ensemble module to HDF5.
    \param ensembleText Plain text of the ensemble module.
    \param group HDF5 group to write into.
    \param name Name of the dataset written in group.
    """
    if not isinstance(ensembleText, str):
        raise TypeError("First parameter 'ensembleText' must be a string.")
    group[name] = ensembleText

def loadPlain(modName, fname):
    r"""!
    Read and import an ensemble from a plain text file (file .py).
    \param modName Name of the Python module. Can be chosen arbitrarily and is
                   assigned to the new module.
    \param fname Name of the module file to load.
    \returns Imported module and plain text of module contents.
    """
    with open(fname, "r") as modf:
        text = modf.read()
    return importEnsemble(modName, text), text

def savePlain(ensembleText, fname):
    r"""!
    Write the text of an ensemble module to a plain text file.
    \param ensembleText Plain text of the ensemble module.
    \param fname Name of file to write to.
    """
    if not isinstance(ensembleText, str):
        raise TypeError("First parameter 'ensembleText' must be a string.")
    with open(fname, "w") as modf:
        modf.write(ensembleText)

def load(modName, fname, ftype=None):
    r"""!
    Load an ensemble from a file.
    \param modName Name of the Python module. Can be chosen arbitrarily and is
                   assigned to the new module.
    \param fname Name of the input file.
    \param ftype Type of the input file, allowed values are `cns.fileio.FileType.PY`
                   and `cns.fileio.FileType.HDF5`.
    \returns The imported ensemble module and plain text of module contents.
    """
    if ftype is None:
        ftype = cns.fileio.fileType(fname)

    if ftype == cns.fileio.FileType.PY:
        # load from Python module
        return loadPlain(modName, fname)
    elif ftype == cns.fileio.FileType.HDF5:
        # load from HDF5 file
        with h5.File(fname, "r") as h5f:
            return loadH5(modName, h5f)
    else:
        # no other file types supported
        raise ValueError("Cannot load ensemble from file type '{}', file = {}" \
                         .format(str(ftype).split(".")[1].lower(), fname))

def readLattice(filename):
    r"""!
    Read a lattice file from YAML file cns.env["latticeDirectory"].
    \param filename The name of the lattice to read.
    \returns An instance of cns.Lattice constructed from the file.
    """
    if cns.fileio.fileType(filename) != cns.fileio.FileType.YAML:
        raise ValueError(f"Cannot read lattice from file {filename}. Wrong file type, only YAML is supported.")
    try:
        with open(str(cns.env["latticeDirectory"]/filename)) as yamlf:
            return yaml.safe_load(yamlf)
    except KeyError:
        raise KeyError("Need to set cns.env[\"latticeDirectory\"] before reading a lattice.") from None
