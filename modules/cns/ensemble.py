"""!
Handle ensembles.
"""

import importlib.util

import yaml

import cns

def importEnsemble(ensembleFile):
    r"""!
    Import an ensemble Python module from a given file location.
    \param ensembleFile Path to the ensemble module to load.
    """

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
