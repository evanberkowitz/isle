"""!
Routines for working with HDF5.
"""

from logging import getLogger
from pathlib import Path

import yaml
import h5py as h5
import numpy as np

from . import Vector, isleVersion, pythonVersion, blazeVersion, pybind11Version
from .random import readStateH5
from .util import parseSlice

def createH5Group(base, name):
    r"""!
    Create a new HDF5 group if it does not yet exist.
    \param base H5 group in which to create the new group.
    \param name Name of the new group relative to base.
    \returns The (potentially newly created) group.
    """

    if name in base:
        if isinstance(base[name], h5.Group):
            return base[name] # there is already a group with that name
        # something else than a group with that name
        raise ValueError(("Cannot create group '{}', another object with the same"\
                         +" name already exists in '{}/{}'").format(name, base.filename, base.name))
    # does not exists yet
    return base.create_group(name)

def writeMetadata(fname, lattice, params, makeActionSrc):
    """!
    Write metadata to HDF5 file.
    Overwrites any existing datasets.
    """

    with h5.File(str(fname), "a") as outf:
        metaGrp = createH5Group(outf, "meta")
        metaGrp["lattice"] = yaml.dump(lattice)
        metaGrp["params"] = yaml.dump(params)
        metaGrp["action"] = makeActionSrc

        vgrp = createH5Group(metaGrp, "version")
        vgrp["isle"] = str(isleVersion)
        vgrp["python"] = str(pythonVersion)
        vgrp["blaze"] = str(blazeVersion)
        vgrp["pybind11"] = str(pybind11Version)

def readMetadata(fname):
    r"""!
    Read metadata on ensemble from HDF5 file.

    \returns Lattice, parameters, makeAction (source code of function)
    """
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    with h5.File(str(fname), "r") as inf:
        try:
            metaGrp = inf["meta"]
            lattice = yaml.safe_load(metaGrp["lattice"][()])
            params = yaml.safe_load(metaGrp["params"][()])
            makeActionSrc = metaGrp["action"][()]
            versions = {name: val[()] for name, val in metaGrp["version"].items()}
        except KeyError as exc:
            getLogger(__name__).error("Cannot read metadata from file %s: %s",
                                              str(fname), str(exc))
            raise
    return lattice, params, makeActionSrc, versions

def initializeNewFile(fname, overwrite, lattice, params, makeActionSrc, extraGroups=[]):
    """!
    Prepare the output file by storing program versions, metadata, and creating groups.
    If `overwrite==False` the file must not exist. If it is True, the file is removed if it exists.
    """

    fname = Path(fname)
    if fname.exists():
        if overwrite:
            fname.unlink()
            getLogger(__name__).info("Output file %s exists -- overwriting", fname)
        else:
            getLogger(__name__).error("Output file %s exists and not allowed to overwrite", fname)
            raise RuntimeError("Ouput file exists")

    with h5.File(str(fname), "w-") as h5f:
        for group in extraGroups:
            createH5Group(h5f, group)

    writeMetadata(fname, lattice, params, makeActionSrc)

def writeTrajectory(h5group, label, phi, actVal, trajPoint):
    r"""!
    Write a trajectory (endpoint) to a HDF5 group.
    Creates a new group with name 'label' and stores
    Configuration, action, and whenther the trajectory was accepted.

    \param h5group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `h5group` to write to.
                 The subgroup must not already exist.
    \param phi Configuration to save.
    \param actVal Value of the action at configuration `phi`.
    \param trajPoint Point on the trajectory that was accepted.
                     `trajPoint==0` is the start point and values `>0` or `<0` are
                     `trajPoint` MD steps after or before the start point.

    \returns The newly created HDF5 group containing the trajectory.
    """

    grp = h5group.create_group(str(label))
    grp["phi"] = phi
    grp["action"] = actVal
    grp["trajPoint"] = trajPoint
    return grp

def writeCheckpoint(h5group, label, rng, trajGrpName, proposer, proposerManager):
    r"""!
    Write a checkpoint to a HDF5 group.
    Creates a new group with name 'label' and stores RNG state
    and a soft link to the trajectory for this checkpoint.

    \param h5group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `h5group` to write to.
                 The subgroup must not already exist.
    \param rng Random number generator whose state to save in the checkpoint.
    \param trajGrpName Name of the HDF5 group containing the trajectory this
                       checkpoint corresponds to.
    \param proposer Proposer used to make the trajectory at this checkpoint.
    \param proposerManager Instance of ProposerManager to handle saving the proposer.

    \returns The newly created HDF5 group containing the checkpoint.
    """

    grp = h5group.create_group(str(label))
    rng.writeH5(grp.create_group("rngState"))
    grp["cfg"] = h5.SoftLink(trajGrpName)
    proposerManager.save(proposer, grp.create_group("proposer"))
    return grp

def loadCheckpoint(h5group, label, proposerManager, action, lattice):
    r"""!
    Load a checkpoint from a HDF5 group.

    \param h5group Base HDF5 group containing checkpoints.
    \param label Name of the subgroup of `h5group` to read from.
    \param proposerManager A ProposerManager to load the proposer
                           including its type.
    \param action Action to construct the proposer with.
    \param lattice Lattice to construct the proposer with.
    \returns (RNG, configuration, proposer)
    """

    grp = h5group[str(label)]
    rng = readStateH5(grp["rngState"])
    phi = Vector(grp["cfg/phi"][()])
    proposer = proposerManager.load(grp["proposer"], action, lattice, rng)
    return rng, phi, proposer

def loadList(h5group, convert=int):
    r"""!
    Load a list of objects from a HDF5 group.

    All entries in `h5group` must have names convertible to `int` by `convert`.

    \param h5group HDF5 group to load from. All elements in that group must be
                   named such that they can be processed by `convert`.
    \param convert Function that takes a group entry name and returns an int.
    \returns List of pairs (key, obj) where key is the name of each object converted to `int`.
    """
    return sorted(map(lambda p: (convert(p[0]), p[1]), h5group.items()),
                  key=lambda item: item[0])

def loadActionValuesFrom(h5obj, configRange=None):
    r"""!
    Load values of the action from a HDF5 file given via a HDF5 object in that file.

    Reads the action from dataset `/action/action` if it exists.
    Otherwise, read action from saved configurations.

    \param fname An arbitrary HDF5 object in the file to read the action from.
    \param configRange `slice` indicating which values of the action to load.
                       Might need to load from `/configuration` to satisfy the constraint.
                       If None, load whichever action values are stored in `/action/action` if
                       it exists or all action if reading from `/configuration`.
    \returns (action, configRange) where
             - action: Numpy array of values of the action.
             - configRange: `slice` indicating the range of configurations
                            the action was laoded for.
    \throws RuntimeError if neither `/action/action` nor `/configuration` exist in the file.
    """

    h5f = h5obj.file

    if "action" in h5f:
        cRange = parseSlice(h5f["action"].attrs["configurations"],
                            minComponents=3)
        # not the cleverest way to handle it but the simplest; should be enough though
        if configRange is None or cRange == configRange:
            return h5f["action/action"][()], cRange

    if "configuration" in h5f:
        if configRange is None:
            configRange = slice(None)
        configs = loadList(h5f["configuration"])[configRange]
        return np.array([grp["action"][()] for _, grp in configs]), configRange

    getLogger(__name__).error("Cannot load action, no configurations or "
                              "separate action found in file %s.", h5f.filename)
    raise RuntimeError("No action found in file")


def loadActionValues(fname, configRange=None):
    r"""!
    Load values of the action from a HDF5 file.

    Reads the action from dataset `/action/action` if it exists.
    Otherwise, read action from saved configurations.

    \param fname Name of the file to load action from.
    \param configRange `slice` indicating which values of the action to load.
                       Might need to load from `/configuration` to satisfy the constraint.
                       If None, load whichever action values are stored in `/action/action` if
                       it exists or all action if reading from `/configuration`.
    \returns (action, configRange) where
             - action: Numpy array of values of the action.
             - configRange: `slice` indicating the range of configurations
                            the action was laoded for.
    \throws RuntimeError if neither `/action/action` nor `/configuration` exist in the file.
    """

    with h5.File(fname, "r") as h5f:
        return loadActionValuesFrom(h5f, configRange)

# def loadWeightsFor(dset):
#     group = dset.parent
#     # f = dset.file

#     # action = f["arction/action"][()]
#     action, configRange = loadActionValuesFrom(dset)
