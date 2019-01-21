"""!
Routines for working with HDF5.
"""

import logging

import yaml
import h5py as h5

from .cpp_wrappers import isleVersion, pythonVersion, blazeVersion, pybind11Version

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
            logging.getLogger(__name__).error("Cannot read metadata from file %s: %s",
                                              str(fname), str(exc))
            raise
    return lattice, params, makeActionSrc, versions

def initializeFile(fname, lattice, params, makeActionSrc, extraGroups=[]):
    """!
    Prepare the output file by storing program versions, metadata, and creating groups.
    The file is not allowed to exist yet.
    """

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
    \param label Name of the subgroup of `group` to write to.
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

def writeCheckpoint(h5group, label, rng, trajGrpName):
    r"""!
    Write a checkpoint to a HDF5 group.
    Creates a new group with name 'label' and stores RNG state
    and a soft link to the trajectory for this checkpoint.

    \param h5group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `group` to write to.
                 The subgroup must not already exist.
    \param rng Random number generator whose state to save in the checkpoint.
    \param trajGrpName Name of the HDF5 group containing the trajectory this
                       checkpoint corresponds to.

    \returns The newly created HDF5 group containing the checkpoint.
    """

    grp = h5group.create_group(str(label))
    rng.writeH5(grp.create_group("rngState"))
    grp["cfg"] = h5.SoftLink(trajGrpName)
    return grp
