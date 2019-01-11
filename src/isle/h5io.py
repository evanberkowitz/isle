"""!
Routines for working with HDF5.
"""

import yaml
import h5py as h5

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

    # TODO write Version(s)  -  write function in h5io

    with h5.File(str(fname), "w") as outf:
        metaGrp = createH5Group(outf, "meta")
        metaGrp["lattice"] = yaml.dump(lattice)
        metaGrp["params"] = yaml.dump(params)
        metaGrp["action"] = makeActionSrc

def readMetadata(fname):
    r"""!
    Read metadata on ensemble from HDF5 file.

    \returns Lattice, parameters, makeAction (source code of function)
    """
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    with h5.File(str(fname), "r") as inf:
        metaGrp = inf["meta"]
        lattice = yaml.safe_load(metaGrp["lattice"][()])
        params = yaml.safe_load(metaGrp["params"][()])
        makeActionSrc = metaGrp["action"][()]
    return lattice, params, makeActionSrc

def writeTrajectory(group, label, phi, act, trajPoint):
    r"""!
    Write a trajectory (endpoint) to a HDF5 group.
    Creates a new group with name 'label' and stores
    Configuration, action, and whenther the trajectory was accepted.

    \param group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `group` to write to.
                 The subgroup must not already exist.
    \param phi Configuration to save.
    \param act Value of the action at configuration `phi`.
    \param trajPoint Point on the trajectory that was accepted.
                     `trajPoint==0` is the start point and values `>0` or `<0` are
                     `trajPoint` MD steps after or before the start point.

    \returns The newly created HDF5 group containing the trajectory.
    """

    grp = group.create_group(str(label))
    grp["phi"] = phi
    grp["action"] = act
    grp["trajPoint"] = trajPoint
    return grp

def writeCheckpoint(group, label, rng, trajGrpName):
    r"""!
    Write a checkpoint to a HDF5 group.
    Creates a new group with name 'label' and stores RNG state
    and a soft link to the trajectory for this checkpoint.

    \param group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `group` to write to.
                 The subgroup must not already exist.
    \param rng Random number generator whose state to save in the checkpoint.
    \param trajGrpName Name of the HDF5 group containing the trajectory this
                       checkpoint corresponds to.

    \returns The newly created HDF5 group containing the checkpoint.
    """

    grp = group.create_group(str(label))
    rng.writeH5(grp.create_group("rngState"))
    grp["cfg"] = h5.SoftLink(trajGrpName)
    return grp
