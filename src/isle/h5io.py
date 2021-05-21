r"""!\file
Routines for working with HDF5.
"""

from logging import getLogger
from pathlib import Path
from itertools import chain

import yaml
import h5py as h5
import numpy as np

from . import Vector, isleVersion, pythonVersion, blazeVersion, pybind11Version
from .random import readStateH5
from .collection import listToSlice, parseSlice, subslice, normalizeSlice

def empty(dtype):
    return h5.Empty(dtype=dtype)

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

def writeDict(h5group, dictionary):
    """!
    Write a `dict` into an HDF5 group by storing each dict element as a dataset.
    """
    for key, value in dictionary.items():
        h5group[key] = value

def loadDict(h5group):
    """!
    Load all datasets from an HDF5 group into a dictionary.
    """
    return {key: dset[()] for key, dset in h5group.items()}

def loadString(dset):
    """!
    Load a string from an HDF5 dataset and return as a Python str object.

    Since version 3.0, h5py loads UTF8 strings as `bytes` objects.
    This function provides uniform behavior across h5py 2.0 and h5py 3.0 by
    always returning `str` objects.
    """
    s = dset[()]
    if isinstance(s, str):
        return s
    return s.decode("utf-8")

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
            lattice = yaml.safe_load(loadString(metaGrp["lattice"]))
            params = yaml.safe_load(loadString(metaGrp["params"]))
            makeActionSrc = loadString(metaGrp["action"])
            versions = {name: loadString(val) for name, val in metaGrp["version"].items()}
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
            raise RuntimeError("Output file exists")

    with h5.File(str(fname), "w-") as h5f:
        for group in extraGroups:
            createH5Group(h5f, group)

    writeMetadata(fname, lattice, params, makeActionSrc)

def writeTrajectory(h5group, label, stage):
    r"""!
    Write a trajectory (endpoint) to a HDF5 group.
    Creates a new group with name 'label' and stores the EvolutionStage.

    \param h5group Base HDF5 group to store trajectory in.
    \param label Name of the subgroup of `h5group` to write to.
                 The subgroup must not already exist.
    \param stage EvolutionStage to save.

    \returns The newly created HDF5 group containing the trajectory.
    """

    grp = h5group.create_group(str(label))
    stage.save(grp)
    return grp

def writeCheckpoint(h5group, label, rng, trajGrpName, evolver, evolverManager):
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
    \param evolver Evolver used to make the trajectory at this checkpoint.
    \param evolverManager Instance of EvolverManager to handle saving the evolver.

    \returns The newly created HDF5 group containing the checkpoint.
    """

    grp = h5group.create_group(str(label))
    rng.writeH5(grp.create_group("rngState"))
    grp["cfg"] = h5.SoftLink(trajGrpName)
    evolverManager.save(evolver, grp.create_group("evolver"))
    return grp

def loadCheckpoint(h5group, label, evolverManager, action, lattice):
    r"""!
    Load a checkpoint from a HDF5 group.

    \param h5group Base HDF5 group containing checkpoints.
    \param label Name of the subgroup of `h5group` to read from.
    \param evolverManager A EvolverManager to load the evolver
                          including its type.
    \param action Action to construct the evolver with.
    \param lattice Lattice to construct the evolver with.
    \returns (RNG, HDF5 group of configuration, evolver)
    """

    grp = h5group[str(label)]
    rng = readStateH5(grp["rngState"])
    cfgGrp = grp["cfg"]
    evolver = evolverManager.load(grp["evolver"], action, lattice, rng)
    return rng, cfgGrp, evolver

def loadConfiguration(h5group, trajIdx=-1, path="configuration"):
    r"""!
    Load a configuration from HDF5.

    \param h5group Base HDF5 group. Configurations must be located at `h5group[path]`.
    \param trajIdx Trajectory index of the configuration to load.
                   This is the number under which the configuration is stored, not a
                   plain index into the array of all configurations.
    \param path Path under `h5group` that contains configurations.

    \returns (configuration, action value)
    """

    configs = loadList(h5group[path])
    # get proper positive index
    idx = configs[-1][0]+trajIdx+1 if trajIdx < 0 else trajIdx
    # get the configuration group with the given index
    cfgGrp = next(pair[1] for pair in loadList(h5group[path]) if pair[0] == idx)
    return Vector(cfgGrp["phi"][()]), cfgGrp["actVal"][()]

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

def loadActionValuesFrom(h5obj, full=False, base="/"):
    r"""!
    Load values of the action from a HDF5 file given via a HDF5 object in that file.

    Reads the action from dataset `/action/action` if it exists.
    Otherwise, read action from saved configurations.

    \param fname An arbitrary HDF5 object in the file to read the action from.
    \param full If True, always read from saved configurations as `/action/action` might
                contain only a subset of all actions.
    \param base Path in HDF5 file under which the action is stored.
    \returns (action, configRange) where
             - action: Numpy array of values of the action.
             - configRange: `slice` indicating the range of configurations
                            the action was loaded for.
    \throws RuntimeError if neither `/action/action` nor `/configuration` exist in the file.
    """

    grp = h5obj.file[base]
    action = None

    if not full and "action" in grp:
        action = grp["action/action"][()]
        cRange = normalizeSlice(parseSlice(grp["action"].attrs["configurations"],
                                           minComponents=3),
                                0, action.shape[0])

    if not full and "weights" in grp:
        action = grp["weights/actVal"][()]
        cRange = normalizeSlice(parseSlice(grp["weights"].attrs["configurations"],
                                           minComponents=3),
                                0, action.shape[0])

    if action is None and "configuration" in grp:
        indices, groups = zip(*loadList(grp["configuration"]))
        action = np.array([grp["actVal"][()] for grp in groups])
        cRange = listToSlice(indices)

    if action is None:
        getLogger(__name__).error("Cannot load action, no configurations or "
                                  "separate action found in file %s.", grp.file.filename)
        raise RuntimeError("No action found in file")

    return action, cRange

def loadActionValues(fname, full=False, base="/"):
    r"""!
    Load values of the action from a HDF5 file.

    Reads the action from dataset `/action/action` if it exists.
    Otherwise, read action from saved configurations.

    \param fname Name of the file to load action from.
    \param full If True, always read from saved configurations as `/action/action` might
                contain only a subset of all actions.
    \param base Path in HDF5 file under which the action is stored.
    \returns (action, configRange) where
             - action: Numpy array of values of the action.
             - configRange: `slice` indicating the range of configurations
                            the action was loaded for.
    \throws RuntimeError if neither `/action/action` nor `/configuration` exist in the file.
    """

    with h5.File(fname, "r") as h5f:
        return loadActionValuesFrom(h5f, full, base)

def loadActionWeightsFor(dset, base="/"):
    r"""!
    Load the weights from the imaginary part of the action for a measurement result.

    The weights are loaded based on the 'configurations' attribute stored in the
    parent group of `dset`.
    This requires the attribute to be stored properly (no `None`) and the file to
    contain the values of the action for all the trajectories the measurement was taken on
    (can be as '/action/action' dataset or '/configuration' group).

    The weights are computed as \f$e^{-i S_{I}}\f$.

    \param dset HDF5 dataset containing the measurement result.
    \param base Path in HDF5 file under which the action is stored.
    \returns np.ndarray of the weights for `dset`.
    """

    # load the configuration slice
    group = dset.parent
    try:
        neededRange = parseSlice(group.attrs["configurations"], minComponents=3)
    except KeyError:
        getLogger(__name__).error("Cannot load weights for dataset %s; "
                                  "there is no 'configurations' attribute", dset)
        raise
    if None in (neededRange.start, neededRange.stop, neededRange.step):
        getLogger(__name__).error("The 'configurations' attibute for dataset %s "
                                  "has not been saved properly, it contains None: %s",
                                  dset, neededRange)
        raise RuntimeError("configSlice of dataset contains None")

    action, actionRange = loadActionValuesFrom(dset, base=base)
    try:
        subRange = subslice(actionRange, neededRange)
    except ValueError:
        # try again, maybe there are enough configurations in the file
        action, actionRange = loadActionValuesFrom(dset, True, base=base)
        subRange = subslice(actionRange, neededRange)

    return np.exp(-1j * np.imag(action[subRange]))

def loadLogWeightsFrom(h5obj, full=False, base="/", weightsGroup="weights"):
    r"""!
    Load logarithmic weights from an HDF5 file given via an HDF5 object in that file.

    Reads the weights from `{base}/{weights}` if it exists.
    This group can be created using the measurement CollectWeights.
    Otherwise, read weights from saved configurations.
    If that is not possible either, raise `RuntimeError`.

    \param h5obj An arbitrary HDF5 object in the file to read the weights from.
    \param base Path in HDF5 file under which the data is stored.
    \param full If True, always read from configurations group instead of collected weights.
    \returns (weights, configRange) where
             - weights: `dict` to numpy arrays of values of the weights.
             - configRange: `slice` indicating the range of configurations
                            the weights were loaded for.
    \throws RuntimeError if neither weights nor configuration groups exist in the file.
    """

    baseGroup = h5obj.file[base]
    log = getLogger(__name__)

    if not full and weightsGroup in baseGroup:
        # load content of weights group as it is without any fancy checks
        wgrp = baseGroup[weightsGroup]
        log.info("Loading weights from %s/%s", h5obj.filename, wgrp.name)
        logWeights = loadDict(wgrp)
        cRange = normalizeSlice(parseSlice(wgrp.attrs["configurations"],
                                           minComponents=3),
                                0, next(iter(logWeights.values())).shape[0])
        return logWeights, cRange

    if "configuration" in baseGroup:
        log.info("Loading weights from %s/%s", h5obj.filename, baseGroup["configuration"].name)
        indices, groups = zip(*loadList(baseGroup["configuration"]))

        # prepare dict for all waights that are present (logWeights might not exist)
        logWeights = {name: np.empty(len(indices), dtype=complex)
                      for name in chain(("actVal",),
                                        groups[0]["logWeights"] if "logWeights" in groups[0] else ())}
        # load weights one trajectory at a time
        for i, grp in enumerate(groups):
            for name in logWeights.keys():
                if name == "actVal":
                    logWeights[name][i] = grp["actVal"][()]
                else:
                    logWeights[name][i] = grp["logWeights/"+name][()]

        return logWeights, listToSlice(indices)

    getLogger(__name__).error("Cannot load weights from file %s, no configurations or "
                              "separate weights group found", baseGroup.file.filename)
    raise RuntimeError("No action found in file")

def loadLogWeights(fname, full=False, base="/"):
    r"""!
    Load logarithmic weights from an HDF5 file given via an HDF5 object in that file.

    Reads the weights from `{base}/{weights}` if it exists.
    This group can be created using the measurement CollectWeights.
    Otherwise, read weights from saved configurations.
    If that is not possible either, raise `RuntimeError`.

    \param fname Name of the file to load action from.
    \param base Path in HDF5 file under which the data is stored.
    \param full If True, always read from configurations group instead of collected weights.
    \returns (weights, configRange) where
             - weights: `dict` to numpy arrays of values of the weights.
             - configRange: `slice` indicating the range of configurations
                            the weights were loaded for.
    \throws RuntimeError if neither weights nor configuration groups exist in the file.
    """

    with h5.File(fname, "r") as h5f:
        return loadLogWeightsFrom(h5f, full, base)
