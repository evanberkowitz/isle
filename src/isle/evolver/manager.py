r"""!\file
\ingroup evolvers
Handlers for I/O from/to HDF5 files.
"""

from logging import getLogger
from pathlib import Path

import h5py as h5

from .evolver import Evolver
from .transform import Transform
from ..meta import classFromSource, sourceOfClass


class EvolverManager:
    r"""! \ingroup evolvers
    Manages evolvers in a file.

    Handles saving and loading of types and parameters of evolvers in a centralized way.
    Uses the file structure descriped in \ref filelayout.
    """

    def __init__(self, fname, evolverTypeLocation="/meta/evolvers",
                 transformTypeLocation="/meta/transforms", definitions={}):
        r"""!
        Initialize for a given file.
        \param fname Path to the file to manage.
        \param evolverTypeLocation Path inside the file where evolver types are to be stored.
        \param transformTypeLocation Path inside the file where transform types are to be stored.
        \param definitions Dictionary of extra definitions to take into account when
               saving/loading evolvers.
        """

        self.evolverTypeLocation = evolverTypeLocation
        self.transformTypeLocation = transformTypeLocation
        self.extraDefinitions = definitions
        # list of all currently existing evolvers and transforms in the file
        # index in this list is index in file
        self._evolvers, self._transforms = self._loadTypes(Path(fname))

    def _loadTypes(self, fname):
        r"""!
        Load all types of evolvers and transforms in file `fname`.
        """

        if not fname.exists():
            getLogger(__name__).info("File to load evolvers from does not exist: %s", fname)
            return []

        with h5.File(fname, "r") as h5f:
            try:
                grp = h5f[self.evolverTypeLocation]
                evolvers = [_retrieveTypeFrom(g, Evolver, self.extraDefinitions)
                            for _, g in sorted(grp.items(), key=lambda p: int(p[0]))]
                getLogger(__name__).info("Loaded evolver types from file %s:\n    %s",
                                         fname,
                                         "\n    ".join(f"{i}: {p}" for i, p in enumerate(evolvers)))
            except KeyError:
                getLogger(__name__).info("No evolvers found in file %s", fname)
                evolvers = []

            try:
                grp = h5f[self.transformTypeLocation]
                transforms = [_retrieveTypeFrom(g, Transform, self.extraDefinitions)
                            for _, g in sorted(grp.items(), key=lambda p: int(p[0]))]
                getLogger(__name__).info("Loaded transform types from file %s:\n    %s",
                                         fname,
                                         "\n    ".join(f"{i}: {p}" for i, p in enumerate(transforms)))
            except KeyError:
                getLogger(__name__).info("No transforms found in file %s", fname)
                transforms = []

        return evolvers, transforms

    def _saveType(self, obj, baseClass, h5file):
        r"""!
        Save the type of an evolver if it is not already stored.
        \param obj Object (<I>not</I> type!) to save.
        \param baseClass Base class of the type to save (Evolver or Transform).
        \param h5file File to save to.
        \returns Index of the type in the file.
        """

        if baseClass == Evolver:
            collection = self._evolvers
            location = self.evolverTypeLocation
        elif baseClass == Transform:
            collection = self._transforms
            location = self.transformTypeLocation
        else:
            raise ValueError(f"Base class not supported: {baseClass}")

        typ = type(obj)

        # check if it is already stored
        for index, stored in enumerate(collection):
            if stored.__name__ == typ.__name__:
                return index

        # else: is not stored yet
        index = len(collection)
        collection.append(typ)

        grp = h5file.create_group(location+f"/{index}")
        _storeTypeTo(obj, grp, baseClass, self.extraDefinitions)

        getLogger(__name__).info("Saved %s number %d: %s",
                                 "evolver" if baseClass == Evolver else "transform",
                                 index, typ)

        return index

    def save(self, obj, h5group):
        r"""!
        Save an evolver or transform including its type.
        \param obj Evolver or Transform object to save.
        \param h5group Group in the HDF5 file to save parameters to.
                       Stores the index of the type in the attribute `__index__` of the group.
        """

        # save the type
        if isinstance(obj, Evolver):
            index = self._saveType(obj, Evolver, h5group.file)
            h5group.attrs["__object_kind__"] = "Evolver"
        elif isinstance(obj, Transform):
            index = self._saveType(obj, Transform, h5group.file)
            h5group.attrs["__object_kind__"] = "Transform"
        else:
            getLogger(__name__).error("Cannot save object of type %s", type(obj))
            raise ValueError(f"Cannot save object {obj}")

        # save the parameters
        obj.save(h5group, self)

        # link to the type
        h5group.attrs["__index__"] = index

    def loadEvolverType(self, index, h5file):
        r"""!
        Load an evolver type from file.
        \param index Index of the type to load. Corresponds to group name in the type location.
        \param h5file HDF5 file to load the type from.
        """
        return _retrieveTypeFrom(h5file[self.evolverTypeLocation+f"/{index}"], Evolver)

    def loadTransformType(self, index, h5file):
        r"""!
        Load an Transform type from file.
        \param index Index of the type to load. Corresponds to group name in the type location.
        \param h5file HDF5 file to load the type from.
        """
        return _retrieveTypeFrom(h5file[self.transformTypeLocation+f"/{index}"], Transform)

    def load(self, h5group, action, lattice, rng):
        r"""!
        Load the type of an evolver or transform and construct an instance from given group.
        The type has to be stored in the 'type location' (see `__init__`) in the same file
        as `h5group`.
        \param h5group Group in the file where evolver/transform parameters are stored.
        \param action Passed to evolver's/transform's constructor.
        \param lattice Passed to evolver's/transform's constructor.
        \param rng Passed to evolver's/transform's constructor.
        """
        kind = h5group.attrs["__object_kind__"]
        if kind == "Evolver":
            return self.loadEvolverType(int(h5group.attrs["__index__"]), h5group.file) \
                       .fromH5(h5group, self, action, lattice, rng)
        if kind == "Transform":
            return self.loadTransformType(int(h5group.attrs["__index__"]), h5group.file) \
                       .fromH5(h5group, self, action, lattice, rng)

        getLogger(__name__).error("Cannot load object from file at location '%s', "
                                  "unsupported kind: %s\n",
                                  h5group.name, kind)
        raise RuntimeError("Cannot load object from file, unsupported kind")

def _storeTypeTo(obj, h5group, baseClass, definitions={}):
    r"""! \ingroup evolvers
    Save a class to HDF5.

    There are three possible scenarios:
     - The class is built into Isle: Only its name is stored.
       It can be reconstructed automatically.
     - The class is custom defined and included in parameter `definitions`:
       Only its name is stored. It needs to be passed to _retrieveTypeFrom() when
       reconstructing.
     - The class is custom defined and not included in `definitions`:
       The full source code of the class's definition is stored.
       This requries that the class is fully self-contained, i.e. not use
       any symbols from outside its own definition (except for `isle`).

    \see _retrieveTypeFrom to load evolvers saved with this function.
    """

    if not isinstance(obj, baseClass):
        getLogger(__name__).error("Can only save instances of subclasses of %s, given %s",
                                  baseClass, type(obj))
        raise ValueError("Not an evolver")

    # get the name of the class
    name = type(obj).__name__
    if name == "__as_source__":
        getLogger(__name__).error("Classes must not be called __as_source__. "
                                  "That name is required for internal use.")
        raise ValueError("Class must not be called __as_source__")

    if obj.__module__.startswith("isle.evolver") or type(obj).__name__ in definitions:
        # builtin or custom
        h5group["__name__"] = name
        getLogger(__name__).info("Saved type %s via its name", name)

    else:
        # store source
        h5group["__name__"] = "__as_source__"
        src = sourceOfClass(type(obj))
        # attempt to reconstruct it to check for errors early
        import isle
        classFromSource(src, {"isle": isle,
                              "evolver": isle.evolver,
                              "evolver.evolver": isle.evolver.evolver,
                              "Evolver": Evolver,
                              "transform": isle.evolver.transform,
                              "evolver.transform": isle.evolver.transform,
                              "evolver.transform.transform": isle.evolver.transform,
                              "Transform": isle.evolver.transform.Transform})
        h5group["__source__"] = src
        getLogger(__name__).info("Saved type %s as source", name)

def _retrieveTypeFrom(h5group, baseClass, definitions={}):
    r"""! \ingroup evolvers
    Retrieves a class from HDF5.

    \param h5group HDF5 group containing name and optionally source a class.
    \param baseClass Required base class of the class retrieved from HDF5.
    \param definitions Dict containing custom definitions. If it contains an entry
                       with the name of the class, it is loaded based on that
                       entry instead of from source code.
    \return Class loaded from file.

    \see _storeTypeTo() to save classes in a supported format.
    """

    # name of the class
    name = h5group["__name__"][()]

    import isle  # get it here so it is not imported unless needed

    if name == "__as_source__": # from source
        try:
            cls = classFromSource(h5group["__source__"][()],
                                  {"isle": isle,
                                   "evolver": isle.evolver,
                                   "evolver.evolver": isle.evolver.evolver,
                                   "Evolver": Evolver,
                                   "transform": isle.evolver.transform,
                                   "evolver.transform": isle.evolver.transform,
                                   "evolver.transform.transform": isle.evolver.transform,
                                   "Transform": isle.evolver.transform.Transform})
        except ValueError:
            getLogger(__name__).error("Source code does not define a class. "
                                      "Cannot load %s.", baseClass)
            raise RuntimeError("Cannot load class from source") from None

    else:  # from name + known definition
        try:
            # try and load a built in class
            if baseClass == Evolver:
                cls = isle.evolver.__dict__[name]
            elif baseClass == Transform:
                cls = isle.evolver.transform.__dict__[name]
            else:
                getLogger(__name__).error("Base class not supported when loading types in evolver"
                                          "manager: %s", baseClass)
                raise ValueError("Base class not supported")

        except KeyError:
            try:
                # provided by user
                cls = definitions[name]
            except KeyError:
                getLogger(__name__).error(
                    "Unable to load class '%s' from source. "
                    "The type is neither built in nor provided through argument 'definitions'.",
                    name)
                raise RuntimeError("Cannot load type from source") from None

    if not issubclass(cls, baseClass):
        getLogger(__name__).error("Loaded type is not derived from target base class: %s, base=%s",
                                  name, baseClass)
        raise RuntimeError("Loaded type is not derived from correct base class")

    return cls
