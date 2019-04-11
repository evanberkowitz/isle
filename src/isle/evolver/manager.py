r"""!\file
\ingroup evolvers
Handlers for I/O from/to HDF5 files.
"""

from inspect import getmodule
from logging import getLogger
from pathlib import Path

import h5py as h5

from .evolver import Evolver
from ..meta import classFromSource, sourceOfClass


class EvolverManager:
    r"""! \ingroup evolvers
    Manages evolvers in a file.

    Handles saving and loading of types and parameters of evolvers in a centralized way.
    Operates with the file structure descriped in
    \todo link to file doc
    """

    def __init__(self, fname, typeLocation="/meta/evolvers", definitions={}):
        r"""!
        Initialize for a given file.
        \param fname Path to the file to manage.
        \param typeLocation Path inside the file where evolver types are to be stored.
        \param definitions Dictionary of extra definitions to take into account when
               saving/loading evolvers. See saveEvolverType() and loadEvolverType().
        """

        self.typeLocation = typeLocation
        self.extraDefinitions = definitions
        # list of all currently existing evolvers in the file
        # index in this list is index in file
        self._evolvers = self._loadTypes(Path(fname))

    def _loadTypes(self, fname):
        r"""!
        Load all types of evolvers in file `fname`.
        """

        if not fname.exists():
            getLogger(__name__).info("File to load evolvers from does not exist: %s", fname)
            return []

        with h5.File(fname, "r") as h5f:
            try:
                grp = h5f[self.typeLocation]
            except KeyError:
                getLogger(__name__).info("No evolvers found in file %s", fname)
                return []

            evolvers = [loadEvolverType(g, self.extraDefinitions)
                         for _, g in sorted(grp.items(), key=lambda p: int(p[0]))]
        getLogger(__name__).info("Loaded evolver types from file %s:\n    %s",
                                 fname,
                                 "\n    ".join(f"{i}: {p}" for i, p in enumerate(evolvers)))
        return evolvers

    def saveType(self, evolver, h5file):
        r"""!
        Save the type of an evolver if it is not already stored.
        \param evolver Evolver object (<I>not</I> type!) to save.
        \param h5file File to save to.
        \returns Index of the evolver type in the file.
        """

        typ = type(evolver)

        # check if it is already stored
        for index, stored in enumerate(self._evolvers):
            if stored.__name__ == typ.__name__:
                return index

        # else: is not stored yet
        index = len(self._evolvers)
        self._evolvers.append(typ)

        grp = h5file.create_group(self.typeLocation+f"/{index}")
        saveEvolverType(evolver, grp, self.extraDefinitions)

        getLogger(__name__).info("Saved evolver number %d: %s", index, typ)

        return index

    def save(self, evolver, h5group):
        r"""!
        Save an evolver including its type.
        \param evolver Evolver object to save.
        \param h5group Group in the HDF5 file to save the evolver's parameters to.
                       Stores the index of the evolver's type in the attribute `__index__`
                       of the group.
        """

        # save the type
        index = self.saveType(evolver, h5group.file)

        # save the parameters
        evolver.save(h5group, self)

        # link to the type
        h5group.attrs["__index__"] = index

    def loadType(self, index, h5file):
        r"""!
        Load an evolver type from file.
        \param index Index of the type to load. Corresponds to group name in the type location.
        \param h5file HDF5 file to load the type from.
        """
        return loadEvolverType(h5file[self.typeLocation+f"/{index}"])

    def load(self, h5group, action, lattice, rng):
        r"""!
        Load an evolver's type and construct an instance from given group.
        The type has to be stored in the 'type location' (see `__init__`) in the same file
        as `h5group`.
        \param h5group Group in the file where evolver parameters are stored.
        \param action Passed to evolver's constructor.
        \param lattice Passed to evolver's constructor.
        \param rng Passed to evolver's constructor.
        """
        return self.loadType(h5group.attrs["__index__"][()], h5group.file) \
                   .fromH5(h5group, self, action, lattice, rng)


def saveEvolverType(evolver, h5group, definitions={}):
    r"""! \ingroup evolvers
    Save an evolver's type to HDF5.

    There are three possible scenarios:
     - The evolver is built into Isle: Only its name is stored.
       It can be reconstructed automatically.
     - The evolver is custom defined and included in parameter `definitions`:
       Only its name is stored. It needs to be passed to loadEvolverType() when
       reconstructing.
     - The evolver is custom defined and not included in `definitions`:
       The full source code of the evolver's definition is stored.
       This requries that the evolver is fully self-contained, i.e. not use
       any symbols from outside its own definition (except for `isle`).

    \see loadEvolverType to load evolvers saved with this function.
    """

    if not isinstance(evolver, Evolver):
        getLogger(__name__).error("Can only save instances of subclasses of "
                                  "isle.evolver.Evolver, given %s",
                                  type(evolver))
        raise ValueError("Not an evolver")

    # get the name of the evolver's class
    name = type(evolver).__name__
    if name == "__as_source__":
        getLogger(__name__).error("Evolvers must not be called __as_source__. "
                                  "That name is required for internal use.")
        raise ValueError("Evolver must not be called __as_source__")

    if evolver.__module__.startswith("isle.evolver") or type(evolver).__name__ in definitions:
        # builtin or custom
        h5group["__name__"] = name
        getLogger(__name__).info("Saved type of evolver %s via its name", name)

    else:
        # store source
        h5group["__name__"] = "__as_source__"
        src = sourceOfClass(type(evolver))
        # attempt to reconstruct it to check for errors early
        import isle
        classFromSource(src, {"isle": isle, "evolvers": isle.evolver, "Evolver": Evolver})
        h5group["__source__"] = src
        getLogger(__name__).info("Saved type of evolver %s as source", name)

def loadEvolverType(h5group, definitions={}):
    r"""! \ingroup evolvers
    Retrieves the class of an evolver from HDF5.

    \param h5group HDF5 group containing name, (source), parameters of an evolver.
    \param definitions Dict containing custom definitions. If it contains an entry
                       with the name of the evolver, it is loaded based on that
                       entry instead of from source code.
    \return Class loaded from file.

    \see saveEvolverType() to save evolvers in a supported format.
    """

    name = h5group["__name__"][()]

    import isle  # get it here so it is not imported unless needed

    if name == "__as_source__": # from source
        try:
            cls = classFromSource(h5group["__source__"][()],
                                  {"isle": isle,
                                   "evolver": isle.evolver,
                                   "evolver.evolver": isle.evolver.evolver,
                                   "Evolver": Evolver})
        except ValueError:
            getLogger(__name__).error("Source code for evolver does not define a class. "
                                      "Cannot load evolver.")
            raise RuntimeError("Cannot load evolver from source") from None

    else:  # from name + known definition
        try:
            # builtin, in the package that Evolver is defined in (i.e. this one)
            # cls = getmodule(Evolver).__dict__[name]
            cls = isle.evolver.__dict__[name]
        except KeyError:
            try:
                # provided by user
                cls = definitions[name]
            except KeyError:
                getLogger(__name__).error(
                    "Unable to load evolver of type '%s' from source. "
                    "The type is neither built in nor provided through argument 'definitions'.",
                    name)
                raise RuntimeError("Cannot load evolver from source") from None

    if not issubclass(cls, Evolver):
        getLogger(__name__).error("Loaded type is not an evolver: %s", name)

    return cls
