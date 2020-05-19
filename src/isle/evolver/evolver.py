r"""!\file
\ingroup evolvers
Base class for evolvers.
"""

from abc import ABCMeta, abstractmethod


class Evolver(metaclass=ABCMeta):
    r"""! \ingroup evolvers
    Abstract base class for evolvers.
    """

    @abstractmethod
    def evolve(self, stage):
        r"""!
        Evolve a configuration and momentum.
        \param stage EvolutionStage at the beginning of this evolution step.
        \returns EvolutionStage at the end of this evolution step.
        """

    @abstractmethod
    def save(self, h5group, manager):
        r"""!
        Save the evolver to HDF5.
        Has to be the inverse of Evolver.fromH5().
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the evolver in.
        """

    @classmethod
    @abstractmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct an evolver from HDF5.
        Create and initialize a new instance from parameters stored via Evolver.save().
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed evolver.
        """

    @abstractmethod
    def report(self):
        r"""!
        Return a string summarizing the evolution since the evolver
        was constructed including by fromH5.
        """
