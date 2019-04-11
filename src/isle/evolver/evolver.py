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
    def evolve(self, phi, pi, actVal, trajPoint):
        r"""!
        Evolve a configuration and momentum.
        \param phi Input configuration.
        \param pi Input Momentum.
        \param actVal Value of the action at phi.
        \param trajPoint 0 if previous trajectory was rejected, 1 if it was accepted.
        \returns In order:
          - New configuration
          - New momentum
          - Action evaluated at new configuration
          - Point along trajectory that was selected
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
