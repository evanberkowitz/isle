r"""!\file
\ingroup evolvers
Base class for evolvers.
"""

from abc import ABCMeta, abstractmethod


class Transform(metaclass=ABCMeta):
    r"""! \ingroup evolvers
    Abstract base class for evolver transforms.
    """

    @abstractmethod
    def forward(self, phi, actVal):
        r"""!
        Transform a configuration from proposal to MC manifold.
        \param phi Configuration on proposal manifold.
        \param actVal Value of the action at phi.
        \returns In order:
          - Configuration on MC manifold.
          - Value of action at configuration on MC manifold.
          - \f$\log \det J\f$ where \f$J\f$ is the Jacobian of the transformation.
        """

    @abstractmethod
    def backward(self, phi, jacobian=False):
        r"""!
        Transform a configuration from MC to proposal manifold.
        \param phi Configuration on MC manifold.
        \returns
            - Configuration on proposal manifold
            - \f$\log \det J\f$ where \f$J\f$ is the Jacobian of the
              *forwards* transformation. `None` if `jacobian==False`.
        """

    @abstractmethod
    def save(self, h5group, manager):
        r"""!
        Save the transform to HDF5.
        Has to be the inverse of Transform.fromH5().
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the transform in.
        """

    @classmethod
    @abstractmethod
    def fromH5(cls, h5group, manager, action, lattice, rng):
        r"""!
        Construct a trasnform from HDF5.
        Create and initialize a new instance from parameters stored via Transform.save().
        \param h5group HDF5 group to load parameters from.
        \param manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param lattice Lattice the simulation runs on.
        \param rng Central random number generator for the run.
        \returns A newly constructed transform.
        """


def backwardTransform(transform, stage):
    """!
    Backwards transform a configuration in an EvolutionStage and compute
    Jacobian for forwards transform if necessary.
    """

    # there is no transform => Jacobian is zero
    if transform is None:
        return stage.phi, 0

    # Jacobian is known and stored in EvolutionStage
    if "logdetJ" in stage.logWeights:
        return transform.backward(stage.phi, False)[0], \
            stage.logWeights["logdetJ"]

    # Jacobian is unknown
    return transform.backward(stage.phi, True)

def forwardTransform(transform, phi, actVal):
    """!
    Forwards transform a configuration and compute Jacobian.
    """

    # there is no transform
    if transform is None:
        return phi, actVal, 0

    # delegate to the actual transform
    return transform.forward(phi, actVal)
