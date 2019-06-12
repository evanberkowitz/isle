r"""!\file
\ingroup evolvers
Transform to shift configurations by a constant.
"""


import numpy as np

from .transform import Transform
from ... import CDVector


class ConstantShift(Transform):
    r"""! \ingroup evolvers
    Transform that shifts configurations by a constant.
    """

    def __init__(self, shift, action, lattSize=None):
        if isinstance(shift, (np.ndarray, CDVector)):
            self.shift = CDVector(shift)
        else:
            if lattSize is None:
                raise ValueError("Argument lattSize must not be None if shift is"
                                 " passed as a scalar")
            self.shift = CDVector(np.full(lattSize, shift, dtype=complex))
        self.action = action

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
        return phi+self.shift, self.action.eval(phi), 0

    def backward(self, phi, jacobian=False):
        r"""!
        Transform a configuration from MC to proposal manifold.
        \param phi Configuration on MC manifold.
        \returns
            - Configuration on proposal manifold
            - \f$\log \det J\f$ where \f$J\f$ is the Jacobian of the
              *forwards* transformation. `None` if `jacobian==False`.
        """
        return phi-self.shift, 0 if jacobian else None

    def save(self, h5group, manager):
        r"""!
        Save the transform to HDF5.
        Has to be the inverse of Transform.fromH5().
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the transform in.
        """
        h5group["shift"] = self.shift

    @classmethod
    def fromH5(cls, h5group, _manager, action, _lattice, _rng):
        r"""!
        Construct a trasnform from HDF5.
        Create and initialize a new instance from parameters stored via Identity.save().
        \param h5group HDF5 group to load parameters from.
        \param _manager EvolverManager responsible for the HDF5 file.
        \param action Action to use.
        \param _lattice Lattice the simulation runs on.
        \param _rng Central random number generator for the run.
        \returns A newly constructed identity transform.
        """
        return cls(h5group["shift"][()], action)
