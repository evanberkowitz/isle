r"""!\file
\ingroup evolvers
Identity transform.
"""


from .transform import Transform


class Identity(Transform):
    r"""! \ingroup evolvers
    Identity transform that returns the input configuration unchanged.
    """

    def forward(self, phi, actVal):
        r"""!
        %Transform a configuration from proposal to MC manifold.
        \param phi Configuration on proposal manifold.
        \param actVal Value of the action at phi.
        \returns In order:
          - Configuration on MC manifold.
          - Value of action at configuration on MC manifold.
          - \f$\log \det J\f$ where \f$J\f$ is the Jacobian of the transformation.
        """
        return phi, actVal, 0

    def backward(self, phi, jacobian=False):
        r"""!
        %Transform a configuration from MC to proposal manifold.
        \param phi Configuration on MC manifold.
        \returns
            - Configuration on proposal manifold
            - \f$\log \det J\f$ where \f$J\f$ is the Jacobian of the
              *forwards* transformation. `None` if `jacobian==False`.
        """
        return phi, 0 if jacobian else None

    def save(self, h5group, manager):
        r"""!
        Save the transform to HDF5.
        Has to be the inverse of Transform.fromH5().
        \param h5group HDF5 group to save to.
        \param manager EvolverManager whose purview to save the transform in.
        """
        # nothing to save

    @classmethod
    def fromH5(cls, _h5group, _manager, _action, _lattice, _rng):
        r"""!
        Construct a transform from HDF5.
        Create and initialize a new instance from parameters stored via Identity.save().
        \param _h5group HDF5 group to load parameters from.
        \param _manager EvolverManager responsible for the HDF5 file.
        \param _action Action to use.
        \param _lattice Lattice the simulation runs on.
        \param _rng Central random number generator for the run.
        \returns A newly constructed identity transform.
        """
        return cls()
