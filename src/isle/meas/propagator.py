"""!
Measurement of propagators
"""

import numpy as np

from logging import getLogger

import isle
from ..memoize import MemoizeMethod

class AllToAll:
    r"""!
    \ingroup meas
    Tabulate single-particle or hole all-to-all propagator.
    Given a field phi, return the inverse of the species-appropriate fermion matrix M[phi].
    The result is a four-index object Minverse[xf,tf,xi,ti].

    \note The result of the most recent call is cached.
          This result gets re-used automatically if the value of the action, trajectory point, and trajectory index
          of subsequent calls are the same.
          The actual configuration is not taken into account!
    """

    def __init__(self, hfm, species, alpha=1):
        self.hfm = hfm
        self.nx = self.hfm.nx()
        self.species = species
        self._alpha = alpha

    @MemoizeMethod(lambda stage, itr: (stage.logWeights["actVal"], stage.trajPoint, itr))
    def __call__(self, stage, itr):
        r"""!
        Compute the all-to-all propagator.
        \returns (Minverse)_{yfxi}, a 4D tensor where y/f is the space/time at the sink and x/i the space/time at the source.
        """

        if np.mod(len(stage.phi), self.nx) != 0:
            getLogger(__name__).error(f"Field configuration does not partition evenly into spatial slices of size nx={nx}")
            # TODO: do something more drastic?  Exit?

        nt = int(len(stage.phi) / self.nx)

        # A large set of sources, one for each spacetime point, an identity matrix.
        # Just as a reminder, it's time-major (space is faster).
        rhss = isle.Matrix(np.eye(self.nx * nt, dtype=complex))

        # Solve M*x = b for all right-hand sides:
        if self._alpha == 1:
            res = np.array(isle.solveM(self.hfm, stage.phi, self.species, rhss), copy=False)
            # As explained in the documentation of isle.solveM,
            # the first index counts the right-hand side M was solved against,
            # the second index is a spacetime index.
            # So we transpose to put source label as the right label.
            # Because rhss is the identity matrix, we can now think of it as a spacetime label as well.
            res = res.T
        else:
            res = np.linalg.solve(isle.Matrix(self.hfm.M(-1j*stage.phi, self.species)), np.array(rhss).T).T

        # Now we transform the propagator into a four-index object with space, time, space, and time indices.
        propagator = res.reshape([nt, self.nx, nt, self.nx])
        propagator = np.transpose(propagator, axes=[1,0,3,2])
        return propagator

    def save(self, base, name):
        r"""!
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup ob base for this measurement.
        """
        ...

    def read(self, group):
        r"""!
        Read from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        ...
