r"""!\file
\ingroup meas

# One-point functions

We can use the bilinear operators with vacuum quantum numbers
described in the SpinSpinCorrelator documentation to compute one-point functions.

All the vacuum quantum number bilinears can be written as combinations of the
number operators,
\f{align}{
    N^p_x & = \left\langle 1-a_x a_x^\dagger \right\rangle
            = \left\langle 1-P_{xx} \right\rangle
    \\
    N^h_x & = \left\langle 1-b_x b_x^\dagger \right\rangle
            = \left\langle 1-H_{xx} \right\rangle
\f}
For example, the expected charge density operator
\f[
    \rho_x = 1-2S^0_x = N^p_x - N^h_x
\f]
(as we use positive particles) and the z-component of spin is given by
\f[
    S^3_x = \frac{1}{2} \left( 1 - N^p_x - N^h_x \right)
\f]

We can transform into a different basis; by default the results are in the spatial basis.
"""


from logging import getLogger

import numpy as np

import isle
from .measurement import Measurement
from ..util import temporalRoller
from ..h5io import createH5Group

class onePointFunctions(Measurement):
    r"""!
    \ingroup meas
    Tabulate one-point correlators.
    """

    def __init__(self, particleAllToAll, holeAllToAll, savePath, configSlice=(None, None, None), transform=None):
        super().__init__(savePath, configSlice)

        # The correlation functions encoded here are between bilinear operators.
        # Since the individual constituents are fermionic, the bilinear is bosonic.
        self.fermionic = False

        self.particle=particleAllToAll
        self.hole=holeAllToAll

        self.data = {k: [] for k in ["N_p", "N_h"]}

        self.transform = transform

        self._einsum_path = None

    def __call__(self, stage, itr):
        """!Record the spin-spin correlators."""

        P = self.particle(stage, itr)
        H = self.hole(stage, itr)

        nx = P.shape[0]
        nt = P.shape[1]

        d = np.eye(nx*nt).reshape(*P.shape) # A Kronecker delta
        if self.transform is None:
            self.transform = np.eye(nx)

        log = getLogger(__name__)

        data={}
        data["N_p"] = d-P
        data["N_h"] = d-H

        # We'll time average and transform at once:
        if self._einsum_path is None:
            self._einsum_path, _ = np.einsum_path("ax,xtxt->a", self.transform, data["N_p"], optimize="optimal")
            log.info("Optimized Einsum path for time averaging and unitary transformation.")


        for correlator in self.data:
            measurement = np.einsum("ax,xtxt->a", self.transform, data[correlator], optimize=self._einsum_path) / nt
            self.data[correlator].append(measurement)


    def save(self, h5group):
        r"""!
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["transform"] = self.transform
        for field in self.data:
            subGroup[field] = self.data[field]

def read(h5group):
    r"""!
    \param h5group HDF5 group which contains the data of this measurement.
    """
    ...
