r"""!\file
\ingroup meas

# One-point functions

We can use the bilinear operators with vacuum quantum numbers
described in the SpinSpinCorrelator documentation to compute one-point functions.

All the vacuum quantum number bilinears can be written as combinations of the
number operators,
\f{align}{
    \left\langle n^p_x \right\rangle
          & = \left\langle a_x^\dagger a_x \right\rangle
            = \left\langle 1-a_x a_x^\dagger \right\rangle
            = \left\langle 1-P_{xx} \right\rangle
    \\
    \left\langle n^h_x \right\rangle
          & = \left\langle b_x^\dagger b_x \right\rangle
            = \left\langle 1-b_x b_x^\dagger \right\rangle
            = \left\langle 1-H_{xx} \right\rangle
\f}
For example, the expected charge density operator
\f[
    \rho_x = 1-2S^0_x = n^p_x - n^h_x
\f]
(as we use positive particles), the total number operator
\f[
    n_x = n^p_x + n^h_x
\f]
and the z-component of spin is given by
\f[
    S^3_x = \frac{1}{2} \left( 1 - n_x \right)
\f]

We can transform into a different basis; by default the results are in the spatial basis.
"""


from logging import getLogger

import numpy as np
import h5py as h5
from pentinsula.h5utils import open_or_pass_file

from .measurement import Measurement, BufferSpec

#TODO: save / retrieve einsum paths.

class OnePointFunctions(Measurement):
    r"""!
    \ingroup meas
    Tabulate one-point correlators.
    """

    ## Set of names of all possible elementary one-point-functions.
    CORRELATOR_NAMES = {"np", "nh"}

    def __init__(self, particleAllToAll, holeAllToAll, savePath, configSlice=(None, None, None), transform=None):
        r"""!
        \param particleAllToAll propagator.AllToAll for particles.
        \param holeAllToAll propagator.AllToAll for holes.
        \param savePath Path in the output file where results are saved.
        \param configSlice `slice` indicating which configurations to measure on.
        \param transform Transformation matrix applied to correlators in position space.
        """

        assert particleAllToAll.hfm.nx() == holeAllToAll.hfm.nx()
        super().__init__(savePath,
                         tuple(BufferSpec(name, (particleAllToAll.hfm.nx(),), complex, name)
                               for name in self.CORRELATOR_NAMES),
                         configSlice)

        # The correlation functions encoded here are between bilinear operators.
        # Since the individual constituents are fermionic, the bilinear is bosonic.
        self.fermionic = False

        self.particle = particleAllToAll
        self.hole = holeAllToAll

        self.transform = transform

        self._einsum_path = None

    def __call__(self, stage, itr):
        """!Record the spin-spin correlators."""

        P = self.particle(stage, itr)
        H = self.hole(stage, itr)

        nx = P.shape[0]
        nt = P.shape[1]

        d = np.eye(nx*nt).reshape(*P.shape) # A Kronecker delta

        log = getLogger(__name__)

        data={}
        data["np"] = d-P
        data["nh"] = d-H

        if self._einsum_path is None:
            if self.transform is None:
                # No need for the transformation, cut the cost:
                self._einsum_path, _ = np.einsum_path("xtxt->x", data['np'], optimize="optimal")
                log.info("Optimized Einsum path for time averaging.")
            else:
                # We'll time average and transform at once:
                self._einsum_path, _ = np.einsum_path("ax,xtxt->a", self.transform, data["np"], optimize="optimal")
                log.info("Optimized Einsum path for time averaging and unitary transformation.")

        if self.transform is None:
            for name, correlator in data.items():
                measurement = np.einsum("xtxt->x", correlator, optimize=self._einsum_path) / nt
                self.nextItem(name)[...] = measurement
        else:
            for name, correlator in data.items():
                measurement = np.einsum("ax,xtxt->a", self.transform, correlator, optimize=self._einsum_path) / nt
                self.nextItem(name)[...] = measurement

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        res = super().setup(memoryAllowance, expectedNConfigs, file, maxBufferSize)
        with open_or_pass_file(file, None, "a") as h5f:
            if self.transform is None:
                h5f[self.savePath]["transform"] = h5.Empty(dtype="complex")
            else:
                h5f[self.savePath]["transform"] = self.transform
        return res

    @classmethod
    def computeDerivedCorrelators(cls, measurements, commonTransform):
        r"""!
        \param measurements a dictionary of measurements that has measurements of `"np"` and `"nh"`
        \param commonTransform A spatial matrix used to transform *all* correlators passed in through `measurements`,
                               i.e. the `transform` attribute of measurement objects.

        Measurements of one-point functions \f$\rho_x\f$, \f$n_x\f$,
        and \f$S^3_x\f$ are built from measurements of \f$n^p_x\f$ and \f$n^h_x\f$
        using the identities above.

        This can be used with the following example codeblock

        ```python
           # meas is an instance of OnePointFunctions
            derived = isle.meas.OnePointFunctions.computeDerivedCorrelators(
                {name: np.asarray(corr) for name, corr in meas.correlators.items()},
                 meas.transform)
        ```

        \returns `dict` with additional one-point functions, built from those already computed.
        """

        nx = next(iter(measurements.values())).shape[1]
        U = commonTransform if commonTransform is not None else np.eye((nx, nx))

        derived = dict()

        derived["rho"] = measurements["np"] - measurements["nh"]
        derived["n"]   = measurements["np"] + measurements["nh"]
        derived["S3"]  = 0.5 * (np.expand_dims(np.sum(U, axis=1), axis=0) - derived["n"])

        return derived
