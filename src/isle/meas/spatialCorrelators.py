r"""!\file
\ingroup meas

# Spatial correlator operators 

We calculate the operator
\f{align}{
       & \langle a^{}_x(t)a^\dag_{k,\sigma}(0)\rangle
       & \langle a^\dag_x(t)a^{}_{k,\sigma}(0)\rangle
\f}

where 

\f[
     a^\dag_{k,\sigma}=\sum_i P_i(k\sigma)a_i^\dag
\f]

and \f$P_i(k\sigma)\f$ are projection operators to a particular \f$k,\sigma\f$ (ie. momentum,band).  
Such projection operators are gotten, for example, from diagonalizing the connectivity matrix.

In terms of the fermion matrix \f$M\f$ and the ensemble of configurations \f$\{\Phi\}\f$ this expression becomes
\f[
    C(x,t)= \frac{1}{N_{cfg}}\sum_{i\in\{\Phi\}}\sum_y M^{-1}[xt,y0;\phi_i]P_y(k\sigma)
\f]

One can also average over initial time sources (taking into account the anti-periodic BCs in time) to increase 
statistics.  (the code does do this)

"""

import h5py as h5
import numpy as np
from pentinsula.h5utils import open_or_pass_file

from .measurement import Measurement, BufferSpec
from ..util import temporalRoller
from ..h5io import createH5Group, empty
from .propagator import AllToAll


class SpatialCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    CORRELATOR_NAMES = {"creation_destruction", "destruction_creation"}

    def __init__(self, allToAll, lattice, savePath, configSlice=slice(None, None, None),
                 transform=None, correlators=CORRELATOR_NAMES):
        r"""!
        \param allToAll propagator.AllToAll for one species.
        \param savePath Path in the output file where results are saved.
        \param configSlice `slice` indicating which configurations to measure on.
        \param transform   Transformation matrix applied to correlators in position space.
        \param correlators Iterable of names of correlators to compute.
                           Defaults to `SingleParticleCorrelator.CORRELATOR_NAMES`.
        """

        _checkCorrNames(correlators, self.CORRELATOR_NAMES)
        super().__init__(savePath,
                         tuple(BufferSpec(name, (lattice.nx(), lattice.nx(), lattice.nt()),
                                          complex, name)
                               for name in correlators),
                         configSlice)

        # The correlation functions encoded here are between single ladder operators.
        self.fermionic = True

        self._inverter = allToAll

# TODO??
        self._path = {c: None for c in correlators}

        self.transform = transform
        self._indices = dict()
        if self.transform is None:
            self._indices["creation_destruction"] = "idf,yixf->xyd"
            self._indices["destruction_creation"] = "idf,xfyi->xyd"
        else:
            self._indices["creation_destruction"] = "idf,bx,yixf,ya->bad"
            self._indices["destruction_creation"] = "idf,bx,xfyi,ya->bad"

        self._einsum_paths = {c: None for c in correlators}

        self._roll = None

    def __call__(self, stage, itr):
        """!Record the single-particle correlators."""

        S = self._inverter(stage, itr)
        nx = S.shape[0]
        nt = S.shape[1]

        d = np.eye(nx*nt).reshape(*S.shape) # A kronecker delta

        if self._roll is None:
            self._roll = np.array([temporalRoller(nt, -t, fermionic=self.fermionic) for t in range(nt)])

        # If there's no transformation needed, we should avoid doing
        # space matrix-matrix-matrix, as it will scale poorly.
        tensors = dict()
        if self.transform is None:
            tensors['destruction_creation'] = (self._roll, S)
            tensors['creation_destruction'] = (self._roll, d-S)
        else:
            tensors['destruction_creation'] = (self._roll, self.transform.T.conj(), S, self.transform)
            tensors['creation_destruction'] = (self._roll, self.transform.T.conj(), d-S, self.transform)

        for c in self._einsum_paths:
            if self._einsum_paths[c] is None:
                self._einsum_paths[c], _ = np.einsum_path(self._indices[c], *tensors[c], optimize='optimal')

        # The temporal roller sums over time, but does not *average* over time.  So, divide by nt:
        for name in self._einsum_paths:
            res = self.nextItem(name)
            np.einsum(self._indices[name], *tensors[name],
                      optimize=self._einsum_paths[name], out=res)
            res /= nt

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        res = super().setup(memoryAllowance, expectedNConfigs, file, maxBufferSize)
        with open_or_pass_file(file, None, "a") as h5f:
            # TODO store the optimization; this line of code doesn't work.
            # subGroup["einsum_path"] = self._path

            if self.transform is None:
                h5f[self.savePath]["transform"] = empty(dtype="complex")
            else:
                h5f[self.savePath]["transform"] = self.transform
        return res


def _checkCorrNames(actual, allowed):
    for name in actual:
        if name not in allowed:
            raise ValueError(f"Unknown correlator: '{name}'. Choose from '{allowed}'")
