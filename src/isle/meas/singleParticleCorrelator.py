r"""!\file
\ingroup meas

# Single-Particle and -Hole operators

On each site we have four ladder operators, the particle destruction and creation
operators, \f$a\f$ and \f$a^\dagger\f$, and the hole operators \f$b\f$, \f$b^\dagger\f$.

Particles also have z-component of spin \f$-\frac{1}{2}\f$ because
\f$[S^3_x, a_y^\dagger] = -\frac{1}{2} a_y^\dagger \delta_{xy}\f$
 and charge \f$+1\f$ because
\f$[\rho_x, a_y^\dagger] = +1 a_y^\dagger \delta_{xy}\f$
(see spinSpinCorrelator.py for more details).
Similarly, holes have z-component of spin \f$-\frac{1}{2}\f$ (the same sign as particles!)
but charge \f$-1\f$.

We can therefore construct operators that create a definite charge but indefinite spin, and vice-versa,
\f{align}{
    q^+_x &= \frac{a_x^\dagger + b_x}{\sqrt{2}}
    &
    q^-_x &= \frac{a_x + b_x^\dagger}{\sqrt{2}}
    \\
    s^+_x &= \frac{a_x + b_x}{\sqrt{2}}
    &
    s^-_x &= \frac{a_x^\dagger + b_x^\dagger}{\sqrt{2}}
\f}
so that \f$[\rho_x, q^\pm_y] = \pm 1           q^\pm_y \delta_{xy}\f$
and     \f$[S^3_x,  s^\pm_y] = \pm \frac{1}{2} s^\pm_y \delta_{xy}\f$
but these guys have indefinite (spin/charge, respectively) quantum number, and therefore mix
under Hamiltonian evoluation.  Diagonalizing would reveal the particle/hole sectors
already discussed, so we do not construct correlation functions with these operators,
or any other linear combinations (there are plenty).

# Correlation functions

Now we can write correlation functions
\f[
    C^{ij}_{xy}(\tau) = \frac{1}{N_t} \sum_t \left\langle i_{x,t+\tau} j_{y,t} \right\rangle
\f]
and we suppress the time dependence for the time being.

We can perform contractions,
\f{align}{
    C^{a a^\dagger}_{xy}(\tau)  &   = \left\langle a_{x} a_{y}^\dagger \right\rangle
                                    = \left\langle P_{xy} \right\rangle
                            &
    C^{a^\dagger a}_{xy}(\tau)  &   = \left\langle a_x^\dagger a_y \right\rangle
                                    = \left\langle \delta_{yx} - a_y a_x^\dagger \right\rangle
                                    = \left\langle \delta_{yx} - P_{yx} \right\rangle
                            \\
    C^{b b^\dagger}_{xy}(\tau)  &   = \left\langle b_{x} b_{y}^\dagger \right\rangle
                                    = \left\langle H_{xy} \right\rangle
                            &
    C^{b^\dagger b}_{xy}(\tau)  &   = \left\langle b_x^\dagger b_y \right\rangle
                                    = \left\langle \delta_{yx} - b_y b_x^\dagger \right\rangle
                                    = \left\langle \delta_{yx} - H_{yx} \right\rangle
\f}
and because of the identity \f$\{a_x,a_y^\dagger\} = \delta_{xy}\f$ one can show that
in expectation value \f$C^{a a^\dagger}_{xy}(\tau) = C^{a^\dagger a}_{yx}(-\tau)\f$
and likewise for holes.  Note that this is true in expectation value and not
configuration-by-configuration.  Therefore it is worthwhile to perform both measurements.

"""

import numpy as np

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, temporalRoller
from ..h5io import createH5Group, empty
from .propagator import AllToAll


def _checkCorrNames(actual, allowed):
    for name in actual:
        if name not in allowed:
            raise ValueError(f"Unknown correlator: '{name}'. Choose from '{allowed}'")

class SingleParticleCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate single-particle correlator.
    """

    CORRELATOR_NAMES = {"creation_destruction", "destruction_creation"}

    def __init__(self, allToAll, savePath, configSlice=slice(None, None, None),
                 transform=None, correlators=CORRELATOR_NAMES):
        r"""!
        \param allToAll propagator.AllToAll for one species.
        \param savePath Path in the output file where results are saved.
        \param configSlice `slice` indicating which configurations to measure on.
        \param transform   Transformation matrix applied to correlators in position space.
        \param correlators Iterable of names of correlators to compute.
                           Defaults to `SingleParticleCorrelator.CORRELATOR_NAMES`.
        """
        super().__init__(savePath, configSlice)

        _checkCorrNames(correlators, self.CORRELATOR_NAMES)

        # The correlation functions encoded here are between single ladder operators.
        self.fermionic = True

        self._inverter = allToAll

        self.correlators = {c: [] for c in correlators}
        self._path = {c: None for c in correlators}

        self.transform = transform
        self._indices = dict()
        if self.transform is None:
            self._indices["creation_destruction"] = "idf,yixf->xyd"
            self._indices["destruction_creation"] = "idf,xfyi->xyd"
        else:
            self._indices["creation_destruction"] = "idf,bx,yixf,ya->bad"
            self._indices["destruction_creation"] = "idf,bx,xfyi,ya->bad"

        self._einsum_paths = {c: None for c in self.correlators}

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

        for c in self.correlators:
            if self._einsum_paths[c] is None:
                self._einsum_paths[c], _ = np.einsum_path(self._indices[c], *tensors[c], optimize='optimal')

        # The temporal roller sums over time, but does not *average* over time.  So, divide by nt:
        for name, correlator in self.correlators.items():
            self.correlators[name].append(np.einsum(self._indices[name],
                                                   *tensors[name],
                                                    optimize=self._einsum_paths[name]) / nt)

    def save(self, h5group):
        r"""!
        Write the transformation and correlators to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)

        for name, correlator in self.correlators.items():
            subGroup[name] = correlator

        if self.transform is None:
            subGroup["transform"] = empty(dtype=complex)
        else:
            subGroup["transform"] = self.transform
        # subGroup["einsum_path"] = self._path # TODO: store the optimization; this line of code doesn't work.

def read(h5group):
    r"""!
    Read the transform and their correlators from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["correlators"][()], h5group["transform"][()]
