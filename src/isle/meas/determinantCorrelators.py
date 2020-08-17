r"""!\file
\ingroup meas
Measurement of single-particle correlator.
"""

import numpy as np

from .measurement import Measurement, BufferSpec
from ..util import temporalRoller


class DeterminantCorrelators(Measurement):
    r"""!
    \ingroup meas
    Tabulate correlator with a creation operator on each site.

    Because particles and holes are fermions, the Wick contractions are totally
    antisymmetric, which makes it particularly easy to compute.

    If P[xf,tf,xi,ti] is an all-to-all particle propagator, then the Wick contractions
    for this very symmetric case can be written as the determinant on the spatial
    indices xf and xi.

    How come?  Consider the correlation function between
        a[n]a[n-1]...a[3]a[2]a[1]               at time t
    and
        adag[1] adag[2] adag[3] ... adag[n]     at time 0
    where the product is over all the sites.

    We must now compute all the Wick contractions.  The first term is obvious,
        + P11 P22 P33 ... Pnn   (suppressing time indices)
    the next obvious one is to contract a[2] with adag[1] and a[1] with adag[2].
    Because this contraction has a crossing, it gets a minus sign.
        - P12 P21 P33 ... Pnn
    How many Wick contractions *are* there?  It's clearly n!
    And, every time there's a crossing, in the contraction we pick up another sign.

    Now compare to this formula for the determinant of A:

    det A = epsilon_{i_1, i_2, i_3, ... i_n} A_{1,i_1} A_{2,i_2} A_{3,i_3} \cdots A_{n,i_n}

    They clearly match!

    So, by taking determinants over the spatial indices we can construct correlators
    for these maximal operators.

    """

    def __init__(self, particleAllToAll, holeAllToAll, lattice, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath,
                         (BufferSpec("P", (lattice.nt(),), complex, "P"),
                          BufferSpec("H", (lattice.nt(),), complex, "H"),
                          BufferSpec("PH", (lattice.nt(),), complex, "PH")),
                         configSlice)

        self.particle = particleAllToAll
        self.hole = holeAllToAll

        # The correlation functions encoded here are between
        # Nx ladder operators.  So, whether the operator is fermionic or not
        # depends on the parity of the number of sites.
        self.fermionic = (1 == np.mod(self.particle.nx, 2))

        self._time_slowest = "xfyi->fixy"
        self._time_averaging = "idf,fi->d"
        self._roll = None

    def __call__(self, stage, itr):
        """!Record the determinant correlators."""

        P = np.einsum(self._time_slowest, self.particle(stage, itr))
        H = np.einsum(self._time_slowest, self.hole(stage, itr))

        # Determinants are real if you use the exponential discretization.
        # But, store complex numbers as a discretization-agnostic
        # TODO: improve this, save a factor of 2 on storage for the exponential case.
        det = {"P": np.zeros(P.shape[0:2], dtype=complex),
               "H": np.zeros(H.shape[0:2], dtype=complex),
               "PH": np.zeros(H.shape[0:2], dtype=complex)}

        nt = det["P"].shape[0]
        time = np.arange(nt)

        for f in time:
            for i in time:
                det["P"][f,i] = np.linalg.det(P[f,i])
                det["H"][f,i] = np.linalg.det(H[f,i])
                det["PH"][f,i] = det["P"][f,i]*det["H"][f,i]

        self._roll = np.array([temporalRoller(nt, -t, fermionic=self.fermionic) for t in range(nt)])
        self.nextItem("P")[...] = np.einsum(self._time_averaging, self._roll/nt, det["P"])
        self.nextItem("H")[...] = np.einsum(self._time_averaging, self._roll/nt, det["H"])

        # If we fully populate particles AND holes the operator is necessarily bosonic.
        self._roll = np.array([temporalRoller(nt, -t, fermionic=False) for t in range(nt)])
        self.nextItem("PH")[...] = np.einsum(self._time_averaging, self._roll/nt, det["PH"])
