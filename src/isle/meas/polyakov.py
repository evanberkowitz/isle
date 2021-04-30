r"""!\file
\ingroup meas

In QCD the Polyakov loop is a Wilson line that wraps the temporal direction,
\f[
    \mathcal{P}(\vec{x}) = \textrm{tr}\left[ \prod_{j=0}^{N_t-1} U_{t}(\vec{x},j)\right]
\f]
where the product is time-ordered and the trace yields a color singlet.
The correlation function of Polyakov loops yields access to the static quark potential,
\f[
    \left\langle \mathcal{P}(\vec{m}) \mathcal{P}^{\dagger}(\vec{n}) \right\rangle
    \propto
    \exp\left(-N_t a V(\vec{m},\vec{n})\right) \left[1+\mathcal{O}\left(\exp(-N_t a \Delta E)\right)\right]
\f]
where \f$V\f$ is the potential between static sources on sites \f$\vec{m}\f$ and \f$\vec{n}\f$,
\f$a\f$ is the lattice spacing,
and \f$\Delta E\f$ is the difference between \f$V\f$ and the first excited energy level of a quark-antiquark pair.
For an enlightening derivation and discussion, see Gattringer \& Lang, *Quantum Chromodynamics on the Lattice* (2010), sections 3.3, 4.5.4, 12.1.
Whether the Polyakov loop correlator is related to a static potential in our case is not *a priori* obvious, as there is no equivalent quantity to the plaquette, which encodes the energy of the gauge fields in QCD.

In our case we can calculate the analogous
\f[
    \mathcal{P}_{xy} = \prod_{j=0}^{N_t-1} F_j = \exp\left(i\sum_j \phi_{xj}\right) \delta_{xy}
    \quad\quad\quad
    (\alpha=1)
\f]
a diagonal matrix whose entries are the Polyakov loops for the spatial sites.
(The trace is trivial because the "gauge group" is \f$U(1)\f$).
We define the argument
\f[
    \Phi_x = \sum_t \phi_{xt}
\f]
and store it as `Phi_x`.  The Polyakov loops are stored as `Polyakov`.
The formation of potentials and correlators are left for analysis.

"""

import numpy as np

import isle
from .measurement import Measurement, BufferSpec

from logging import getLogger

class Polyakov(Measurement):
    r"""!
    \ingroup meas
    Tabulate the Polyakov loop.
    """

    def __init__(self, basis, lattice, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath,
                         (BufferSpec("Phi_x", (lattice.nx(),), complex, "Phi_x"),
                          BufferSpec("P", (lattice.nx(),), complex, "Polyakov")),
                         configSlice)

        self.basis = basis
        self.nt = lattice.nt()

        try:
            if self.basis == isle.action.HFABasis.PARTICLE_HOLE:
                self.forward = 1j
            elif self.basis == isle.action.HFABasis.SPIN:
                self.forward = 1
        except:
            getLogger(__name__).exception(f"Unknown basis, {self.basis}.")


    def __call__(self, stage, itr):
        """!Record the sum_t phi_xt and the Polyakov loops."""
        Phi_x = np.sum(np.reshape(stage.phi, (self.nt, -1)), axis=0)
        self.nextItem("Phi_x")[...] = Phi_x
        self.nextItem("P")[...] = np.exp(self.forward * Phi_x)
