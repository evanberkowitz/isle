r"""!\file
\ingroup meas

# Spin Operators

Spin-spin correlation functions are correlation functions between local spin operators \f$S_x^i\f$ where \f$x\f$ is a lattice site and \f$i\f$ runs over the indices of the Pauli matrices,
\f[
    S_x^i = \frac{1}{2} \sum_{ss'} c_{xs} \sigma^i_{ss'} c^\dagger_{xs'}
\f]
and where \f$c\f$ is a doublet of operators,
\f[
    c_{xs} = \left(\begin{array}{c} a_x \\ (-\sigma_\kappa)^x b_x^\dagger \end{array}\right)
\f]
where \f$\sigma_\kappa\f$ is +1 on bipartite lattices and -1 on non-bipartite lattices, following the convention of Isle.
In Buividovich, Smith, Ulybyshev, and von Smekal [1807.07025](https://arxiv.org/abs/1807.07025) [PRB 98 235129](https://doi.org/10.1103/PhysRevB.98.235129) they also define, just after (1),
the electric charge operator
\f[
    \rho_x = c_{x,\uparrow}^\dagger c_{x,\uparrow} + c_{x,\downarrow}^\dagger c_{x,\downarrow} - 1
\f]
which can be rewritten as \f$ \rho_x = 1-2 S^0_x\f$, where the \f$0^{th}\f$ Pauli matrix is the \f$2\times2\f$ identity matrix.  The \f$S\f$ operators are Hermitian.

Rewriting those operators into the Isle basis,
\f{align}{
    S^0_x &= \frac{1}{2} \left[ a_x a_x^\dagger - b_x b_x^\dagger +1 \right] \\
    S^1_x &= \frac{1}{2} (-\sigma_\kappa)^x \left[ b_x^\dagger a_x^\dagger + a_x b_x \right] \\
    S^2_x &= \frac{i}{2} (-\sigma_\kappa)^x \left[ b_x^\dagger a_x^\dagger - a_x b_x \right] \\
    S^3_x &= \frac{1}{2} \left[ a_x a_x^\dagger + b_x b_x^\dagger -1 \right]
\f}
where the \f$\sigma_\kappa\f$ squares away when two \f$b\f$ operators are multiplied.
The three spin operators obey the commutation relation
\f[
    \left[ S_x^i, S_y^j \right] = i \delta_{xy} \epsilon^{ijk} S_x^k
\f]
which I have checked explicitly for \f$(i,j)=(1,2)\f$ by writing out the operators in all their glory and using the anticommutation properties of \f$a\f$ and \f$b\f$.
By a similar exercise one may show
\f[
    \left[ S_x^0, S_y^j \right] = 0
\f]
which I have checked explicitly for \f$j=3\f$.
// TODO: check commutation rules with the tight-binding Hamiltonian and interaction terms.

Single-particle and single-hole operators \f$\mathcal{O}\f$ with a definite third component of spin \f$s_3\f$ obeys the eigenvalue equation
\f[
    [ S_x^3, \mathcal{O}_y ] = s_3 \mathcal{O}_y \delta_{xy}.
\f]
This equation is satisfied when
\f$(\mathcal{O},s_3) = (a, +\frac{1}{2})\f$,
\f$(\mathcal{O},s_3) = (a^\dagger, -\frac{1}{2})\f$,
\f$(\mathcal{O},s_3) = (b, +\frac{1}{2})\f$, and
\f$(\mathcal{O},s_3) = (b^\dagger, -\frac{1}{2})\f$.

Single-particle and single-hole operators \f$\mathcal{O}\f$ with a definite electric charge \f$q\f$ obeys the eigenvalue equation
\f[
    [ \rho_x, \mathcal{O}_y ] = q \mathcal{O}_y \delta_{xy}.
\f]
Since \f$\rho = 1-2S^0\f$, we can check the commutator with \f$2S^0\f$.
I have checked that this equation is satisfied when
\f$(\mathcal{O},q) = (a, +1)\f$,
\f$(\mathcal{O},q) = (a^\dagger, -1)\f$,
\f$(\mathcal{O},q) = (b, -1)\f$, and
\f$(\mathcal{O},q) = (b^\dagger, +1)\f$.
Note that the signs are different.
Presumably there are eigenoperators for \f$S^{1,2}\f$ that are linear combinations of \f$a\f$ and \f$b\f$ with some \f$i\f$s and daggers, but I cannot be bothered to find them.


# Correlation Functions

Now we can write correlation functions
\f[
    C^{ij}_{xy}(\tau) = \frac{1}{N_t} \sum_t \left\langle S^{i}_{x,t+\tau} S^{j}_{y,t}{}^\dagger \right\rangle
\f]
and we don't need to track time separately, until we start analyzing how to actually use this correlation function.

The simplest correlation function is \f$C^{11}\f$,
\f{align}{
    \left\langle S^{1}_{x} S^{1}_{y}{}^\dagger \right\rangle
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle \left[ b_x^\dagger a_x^\dagger + a_x b_x \right] \left[ b_y^\dagger a_y^\dagger + a_y b_y \right] \right\rangle \\
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle a_x b_x a_y b_y + a_x b_x b_y^\dagger a_y^\dagger + b_x^\dagger a_x^\dagger a_y b_y + b_x^\dagger a_x^\dagger b_y^\dagger a_y^\dagger \right\rangle \\
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle a_x a_y^\dagger b_x b_y^\dagger + a_x^\dagger a_y b_x^\dagger b_y \right\rangle \\
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle a_x a_y^\dagger b_x b_y^\dagger + (\delta_{yx} - a_y a_x^\dagger)(\delta_{yx} - b_y b_x^\dagger) \right\rangle \\
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle P_{xy} H_{xy} + (\delta_{yx} - P_{yx})(\delta_{yx} - H_{yx}) \right\rangle \\
\f}
where we have taken advantage of the anticommutator rules,
the fact that we will only get a non-zero result if we have the same number of \f$a\f$s as \f$a^\dagger\f$s (and likewise for \f$b\f$),
and used the fact that the Wick contraction of \f$a_x a_y^\dagger = (M^p)^{-1}_{xy} \equiv P_{xy}\f$, defining \f$P\f$,
and similarly for holes \f$b_x b_y^\dagger = (M^h)^{-1}_{xy} \equiv H_{xy}\f$.

Computing \f$C^{22}_{xy}\f$ requires
\f{align}{
    \left\langle S^{2}_{x} S^{2}_{y}{}^\dagger \right\rangle
    &= \frac{1}{4} (-\sigma_\kappa)^{x+y} \left\langle \left[ b_x^\dagger a_x^\dagger - a_x b_x \right] \left[ a_y b_y - b_y^\dagger a_y^\dagger \right] \right\rangle \\
\f}
and when you write out all the operators in their complete glory, you find that you reproduce the non-vanishing operator content in \f$\left\langle S^{1}_{x} S^{1}_{y}{}^\dagger \right\rangle\f$,
so \f$C^{22}_{xy} = C^{11}_{xy}\f$ (the vanishing operators have the opposite sign).

We now move to \f$C^{33}_{xy}\f$, which is less trivial because of terms like \f$aa^\dagger\f$ in the individual operators,
so that there are terms with four \f$a\f$ operators, and so-called disconnected diagrams:
\f{align}{
    \left\langle S^{3}_{x} S^{3}_{y}{}^\dagger \right\rangle
    &= \frac{1}{4} \left\langle \left[ a_x a_x^\dagger + b_x b_x^\dagger -1 \right] \left[ a_y a_y^\dagger + b_y b_y^\dagger -1 \right] \right\rangle \\
    &= \frac{1}{4} \left\langle a_x a_x^\dagger a_y a_y^\dagger + a_x a_x^\dagger b_y b_y^\dagger - a_x a_x^\dagger - a_y a_y^\dagger + (a \leftrightarrow b) +1 \right\rangle \\
    &= \frac{1}{4} \left\langle a_x (\delta_{yx} - a_y a_x^\dagger ) a_y^\dagger + a_x a_x^\dagger b_y b_y^\dagger - a_x a_x^\dagger - a_y a_y^\dagger + (a \leftrightarrow b) +1 \right\rangle \\
    &= \frac{1}{4} \left\langle - a_x a_y a_x^\dagger a_y^\dagger + a_x a_x^\dagger b_y b_y^\dagger + a_x a_y^\dagger \delta_{yx} - a_x a_x^\dagger - a_y a_y^\dagger + (a \leftrightarrow b) +1 \right\rangle \\
    &= \frac{1}{4} \left\langle P_{xx}P_{yy} - P_{yx} P_{xy} + P_{xx} H_{yy} + P_{xy} - P_{xx} - P_{yy} + (P \leftrightarrow H) + 1 \right\rangle
\f}

Something left undefined in Buividovich et al. are the mixed correlators, \f$C^{12}_{xy}\f$, which requires
\f{align}{
    \left\langle S^{1}_{x} S^{2}_{y}{}^\dagger \right\rangle
    &= -\frac{i}{4}(-\sigma_\kappa)^{x+y} \left\langle \left[ b_x^\dagger a_x^\dagger + a_x b_x \right] \left[ a_y b_y - b_y^\dagger a_y^\dagger \right] \right\rangle   \\
    &= -\frac{i}{4}(-\sigma_\kappa)^{x+y} \left\langle b_x^\dagger a_x^\dagger a_y b_y - a_x b_x b_y^\dagger a_y^\dagger + a_x b_x a_y b_y - b_x^\dagger a_x^\dagger b_y^\dagger a_y^\dagger \right\rangle   \\
    &= -\frac{i}{4}(-\sigma_\kappa)^{x+y} \left\langle b_x^\dagger b_y a_x^\dagger a_y - b_x b_y^\dagger a_x a_y^\dagger \right\rangle   \\
    &= -\frac{i}{4}(-\sigma_\kappa)^{x+y} \left\langle (\delta_{yx} - b_y b_x^\dagger) (\delta_{yx} - a_y a_x^\dagger) - b_x b_y^\dagger a_x a_y^\dagger \right\rangle   \\
    &= -\frac{i}{4}(-\sigma_\kappa)^{x+y} \left\langle (\delta_{yx} - H_{yx}) (\delta_{yx} - P_{yx}) - H_{xy} P_{xy} \right\rangle   \\
\f}
and \f$C^{03}_{xy}\f$, requiring
\f{align}{
    \left\langle S^{0}_{x} S^{3}_{y}{}^\dagger \right\rangle
    &= \frac{1}{4} \left\langle \left[ a_x a_x^\dagger - b_x b_x^\dagger +1 \right] \left[ a_x a_x^\dagger + b_x b_x^\dagger -1 \right] \right\rangle \\
    &= \frac{1}{4} \left\langle P_{xx}P_{yy} - P_{xy}P_{yx} + P_{xx}H_{yy} - H_{xx}P_{yy} - H_{xx} H_{yy} + H_{xy} H_{yx} + P_{xy} - P_{xx} + P_{yy} - H_{xy} + H_{xx} + H_{yy} - 1\right\rangle
\f}
while \f$C^{21}_{xy}\f$ and \f$C^{30}_{xy}\f$ are defined analogously.
Once the all-dagger or no-dagger operators are dropped, it is easy to see that
\f{align}{
    \left\langle S^1_x S^2_y{}^\dagger \right\rangle &= - \left\langle S^2_x S^1_y{}^\dagger\right\rangle
\f}
but unfortunately no such simple relation holds between (0,3) and (3,0).

Note that 1 and 2 cannot mix with 0 or 3 because each term would not have the right constituent operator content to contract completely.


"""


from logging import getLogger

import numpy as np

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, rollTemporally
from ..h5io import createH5Group

class spinSpinCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate spin-spin correlators.
    """

    def __init__(savePath, configSlice):
        super().__init__(savePath, configSlice)

    def __call__(self, phi, action, itr):
        """!Record the spin-spin correlators."""

    def save(self, h5group):
        r"""!
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        ...

def read(h5group):
    r"""!
    \param h5group HDF5 group which contains the data of this measurement.
    """
    ...
