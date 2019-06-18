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
where \f$\sigma_\kappa\f$ is +1 on bipartite lattices and must be -1 on non-bipartite lattices, following the convention of Isle (bipartite graphs can also have \f$\sigma_\kappa=-1\f$).
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

One may also construct spin raising and lowering operators in the standard way,
\f{align}{
    S^+_x = S^1_x + i S^2_x &= (-\sigma_\kappa)^x a_x b_x \\
    S^-_x = S^1_x - i S^2_x &= (-\sigma_\kappa)^x b^\dagger_x a^\dagger_x
\f}
which obey the eigenvalue relations
\f[
    [S_x^3, S^\pm_y ] = \pm S^\pm_y \delta_{xy},
\f]
which can be shown using the single-particle and single-hole eigenvalue equations and the identity \f$[A, BC] = [A,B]C + B[A,C]\f$.

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

We can also calculate the charge-charge correlator,
\f{align}{
    \left\langle \rho_x \rho_y^\dagger \right\rangle
     = \left\langle (1-2S_x^0) (1-2S_y^0) \right\rangle
    &= \left\langle 1 - 2 S_x^0 - 2 S_y^0 + 4 S_x^0 S_y^0 \right\rangle  \\
    &= \left\langle 1 -(a_x a_x^\dagger - b_x b_x^\dagger + 1) - (a_y a_y^\dagger - b_y b_y^\dagger + 1) + (a_x a_x^\dagger - b_x b_x^\dagger + 1)(a_y a_y^\dagger - b_y b_y^\dagger + 1) \right\rangle \\
    &= \left\langle a_x a_x^\dagger a_y a_y^\dagger - a_x a_x^\dagger b_y b_y^\dagger - b_x b_x^\dagger a_y a_y^\dagger + b_x b_x^\dagger b_y b_y^\dagger \right\rangle \\
    &= \left\langle a_x (\delta_{yx} - a_y a_x^\dagger ) a_y^\dagger - a_x a_x^\dagger b_y b_y^\dagger - b_x b_x^\dagger a_y a_y^\dagger + b_x (\delta_{yx} - b_y b_x^\dagger) b_y^\dagger \right\rangle \\
    &= \left\langle P_{xx}P_{yy} - P_{xy}P_{yx} + P_{xy}\delta_{yx} - P_{xx}H_{yy} + (P \leftrightarrow H) \right\rangle
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

Note that 1 and 2 cannot mix with 0 or 3 because each term would not have the right constituent operator content to contract completely.  This is very curious, and seems kind of broken.
I think if we had a spin chemical potential that mixed \f$a\f$ and \f$b\f$ operators, such as \f$\mu S^1\f$, we would not be able to split the operators into two species, because the bilinear piece would allow one species to mix with the other while propagating,
so that the contractions, rather than considering \f$a\f$ and \f$b\f$ separately, would consider \f$c\f$ operators with more entries and the chemical potential would introduce an off-diagonal component.
Working out the details of this is a question for a different time, however.

We can also think of correlators between \f$S^+\f$ and \f$S^-\f$,
\f{align}{
    \left\langle S^+_x S^-_y \right\rangle
        &= (-\sigma_\kappa)^{x+y}\left\langle a_x b_x b^\dagger_y a^\dagger_y \right\rangle \\
        &= (-\sigma_\kappa)^{x+y}\left\langle P_{xy} H_{xy}\right\rangle    \\
    \left\langle S^-_x S^+_y \right\rangle
        &= (-\sigma_\kappa)^{x+y}\left\langle b^\dagger_x a^\dagger_x a_y b_y  \right\rangle \\
        &= (-\sigma_\kappa)^{x+y}\left\langle (\delta_{yx}- b_y b^\dagger_x)(\delta_{yx} - a_y a^\dagger_x)  \right\rangle \\
        &= (-\sigma_\kappa)^{x+y}\left\langle (\delta_{yx}- H_{yx})(\delta_{yx} - P_{yx})  \right\rangle \\
\f}
# TODO: These have a name; I should find it.
At half filling on a bipartite lattice, the cost to create or destroy a spin from the vacuum should be equal and the correlators should match, in the limit of large statistics.


# Order Parameters

In Meng and Wessel ([PRB 78, 224416 (2008)](http://dx.doi.org/10.1103/PhysRevB.78.224416)), they define the equal-time structure factor
\f[
    S_{AF} = \frac{1}{V}\sum_{xy} \epsilon_x \epsilon_y \left\langle S^3_x S^3_y \right\rangle
\f]
where \f$\epsilon_x=\pm1\f$ depending on the parity of the site (they study a square lattice), and each sum is over space, \f$V\f$ is the spatial volume, and \f$S_{AF}\f$ is extensive so that \f$S_{AF}/V\f$ scales like a constant towards the infinite volume limit.
When this order parameter is nonzero, the system is said to exhibit *antiferromagnetic* or *Néel* order.

This definition of \f$S_{AF}\f$ is what we get when we take the equal-time limit of our correlator,
\f[
    S_{AF} = U_{\Lambda x}C^{33}_{xy}(0) U^\dagger_{y \Lambda'}
\f]
where \f$U\f$ and its dagger are unitary matrices that diagonalize the hopping matrix
and pick the irreps \f$\Lambda\f$ and \f$\Lambda'\f$ to be those that correspond to sign-alternating but otherwise constant normalized wavefunctions (the normalizations each provide a \f$1/\sqrt{V}\f$).
Put another way, we can fourier transform in both spatial coordinates and take the highest frequency mode for each (which is the zero momentum doubler).
Such an alternating irrep is always there on a bipartite lattice.

What's nice is that using our correlator formulation we get access to a lot more order parameters (picking different irreps) at once,
and that our definition generalizes to non-bipartite lattices immediately (diagonalizing the hopping matrix is what is fundamental about the fourier transform).

We can construct the CDW order parameter similarly, substituting the charge density operator for \f$S^3\f$.
Similarly, we can build whole correlators and the order parameters are the zero-time correlators on the diagonal in the irrep basis.

In Lang's 2010 thesis he defines the antiferromagnetic susceptibility \f$\chi_{AF}\f$ [see eq. (1.16)] which we can extract from
\f[
    \chi_{AF} = - U_{\Lambda x} C^{+-}_{xy}(0) U^\dagger_{y \Lambda'}
\f]
where the irreps are the sign-alternating irreps.
I have reproduced his exact result [from the first line in (1.16)] on two spatial sites for a variety of \f$U\f$s and \f$\beta\f$s.
In (1.16) carefully take note of the signs, and of the fact that the sums are over \f$\ell\neq\ell'\f$.

In principle we can construct a four-by-four matrix of correlators (3 spins + the charge) for diagonalization.
In practice this may not be cost effective / sensible, especially if the different operators are separately conserved.
If they mix into one another, one should indeed perform this diagonalization.

There is another subtlety in terms of projection: what should we do if the graph is not orientable?
In other words, if we can parallel transport a spin-z-up particle around and wind up with a spin-z-down particle, the story told above doesn't make sense.
In that case we must have omitted something about the spin connection, which is currently beyond the scope of our code.

Buividovich et al. study a slightly different order parameter, the squared sum (see their equation 14).
In their discussion they say that they would like the order parameter to be the difference of spins between the sublattices, presumably
\f[
    \left\langle \left(\sum_{x \in A} - \sum_{x\in B}\right) S^3_x \right\rangle
\f]
but without introducing a bias and extrapolating the bias to zero (as in spontaneous symmetry breaking), while going to the thermodynamic limit, one finds the order parameter vanishes.
Instead, they study "the square of the total spin per sublattice",
\f[
    \left\langle \left(\sum_{x \in A} S^3_x \right)^2 + \left(\sum_{x \in B} S^3_x \right)^2 \right\rangle,
\f]
which is two of the three terms you would get if you squared the straight A-B difference.
The reason for the omission of the cross-terms is apparently the clarity of the signal; the physical justification is not provided.
It is not clear how they distinguish this from an indication of the square of the magnetization per sublattice,
\f[
    \left\langle \left(\sum_{x \in A} + \sum_{x\in B}\right) S^3_x \right\rangle \longrightarrow \left\langle \left(\sum_{x \in A} S^3_x \right)^2 + \left(\sum_{x \in B} S^3_x \right)^2 \right\rangle
\f]
(note the plus sign between the sums on the left).
If you wish to construct their order parameter, you need matrices that are projectors to just A or just B, not to irreps, and then to sum.
That is,
\f[
    \left\langle \left(\sum_{x \in A} S^3_x \right)^2 + \left(\sum_{x \in B} S^3_x \right)^2 \right\rangle
    =
    U_{\Lambda x'}\left(A_{x'x} C^{33}_{xy} A_{yy'} + B_{x'x} C^{33}_{xy} B_{yy'}\right) U^\dagger_{y' \Lambda'},
\f]
where the irreps are both the constant-everywhere irrep.  Note the lack of \f$ACB\f$ and \f$BCA\f$ terms.

They also study the squared magnetization, (16),
\f{align}{
    \left\langle (m^i)^2 \right\rangle
    &= \left\langle \left(\frac{1}{V} \sum_x S^i_x \right)^2 \right\rangle \\
    &= \frac{1}{V} U_{\Lambda x}C^{ii}_{xy}(0) U^\dagger_{y \Lambda'}
\f}
which can be expressed as an irrep-projected equal-time spin-spin correlator, just like the \f$S_{AF}\f$ of Meng and Wessel,
but rather than taking the doubler, simply take the 0-momentum projection on each index.

### Summary

Because we do not assume a bipartite graph, we implement Meng and Wessel's \f$S_{AF}\f$ and Buividovich et al.'s \f$\left\langle (m^i)^2 \right\rangle\f$, and associated order parameters constructed in good irreps.

"""


from logging import getLogger

import numpy as np

import isle
from .measurement import Measurement
from ..util import spaceToSpacetime, temporalRoller
from ..h5io import createH5Group

class SpinSpinCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate spin-spin correlators.
    """

    def __init__(self, particleAllToAll, holeAllToAll, savePath, configSlice=(None, None, None), projector=None, sigmaKappa=-1):
        super().__init__(savePath, configSlice)

        # The correlation functions encoded here are between bilinear operators.
        # Since the individual constituents are fermionic, the bilinear is bosonic.
        self.fermionic = False


        self.sigmaKappa = sigmaKappa

        self.particle=particleAllToAll
        self.hole=holeAllToAll

        self.rho_rho = []
        self.S1_S1 = []
        self.S3_S3 = []
        self.Splus_Sminus = []
        self.Sminus_Splus = []

        if projector is None:
            # TODO: warn when diagonalizing the hopping matrix.
            _, self.irreps = np.linalg.eigh(isle.Matrix(allToAll.hfm.kappaTilde()))
            self.irreps = self.irreps.T
        else:
            self.irreps = projector.T

        self.irreps = np.matrix(self.irreps)


    def __call__(self, phi, action, itr):
        """!Record the spin-spin correlators."""

        P = self.particle(phi, action, itr)
        H = self.hole(phi, action, itr)

        nx = P.shape[0]
        nt = P.shape[1]

        d = np.eye(nx*nt).reshape(*P.shape) # A Kronecker delta

        # TODO: store some einsum paths

        PxxPyy = np.einsum("xfxf,yiyi->xfyi", P, P, optimize="optimal")
        PxxHyy = np.einsum("xfxf,yiyi->xfyi", P, H, optimize="optimal")
        dxxdyy = np.einsum("xfxf,yiyi->xfyi", d, d, optimize="optimal")
        HxxPyy = np.einsum("xfxf,yiyi->xfyi", H, P, optimize="optimal")
        HxxHyy = np.einsum("xfxf,yiyi->xfyi", H, H, optimize="optimal")

        PxyPyx = np.einsum("xfyi,yixf->xfyi", P, P, optimize="optimal")
        Pxydyx = np.einsum("xfyi,yixf->xfyi", P, d, optimize="optimal")
        Hxydyx = np.einsum("xfyi,yixf->xfyi", H, d, optimize="optimal")
        HxyHyx = np.einsum("xfyi,yixf->xfyi", H, H, optimize="optimal")

        PyxHyx = np.einsum("yixf,yixf->xfyi", P, H, optimize="optimal")
        Pyxdyx = np.einsum("yixf,yixf->xfyi", P, d, optimize="optimal")
        Hyxdyx = np.einsum("yixf,yixf->xfyi", H, d, optimize="optimal")
        dyxdyx = np.einsum("yixf,yixf->xfyi", d, d, optimize="optimal")

        PxyHxy = np.einsum("xfyi,xfyi->xfyi", P, H, optimize="optimal")
        Pxydyx = np.einsum("xfyi,yixf->xfyi", P, d, optimize="optimal")
        Hxydyx = np.einsum("xfyi,yixf->xfyi", H, d, optimize="optimal")

        Pxxdyy = np.einsum("xfxf,yiyi->xfyi", P, d)
        dxxPyy = np.einsum("xfxf,yiyi->xfyi", d, P)
        Hxxdyy = np.einsum("xfxf,yiyi->xfyi", H, d)
        dxxHyy = np.einsum("xfxf,yiyi->xfyi", d, H)

        rho_rho = (PxxPyy + HxxHyy) - (PxyPyx + HxyHyx) + (Pxydyx + Hxydyx) - (PxxHyy+HxxPyy)
        S1_S1 = 0.25*(PxyHxy+ dyxdyx - Pyxdyx - Hyxdyx)
        S3_S3 = 0.25*((PxxPyy + HxxHyy) - (PxyPyx + HxyHyx) + (PxxHyy + HxxPyy) + (Pxydyx+Hxydyx) - (Pxxdyy+Hxxdyy) - (dxxPyy+dxxHyy) + dxxdyy)
        Splus_Sminus = PxyHxy
        Sminus_Splus = (dyxdyx - Pyxdyx - Hyxdyx + PyxHyx)

        self._roll = np.array([temporalRoller(nt, -t, fermionic=self.fermionic) for t in range(nt)])

        # Project to irreps:
        # rho_rho = np.einsum("idf,bx,xfyi,ya->bad", self._roll, self.irreps, rhoxrhoy, self.irreps.H) / nt

        # Just leave with spatial indices:
        for observable, storage in zip(
                (rho_rho,       S1_S1,      S3_S3,      Splus_Sminus,       Sminus_Splus),
                (self.rho_rho,  self.S1_S1, self.S3_S3, self.Splus_Sminus,  self.Sminus_Splus)
                ):
            time_averaged = np.einsum("idf,xfyi->xyd", self._roll, observable) / nt
            storage.append(time_averaged)


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
