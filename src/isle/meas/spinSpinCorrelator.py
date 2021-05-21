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
where \f$\sigma_\kappa\f$ is +1 on bipartite lattices and must be –1 on non-bipartite lattices, following the convention of Isle (bipartite graphs can also have \f$\sigma_\kappa=–1\f$).
In Buividovich, Smith, Ulybyshev, and von Smekal [1807.07025](https://arxiv.org/abs/1807.07025) [PRB 98 235129](https://doi.org/10.1103/PhysRevB.98.235129) they also define, just after (1),
the electric charge operator
\f[
    \rho_x = c_{x,\uparrow}^\dagger c_{x,\uparrow} + c_{x,\downarrow}^\dagger c_{x,\downarrow} – 1
\f]
which can be rewritten as \f$ \rho_x = 1–2 S^0_x\f$, where the \f$0^{th}\f$ Pauli matrix is the \f$2\times2\f$ identity matrix.  The \f$S\f$ operators are Hermitian.

Rewriting those operators into the Isle basis,
\f{align}{
    \rho_x &= n^a_x–n^b_x = a_x^\dagger a_x – b_x^\dagger b_x                                       \\
    S^0_x &= \frac{1}{2} \left[ a_x a_x^\dagger – b_x b_x^\dagger +1 \right]                    \\
    S^1_x &= \frac{1}{2} (–\sigma_\kappa)^x \left[ b_x^\dagger a_x^\dagger + a_x b_x \right]    \\
    S^2_x &= \frac{i}{2} (–\sigma_\kappa)^x \left[ b_x^\dagger a_x^\dagger – a_x b_x \right]    \\
    S^3_x &= \frac{1}{2} \left[ a_x a_x^\dagger + b_x b_x^\dagger –1 \right]                    \\
    n_x   &= a_x^\dagger a_x + b_x^\dagger b_x
\f}
where the \f$\sigma_\kappa\f$ squares away when two \f$b\f$ operators are multiplied and we introduce the total-number operator \f$n\f$ with no superscript.
Note the operator ordering in \f$\rho\f$ is opposite from \f$S^0\f$, and that particles are positively charged.

The three spin operators obey the commutation relation
\f[
    \left[ S_x^i, S_y^j \right] = i \delta_{xy} \epsilon^{ijk} S_x^k
\f]
which may be checked explicitly by writing out the operators in all their glory and using the anticommutation properties of \f$a\f$ and \f$b\f$.
By a similar exercise one may show
\f[
    \left[ S_x^0, S_y^j \right] = 0.
\f]

Single-particle and single-hole operators \f$\mathcal{O}\f$ with a definite third component of spin \f$s_3\f$ obeys the eigenvalue equation
\f[
    [ S_x^3, \mathcal{O}_y ] = s_3 \mathcal{O}_y \delta_{xy}.
\f]
This equation is satisfied when
\f$(\mathcal{O},s_3) = (a, +\frac{1}{2})\f$,
\f$(\mathcal{O},s_3) = (a^\dagger, –\frac{1}{2})\f$,
\f$(\mathcal{O},s_3) = (b, +\frac{1}{2})\f$, and
\f$(\mathcal{O},s_3) = (b^\dagger, –\frac{1}{2})\f$.

Single-particle and single-hole operators \f$\mathcal{O}\f$ with a definite electric charge \f$q\f$ obeys the eigenvalue equation
\f[
    [ \rho_x, \mathcal{O}_y ] = q \mathcal{O}_y \delta_{xy}.
\f]
Since \f$\rho = 1–2S^0\f$, we can check the commutator with \f$2S^0\f$.
This equation is satisfied when
\f$(\mathcal{O},q) = (a, –1)\f$,
\f$(\mathcal{O},q) = (a^\dagger, +1)\f$,
\f$(\mathcal{O},q) = (b, +1)\f$, and
\f$(\mathcal{O},q) = (b^\dagger, –1)\f$.
Note that the signs are distributed differently from the \f$S^3\f$ eigenequation.
There are also eigenoperators of \f$S^{1}\f$,
\f$(\mathcal{O},q) = (a+b^\dagger, +\frac{1}{2})\f$,
\f$(\mathcal{O},q) = (a–b^\dagger, –\frac{1}{2})\f$,
\f$(\mathcal{O},q) = (a^\dagger+b, –\frac{1}{2})\f$, and
\f$(\mathcal{O},q) = (a^\dagger–b, +\frac{1}{2})\f$,
and eigenoperators of \f$S^{2}\f$,
\f$(\mathcal{O},q) = (a+ib^\dagger, +\frac{1}{2})\f$,
\f$(\mathcal{O},q) = (a–ib^\dagger, –\frac{1}{2})\f$,
\f$(\mathcal{O},q) = (a^\dagger+ib, +\frac{1}{2})\f$, and
\f$(\mathcal{O},q) = (a^\dagger–ib, –\frac{1}{2})\f$,
but we list these only for completeness' sake.

One may also construct spin raising and lowering operators in the standard way,
\f{align}{
    S^+_x = S^1_x + i S^2_x &= (–\sigma_\kappa)^x a_x b_x \\
    S^–_x = S^1_x – i S^2_x &= (–\sigma_\kappa)^x b^\dagger_x a^\dagger_x
\f}
which obey the eigenvalue relations
\f[
    [S_x^3, S^\pm_y ] = \pm S^\pm_y \delta_{xy},
\f]
which can be shown using the single-particle and single-hole eigenvalue equations and the identity \f$[A, BC] = [A,B]C + B[A,C]\f$.

The construction of the number operators can be done in a similar fashion,
\f{align}{
    \delta_{xx} – n^p_x &= S^0_x + S^3_x = a_x a^\dagger_x = \delta_{xx} – a^\dagger_x a_x
    \\
    n^h_x &= S^0_x – S^3_x = –b_x b^\dagger_x + \delta_{xx} = –\delta_{xx} + b^\dagger_x b_x + \delta_{xx} = b^\dagger_x b_x.
\f}
We can of course drop the constant term in the first definition.
We use the shorthand
\f{align}{
    n_x &= n^p_x + n^h_x = \delta_{xx}-2 S^3_x
    \\
    \text{as in }
    \rho_x &= n^p_x - n^h_x = \delta_{xx}-2 S^0_x
\f}

## Charge ≠ 0

All the above bilinears have charge zero, since \f$[\rho,\cdot]=0\f$.
However, there are two bilinears that are missing from the above operators, and they have nonzero charge.
(To get a charge of 1, one must have an odd number of particle and hole operators.)
The two operators are \f$a^\dagger b\f$ and \f$b^\dagger a\f$ with charge ±2,
\f{align}{
    [\rho_x, a_y^\dagger b_y] = &+ 2 \delta_{xy} a_y^\dagger b_y \\
    [\rho_x, b_y^\dagger a_y] = &– 2 \delta_{xy} b_y^\dagger a_y.
\f}
which we'll indicate with double symbols, \f${}^+_+\f$ and \f${}^-_-\f$, respectively.
As they are charged, these are best interpreted as creation and annihilation operators.
It is easy to check that these operators have \f$[S^z,\cdot]=0\f$.

# Correlation Functions

Now we can write correlation functions
\f[
    C^{ij}_{xy}(\tau) = \frac{1}{N_t} \sum_t \left\langle S^{i}_{x,t+\tau} S^{j}_{y,t} \right\rangle
\f]
and we don't need to track time separately, until we start analyzing how to actually use this correlation function.

The simplest correlation function is \f$C^{+–}_{xy}\f$,
\f{align}{
    C^{+–}_{xy}     = \left\langle S^+_x S^–_y \right\rangle
                    = (–\sigma_\kappa)^{x+y}\left\langle a_x b_x b_y^\dagger a_y^\dagger \right\rangle
                &   = (-\sigma_\kappa)^{x+y}\left\langle P_{xy} H_{xy} \right\rangle
                \\
    C^{–+}_{xy}     = \left\langle S^–_x S^+_y \right\rangle
                    = (–\sigma_\kappa)^{x+y}\left\langle a_x^\dagger b_x^\dagger b_y a_y \right\rangle
                &   = (–\sigma_\kappa)^{x+y}\left\langle (\delta_{yx} – b_y b_x^\dagger)(\delta_{yx} – a_y a_x^\dagger) \right\rangle
                \\
                &   = (–\sigma_\kappa)^{x+y}\left\langle (\delta_{yx} – H_{yx})(\delta_{yx} – P_{yx}) \right\rangle
\f}
and \f$C^{–+}\f$, where we have used anticommutator rules to push all the daggered operators to the right to apply the Wick contraction rule,
contracting particles into \f$P\f$ propagators and holes into \f$H\f$ propagators.

Also simple (to write) are correlations between number operators,
\f{align}{
    C^{ph}_{xy}     = \left\langle N^p_x N^h_y \right\rangle
                    = \left\langle (\delta_{xx} - a_x a_x^\dagger) (\delta_{yy} - b_y b_y^\dagger) \right\rangle
                &   = \left\langle (\delta_{xx} - P_{xx}) (\delta_{yy} - H_{yy}) \right\rangle
                \\
    C^{pp}_{xy}     = \left\langle N^p_x N^p_y \right\rangle
                    = \left\langle (\delta_{xx} - a_x a_x^\dagger) (\delta_{yy} - a_y a_y^\dagger) \right\rangle
                &   = \left\langle \delta_{xx}\delta_{yy} - \delta_{xx} a_y a_y^\dagger - a_x a_x^\dagger \delta_{yy} + a_x a_x^\dagger a_y a_y^\dagger \right\rangle
                \\
                &   = \left\langle \delta_{xx}\delta_{yy} - \delta_{xx} a_y a_y^\dagger - a_x a_x^\dagger \delta_{yy} + a_x (\delta_{yx} - a_y a_x^\dagger) a_y^\dagger \right\rangle
                \\
                &   = \left\langle \delta_{xx}\delta_{yy} - \delta_{xx} P_{yy} - P_{xx} \delta_{yy} + \delta_{yx} P_{xy} – P_{xy} P_{yx} + P_{xx} P_{yy}  \right\rangle
\f}
and what we get by exchanging particles with holes,
\f{align}{
    C^{hp}_{xy}     = \left\langle N^h_x N^p_y \right\rangle
                &   = \left\langle (\delta_{xx} - H_{xx}) (\delta_{yy} - P_{yy}) \right\rangle
                \\
    C^{hh}_{xy}     = \left\langle N^p_x N^p_y \right\rangle
                &   = \left\langle \delta_{xx}\delta_{yy} - \delta_{xx} H_{yy} - H_{xx} \delta_{yy} + \delta_{yx} H_{xy} – H_{xy} H_{yx} + H_{xx} H_{yy}  \right\rangle
\f}
though these are more complicated to _compute_ because they have so-called disconnected diagrams.  However, with all-to-all propagators, it's all the same.

We can build correlations between the spin operators themselves.
For example,
\f{align}{
    C^{11} = \left\langle S^{1}_{x} S^{1}_{y}{} \right\rangle
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle \left[ b_x^\dagger a_x^\dagger + a_x b_x \right] \left[ b_y^\dagger a_y^\dagger + a_y b_y \right] \right\rangle \\
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle a_x b_x a_y b_y + a_x b_x b_y^\dagger a_y^\dagger + b_x^\dagger a_x^\dagger a_y b_y + b_x^\dagger a_x^\dagger b_y^\dagger a_y^\dagger \right\rangle \\
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle a_x a_y^\dagger b_x b_y^\dagger + a_x^\dagger a_y b_x^\dagger b_y \right\rangle \\
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle a_x a_y^\dagger b_x b_y^\dagger + (\delta_{yx} – a_y a_x^\dagger)(\delta_{yx} – b_y b_x^\dagger) \right\rangle \\
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle P_{xy} H_{xy} + (\delta_{yx} – P_{yx})(\delta_{yx} – H_{yx}) \right\rangle \\
\f}
Computing \f$C^{22}_{xy}\f$ requires
\f{align}{
    \left\langle S^{2}_{x} S^{2}_{y} \right\rangle
    &= \frac{1}{4} (–\sigma_\kappa)^{x+y} \left\langle \left[ b_x^\dagger a_x^\dagger – a_x b_x \right] \left[ a_y b_y – b_y^\dagger a_y^\dagger \right] \right\rangle \\
\f}
and when you write out all the operators in their complete glory, you find that you reproduce the non-vanishing operator content in \f$\left\langle S^{1}_{x} S^{1}_{y} \right\rangle\f$,
so \f$C^{22}_{xy} = C^{11}_{xy}\f$ (the vanishing operators have the opposite sign) configuration-by-configuration.

In fact, using the definition of the spin-raising and -lowering operators, one finds
\f[
    C^{11}_{xy} + C^{22}_{xy}
    =
    \frac{1}{2}\left(C^{+–}_{xy}+C^{–+}_{xy}\right)
\f]
so that
\f[
    C^{11}_{xy} = C^{22}_{xy}
    =
    \frac{1}{4}\left(C^{+–}_{xy}+C^{–+}_{xy}\right),
\f]
which you can directly verify with the explict contractions above.
Similarly, knowing that the four-dagger and no-dagger terms vanish, it is easy to show
\f[
         \left\langle S^{1}_{x}S^{2}_y\right\rangle
    =
        –\left\langle S^{2}_{x}S^{1}_y\right\rangle
\f]
(again, because the four- and no-dagger terms vanish) so that
\f[
    C^{12}_{xy} = – C^{21}_{xy}
    =
    \frac{i}{4}\left(C^{+–}_{xy}–C^{–+}_{xy}\right)
\f]
because
\f[
    C^{12}_{xy}-C^{21}_{xy} = \frac{i}{2} \left(C^{+–}_{xy} – C^{–+}_{xy}\right).
\f]
These mixed-spin correlators are not defined in Buividovich et al.

The other two spins \f$S^0\f$ and \f$S^3\f$ do not enjoy such simplifications because
each term in those operators can be contracted with itself, so there are no zero- or four-dagger
operators which may be dropped from the Wick contractions.
We are stuck computing four correlators and taking advantage of the one-point functions,
\f{align}{
        C^{00}_{xy} &= \frac{1}{4}\left(
        C^{pp}_{xy} + C^{hh}_{xy} - C^{ph}_{xy} - C^{hp}_{xy}
        +
        \left\langle 1 -N^p_x - N^p_y+ N^h_x +N^h_y \right\rangle
        \right)
        \\
    C^{03}_{xy} &= \frac{1}{4}\left(
        C^{pp}_{xy} - C^{hh}_{xy} + C^{ph}_{xy} - C^{hp}_{xy}
        +
        \left\langle 1 - N^p_x - N^p_y + N^h_x - N^h_y\right\rangle
        \right)
        \\
    C^{30}_{xy} &= \frac{1}{4}\left(
        C^{pp}_{xy} - C^{hh}_{xy} - C^{ph}_{xy} + C^{hp}_{xy}
        +
        \left\langle 1 - N^p_x - N^p_y - N^h_x + N^h_y \right\rangle
        \right)
        \\
    C^{33}_{xy} &= \frac{1}{4}\left(
        C^{pp}_{xy} + C^{hh}_{xy} + C^{ph}_{xy} + C^{hp}_{xy}
        +
        \left\langle 1 - N^p_x - N^p_y - N^h_x - N^h_y \right\rangle
        \right)
        \\
    \text{and we define }
    C^{\rho\rho}_{xy} &= C^{pp}_{xy} + C^{hh}_{xy} - C^{ph}_{xy} - C^{hp}_{xy},
        \\
    C^{\rho n}_{xy}   &= C^{pp}_{xy} - C^{hh}_{xy} + C^{ph}_{xy} - C^{hp}_{xy},
        \\
    C^{n \rho}_{xy}   &= C^{pp}_{xy} - C^{hh}_{xy} - C^{ph}_{xy} + C^{hp}_{xy},
        \\
    \text{and }
    C^{nn}_{xy} &= C^{pp}_{xy} + C^{hh}_{xy} + C^{ph}_{xy} + C^{hp}_{xy}
\f}
so that the correlator between two charge density operators \f$C^{\rho\rho}_{xy}=\left\langle\rho_x\rho_y\right\rangle\f$ and
the correlator between two total-number operators \f$C^{nn}_{xy} = \left\langle  n_x n_y\right\rangle\f$ where \f$n\f$ counts both particles and holes.
(In the first four above relations that require one-point functions 1 indicates a matrix full of ones---not an identity matrix in x and y.  Similarly \f$N_x\f$ changes in x but is constant in y, and vice-versa for \f$N_y\f$.)

Note that 1 and 2 cannot mix with 0 or 3 because each term would not have the right constituent operator content to contract completely.  This is very curious.
I think if we had a spin chemical potential that mixed \f$a\f$ and \f$b\f$ operators, such as \f$\mu S^1\f$, we would not be able to split the operators into two species, because the bilinear piece would allow one species to mix with the other while propagating,
so that the contractions, rather than considering \f$a\f$ and \f$b\f$ separately, would consider \f$c\f$ operators with more entries and the chemical potential would introduce an off-diagonal component.
Working out the details of this is a question for a different time, however.

The charge +2 and –2 operators cannot be contracted with any of the other bilinears, by conservation of charge.
However, they can be contracted with each other.
\f{align}{
    C^{{}^+_+{}^-_-} = \left\langle a_x^\dagger b_x b_y^\dagger a_y \right\rangle
                    &= \left\langle a_x^\dagger a_y b_x b_y^\dagger \right\rangle                   \\
                    &= \left\langle (\delta_{yx} - a_y a_x^\dagger) b_x b_y^\dagger \right\rangle   \\
                    &= \left\langle (\delta_{yx} - P_{yx}) H_{xy} \right\rangle                         \\
                    \\
    C^{{}^-_-{}^+_+} = \left\langle b_x^\dagger a_x a_y^\dagger b_y\right\rangle
                    &= \left\langle b_x^\dagger b_y a_x a_y^\dagger \right\rangle                   \\
                    &= \left\langle (\delta_{yx} - b_y b_x^\dagger) a_x a_y^\dagger \right\rangle   \\
                    &= \left\langle (\delta_{yx} - H_{yx}) P_{xy} \right\rangle;                        \\
\f}
recall that the double-plus operator and double-minus operator are Hermitian conjugates.


# Conserved Quantities

When the Hamiltonian takes a Hubbard-Coulomb-like form,
\f[
    H = \sum_{xy}–\left(a_x^\dagger K_{xy} a_y + \sigma_\kappa b_x^\dagger K_{xy} b_y\right) + \frac{1}{2} \sum_{xy} \rho_x V_{xy} \rho_y
\f]
some combinations of bilinears may be conserved.  For example, we can calculate the commutator
\f{align}{
    [H, \rho_z]
        &= \left[–\sum_{xy} a_x^\dagger K_{xy} a_y + \sigma_\kappa b_x^\dagger K_{xy} b_y, \rho_z\right]   \\
        &= \left(\sum_x – a_x^\dagger K_{xz} a_z + \sum_y a_z^\dagger K_{zy} a_y\right) – \left(\sigma_\kappa (a \leftrightarrow b)\right) \\
\f}
where we immediately dropped the interaction term since the charge operator commutes with itself,
and the relative minus sign between the particle and hole arises because particles and holes have opposite charge.
If we sum \f$z\f$ over all space the two terms cancel, so that the total charge is conserved.
One similarly finds the total spins conserved,
\f[
    \left[H, \sum_z S^i_z \right] = 0
\f]
for \f$i\in\{1,2,3\}\f$.
When the operators are conserved, their correlation functions are constant,
\f{align}{
    C^{\rho\rho}_{++}(\tau)
        &= \frac{1}{\mathcal{Z}}\textrm{tr}\left[ \rho_+(\tau) \rho_+(0) e^{–\beta H}\right] \\
        &= \frac{1}{\mathcal{Z}}\textrm{tr}\left[ e^{+H \tau }\rho_+(0) e^{–H\tau} \rho_+(0) e^{–\beta H}\right] \\
        &= \frac{1}{\mathcal{Z}}\textrm{tr}\left[ \rho_+(0) e^{–H\tau} \rho_+(0) e^{–(\beta–\tau) H}\right] \\
        &= \frac{1}{\mathcal{Z}}\textrm{tr}\left[ \rho_+(0) \rho_+(0) e^{–\beta H}\right]
\f}
where we used the fact that \f$\rho_+\f$ commutes with the Hamiltonian in the last step.
We can turn this relation on its head and get an estimate for the equal-time correlator \f$\left\langle\rho_+\rho_+\right\rangle\f$ by averaging \f$C^{\rho\rho}_{++}(\tau)\f$ over the temporal separation \f$\tau\f$,
\f[
    \left\langle \rho_+ \rho_+ \right\rangle = \frac{1}{N_t} \sum_\tau C^{\rho\rho}_{++}(\tau)
\f]
This same observation holds for the total spin operators \f$S^i_+\f$ (and therefore also for \f$S^\pm_+\f$), with the Hamiltonian shown above.
On small test examples one observes numerically that the correlators are flat with the exponential discretization and seem linear with time in the diagonal discretization; averaging properly still yields good values.


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
    \chi_{AF} = – U_{\Lambda x} C^{+–}_{xy}(0) U^\dagger_{y \Lambda'}
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
    \left\langle \left(\sum_{x \in A} – \sum_{x\in B}\right) S^3_x \right\rangle
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
The CDW parameter \f$q^2\f$ in Buividovich et al. is constructed similarly but with the charge operator.

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
import h5py as h5
from pentinsula.h5utils import open_or_pass_file

from .measurement import Measurement, BufferSpec
from ..util import temporalRoller, signAlternator


class SpinSpinCorrelator(Measurement):
    r"""!
    \ingroup meas
    Tabulate spin-spin correlators.
    """

    ## Set of names of all possible elementary spin-spin correlators.
    CORRELATOR_NAMES = {"np_np", "nh_np", "np_nh", "nh_nh",
                        "Splus_Sminus", "Sminus_Splus",
                        '++_--', '--_++'}

    ## Set of names of derived correlators that can be constructed from spin-spin correlators.
    DERIVED_CORRELATOR_NAMES_SPIN_ONLY = {"S1_S1", "S1_S2", "rho_rho", "rho_n", "n_rho", "n_n"}

    ## Set of names of derived correlators that need one-point correlators to be constructed.
    DERIVED_CORRELATOR_NAMES_ONE_POINT = {"S0_S0", "S0_S3", "S3_S0", "S3_S3"}

    ## Set of names of all derived correlators
    DERIVED_CORRELATOR_NAMES = {*DERIVED_CORRELATOR_NAMES_SPIN_ONLY,
                                *DERIVED_CORRELATOR_NAMES_ONE_POINT}

    def __init__(self, particleAllToAll, holeAllToAll, lattice, savePath, configSlice=(None, None, None),
                 transform=None, sigmaKappa=-1, correlators=CORRELATOR_NAMES):
        r"""!
        \param particleAllToAll propagator.AllToAll for particles.
        \param holesAllToAll propagator.AllToAll for holes.
        \param savePath Path in the output file where results are saved.
        \param configSlice `slice` indicating which configurations to measure on.
        \param transform Transformation matrix applied to correlators in position space.
        \param sigmaKappa \f$\sigma_\kappa\f$ from the fermion action.
        \param correlators Iterable of names of correlators to compute.
                           Defaults to SpinSpinCorrelator.CORRELATOR_NAMES.
        """

        if correlators is None:
            correlators = self.CORRELATOR_NAMES
        _checkCorrNames(correlators, self.CORRELATOR_NAMES)

        super().__init__(savePath,
                         tuple(BufferSpec(name, (lattice.nx(), lattice.nx(), lattice.nt()),
                                          complex, name)
                               for name in correlators),
                         configSlice)

        # The correlation functions encoded here are between bilinear operators.
        # Since the individual constituents are fermionic, the bilinear is bosonic.
        self.fermionic = False

        self.sigmaKappa = sigmaKappa

        self.particle = particleAllToAll
        self.hole = holeAllToAll
        self.transform = transform
        self.correlators = correlators

        self._einsum_paths = {}
        self._cache = {}

    def __call__(self, stage, itr):
        """!Record the spin-spin correlators."""

        P = self.particle(stage, itr)
        H = self.hole(stage, itr)

        nx = P.shape[0]
        nt = P.shape[1]

        # This puts in all needed factors of (-sigmaKappa)^{x+y}!
        # For operators which DON'T need it, it does nothing---the reason
        # those operators don't need it is that the sign squares away.
        if self.sigmaKappa == +1:
            # This could be done without a test on sigmaKappa, since signAlternator
            # returns the identity matrix if sigmaKappa is -1, but then we'd be doing
            # 1-matrix-1 multiplication for no reason, and it's not cheap.
            Sigma = signAlternator(nx, self.sigmaKappa)
            P = np.einsum('ax,xfyi,yb->afbi', Sigma, P, Sigma, optimize="optimal")
            H = np.einsum('ax,xfyi,yb->afbi', Sigma, H, Sigma, optimize="optimal")

        d = np.eye(nx*nt).reshape(*P.shape) # A Kronecker delta
        roll = np.array([temporalRoller(nt, -t, fermionic=self.fermionic) for t in range(nt)])

        log = getLogger(__name__)

        # TODO: store some einsum paths

        for name in self.correlators:
            ataCorrelator = self._allToAllCorrelator(name, P, H, d)

            # It is major savings to avoid two matrix-matrix multiplies, so it is
            # worthwhile to test for a transform and only add those multiplies in if needed.
            if self.transform is not None:
                if "idf,bx,xfyi,ya->bad" not in self._einsum_paths:
                    self._einsum_paths["idf,bx,xfyi,ya->bad"], _ = np.einsum_path("idf,bx,xfyi,ya->bad", roll, self.transform.T.conj(),
                                                                                  ataCorrelator, self.transform, optimize="optimal")
                    log.info("Optimized Einsum path for time averaging and transform application.")

                res = self.nextItem(name)
                np.einsum("idf,bx,xfyi,ya->bad",
                          roll,
                          self.transform.T.conj(),
                          ataCorrelator,
                          self.transform,
                          out=res,
                          optimize=self._einsum_paths["idf,bx,xfyi,ya->bad"])
                res /= nt
            else:
                if "idf,xfyi->xyd" not in self._einsum_paths:
                    self._einsum_paths["idf,xfyi->xyd"], _ = np.einsum_path("idf,xfyi->xyd", roll, ataCorrelator, optimize="optimal")
                    log.info("Optimized Einsum path for time averaging in position space.")

                res = self.nextItem(name)
                np.einsum("idf,xfyi->xyd", roll, ataCorrelator,
                          optimize=self._einsum_paths["idf,xfyi->xyd"], out=res)
                res /= nt

        # Any additional correlators can be derived by identities explained above.
        # They can be computed by SpinSpinCorrelator.computeDerivedCorrelators().

        self._cache.clear()

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        """!
        Override in order to save 'transform'.
        """
        res = super().setup(memoryAllowance, expectedNConfigs, file, maxBufferSize)
        with open_or_pass_file(file, None, "a") as h5f:
            if self.transform is None:
                h5f[self.savePath]["transform"] = h5.Empty(dtype="complex")
            else:
                h5f[self.savePath]["transform"] = self.transform
        return res

    def _getEinsumPath(self, path, d):
        """!
        Return an optimized einsum path.
        """
        try:
            return self._einsum_paths[path]
        except KeyError:
            self._einsum_paths[path], _ = np.einsum_path(path, d, d, optimize="optimal")
            getLogger(__name__).info("Optimized Einsum path %s", path)
            return self._einsum_paths[path]

    def _getTensor(self, name, P, H, d):
        """!
        Return a component tensor of an all-to-all correlator.
        Build and cache it if necessary.
        """
        try:
            return self._cache[name]
        except KeyError:
            tensor = _TENSOR_CONSTRUCTORS[name](P, H, d, self._getEinsumPath)
            self._cache[name] = tensor
            return self._cache[name]

    def _allToAllCorrelator(self, name, P, H, d):
        """!
        Compute an all-to-all correlator of given name from all-to-all
        propagators P and H and Kronecker delta d.
        """

        def t(n):
            return self._getTensor(n, P, H, d)

        if name == "Splus_Sminus":
            return t("PxyHxy")
        if name == "Sminus_Splus":
            return t("dyxdyx") - t("Pyxdyx") - t("Hyxdyx") + t("PyxHyx")
        if name == "np_nh":
            return t("dxxdyy") - t("Pxxdyy") - t("dxxHyy") + t("PxxHyy")
        if name == "np_np":
            return t("dxxdyy") - t("dxxPyy") - t("Pxxdyy") + t("Pxydyx") - t("PxyPyx") + t("PxxPyy")
        if name == "nh_np":
            return t("dxxdyy") - t("Hxxdyy") - t("dxxPyy") + t("HxxPyy")
        if name == "nh_nh":
            return t("dxxdyy") - t("dxxHyy") - t("Hxxdyy") + t("Hxydyx") - t("HxyHyx") + t("HxxHyy")
        if name == "++_--":
            return t("Hxydyx") - t("HxyPyx")
        if name == "--_++":
            return t("Pxydyx") - t("PxyHyx")
        else:
            raise ValueError(f"Unknown all-to-all correlator: {name}")

    @classmethod
    def computeDerivedCorrelators(cls, measurements, commonTransform, correlators=None):
        r"""!
        \param measurements a dictionary of measurements that has measurements of `"Splus_Sminus"`,
                            `"Sminus_Splus"`, `"np_np"`, `"np_nh"`, `"nh_np"`, and `"nh_nh"`
                            (and other fields are allowed).
        \param commonTransform A spatial matrix used to transform *all* correlators passed in through `measurements`,
                               i.e. the `transform` attribute of measurement objects.
        \param correlators an iterable of correlators you wish to compute.
                           If `None`, the default, uses all possible identities from the above bilinear correlators.
                           If one-point measurements from onePointFunctions.py `"np"` and `"nh"` are included,
                           additional identities will be leveraged to construct more two-point functions.

        Uses the identities above to reframe data in terms of different operators.

        You can sensibly do this on vectors of bootstrapped averages, since everything is linear.

        \returns `dict` with additional reconstructed correlators
        `"S1_S1"`, `"S1_S2"`, `"rho_rho"`, `"rho_n"`, `"n_rho"`, `n_n`.  If `"np"` and `"nh"` are available,
        also constructed are `"S0_S0"`, `"S0_S3"`, `"S3_S0"`, and `"S3_S3"`.
        """

        log = getLogger(__name__)

        if correlators is None:
            if "np" in measurements and "nh" in measurements:
                correlators = SpinSpinCorrelator.DERIVED_CORRELATOR_NAMES
                log.info("Selecting full derived spin-spin correlators: %s", correlators)
            else:
                correlators = SpinSpinCorrelator.DERIVED_CORRELATOR_NAMES_SPIN_ONLY
                log.info("Selecting only partial derived spin-spin correlators, "
                         "no one point data is available: %s", correlators)

        derived = dict()

        # These are easy to think about, if one is measured they are all measured.
        if "S1_S1" in correlators:
            derived["S1_S1"] = 0.25 *(measurements["Splus_Sminus"] + measurements["Sminus_Splus"])
        if "S1_S2" in correlators:
            derived["S1_S2"] = 0.25j*(measurements["Splus_Sminus"] - measurements["Sminus_Splus"])
        if "rho_rho" in correlators:
            derived["rho_rho"] = measurements["np_np"] + measurements["nh_nh"] - measurements["np_nh"] - measurements["nh_np"]
        if "rho_n" in correlators:
            derived["rho_n"] = measurements["np_np"] - measurements["nh_nh"] + measurements["np_nh"] - measurements["nh_np"]
        if "n_rho" in correlators:
            derived["n_rho"] = measurements["np_np"] - measurements["nh_nh"] - measurements["np_nh"] + measurements["nh_np"]
        if "n_n" in correlators:
            derived["n_n"] = measurements["np_np"] + measurements["nh_nh"] + measurements["np_nh"] + measurements["nh_np"]

        if any(name in correlators for name in SpinSpinCorrelator.DERIVED_CORRELATOR_NAMES_ONE_POINT):
            # TODO: These are hard to think about generally, if they're not measured with equal frequency.
            # NB:   This contains the assumption that they're measured at *the same* frequency

            # This assertion checks the number of measurements and the spatial volume match.
            assert measurements['np'].shape == measurements["np_np"].shape[0:2]

            nm = measurements["np"].shape[0]        # number of measurements
            nx = measurements["np"].shape[1]        # space
            nt = measurements["np_np"].shape[-1]    # time

            # Kronecker deltas
            dx = np.eye(nx)
            dt = np.eye(nt)

            U = commonTransform if commonTransform is not None else np.eye(nx)

            # TODO: These need U <---> U.conj().T when the convention is harmonized as part of #30.
            # TODO: think carefully about whether < n > also needs to swap .conj()s
            one = np.einsum('ax,xx,tt,yy,yb->abt', U.conj().T, dx, dt, dx, U, optimize="optimal")
            npx = np.einsum('ma,tt,yy,yb->mabt', measurements['np'],        dt, dx, U,          optimize="optimal")
            npy = np.einsum('mb,tt,xx,ax->mabt', measurements['np'].conj(), dt, dx, U.conj().T, optimize="optimal")
            nhx = np.einsum('ma,tt,yy,yb->mabt', measurements['nh'],        dt, dx, U,          optimize="optimal")
            nhy = np.einsum('mb,tt,xx,ax->mabt', measurements['nh'].conj(), dt, dx, U.conj().T, optimize="optimal")

            if "S0_S0" in correlators:
                derived["S0_S0"] = 0.25*(derived["rho_rho"] + one - npx - npy + nhx + nhy)
            if "S0_S3" in correlators:
                derived["S0_S3"] = 0.25*(derived["rho_n"]   + one - npx - npy + nhx - nhy)
            if "S3_S0" in correlators:
                derived["S3_S0"] = 0.25*(derived["n_rho"]   + one - npx - npy - nhx + nhy)
            if "S3_S3" in correlators:
                derived["S3_S3"] = 0.25*(derived["n_n"]     + one - npx - npy - nhx - nhy)

        return derived


def _checkCorrNames(actual, allowed):
    for name in actual:
        if name not in allowed:
            raise ValueError(f"Unknown correlator: '{name}'. Choose from '{allowed}'")


# Contractions always result in a tensor xfyi, where xf are space/time at the sink
# and yi are space/time at the source.  Because we have to move all the daggers to the right
# to Wick contract ladder operators into propagators, however, the indices on the propagators
# need not be in the same order.
#
# Contractions are grouped into 4 categories,
#   xfxf,yiyi->xfyi
#   xfyi,xfyi->xfyi
#   xfyi,yixf->xfyi
#   yixf,yixf->xfyi
#
# Note that other orders are all already covered
#   yixf,xfyi = xfyi,yixf
#   yiyi,xfxf = xfxf,yiyi
#
# Even in those four categories, there may be enormous redundencies.
# For example, the kronecker delta is symmetric.
_TENSOR_CONSTRUCTORS = {
    "dxxdyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", d, d, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "dxxPyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", d, P, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "dxxHyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", d, H, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "Pxxdyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", P, d, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "PxxPyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", P, P, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "PxxHyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", P, H, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "Hxxdyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", H, d, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "HxxPyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", H, P, optimize=getPath("xfxf,yiyi->xfyi", d)),
    "HxxHyy": lambda P, H, d, getPath:
        np.einsum("xfxf,yiyi->xfyi", H, H, optimize=getPath("xfxf,yiyi->xfyi", d)),

    # dxydxy = dyxdyx
    # dxyPxy = Pxydxy
    # dxyHxy = Hxydyx
    # Pxydxy = Pxydyx
    # PxyPxy cannot appear by Pauli exclusion
    "PxyHxy": lambda P, H, d, getPath:
        np.einsum("xfyi,xfyi->xfyi", P, H, optimize=getPath("xfyi,xfyi->xfyi", d)),
    # Hxydxy = Hxydyx
    # HxyPxy = PxyHxy
    # HxyHxy cannot appear by Pauli exclusion

    # dxydyx = dyxdyx
    # dxyPyx = Pyxdyx
    # dxyHyx = Hyxdyx
    "Pxydyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", P, d, optimize=getPath("xfyi,yixf->xfyi", d)),
    "PxyPyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", P, P, optimize=getPath("xfyi,yixf->xfyi", d)),
    "PxyHyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", P, H, optimize=getPath("xfyi,yixf->xfyi", d)),
    "Hxydyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", H, d, optimize=getPath("xfyi,yixf->xfyi", d)),
    "HxyPyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", H, P, optimize=getPath("xfyi,yixf->xfyi", d)),
    "HxyHyx": lambda P, H, d, getPath:
        np.einsum("xfyi,yixf->xfyi", H, H, optimize=getPath("xfyi,yixf->xfyi", d)),

    "dyxdyx": lambda P, H, d, getPath:
        np.einsum("yixf,yixf->xfyi", d, d, optimize=getPath("yixf,yixf->xfyi", d)),
    # dyxPyx = Pyxdyx
    # dyxHyx = Hyxdyx
    "Pyxdyx": lambda P, H, d, getPath:
        np.einsum("yixf,yixf->xfyi", P, d, optimize=getPath("yixf,yixf->xfyi", d)),
    # PyxPyx cannot appear by Pauli exclusion.
    "PyxHyx": lambda P, H, d, getPath:
        np.einsum("yixf,yixf->xfyi", P, H, optimize=getPath("yixf,yixf->xfyi", d)),
    "Hyxdyx": lambda P, H, d, getPath:
        np.einsum("yixf,yixf->xfyi", H, d, optimize=getPath("yixf,yixf->xfyi", d)),
    # HyxPyx = PyxHyx
    # HyxHyx cannot appear by Pauli exclusion.
}
