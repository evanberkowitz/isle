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
which I have checked explicitly for \f$(i,j)=(1,2)\f$ by writing out the operators in all their glory and using the anticommutation properties of \f$a\f$ and \f$b\f$.
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
I have checked that this equation is satisfied when
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
    C^{12}_{xy}+C^{21}_{xy} = \frac{i}{2} \left(C^{+–}_{xy} – C^{–+}_{xy}\right).
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

import isle
from .measurement import Measurement
from ..util import temporalRoller, signAlternator
from ..h5io import createH5Group

fields = [
    'Splus_Sminus', 'Sminus_Splus',
    'np_np', 'np_nh', 'nh_np', 'nh_nh',
    '++_--', '--_++',
]

class measurement(Measurement):
    r"""!
    \ingroup meas
    Tabulate spin-spin correlators.
    """

    def __init__(self, particleAllToAll, holeAllToAll, savePath, configSlice=(None, None, None), transform=None, sigmaKappa=-1):
        super().__init__(savePath, configSlice)

        # The correlation functions encoded here are between bilinear operators.
        # Since the individual constituents are fermionic, the bilinear is bosonic.
        self.fermionic = False


        self.sigmaKappa = sigmaKappa

        self.particle=particleAllToAll
        self.hole=holeAllToAll

        self.data = {k: [] for k in [
            # Directly computed
            "Splus_Sminus", "Sminus_Splus",
            "np_np", "np_nh", "nh_np", "nh_nh",
            "++_--", "--_++",
            # And their combinations:
            # "S1_S1", "S1_S2",
            # "rho_rho", "rho_n", "n_rho", "n_n"
            ]}

        self.transform = transform

        self._einsum_paths = {}

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
            P = np.einsum('ax,xfyi,yb->afyb', Sigma, P, Sigma, optimize="optimal")
            H = np.einsum('ax,xfyi,yb->afyb', Sigma, H, Sigma, optimize="optimal")


        d = np.eye(nx*nt).reshape(*P.shape) # A Kronecker delta

        log = getLogger(__name__)

        # TODO: store some einsum paths

        # Contractions always result in a tensor xfyi, where xf are space/time at the sink
        # and yi are space/time at the source.  Because we have to move all the daggers to the right
        # to Wick contract ladder operators into propagators, however, the indices on the propagators
        # need not be in the same order.

        # Contractions are grouped into 4 categories,
        #   xfxf,yiyi->xfyi
        #   xfyi,xfyi->xfyi
        #   xfyi,yixf->xfyi
        #   yixf,yixf->xfyi
        #
        # Note that other orders are all ready covered
        #   yixf,xfyi = xfyi,yixf
        #   yiyi,xfxf = xfxf,yiyi
        #
        # Even in those four categories, there may be enormous redundencies.
        # For example, the kronecker delta is symmetric.
        if "xfxf,yiyi->xfyi" not in self._einsum_paths:
            self._einsum_paths["xfxf,yiyi->xfyi"], _ = np.einsum_path("xfxf,yiyi->xfyi", d, d, optimize="optimal")
            log.info("Optimized Einsum path xfxf,yiyi->xfyi")

        dxxdyy = np.einsum("xfxf,yiyi->xfyi", d, d, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        dxxPyy = np.einsum("xfxf,yiyi->xfyi", d, P, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        dxxHyy = np.einsum("xfxf,yiyi->xfyi", d, H, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        Pxxdyy = np.einsum("xfxf,yiyi->xfyi", P, d, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        PxxPyy = np.einsum("xfxf,yiyi->xfyi", P, P, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        PxxHyy = np.einsum("xfxf,yiyi->xfyi", P, H, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        Hxxdyy = np.einsum("xfxf,yiyi->xfyi", H, d, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        HxxPyy = np.einsum("xfxf,yiyi->xfyi", H, P, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])
        HxxHyy = np.einsum("xfxf,yiyi->xfyi", H, H, optimize=self._einsum_paths["xfxf,yiyi->xfyi"])

        if "xfyi,xfyi->xfyi" not in self._einsum_paths:
            self._einsum_paths["xfyi,xfyi->xfyi"], _ = np.einsum_path("xfyi,xfyi->xfyi", d, d, optimize="optimal")
            log.info("Optimized Einsum path xfyi,xfyi->xfyi")

    #   dxydxy = dyxdyx
    #   dxyPxy = Pxydxy
    #   dxyHxy = Hxydyx
    #   Pxydxy = Pxydyx
    #   PxyPxy cannot appear by Pauli exclusion
        PxyHxy = np.einsum("xfyi,xfyi->xfyi", P, H, optimize=self._einsum_paths["xfyi,xfyi->xfyi"])
    #   Hxydxy = Hxydyx
    #   HxyPxy = PxyHxy
    #   HxyHxy cannot appear by Pauli exclusion

        if "xfyi,yixf->xfyi" not in self._einsum_paths:
            self._einsum_paths["xfyi,yixf->xfyi"], _ = np.einsum_path("xfyi,yixf->xfyi", d, d, optimize="optimal")
            log.info("Optimized Einsum path xfyi,yixf->xfyi")

    #   dxydyx = dyxdyx
    #   dxyPyx = Pyxdyx
    #   dxyHyx = Hyxdyx
        Pxydyx = np.einsum("xfyi,yixf->xfyi", P, d, optimize=self._einsum_paths["xfyi,yixf->xfyi"])
        PxyPyx = np.einsum("xfyi,yixf->xfyi", P, P, optimize=self._einsum_paths["xfyi,yixf->xfyi"])
        PxyHyx = np.einsum("xfyi,yixf->xfyi", P, H, optimize=self._einsum_paths["xfyi,yixf->xfyi"])
        Hxydyx = np.einsum("xfyi,yixf->xfyi", H, d, optimize=self._einsum_paths["xfyi,yixf->xfyi"])
        HxyPyx = np.einsum("xfyi,yixf->xfyi", H, P, optimize=self._einsum_paths["xfyi,yixf->xfyi"])
        HxyHyx = np.einsum("xfyi,yixf->xfyi", H, H, optimize=self._einsum_paths["xfyi,yixf->xfyi"])

        if "yixf,yixf->xfyi" not in self._einsum_paths:
            self._einsum_paths["yixf,yixf->xfyi"], _ = np.einsum_path("yixf,yixf->xfyi", d, d, optimize="optimal")
            log.info("Optimized Einsum path yixf,yixf->xfyi")

        dyxdyx = np.einsum("yixf,yixf->xfyi", d, d, optimize=self._einsum_paths["yixf,yixf->xfyi"])
    #   dyxPyx = Pyxdyx
    #   dyxHyx = Hyxdyx
        Pyxdyx = np.einsum("yixf,yixf->xfyi", P, d, optimize=self._einsum_paths["yixf,yixf->xfyi"])
    #   PyxPyx cannot appear by Pauli exclusion.
        PyxHyx = np.einsum("yixf,yixf->xfyi", P, H, optimize=self._einsum_paths["yixf,yixf->xfyi"])
        Hyxdyx = np.einsum("yixf,yixf->xfyi", H, d, optimize=self._einsum_paths["yixf,yixf->xfyi"])
    #   HyxPyx = PyxHyx
    #   HyxHyx cannot appear by Pauli exclusion.

        data = dict()

        data["Splus_Sminus"] = PxyHxy
        data["Sminus_Splus"] = dyxdyx - Pyxdyx - Hyxdyx + PyxHyx

        data["np_nh"] = dxxdyy - Pxxdyy - dxxHyy + PxxHyy
        data["np_np"] = dxxdyy - dxxPyy - Pxxdyy + Pxydyx - PxyPyx + PxxPyy
        data["nh_np"] = dxxdyy - Hxxdyy - dxxPyy + HxxPyy
        data["nh_nh"] = dxxdyy - dxxHyy - Hxxdyy + Hxydyx - HxyHyx + HxxHyy

        data["++_--"] = Hxydyx - HxyPyx
        data["--_++"] = Pxydyx - PxyHyx

        self._roll = np.array([temporalRoller(nt, -t, fermionic=self.fermionic) for t in range(nt)])

        time_averaged = dict()
        for correlator in data:

            # It is major savings to avoid two matrix-matrix multiplies, so it is
            # worthwhile to test for a transform and only add those multiplies in if needed.
            if self.transform is not None:
                if "idf,bx,xfyi,ya->bad" not in self._einsum_paths:
                    self._einsum_paths["idf,bx,xfyi,ya->bad"], _ = np.einsum_path("idf,bx,xfyi,ya->bad", self._roll, self.transform.T.conj(), data[correlator], self.transform, optimize="optimal")
                    log.info("Optimized Einsum path for time averaging and transform application.")

                time_averaged[correlator] = np.einsum("idf,bx,xfyi,ya->bad",
                                    self._roll,
                                    self.transform.T.conj(),
                                    data[correlator],
                                    self.transform,
                                    optimize=self._einsum_paths["idf,bx,xfyi,ya->bad"]) / nt
            else:
                if "idf,xfyi->xyd" not in self._einsum_paths:
                    self._einsum_paths["idf,xfyi->xyd"], _ = np.einsum_path("idf,xfyi->xyd", self._roll, data[correlator], optimize="optimal")
                    log.info("Optimized Einsum path for time averaging in position space.")

                time_averaged[correlator] = np.einsum("idf,xfyi->xyd",
                                    self._roll,
                                    data[correlator],
                                    optimize=self._einsum_paths["idf,bx,xfyi,ya->bad"]) / nt

        # Now we can use identites demonstrated above to build other two-point functions:
        # time_averaged["S1_S1"] = 0.25 *(time_averaged["Splus_Sminus"] + time_averaged["Sminus_Splus"])
        # time_averaged["S1_S2"] = 0.25j*(time_averaged["Splus_Sminus"] - time_averaged["Sminus_Splus"])
        # time_averaged["rho_rho"] = time_averaged["np_np"] + time_averaged["nh_nh"] - time_averaged["np_nh"] - time_averaged["nh_np"]
        # time_averaged["rho_n"]   = time_averaged["np_np"] - time_averaged["nh_nh"] + time_averaged["np_nh"] - time_averaged["nh_np"]
        # time_averaged["n_rho"]   = time_averaged["np_np"] - time_averaged["nh_nh"] - time_averaged["np_nh"] + time_averaged["nh_np"]
        # time_averaged["n_n"]     = time_averaged["np_np"] + time_averaged["nh_nh"] + time_averaged["np_nh"] + time_averaged["nh_np"]

        for correlator in self.data:
            self.data[correlator].append(time_averaged[correlator])

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
    \param h5group HDF5 group which contains the data for this measurement.

    \returns A dictionary of measurements and a transformation, \n
    {   \n
        `Splus_Sminus`: \f$S^+_{xf}S^-_{yi}\f$,                         \n
        `Sminus_Splus`: \f$S^-_{xf}S^+_{yi}\f$,                         \n
        `np_np`       : \f$n^p_{xf}n^p_{yi}\f$,                         \n
        `np_nh`       : \f$n^p_{xf}n^h_{yi}\f$,                         \n
        `nh_np`       : \f$n^h_{xf}n^p_{yi}\f$,                         \n
        `nh_nh`       : \f$n^h_{xf}n^h_{yi}\f$,                         \n
        `++_--`       : \f$(a^\dagger b)_{xf} (b^\dagger a)_{yi}\f$,    \n
        `--_++`       : \f$(b^\dagger a)_{xf} (a^\dagger b)_{yi}\f$,    \n
        `spinSpin-transform`: A transformation from position space to another space.
    }
    """

    try:
        data = {key: h5group[key][()] for key in h5group}

        if type(data['transform']) is h5.Empty:
            data['transform'] = None

        data['spinSpin-transform'] = data['transform']
        del data['transform']

        return data
    except:
        raise KeyError(f"Problem reading spinSpinCorrelator measurements from {h5group.name}")

def complete(measurements, correlators=["S1_S1", "S1_S2", "rho_rho", "rho_n", "n_rho", "n_n",
                                        "S3_S3", "S3_S0", "S0_S3", "S0_S0"
                                        ]):

    requiresOnePoint = ["S3_S3", "S3_S0", "S0_S3", "S0_S0"]

    rest = dict()

    if "S1_S1" in correlators:
        rest["S1_S1"]   = 0.25 *(measurements["Splus_Sminus"] + measurements["Sminus_Splus"])
    if "S1_S2" in correlators:
        rest["S1_S2"]   = 0.25j*(measurements["Splus_Sminus"] - measurements["Sminus_Splus"])
    if "rho_rho" in correlators:
        rest["rho_rho"] = measurements["np_np"] + measurements["nh_nh"] - measurements["np_nh"] - measurements["nh_np"]
    if "rho_n" in correlators:
        rest["rho_n"]   = measurements["np_np"] - measurements["nh_nh"] + measurements["np_nh"] - measurements["nh_np"]
    if "n_rho" in correlators:
        rest["n_rho"]   = measurements["np_np"] - measurements["nh_nh"] - measurements["np_nh"] + measurements["nh_np"]
    if "n_n" in correlators:
        rest["n_n"]     = measurements["np_np"] + measurements["nh_nh"] + measurements["np_nh"] + measurements["nh_np"]

    log = getLogger(__name__)

    if "np" in measurements and "nh" in measurements:
        log.info("Constructing bilinears that require one-point measurements.")
        if "S3_S3" in correlators:
            ...
        if "S3_S0" in correlators:
            ...
        if "S0_S3" in correlators:
            ...
        if "S0_S0" in correlators:
            ...
    else:
        log.info("One-point measurements are needed to construct S3_S3, S3_S0, S0_S3, S0_S0 bilinears.")

    return {**measurements, **rest}
