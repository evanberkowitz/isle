from pathlib import Path
import isle

# Physics parameters
latticeFile = Path("two_sites.yml")   # input lattice
nt = 6                          # number of time slices
U = 2                           # Hubbard parameter
beta = 5                        # Inverse temperature
mu = 0                          # Chemical potential
sigmaKappa = -1

delta = beta / nt
UTilde = U * delta
muTilde = mu * delta
name = "{}.nt{}.U{}.beta{}.mu{}".format(latticeFile.stem, nt, U, beta, mu)


lattice = isle.ensemble.readLattice(latticeFile)
lattice.nt(nt)
kappaTilde = lattice.hopping() * delta  # actually \tilde{kappa}

# Evolution / HMC Information
nTherm = 3000               # number of thermalization trajectories
nLeapfrogTherm = 8          # number of steps in the leapfrog at the beginning of thermalization
nLeapfrog = 3               # production leapfrog steps
nProduction = 10000         # number of production trajectories

hamiltonian = isle.Hamiltonian(isle.HubbardGaugeAction(UTilde),
                              isle.HubbardFermiAction(kappaTilde, muTilde, sigmaKappa))

thermalizer = isle.hmc.LinearStepLeapfrog(hamiltonian, (1, 1), (nLeapfrogTherm, nLeapfrog), nTherm-1)
proposer = isle.hmc.ConstStepLeapfrog(hamiltonian, 1, nLeapfrog)

rng = isle.random.NumpyRNG(1075)
initialConfig = isle.Vector(rng.normal(0, UTilde**(1/2), lattice.lattSize())+0j)
