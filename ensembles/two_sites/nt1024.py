from pathlib import Path
import cns

# Physics parameters
latticeFile = Path("two_sites.yml")   # input lattice
nt = 1024                        # number of time slices
U = 2                           # Hubbard parameter
beta = 5                        # Inverse temperature
mu = 0                          # Chemical potential
sigmaKappa = -1

delta = beta / nt
UTilde = U * delta
muTilde = mu * delta
name = "{}.nt{}.U{}.beta{}.mu{}".format(latticeFile.stem, nt, U, beta, mu)


lattice = cns.ensemble.readLattice(latticeFile)
lattice.nt(nt)
kappaTilde = lattice.hopping() * delta  # actually \tilde{kappa}

# Evolution / HMC Information
nTherm = 2000               # number of thermalization trajectories
nLeapfrogTherm = 32          # number of steps in the leapfrog at the beginning of thermalization
nLeapfrog = 24               # production leapfrog steps
nProduction = 10000         # number of production trajectories

hamiltonian = cns.Hamiltonian(cns.HubbardGaugeAction(UTilde),
                              cns.HubbardFermiAction(kappaTilde, muTilde, sigmaKappa))

thermalizer = cns.hmc.LinearStepLeapfrog(hamiltonian, (0.25, 0.5), (nLeapfrogTherm, nLeapfrog), nTherm-1)
proposer = cns.hmc.ConstStepLeapfrog(hamiltonian, 0.5, nLeapfrog)

rng = cns.random.NumpyRNG(1075)
initialConfig = cns.Vector(rng.normal(0, UTilde**(1/2), lattice.lattSize())+0j)
