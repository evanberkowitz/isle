import cns

# Physics parameters
latticeFile = "c60_ipr.yml"   # input lattice
nt = 12                          # number of time slices
U = 1                           # Hubbard parameter
beta = 2                        # Inverse temperature
mu = 0                          # Chemical potential
sigma_kappa = -1

delta = beta / nt
UTilde = U * delta
muTilde = mu * delta
name = "{}.nt{}.U{}.beta{}.mu{}".format(latticeFile.split(".")[0],nt,U,beta,mu)


lattice = cns.ensemble.readLattice(latticeFile)
lattice.nt(nt)
kappaTilde = lattice.hopping() * delta  # actually \tilde{kappa}

# Evolution / HMC Information
nTherm = 1000               # number of thermalization trajectories
nLeapfrogTherm = 6          # number of steps in the leapfrog at the beginning of thermalization
nLeapfrog = 3               # production leapfrog steps
nProduction = 10000         # number of production trajectories

hamiltonian = cns.Hamiltonian(cns.HubbardGaugeAction(UTilde),
                          cns.HubbardFermiAction(kappaTilde, muTilde, sigma_kappa))

thermalizer = cns.hmc.LinearStepLeapfrog(hamiltonian, (1, 1), (nLeapfrogTherm, nLeapfrog), nTherm-1)
proposer = cns.hmc.ConstStepLeapfrog(hamiltonian, 1, nLeapfrog)

rng = cns.random.NumpyRNG(1075)
initialConfig = cns.Vector(rng.normal(0, UTilde**(1/2), lattice.lattSize())+0j)
