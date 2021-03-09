"""
Benchmark of a single HMC step
"""

import itertools
import timeit
import yaml
import pickle
import isle
import isle.drivers
import numpy as np

# Name of the lattice.
LATTICE = "c60_ipr"
OUTFILE = "hmc-benchmark_NT"

NTs = (16,24,32,40,48,56,64)#(16,32,64)
#NXs = (4,8,12,16,24,32,64,96)
NREP = 50

def makeAction(lattice, params):
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lattice)) \
         + isle.action.makeHubbardFermiAction(lattice,
                                            params.beta,
                                            params.tilde("mu", lattice),
                                            params.sigmaKappa,
                                            params.hopping,
                                            params.basis,
                                            params.algorithm
        )

def benchmark_hmc_step():
    params = isle.util.parameters(
        beta=3,         # inverse temperature
        U=2,            # on-site coupling
        mu=0,           # chemical potential
        sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                        # (+1 only allowed for bipartite lattices)

        # Those three control which implementation of the action gets used.
        # The values given here are the defaults.
        # See documentation in docs/algorithm.
        hopping=isle.action.HFAHopping.DIA,
        basis=isle.action.HFABasis.PARTICLE_HOLE,
        algorithm=isle.action.HFAAlgorithm.DIRECT_SINGLE
    )

    rng = isle.random.NumpyRNG(1075)

    timings = {
            f"HMC": [],
    }

    for nt in NTs:
        print(f"Benchmarking Nt = {nt}")
        # create a lattice
        lattice = isle.LATTICES[LATTICE]

        #
        lattice.nt(nt)

        action = makeAction(lattice, params)

        phi_initial = isle.Vector(
            rng.normal( 0,
                        params.tilde("U", lattice)**(1/2),
                        lattice.lattSize()
            ) + 0j
        )

        evolver = isle.evolver.ConstStepLeapfrog(action, 1, 5, rng)

        # Generate a random initial stage.
        # Note that configurations must be vectors of complex numbers.
        stage = isle.evolver.EvolutionStage(
            phi_initial,
            action.eval(phi_initial),
            1
        )

        timings[f"HMC"].append(
            np.min( timeit.repeat(
                "evolver.evolve(stage)",
                globals= {"evolver": evolver, "stage": stage},
                number = NREP,
                repeat = 5)
            ) * 1e+3 / NREP
        )

    pickle.dump(
        {
            "xlabel": "Nt",
            "ylabel": "time [ms]",
            "xvalues": list(NTs),
            "results": timings
        },
        open("HMC_C60.ben", "wb")
    )

def main():
    benchmark_hmc_step()

if __name__ == "__main__":
    main()
