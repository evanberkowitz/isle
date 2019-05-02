"""
Example script to show how to use Isle with MPI through mpi4py.

Tunes leapfrog integrators for multiple different ensembles and spreads the
work load over available MPI ranks.

This script is designed to be run through mpiexec (or equivalent) directly.
You can also run it via the mpi4py.futures module but this would not
initialize Isle's command line interface on every rank.

One rank is reserved for managing tasks (the 'master') but there is not a lot to manage
in this case because there is almost no pre-/post-processing of tasks.
It can thus be useful launch one more task than you have CPU cores.
On a personal machine with 4 cores and OpenMPI this can be done via
  mpiexec --host localhost:5 -n 5 python3 mpiTuning.py
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger
from pathlib import Path
from itertools import product

import yaml
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers

BASE_PATH = Path(__file__).resolve().parent
LATTICE_FILE = BASE_PATH/"../../resources/lattices/four_sites.yml"


BASE_PARAMS = isle.util.parameters(
    beta=None,      # inverse temperature, task dependent
    U=None,         # on-site coupling, task dependent
    mu=0,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)
    hopping=isle.action.HFAHopping.EXP,  # exponential discretization for fermion matrix
)
# Iterate over all combinations of those values.
US = (3, 4, 5)
BETAS = (3, )
NTS = (8, )


def fnameSuffix(lattice, params):
    """
    Generate a suffix for an output file name based on the parameters.
    """
    return f".nt{lattice.nt()}.beta{params.beta}.U{params.U}.h5"


def iterParams(*args):
    """
    Iterate over all combinations of parameters.
    Passes args along to allow passing constant arguments to workers.
    """

    # Load the lattice once in the master process.
    lattice = isle.fileio.yaml.loadLattice(LATTICE_FILE)

    # Run time depends on nt, so vary nt first such that processes that get scheduled
    # close to each other in time have similar run times.
    for nt, beta, U in product(NTS, BETAS, US):
        params = BASE_PARAMS.replace(beta=beta, U=U)
        lat = isle.Lattice(lattice)
        lat.nt(nt)
        # Instances of Lattice and Parameter cannot be pickled and thus MPI can't send
        # them to other ranks directly. Strings can be pickled though, so just use the
        # existing support for YAML.
        yield (yaml.dump(lat), yaml.dump(params), *args)


### Function to construct actions.
# This function has to construct and return all actions to be used in HMC.
# It is saved as source code to the output file and read back in by any
# subsequent process. It can therefore not rely on any external (global) state
# and must define / import everything it needs!
#
# Parameters:
#     lat - An instance of isle.Lattice containing the hopping matrix
#           and number of time steps.
#     params - Dataclass instance of parameters, see above (PARAMS).
def makeAction(lat, params):
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping)


# This function runs on the worker ranks and performs the actual calculations.
# Leapfrog is tuned for each ensemble essentially in isolation.
def tuneForEnsemble(latticeYAML, paramsYAML, clArgs):
    """
    Tune the leapfrog integrator for ont ensemble with given lattice and parameters.
    """

    rank = MPI.COMM_WORLD.rank
    log = getLogger(f"{__name__}[{rank}]")

    # Re-construct the actual Objects from strings here in the worker task.
    lattice = yaml.safe_load(latticeYAML)
    params = yaml.safe_load(paramsYAML)

    # A unique output file name for this set or parameters.
    outputFile = "mpiexample"+fnameSuffix(lattice, params)

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(rank + 831)

    # Set up a fresh HMC driver just for this ensemble.
    hmcState = isle.drivers.hmc.newRun(lattice, params, rng, makeAction,
                                       outputFile, clArgs.overwrite)

    # Generate a random initial condition.
    phi = isle.Vector(rng.normal(0,
                                 params.tilde("U", lattice)**(1/2),
                                 lattice.lattSize())
                      +0j)

    # Run thermalization.
    log.info("Thermalizing")
    # Pick an evolver which linearly decreases the number of MD steps from 20 to 5.
    # The number of steps (99) must be one less than the number of trajectories below.
    evolver = isle.evolver.LinearStepLeapfrog(hmcState.action, (1, 1), (20, 5), 99, rng)
    # Thermalize configuration for 100 trajectories without saving anything.
    phi, *_ = hmcState(phi, evolver, 100, saveFreq=0, checkpointFreq=0)
    # Reset the internal counter so we start saving configs at index 0.
    hmcState.resetIndex()

    log.info("Tuning")
    # Construct the auto-tuner with default settings, see the documentation of the
    # constructor for a list of all supported parameters.
    evolver = isle.evolver.LeapfrogTuner(hmcState.action, 1, 1, rng, outputFile)
    # Run evolution with the tuner for an indefinite number of trajectories,
    hmcState(phi, evolver, None, saveFreq=1, checkpointFreq=0)


def main():
    # Initialize the command line interface on every MPI rank.
    # This clobbers up the output a little bit but saves us organizing this more.
    parser = isle.cli.makeDefaultParser(description="Tune integrator for multiple ensembles")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file.")
    clArgs = isle.initialize(parser)

    with MPICommExecutor() as executor:
        if executor is not None:
            # This code runs only on the master rank.
            # Submit jobs for worker threads with all combinations of parameters.
            # There is no need to iterate through tasks in order as tuneForEnsemble does not
            # return anything, so use unordere=True to detect exceptions as early as possible below.
            results = executor.starmap(tuneForEnsemble, iterParams(clArgs), unordered=True)
            # Iterate through results to trigger any exceptions caught by the executor.
            for _ in results:
                pass


if __name__ == "__main__":
    main()
