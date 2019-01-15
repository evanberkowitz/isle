"""!
Example script to show how to set up the HMC driver for basic production.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers


### Specify parameters.
# isle.util.parameters takes arbitrary keyword arguments, constructs a new dataclass,
# and stores the function arguments in an instance of it.
# The object is written to the output file and read back in by all subsequent processes.
# Use this to store all physical and model parameters and make them accessible later.
#
# Note that all objects stored in here must be representable in and constructible
# from YAML. You need to register new handlers if you have custom types.
PARAMS = isle.util.parameters(
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
    variant=isle.action.HFAVariant.ONE
)

# Set the number of time slices.
# This is stored in isle.Lattice and does not go into the above object.
NT = 16


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
    # Import all this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.Hamiltonian(isle.action.HubbardGaugeAction(params.tilde("U", lat)),
                            isle.action.makeHubbardFermiAction(lat,
                                                               params.beta,
                                                               params.tilde("mu", lat),
                                                               params.sigmaKappa,
                                                               params.hopping,
                                                               params.basis,
                                                               params.variant))


def main():
    # Initialize Isle.
    # This sets up the command line interface, defines an argument parser for an HMC
    # command, and parses and returns arguments.
    args = isle.initialize("hmc", name="basic-hmc")

    # Get a logger. Use this instead of print() to output any and all information.
    log = getLogger("HMC")

    # Load the spatial lattice.
    lat = isle.fileio.yaml.loadLattice("resources/lattices/four_sites.yml")
    # Lattice files usually only contain information on the spatial lattice
    # to be more flexible. Set the number of time slices here.
    lat.nt(NT)

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(1075)

    # Set up a fresh HMC driver.
    # It handles all HMC evolution as well as I/O.
    hmcState = isle.drivers.hmc.init(lat, PARAMS, rng, makeAction, args.outfile,
                                     args.overwrite, startIdx=0)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi = isle.Vector(rng.normal(0,
                                 (PARAMS.U * PARAMS.beta / lat.nt())**(1/2),
                                 lat.lattSize())
                      +0j)

    # Run thermalization.
    log.info("Thermalizing")
    # Pick a proposer towhich linearly decreases the number of MD steps from 20 to 5.
    # The number of steps (99) must be one less than the number of trajectories below.
    proposer = isle.proposers.LinearStepLeapfrog(hmcState.ham, (1, 1), (20, 5), 99)
    # Thermalize configuration for 100 trajectories without saving anything.
    phi = hmcState(phi, proposer, 100, saveFreq=0, checkpointFreq=0)
    # Reset the internal counter so we start saving configs at index 0.
    hmcState.resetIndex()

    # Run production.
    log.info("Producing")
    # Pick a new proposer with a constant number of steps to get a reproducible ensemble.
    proposer = isle.proposers.ConstStepLeapfrog(hmcState.ham, 1, 5)
    # Produce configurations and save in intervals of 2 trajectories.
    # Place a checkpoint every 10 trajectories.
    phi = hmcState(phi, proposer, 100, saveFreq=2, checkpointFreq=10)

    # That is it, clean up happens automatically.

if __name__ == "__main__":
    main()
