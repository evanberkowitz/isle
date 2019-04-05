"""
Example that demonstrates how to set up a custom argument parser
and start a run by thermalizing a configuration.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger
from pathlib import Path

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers


# Load lattice from this file.
LATTICE = Path(__file__).resolve().parent/"../../resources/lattices/four_sites.yml"

### Specify parameters.
# Compared to fullHMCEvolution.py, we are skipping the algorithmic parameters
# hopping, basis, and variant and use their default values.
PARAMS = isle.util.parameters(
    beta=2.3,         # inverse temperature
    U=4.1,            # on-site coupling
    mu=0,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
)

# Set the number of time slices.
# This is stored in isle.Lattice and does not go into the above object.
NT = 8


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
                                             params.sigmaKappa)


def _init():
    """Initialize Isle."""

    # Define a custom command line argument parser.
    # Isle uses Python's argparse package and the following returns a new parser
    # which is set up to accept some default arguments like --version, --verbose, or --log.
    parser = isle.cli.makeDefaultParser(defaultLog="therm_example.log",
                                        description="Example script to thermalize a configuration")
    # Add custom arguments to control this script in particular.
    parser.add_argument("outfile", help="Output file", type=Path)
    parser.add_argument("-n", "--ntrajectories", type=int, metavar="N", required=True,
                        help="Generate N trajectories")
    parser.add_argument("-s", "--save-freq", type=int, metavar="SF", default=1,
                        help="Save every SF trajectories (default=1)")
    parser.add_argument("-c", "--checkpoint-freq", type=int, metavar="CF", default=None,
                        help="Write a checkpoint every CF trajectories (default=0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file.")

    # Initialize Isle and process arguments with the new parser.
    args = isle.initialize(parser)

    # If no checkpoint frequency is given, save a checkpoint for the last
    # configuration to make sure the run can be continued.
    if args.checkpoint_freq is None:
        # The number of saved configurations.
        nsaved = args.ntrajectories // args.save_freq
        # Place a checkpoint only for the last saved configuration.
        args.checkpoint_freq = nsaved * args.save_freq

    return args


def main():
    args = _init()

    # Load the spatial lattice.
    lat = isle.fileio.yaml.loadLattice(LATTICE)
    # Store nt.
    lat.nt(NT)

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(1337)

    # Set up a fresh HMC driver.
    hmcState = isle.drivers.hmc.newRun(lat, PARAMS, rng, makeAction,
                                       args.outfile, args.overwrite)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi = isle.Vector(rng.normal(0,
                                 PARAMS.tilde("U", lat)**(1/2),
                                 lat.lattSize())
                      +0j)

    # Run thermalization.
    getLogger(__name__).info("Thermalizing")
    # Pick an evolver which linearly decreases the number of MD steps from 20 to 5.
    evolver = isle.evolver.LinearStepLeapfrog(hmcState.action, (1, 1), (20, 5),
                                              args.ntrajectories-1, rng)
    # Thermalize configuration using the command line arguments.
    hmcState(phi, evolver, args.ntrajectories,
             saveFreq=args.save_freq, checkpointFreq=args.checkpoint_freq)


if __name__ == "__main__":
    main()
