"""
Example script to show how to thermalize and tune a leapfrog intergator.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger
from pathlib import Path

import h5py as h5

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers


### Specify input / output files
# Write all tuning data to this file.
TUNEFILE = "tuning-example.h5"
# Write all production results to this file.
PRODFILE = "tuning-prod-example.h5"
# Name of the lattice.
LATTICE = "four_sites"

### Specify parameters.
# isle.util.parameters takes arbitrary keyword arguments, constructs a new dataclass,
# and stores the function arguments in an instance of it.
# The object is written to the output file and read back in by all subsequent processes.
# Use this to store all physical and model parameters and make them accessible later.
#
# Note that all objects stored in here must be representable in and constructible
# from YAML. You need to register new handlers if you have custom types.
PARAMS = isle.util.parameters(
    beta=3.,         # inverse temperature
    U=2.,            # on-site coupling
    mu=1.,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)

    # Those three control which implementation of the action gets used.
    # The values given here are the defaults.
    # See documentation in docs/algorithm.
    hopping=isle.action.HFAHopping.DIA,
    basis=isle.action.HFABasis.PARTICLE_HOLE,
    algorithm=isle.action.HFAAlgorithm.DIRECT_SINGLE
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
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.algorithm)


def tune(rng):
    """
    Thermalize and then tune a leapfrog integrator.
    """

    # Get a logger. Use this instead of print() to output any and all information.
    log = getLogger("HMC")

    # Load the spatial lattice.
    lat = isle.LATTICES[LATTICE]
    # Lattice files usually only contain information on the spatial lattice
    # to be more flexible. Set the number of time slices here.
    lat.nt(NT)

    # Set up a fresh HMC driver.
    # It handles all HMC evolution as well as I/O.
    # Last argument forbids the driver to overwrite any existing data.
    hmcState = isle.drivers.hmc.newRun(lat, PARAMS, rng, makeAction,
                                       TUNEFILE, True)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi = isle.Vector(rng.normal(0,
                                 PARAMS.tilde("U", lat)**(1/2),
                                 lat.lattSize())
                      +0j)

    # Run thermalization; tuning works best on thermalized configurations.
    log.info("Thermalizing")
    # Pick an evolver which linearly decreases the number of MD steps from 20 to 5.
    # The number of steps (99) must be one less than the number of trajectories below.
    evolver = isle.evolver.LinearStepLeapfrog(hmcState.action, (1, 1), (20, 5), 299, rng)
    # Thermalize configuration for 100 trajectories without saving anything.
    evStage = hmcState(phi, evolver, 300, saveFreq=0, checkpointFreq=0)
    # Reset the internal counter so we start saving configs at index 0.
    hmcState.resetIndex()

    log.info("Tuning")
    # Construct the auto-tuner with default settings, see the documentation of the
    # constructor for a list of all supported parameters.
    # Start with an nstep of 1 which should be sufficiently small to produce an acceptance
    # rate close to zero in this case.
    # This is good as it anchors the fit, starting close to the expected optimum can make
    # the fit fail and slow down tuning.
    evolver = isle.evolver.LeapfrogTuner(hmcState.action, 1, 1, rng, TUNEFILE,targetAccRate=0.7,targetConfIntProb=0.0001,runsPerParam=(50,500),maxRuns=50)
    # Run evolution with the tuner for an indefinite number of trajectories,
    # LeapfrogTuner is in charge of terminating the run.
    # Checkpointing is not supported by the tuner, so checkpointFreq must be 0.
    hmcState(evStage, evolver, None, saveFreq=1, checkpointFreq=0)
    print(evolver.currentParams())
    print(evolver.tunedParameters())

def produce(rng):
    """
    Produce configurations starting at a thermalized one and using
    a tuned leapfrog evolver.
    """

    # Get a logger. Use this instead of print() to output any and all information.
    log = getLogger("HMC")

    # This is how the metadata from a previous run can be loaded in case it is not
    # avilable in the script (in contrast to this example).
    # The function to make an action is only loaded as source code, but that is
    # sufficient for the HMC driver.
    lat, params, makeActionSrc, _ = isle.h5io.readMetadata(TUNEFILE)

    # Set up a fresh HMC driver and write to the production file
    hmcState = isle.drivers.hmc.newRun(lat, params, rng, makeActionSrc,
                                       PRODFILE, True)

    # We can't use isle.drivers.hmc.continueRun because there are no checkpoints
    # to continue from, so load configuration and evolver manually.
    with h5.File(TUNEFILE, "r") as h5f:
        # Load the last saved configuration.
        phi, _ = isle.h5io.loadConfiguration(h5f)
        # Construct a ConstStepLeapfrog evolver from tuned parameters.
        evolver = isle.evolver.LeapfrogTuner.loadTunedEvolver(h5f, hmcState.action, rng)

    # Run production.
    log.info("Producing")
    # Produce configurations and save in intervals of 2 trajectories.
    # Place a checkpoint every 10 trajectories.
    hmcState(phi, evolver, 5000, saveFreq=1, checkpointFreq=10)


def main():
    # Initialize Isle.
    # This sets up the command line interface, defines a barebones argument parser,
    # and parses and returns parsed arguments.
    # More complex parsers can be automatically defined or passed in manually.
    # See, e.g., `hmcThermalization.py` or `measure.py` examples.
    isle.initialize("default")

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(105)

    tune(rng)
    produce(rng)


if __name__ == "__main__":
    main()
