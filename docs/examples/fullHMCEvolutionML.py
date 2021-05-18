"""
Example script to show how to set up the HMC driver for production including thermalization.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger
import numpy as np
# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers

# Name of the lattice.
LATTICE = "four_sites"

### Specify parameters
# isle.util.parameters takes arbitrary keyword arguments, constructs a new dataclass,
# and stores the function arguments in an instance of it.
# The object is written to the output file and read back in by all subsequent processes.
# Use this to store all physical and model parameters and make them accessible later.
#
# Note that all objects stored in here must be representable in and constructible
# from YAML. You need to register new handlers if you have custom types.
PARAMS = isle.util.parameters(
    beta=6.,         # inverse temperature
    U=4.,            # on-site coupling
    mu=0,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)

    # Those three control which implementation of the action gets used.
    # The values given here are the defaults.
    # See documentation in docs/algorithm.
    hopping=isle.action.HFAHopping.EXP,
    basis=isle.action.HFABasis.PARTICLE_HOLE,
    algorithm=isle.action.HFAAlgorithm.ML_APPROX_FORCE,
    allowShortcut = False,
    module_path="/p/project/cjjsc37/john/testing/isle/docs/examples/NNgHMC_models/NNgModel_4sitesU4B6Nt16.pt")
# Set the number of time slices.
# This is stored in isle.Lattice and does not go into the above object.
NT = 16

### Specify input / output files
# Write all data to this file.
OUTFILE = f"NNgHMCData/NNgHMC_4sitesNmd3_U{PARAMS.U}B{PARAMS.beta}Nt{NT}.h5"


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

    return  isle.action.HubbardGaugeActionML(params.tilde("U", lat)) \
             + isle.action.makeHubbardFermiActionMLApprox(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.algorithm,
                                             params.allowShortcut,params.module_path)


def main():
    # Initialize Isle.
    # This sets up the command line interface, defines a barebones argument parser,
    # and parses and returns parsed arguments.
    # More complex parsers can be automatically defined or passed in manually.
    # See, e.g., `hmcThermalization.py` or `measure.py` examples.
    isle.initialize("default")

    # Get a logger. Use this instead of print() to output any and all information.
    log = getLogger("HMC")

    # Load the spatial lattice.
    # Note: This command loads a lattice that is distributed together with Isle.
    #       In order to load custom lattices from a file, use
    #       either  isle.LATTICES.loadExternal(filename)
    #       or  isle.fileio.yaml.loadLattice(filename)
    lat = isle.LATTICES[LATTICE]
    # Lattice files usually only contain information on the spatial lattice
    # to be more flexible. Set the number of time slices here.
    lat.nt(NT)

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(1075)

    # Set up a fresh HMC driver.
    # It handles all HMC evolution as well as I/O.
    # Last argument forbids the driver to overwrite any existing data.
    hmcState = isle.drivers.hmc.newRun(lat, PARAMS, rng, makeAction,
                                       OUTFILE, False)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi = isle.Vector(rng.normal(0,
                                 PARAMS.tilde("U", lat)**(1/2),
                                 lat.lattSize())
                      +0j)




    # Run thermalization.
    log.info("Thermalizing")
    # Pick an evolver which linearly decreases the number of MD steps from 20 to 5.
    # The number of steps (99) must be one less than the number of trajectories below.
    evolver = isle.evolver.LinearStepLeapfrog(hmcState.action, (1, 1), (20, 5), 99, rng)
    # Thermalize configuration for 100 trajectories without saving anything.
    evStage = hmcState(phi, evolver, 100, saveFreq=0, checkpointFreq=0)
    # Reset the internal counter so we start saving configs at index 0.
    hmcState.resetIndex()

    # Run production.
    log.info("Producing")
    
    # Pick a new evolver with a constant number of steps to get a reproducible ensemble.
    evolver = isle.evolver.ConstStepLeapfrog(hmcState.action, 1,3, rng)
    # Produce configurations and save in intervals of 2 trajectories.
    # Place a checkpoint every 10 trajectories.
    hmcState(phi, evolver,10000, saveFreq=1, checkpointFreq=10)

    # That is it, clean up happens automatically.

if __name__ == "__main__":
    main()
