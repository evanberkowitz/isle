"""
Example that demonstrates how to continue a run with a different evolver.
"""

# All output should go through a logger for better control and consistency.
from logging import getLogger

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers


def _buildEvolver(previous, hmcState):
    """Construct an evolver for production based on the one used for thermalization."""

    # Try to read the parameters for leapfrog from the old proposer.
    if isinstance(previous, isle.evolver.LinearStepLeapfrog):
        # There is no way to get the current values of there paramters from the evolver.
        # So assume it has finished and just use last values for trajectory length
        # and number of MD steps.
        length = previous.lengthRange[1]
        nstep = previous.nstepRange[1]

    elif isinstance(previous, isle.evolver.ConstStepLeapfrog):
        # Copy the exact parameters.
        length = previous.length
        nstep = previous.nstep

    else:
        # Cannot handle any evolver that is not one of the above.
        getLogger(__name__).error("Unable to extract leapfrog parameters from evolver of type %s",
                                  type(previous))
        raise RuntimeError("Unable to extract leapfrog parameters")

    # Define a new evolver that alternates between leapfrog integration and
    # jumps py 2*pi in the field variables.
    return isle.evolver.Alternator((
        # Use leapfrog with constant parameters for 100 trajectories.
        (100, isle.evolver.ConstStepLeapfrog(hmcState.action, length, nstep, hmcState.rng)),
        # Use 2*pi jumps once but attempt to jump nt*nx times on random sites,
        # accepting or rejecting each jump individually.
        (1, isle.evolver.TwoPiJumps(hmcState.lattice.lattSize(), 1, hmcState.action,
                                    hmcState.lattice, hmcState.rng))
    ))


def main():
    # Initialize Isle.
    # Sets up the command line interface and parses arguments using the 'continue' parser.
    # This is the same parser that is used for the `isle continue` command that is
    # installed alongside Isle.
    args = isle.initialize("continue")

    # Set up an HMC driver based on the results of an existing run, loading
    # the action and all parameters.
    # This reads the indicated checkpoint (latest by default) from the file
    # and extracts the save and checkpoint frequencies based on how often data
    # was saved / checkpointed to the file before.
    hmcState, evStage, evolver, \
        saveFreq, checkpointFreq = isle.drivers.hmc.continueRun(args.infile, args.outfile,
                                                                args.initial, args.overwrite)

    # If the used requested a particular save / checkpoint frequency,
    # use that instead of the ones read from file.
    if args.save_freq is not None:
        saveFreq = args.save_freq
    if args.checkpoint_freq is not None:
        checkpointFreq = args.checkpoint_freq

    # Build a new evolver.
    evolver = _buildEvolver(evolver, hmcState)

    # Run the driver with the input data and user specified parameters.
    getLogger(__name__).info("Starting evolution")
    hmcState(evStage, evolver, args.ntrajectories,
             saveFreq=saveFreq, checkpointFreq=checkpointFreq)


if __name__ == "__main__":
    main()
