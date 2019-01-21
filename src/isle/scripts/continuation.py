r"""! \file
Script to continue a HMC run.

Run the script via continuation.main().
"""


from logging import getLogger

from isle.drivers.hmc import continueRun


def main(args):
    r"""!
    Perform a continuation run.
    \param args Parsed command line arguments.
    """

    hmcState, phi, proposer, \
        saveFreq, checkpointFreq = continueRun(args.infile, args.outfile,
                                               args.initial, args.overwrite)

    if args.save_freq is not None:
        saveFreq = args.save_freq
    if args.checkpoint_freq is not None:
        checkpointFreq = args.checkpoint_freq

    getLogger("continuation").info("Starting evolution")
    phi = hmcState(phi, proposer, args.ntrajectories,
                   saveFreq=saveFreq, checkpointFreq=checkpointFreq)
