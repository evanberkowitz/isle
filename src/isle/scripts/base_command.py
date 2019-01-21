r"""! \file
The base command line script for Isle.
Dispatches calls to other, specific scripts.
"""

from logging import getLogger

from .. import initialize
from . import continuation


def _run_show(args):
    # import on demand in case matplotlib is not installed
    from .show import main
    main(args)

def main():
    """!Run Isle's base script. Dispatches to other scripts based on command line arguments."""

    commands = ("show", "continue")
    mains = (_run_show, continuation.main)
    descriptions = ("Pulls all data it can from a file in a format supported by Isle, "
                    "prints, and visualizes that data. Supported file types are HDF5 and YAML. "
                    "Select a reporter via -r to choose which information to show.",

                    "Perform a continuation HMC run. Loads a checkpoint and produce more "
                    "configurations with the same parameters and proposer.")

    args = initialize(commands,
                      name="isle",
                      description="Base utility program of Isle. Dispatches to sub-commands. "
                      "Use -h for a sub-command to get more information.",
                      epilog="See https://github.com/jl-wynen/isle",
                      subdescriptions=descriptions)

    try:
        mains[commands.index(args.cmd)](args)
    except:
        getLogger("isle").exception("Failed to execute commad %s", args.cmd)
        exit(1)
