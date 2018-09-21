"""!
Utilities for command line interfaces.

The default interface can be set up using isle.cli.init().
More control is available through the lower level functions.
"""

import argparse

import isle
from . import fileio

def _sliceArgType(arg):
    "!Parse an argument in slice notation"
    return slice(*map(lambda x: int(x) if x else None, arg.split(":")))

def makeDefaultParser(name, description):
    "!Return a new default parser object."
    parser = argparse.ArgumentParser(prog=name,
                                     description=description)
    parser.add_argument("--version", action="version",
                        version=isle.__version__)
    return parser

def addHMCArgs(parser):
    "!Add arguments for HMC to parser."
    parser.add_argument("-i", "--input", help="Input file.",
                        type=fileio.pathAndType, dest="infile")
    parser.add_argument("-o", "--output", help="Output file",
                        type=fileio.pathAndType, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("-c", "--continue", action="store_true",
                        help="Continue from previous run", dest="cont")
    parser.add_argument("-n", "--nproduction", type=int,
                        help="Number of production trajectories")
    parser.add_argument("-t", "--ntherm", type=int, default=0,
                        help="Number of thermalization trajectories. Defaults to 0.")
    parser.add_argument("-s", "--save-freq", type=int, default=1,
                        help="Frequency with which configurations are saved. Defaults to 1.")
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help="Checkpoint frequency relative to measurement frequency. Defaults to 0.")
    parser.add_argument("--no-checks", action="store_true",
                        help="Disable consistency checks")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Make output more verbose, stacks.")

def addMeasArgs(parser):
    "!Add arguments for measurements to parser."
    parser.add_argument("-i", "--input", help="Input file",
                        type=fileio.pathAndType, dest="infile")
    parser.add_argument("-o", "--output", help="Output file",
                        type=fileio.pathAndType, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file.")
    parser.add_argument("-n", type=_sliceArgType, default=slice(-1),
                        help="Select which trajectories to process. In slice notation without spaces.")

def addReportArgs(parser):
    "!Add arguments for measurements to parser."
    parser.add_argument("input", help="Input file",
                        type=fileio.pathAndType)

def parseArguments(cmd, name, description):
    """!
    Parse command line arguments.

    \param cmd Command(s) to parse arguments for.
               Supported commands are `'hmc', 'meas'`.
               Can be a string or a list of strings.
               In case of a string, the arguments for that command are registered
               for the base program.
               In case of a list, a subcommand is added for each element of the list.
    \param name Name of the program for the help message.
    \param description Short description of the program for the help message.
    """

    cmdArgMap = {"hmc": addHMCArgs,
                 "meas": addMeasArgs,
                 "report": addReportArgs}

    if cmd is not None:
        if isinstance(cmd, str):
            parser = makeDefaultParser(cmd if name is None else name,
                                       description)
            cmdArgMap[cmd](parser)

        else:
            parser = makeDefaultParser(name, description)
            subp = parser.add_subparsers(title="Commands")
            for subcmd in cmd:
                cmdArgMap[subcmd](subp.add_parser(subcmd))
        args = parser.parse_args()

    else:
        args = None

    return args


def init(cmd=None, name=None, description=None):
    """!
    Initialize command line interface.

    \param cmd See isle.cli.parseArguments().
    \param name See isle.cli.parseArguments().
    \param description See isle.cli.parseArguments().
    """

    args = parseArguments(cmd, name, description)

    return args
