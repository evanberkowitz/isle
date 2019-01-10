"""! \file
Utilities for command line interfaces.

The default interface can be set up using isle.cli.init().
More control is available through the lower level functions.
"""

from abc import ABCMeta, abstractmethod
import argparse
import contextlib
import logging
import random
import shutil
import sys
import time

import isle
from . import fileio


# the active progress bar
# Yes, yes a global variable but we need singletons here!
_activeBar = None

## Unicode ellipsis string.
ELLIPSIS = "…"



########################################################################
# General stuff
#

def terminalWidth():
    """!
    Return the current number of columns of the terminal.

    \note This does not give the proper size if `sys.stdout` is
          redirected to a file.
    """
    return shutil.get_terminal_size().columns

def stderrConnectedToTerm():
    """!Return True if stderr is connected to a terminal, False otherwise."""
    return sys.stderr.isatty()



########################################################################
# Progress bars
#

class ETA:
    """!
    Estimate the time of arrival for iterations.

    The ETA is computed from a linear regression to the starting time
    and current time (time at execution of the __call__ method).
    This is not very stable for strongly changing durations of individual iterations
    but gives a good estimate for mostly stable durations.

    The start time is only set after the first iteration
    (calling __call__ with current > 0).
    This is because the first iteration might perform some expensive setup
    operations the following iterations do not have to repeat.
    In such a case, using the time at current=0 would introduce a bias
    to the ETA.

    Given an initial iteration xi which was started at time ti
    and a current iteration xc and current time tc
    the estimated final time (ETA) is
        tf = mc * (xf - xi) + ti
    where
        mc = (tc - ti) / (xc - xi).
    """

    def __init__(self, target):
        """!Initialize with a given target iteration number."""

        if target <= 1:
            raise ValueError(f"Target iteration of ETA must be > 1, got {target}")

        self.targetIteration = target
        self._ti = None  # initial time
        self._xi = None  # iteration when initial time was measured

    def __call__(self, current):
        r"""!
        Estimate the time of arrival given a current iteration.
        \param current Iteration number the loop is currently at.
        \returns - Estimated time of arrival in seconds since epoch.
                 - `None` if no starting time has been set yet or `current`
                   is below or equal to starting iteration.
        """

        # can't estimate time yet
        if self._ti is None:
            # initial time is time after first iteration or later
            if current > 0:
                self._ti = time.time()
                self._xi = current
            return None

        # The method might be called multiple times during the same iteration.
        # But cannot estimate while still at initial iteration (xi).
        if current <= self._xi:
            return None

        # do linear regression
        tc = time.time()
        return (tc-self._ti)/(current-self._xi) * (self.targetIteration-self._xi) + self._ti

    def reset(self, target):
        """!Re-initialize with a given target iteration number."""

        if target <= 1:
            raise ValueError(f"Target iteration of ETA must be > 1, got {target}")

        self.targetIteration = target
        self._ti = None  # initial time
        self._xi = None  # iteration when initial time was measured




class Progressbar(metaclass=ABCMeta):
    r"""!
    Abstract base class for progress bars.

    \warning Any and all terminal output must be made through TerminalProgressbar.write()
             while a progress bar is active!
             Otherwise it will interfere with the progress bar and might get erased
             when the bar is redrawn.
             When set up correctly (via `setupLogging()`), loggers handle output properly.
    """

    def __init__(self, message=""):

        r"""!
        Construct a new base progress bar.

        \warning Never have more than one active instance at the same time,
                 they interfere with each other.

        \param message A string that is displayed in front of the actual bar
                       and realted information.
        """

        self._message = message
        self._startTime = time.time()

    @abstractmethod
    def advance(self, amount=1):
        """!Advance the bar by a given amount, redraws the bar."""
        raise NotImplementedError()

    @abstractmethod
    def clear(self):
        """!Clear the current line of output."""
        raise NotImplementedError()

    @abstractmethod
    def redraw(self):
        """!Clear the current bar and draw a new one."""
        raise NotImplementedError()

    @abstractmethod
    def draw(self):
        """!Draw the bar."""
        raise NotImplementedError()

    def write(self, msg):
        r"""!
        Write a message to the terminal.

        \attention Use exclusively this function to write to the terminal
                   while a progress bar is being displayed.
        """
        sys.stderr.write(msg)

    def finalize(self):
        """!Remove bar from screen and show a message showing the run time."""
        sys.stderr.write(f"{self._message} finished after {(time.time()-self._startTime):.1f}s "
                         "at "+time.strftime("%H:%M:%S", time.localtime())+" \n")




class TerminalProgressbar(Progressbar):
    r"""!
    A progress bar that is shown in the terminal via sys.stdio.

    Needs to modify the current line of output in order to animate the progress bar.
    This is only possible when the output is indeed connected to a terminal.
    If sys.stderr is piped into a file, this class cannot operate properly.

    \warning Any and all terminal output must be made through TerminalProgressbar.write()
             while a progress bar is active!
             Otherwise it will interfere with the progress bar and might get erased
             when the bar is redrawn.
             When set up correctly (via `setupLogging()`), loggers handle output properly.
    """

    # Escape sequence to clear the current line right of the cursor.
    _CLEAR = "[K"
    # Escape sequence to move the cursor to the beginning of the current line.
    _FRONT = "[G"


    class _FillingBar:
        """!A bar that fills up over time approaching a target."""

        def __init__(self, length, filledChar, emptyChar):
            self.length = length
            self._filledChar = filledChar
            self._emptyChar = emptyChar

        def construct(self, current, target):
            """!Construct a string representing the bar from a current and target fill status."""
            nfilled = int(current / target * self.length)
            return self._filledChar*nfilled + self._emptyChar*(self.length-nfilled)


    class _OscillatingBar:
        """!A 'bar' that oscillates randomly for cases where no target is known."""

        # element [i][j] transitions from height i to height j
        _PIECES = [["⠤", "⠴", "⠼"],
                   ["⠲", "⠒", "⠚"],
                   ["⠹", "⠙", "⠉"]]

        # extended 4-dot pieces, looks a bit odd because the lower most dots are very low
        # _PIECES = [
        #     ["⣀", "⣠", "⣰", "⣸"],
        #     ["⢤", "⠤", "⠴", "⠼"],
        #     ["⢲", "⠲", "⠒", "⠚"],
        #     ["⢹", "⠹", "⠙", "⠉"]
        # ]

        def __init__(self, length):
            self.length = length
            self._rng = random.Random()
            self._currentHeight = 1
            self._barStr = self._PIECES[self._currentHeight][self._currentHeight]*length

        def construct(self, _current, _target):
            """!Construct a string representing the bar, arguments are ignored."""

            h = self._rng.randint(0, 2)  # new height
            # shift by one and add new element
            self._barStr = self._barStr[1:]+self._PIECES[self._currentHeight][h]
            self._currentHeight = h
            return self._barStr

    def __init__(self, target, message="", barLength=40,
                 barChar="≣", emptyChar="-"):

        r"""!
        Construct a new progress bar.

        \warning Never have more than one active instance at the same time,
                 they interfere with each other.

        \param target Targeted number of iterations.
                      May be `None` in which case the bar indicates only
                      general progress without showing how far away the goal is.
        \param message A string that is displayed in front of the actual bar
                       and realted information.
        \param barLength Number of characters the bar itself occupies in the terminal.
                         Does not include ETA and iteration counter.
        \param barChar Single character to use for the filled portion of the bar.
        \param emptyChar Single character to use for the not yet filled portion of the bar.
        """

        super().__init__(message)
        self._target = target

        self._current = 0
        # ETA's __init__ makes sure that target > 0
        self._eta = ETA(target) if target else None
        self._bar = self._FillingBar(barLength, barChar, emptyChar) \
            if target \
               else self._OscillatingBar(barLength)

        # format string for text after bar
        if target:
            targetStr = f"{target:d}"
            self._postFmt = "] ({:"+str(len(targetStr))+"d}/"+targetStr+") "
        else:
            self._postFmt = "] ({:3d}/?)"


    def advance(self, amount=1):
        """!Advance the bar by a given amount, redraws the bar."""
        self._current += amount
        self.redraw()

    def clear(self):
        """!Clear the current line of output."""
        sys.stderr.write(self._FRONT+self._CLEAR)

    def redraw(self):
        """!Clear the current bar and draw a new one."""
        # enough to go to front, don't need to clear the line
        sys.stderr.write(self._FRONT)
        self.draw()

    def draw(self):
        """!Draw the bar into the terminal at the current cursor position."""

        # format string before bar
        if self._eta:
            eta = self._eta(self._current)
            pre = "  ETA: " \
                + (time.strftime("%H:%M:%S", time.localtime(eta)) if eta else "??:??:??") \
                + " ["
        else:
            pre = " ["
        # format string after bar
        post = self._postFmt.format(self._current)

        # current total length of a line in the terminal
        lineLength = terminalWidth()
        # length available for messages
        availLength = lineLength - len(pre) - len(post) - self._bar.length

        if availLength < 10:
            # not enough space to display everything, only show message
            out = self._message[:lineLength-4]+" ["+ELLIPSIS+"]"

        else:
            # add spaces after message or abbreviate message depending on availLength
            spaceLength = availLength - len(self._message)
            msg = self._message+" "*spaceLength \
                if spaceLength >= 0 \
                   else self._message[:availLength-1]+ELLIPSIS

            # construct the full output string
            out = msg+pre+self._bar.construct(self._current, self._target)+post

        sys.stderr.write(out)
        sys.stderr.flush()

    def write(self, msg):
        r"""!
        Write a message to the terminal.

        Clears the current progress bar on screen, writes the message,
        appends a newline if needed and redraws the bar.

        \attention Use exclusively this function to write to the terminal
                   while a progress bar is being displayed.
        """

        self.clear()
        if not msg.endswith("\n"):
            sys.stderr.write(msg+"\n")
        else:
            sys.stderr.write(msg)
        self.draw()

    def finalize(self):
        """!Remove bar from screen and print a message showing the run time."""
        self.clear()
        sys.stderr.write(f"{self._message} finished after {(time.time()-self._startTime):.1f}s "
                         "at "+time.strftime("%H:%M:%S", time.localtime())+" \n")




class FileProgressbar(Progressbar):
    r"""!
    A progress indicator that writes individual update messages.

    This is not really a progress 'bar' as it only prints simple messages
    indicating progress, not an animated bar.
    Is still writes to `sys.stderr` though, not directly to a file.
    Use this class if `sys.stderr` is not connected to a terminal.

    \warning Even though normal output does not interfere with this progress bar,
             it is still better to use FileProgressbar.write() instead of
             plain `print()` for uniformity with TerminalProgressbar.
             When set up correctly (via `setupLogging()`), loggers handle output properly.
    """

    def __init__(self, target, message="", updateRate=1):

        r"""!
        Construct a new progress bar.

        \warning Never have more than one active instance at the same time,
                 they interfere with each other.

        \param target Targeted number of iterations.
                      May be `None` in which case the bar indicates only
                      general progress without showing how far away the goal is.
        \param message A string that is displayed in front of the actual bar
                       and realted information.
        \param updateRate The bar is only redrawn after updateRate number of steps.
        """

        super().__init__(message)

        self._target = target
        self._updateRate = updateRate

        self._current = 0
        self._lastUpdated = -updateRate
        # ETA's __init__ makes sure that target > 0
        self._eta = ETA(target) if target else None

        # format string for a counter
        if target:
            targetStr = f"{target:d}"
            self._counterFmt = " ({:"+str(len(targetStr))+"d}/"+targetStr+") "
        else:
            self._counterFmt = " ({:3d}/?)"


    def advance(self, amount=1):
        """!Advance the bar by a given amount, redraws the bar."""
        self._current += amount
        if self._current - self._updateRate >= self._lastUpdated:
            self.redraw()
            # go to nearest multiple of updateRate less than current
            self._lastUpdated = (self._current // self._updateRate)*self._updateRate

    def clear(self):
        """!Do nothing, cannot easily erase content from files."""

    def redraw(self):
        """!Just call draw()."""
        self.draw()

    def draw(self):
        """!Print progress information."""

        # format progress indication string
        if self._eta:
            eta = self._eta(self._current)
            progStr = "  ETA: " \
                + (time.strftime("%H:%M:%S", time.localtime(eta)) if eta else "??:??:??")
        else:
            progStr = ""
        # format string after bar
        progStr += self._counterFmt.format(self._current)

        sys.stderr.write(self._message+progStr+"\n")
        sys.stderr.flush()

    def write(self, msg):
        r"""!
        Write out a message.

        Just redirects to sys.stderr.

        \attention Use exclusively this function to write to the terminal
                   while a progress bar is being displayed.
        """

        sys.stderr.write(msg)

    def finalize(self):
        """!Print a message showing the run time."""
        sys.stderr.write(f"{self._message} finished after {(time.time()-self._startTime):.1f}s "
                         "at "+time.strftime("%H:%M:%S", time.localtime())+" \n")




def makeProgressbar(target, message="", updateRate=1):
    r"""!
    Construct a Progressbar.

    Selects either TerminalProgressbar or FilePRogressbar
    depending on whether `sys.stderr` is connected to a terminal
    or not.

    \warning Never have more than one active instance at the same time,
             they interfere with each other.

    \param target Targeted number of iterations.
    \param message String to display with the progressbar.
    \param updateRate The bar is only redrawn after updateRate number of steps.
                      Only used when writing to file.
    \returns A new Progressbar instance.
    """

    if stderrConnectedToTerm():
        return TerminalProgressbar(target, message)

    return FileProgressbar(target, message, updateRate)


@contextlib.contextmanager
def trackProgress(target, message="", updateRate=1):
    r"""!
    A context manager to track progress of an operation via a progress bar.

    Sets up and returns a new progress bar.
    The caller needs to advance that bar themselves.

    \param target Target number of steps to track.
                  Can be `None` in which case the bar only indicates that something,
                  happens, not how far away the goal is.
    \param message Message to display in front of the progress bar.
    \returns A newly constructed instance of Progressbar.
    \throws RuntimeError is a progress bar is already active.
    """

    global _activeBar
    if _activeBar is not None:
        logging.getLogger(__name__).error("Cannot construct a new progress bar, "
                                          "another one is already active.")
        raise RuntimeError("A progress bar is already active.")

    try:
        _activeBar = makeProgressbar(target, message, updateRate)
        yield _activeBar
        _activeBar.finalize()  # success => clean up

    except:
        # failure => leave bar visible and advance a line
        sys.stderr.write("\n")
        raise

    finally:
        # in any case the bar is now done
        _activeBar = None

def progressRange(start, stop=None, step=1, message="", updateRate=1):
    r"""!
    Like built in `range()` but indicates progress using a Progressbar.

    Parameters `start`, `stop`, and `step` behave the same way as for
    the built in `range()` generator.
    `message` is displayed in front of the progress bar.
    """

    # mimic behavior of built in range
    if stop is None:
        stop = start
        start = 0

    with trackProgress(stop-start, message, updateRate) as pbar:
        for cur in range(start, stop, step):
            yield cur
            # advance only up to stop
            # If stop-start is not a multiple of step, advance(step) would overshoot.
            pbar.advance(min(step, stop-cur))



########################################################################
# Logging
#

class ProgressStream:
    """!
    A barbones output stream that writes to `sys.stderr` or
    directs the output through a progress bar if one is active.
    """

    def write(self, msg):
        """!Write msg to `sys.stderr`."""

        if _activeBar is not None:
            # The c++ logger sends spurious empty lines,
            # just gobble them up.
            if msg.strip():
                _activeBar.write(msg)
        else:
            sys.stderr.write(msg)

class ColorFormatter(logging.Formatter):
    """!
    A logging formatter that uses colors for loglevel and logger name.

    Colors are encoded using ANSI escape sequences. If they are not supported,
    the output shows extra characters. You should therefore pick a different
    formatter if you write to file.
    """

    ## Colorized level names.
    LEVELNAMES = {
        logging.DEBUG: "[94mDEBUG[0m   ",
        logging.INFO: "INFO    ",
        logging.WARNING: "[33mWARNING[0m ",
        logging.ERROR: "[31mERROR[0m   ",
        logging.CRITICAL: "[91mCRITICAL[0m",
    }

    def format(self, record):
        "!Format a record using colors."
        # Hack the colors into the record itself and let super do the heavy lifting.
        record.levelname = self.LEVELNAMES[record.levelno]
        record.name = f"[1m{record.name}[0m"
        return super().format(record)


def setupLogging(logfile=None, verbosity=0):
    r"""!
    Set up Python's logging framework.

    The root logger is set up to output to terminal and file.
    The former shows colored output if `sys.stderr` is connected to a terminal
    and is aware of Progressbar.

    \param logfile Write log to this file.
                   If `None`, no file output is performed.
    \param verbosity Set logging level.
                     The minimum level for each handler is
                     `verbosity` | terminal | file
                     ----------- | -------- | ----
                           0     | WARNING  | INFO
                           1     | INFO     | INFO
                           2     | DEBUG    | DEBUG

    \throws RuntimeError if this function is called more than once.
                         It is safe to discard this exception,
                         logging will still be set up properly.
    """

    if setupLogging.isSetUp:
        logging.getLogger(__name__).error("Called setupLogging a second time."
                                          "This function must be called *exactly* once.")
        raise RuntimeError("Logging already set up")

    if verbosity > 2:
        # can't be any noisier than that
        verbosity = 2

    # need at least this level so all messages get out
    minLoglevel = logging.DEBUG if verbosity == 2 else logging.INFO

    # No need for keeping track of threads in Python.
    # In C++, all bets are off.
    logging.logThreads = 0

    # configure the root logger
    logger = logging.getLogger("")
    logger.setLevel(minLoglevel)

    if logfile:
        # output to file at least at level INFO and w/o colors
        fh = logging.FileHandler(logfile, "w")
        fh.setLevel(minLoglevel)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s in %(name)s: %(message)s",
                                          "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

    # and output to STDERR based on verbosity and possibly w/ colors
    ch = logging.StreamHandler(stream=ProgressStream())
    ch.setLevel((logging.WARNING, logging.INFO, logging.DEBUG)[verbosity])
    if stderrConnectedToTerm():
        ch.setFormatter(ColorFormatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
                                       "%H:%M:%S"))
    else:
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s in %(name)s: %(message)s",
                                          "%H:%M:%S"))
    logger.addHandler(ch)

    # done => Never run this code again!
    setupLogging.isSetUp = True

setupLogging.isSetUp = False



########################################################################
# Argument parsing
#

def _sliceArgType(arg):
    """!Parse an argument in slice notation"""
    return slice(*map(lambda x: int(x) if x else None, arg.split(":")))

def addDefaultArgs(parser, log="none"):
    """!Add default arguments common to all commands."""
    parser.add_argument("--version", action="version",
                        version=f"Isle {isle.__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Make output more verbose, stacks.")
    parser.add_argument("--log", default=log,
                        help="Specify log file name. Set to none to not write log file.")
    return parser

def addHMCArgs(parser):
    """!Add arguments for HMC to parser."""
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
                        help="Checkpoint frequency relative to measurement frequency. "
                        "Defaults to 0.")
    parser.add_argument("--no-checks", action="store_true",
                        help="Disable consistency checks")
    return parser

def addMeasArgs(parser):
    """!Add arguments for measurements to parser."""
    parser.add_argument("-i", "--input", help="Input file",
                        type=fileio.pathAndType, dest="infile")
    parser.add_argument("-o", "--output", help="Output file",
                        type=fileio.pathAndType, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file.")
    parser.add_argument("-n", type=_sliceArgType, default=slice(-1),
                        help="Select which trajectories to process. "
                        "In slice notation without spaces.")
    return parser

def addShowArgs(parser):
    """!Add arguments for reporting to parser."""

    reporters = ["overview", "lattice", "correlator"]

    class _ReportAction(argparse.Action):
        """!custom action to parse reporters."""
        def __call__(self, parser, namespace, values, option_string=None):
            if "all" in values:
                setattr(namespace, self.dest, reporters)
            else:
                setattr(namespace, self.dest, values.split(","))

    parser.add_argument("input", help="Input file",
                        type=fileio.pathAndType)
    parser.add_argument("-r", "--report", action=_ReportAction, metavar="", default=["overview"],
                        help="Comma separated list of reporters to use. Allowed values are ["
                        +",".join(reporters)+",all] Defaults to overview.")
    return parser

def parseArguments(cmd, name, description, epilog, subdescriptions):
    r"""!
    Parse command line arguments.

    \param cmd Command(s) to parse arguments for.
               Supported commands are `'hmc', 'meas'`.
               Can be a string or a list of strings.
               In case of a string, the arguments for that command are registered
               for the base program.
               In case of a list, a subcommand is added for each element of the list.
    \param name Name of the program for the help message.
    \param description Short description of the program for the help message.
    \param epilog Argparse epilog message to show below other help messages.
    \param subdescriptions List of descriptions for subparsers in case cmd contains
                           more than one command.
    """

    cmdArgMap = {"hmc": addHMCArgs,
                 "meas": addMeasArgs,
                 "show": addShowArgs}
    defaultLog = {"hmc": "isle.hmc.log",
                  "meas": "isle.meas.log",
                  "show": "none"}

    if cmd is not None:
        if isinstance(cmd, str):
            # just one command
            parser = argparse.ArgumentParser(prog=cmd if name is None else name,
                                             description=description,
                                             epilog=epilog)
            cmdArgMap[cmd](addDefaultArgs(parser, log=defaultLog[cmd]))

        else:
            # multiple commands
            parser = argparse.ArgumentParser(prog=name,
                                             description=description,
                                             epilog=epilog)
            addDefaultArgs(parser)
            subp = parser.add_subparsers(title="Commands", dest="cmd", required=True)
            for i, subcmd in enumerate(cmd):
                subParser = subp.add_parser(subcmd, epilog=epilog,
                                            description=subdescriptions[i]
                                            if subdescriptions else None)
                cmdArgMap[subcmd](addDefaultArgs(subParser, log=defaultLog[subcmd]))
        args = parser.parse_args()

    else:
        args = None

    return args



########################################################################
# The one function to control all the rest.
#

def init(cmd=None, name=None, description=None, epilog=None, subdescriptions=None):
    r"""!
    Initialize command line interface.

    \param cmd See isle.cli.parseArguments().
    \param name See isle.cli.parseArguments().
    \param description See isle.cli.parseArguments().
    \param epilog See isle.cli.parseArguments().
    \param subdescriptions See isle.cli.parseArguments().
    """

    args = parseArguments(cmd, name, description, epilog, subdescriptions)
    setupLogging(None if args.log.lower() == "none" else args.log,
                 verbosity=args.verbose)

    return args
