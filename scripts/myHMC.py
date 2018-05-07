#!/usr/bin/env python3

r"""!
\file
\brief Basic HMC without measurements.
\ingroup scripts

Runs HMC for an ensemble and saves configurations to disk.
Can start a new calculation or continue from an old one.
Use command line argument `-h` or `--help` to get information on supported arguments.

### Input
The only positional argument to this script is a file containing an ensemble.
This file can be either a Python module or an HDF5 file with a dataset called 'ensemble'
in the root.
In case of a continuation run, only HDF5 files are allowed and must probide a checkpoint
to start off of. The format of HDF5 files for continuation runs must match the format
descibed under 'Output' below.

An ensemble must provide the following variables:
- `hamiltonian`: An instance of `cns.Hamiltonian` passed to cns.hmc.hmc.
- `thermalizer`: A proposer used during thermalization.
- `proposer`: A proposer used during production.
- `rng`: A random number generator. Is only used for new runs. For continuation runs,
         the rng is loaded from a checkpoint.
- `initialConfig`: Initial gauge configuration. Is only used for new runs.
                   For continuation runs, it is is loaded from a checkpoint.
- `name`: A string name of the ensemble. Only needed if no name for the output file is given.

### Output
A single HDF5 file containing:
- `ensemble`: Dataset holding the text of the ensemble module.
- `configuration`: Group holding groups labeled with just the trajectory index,
                   no other characters. Each subgroup contains datasets holding
                   configuration (`phi`), the action with that configuration (`action`),
                   and whether the trajectory was accepted (`acceptance`).
- `checkpoint`:  Group containing groups labeled with just the trajectory index, no
                 other charaters. Each subgroup contains a link to a configuration
                 and a group holding the RNG state.

See cns.meas.writeConfiguration.WriteConfiguration for more information on the output.

If no output file is given, its name is either deduced from the ensemble name for new runs
or it is identical to the input file for continuation runs.
`hmc.py` will not overwrite any old data unless the `--overwrite` command line argument
is given.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import h5py as h5

import core
core.prepare_module_import()
import cns
import cns.meas

def main(args):
    """!Run HMC."""

    ensemble, phi, rng, itrOffset = _setup(args)
    checks = [] if args.no_checks else [(20, cns.checks.realityCheck)]

    if not args.cont and args.ntherm > 0:
        print("thermalizing")
        phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.thermalizer,
                          args.ntherm, rng,
                          [
                           (1,cns.meas.AcceptancRate()),
                              (args.ntherm/10,
                               cns.meas.Progress("Thermalization", args.ntherm)),
                          ],
                          checks)

    print("running production")
    phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.proposer,
                      args.nproduction, rng,
                      [
                          (args.save_freq,
                           cns.meas.WriteConfiguration(str(args.outfile[0]),
                                                       "/configuration/{itr}",
                                                       args.checkpoint_freq,
                                                       "/checkpoint/{itr}")),
                          (500, cns.meas.Progress("Production",
                                                  args.nproduction+itrOffset-1,
                                                  itrOffset)),
                      ],
                      checks,
                      itrOffset)

def _setup(args):
    """!Setup the HMC run; load input data and prepare output file."""

    # setup environment
    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    ensemble, ensembleText = cns.ensemble.load(args.infile[0].stem, args.infile[0])

    if args.outfile is None:
        if args.cont:
            args.outfile = args.infile[0]
        else:
            args.outfile = ensemble.name+".h5"
        print("Set output file to '{}'".format(args.outfile))
    args.outfile = cns.fileio.pathAndType(args.outfile)

    # prepare output file
    # delete if necessary and save ensemble
    if args.outfile[0].exists():
        if (not args.cont) or (args.cont and args.infile[0] != args.outfile[0]):
            if args.overwrite:
                args.outfile[0].unlink()  # just erase the file so we can safely write to it
                with h5.File(args.outfile[0], "w") as cfgf:
                    cns.ensemble.saveH5(ensembleText, cfgf)
            else:
                print("Error: Output file already exists."
                      +" Use --overwrite to overwrite it or do acontinuation run with identical input and output files.")
                sys.exit(1)
        # else: continuation run with equal infile and outfile
        # Don't do anything here, just write to the file.
    else:
        with h5.File(args.outfile[0], "w") as cfgf:
            cns.ensemble.saveH5(ensembleText, cfgf)

    return (ensemble, *_initialState(args, ensemble))

def _initialStateFromHDF5(fname, overwrite):
    """!Load initial state (config, rng, and trajectory index) from last checkpoint."""

    with h5.File(fname, "r+" if overwrite else "r") as h5f:
        # get names (int) of all configurations and checkpoints in ascending order
        cfgNames = sorted(map(int, h5f["configuration"].keys()))
        chkptNames = sorted(map(int, h5f["checkpoint"].keys()))

        # check for consistency
        if chkptNames[-1] < cfgNames[-1]:
            if overwrite:
                print(f"Warning: last checkpoint is less than last configuration in file {fname}:"
                      +" {} vs {}. Overwriting old configurations.".format(chkptNames[-1], cfgNames[-1]))
                # erase configuration newer than last checkpoint
                for i in range(cfgNames.index(chkptNames[-1])+1, len(cfgNames)):
                    cfgGrp = h5f["configuration"]
                    del cfgGrp[str(cfgNames[i])]
            else:
                print(f"Error: last checkpoint is less than last configuration in file {fname}:"
                      +" {} vs {}.".format(chkptNames[-1], cfgNames[-1])
                      +" Use --overwrite to replace old configurations after last checkpoint.")
                sys.exit(1)
        elif chkptNames[-1] > cfgNames[-1]:
            print(f"Error: last checkpoint is older than last configuration in file {fname}:"
                  +" {} vs {}.".format(chkptNames[-1], cfgNames[-1]))
            sys.exit(1)

        return cns.Vector(np.array(h5f["checkpoint"][str(chkptNames[-1])]["cfg"]["phi"])),\
               cns.random.readStateH5(h5f["checkpoint"][str(chkptNames[-1])]["rng_state"]),\
               chkptNames[-1]+1  # +1 because we want to point to next trajectory

def _initialState(args, ensemble):
    """!Get the initial state for HMC tuple of (config, rng, trajectory index)."""
    if args.cont:
        if args.infile[1] != cns.fileio.FileType.HDF5:
            print("Error: Can only load initial state for continuation run from HDF5 file."
                  +"Given file type is {}".format(args.infile[1]))
        return _initialStateFromHDF5(str(args.infile[0]), args.overwrite)
    else:
        return ensemble.initialConfig, ensemble.rng, 0

def _parseArgs():
    """!Parse command line arguments."""
    parser = argparse.ArgumentParser(description="""
    Basic HMC without any measurements.
    """)
    parser.add_argument("infile", help="Input file. Python module or HDF5 file",
                        type=cns.fileio.pathAndType)
    requiredGrp = parser.add_argument_group("required named arguments")
    requiredGrp.add_argument("-n", "--nproduction", type=int, required=True,
                             help="Number of production trajectories")
    parser.add_argument("-o", "--output", help="Output file",
                        type=Path, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("-c", "--continue", action="store_true",
                        help="Continue from previous run", dest="cont")
    parser.add_argument("-t", "--ntherm", type=int, default=0,
                        help="Number of thermalization trajectories."
                        +" Is ignored if doing a continuation run. Defaults to 0.")
    parser.add_argument("-s", "--save-freq", type=int, default=10,
                        help="Frequency with which configurations are saved. Defaults to 10.")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                        help="Checkpoint frequency relative to measurement frequency. Defaults to 100.")
    parser.add_argument("--no-checks", action="store_true",
                        help="Disable consistency checks")
    return parser.parse_args()

if __name__ == "__main__":
    main(_parseArgs())
