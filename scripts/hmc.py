#!/usr/bin/env python3
"""!
Basic HMC without measurements.

\todo document

needs config and chkpt names to be just an int: itr
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

    if not args.cont and ensemble.nTherm > 0:
        print("thermalizing")
        phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.thermalizer,
                          ensemble.nTherm, rng,
                          [
                              (ensemble.nTherm/10,
                               cns.meas.Progress("Thermalization", ensemble.nTherm)),
                          ],
                          [(20, cns.checks.realityCheck)])

    print("running production")
    phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.proposer,
                      ensemble.nProduction, rng,
                      [
                          (10, cns.meas.WriteConfiguration(str(args.outfile[0]),
                                                           "/configuration/{itr}",
                                                           100,
                                                           "/checkpoint/{itr}")),
                          (500, cns.meas.Progress("Production", ensemble.nProduction)),
                      ],
                      [(20, cns.checks.realityCheck)],
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
    parser.add_argument("infile", help="", type=cns.fileio.pathAndType)
    parser.add_argument("-o", "--output", help="Output file",
                        type=Path, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    parser.add_argument("-c", "--continue", action="store_true",
                        help="Continue from previous run", dest="cont")
    return parser.parse_args()

if __name__ == "__main__":
    main(_parseArgs())
