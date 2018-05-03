#!/usr/bin/env python3
"""!
Run measurements on existing configurations.
"""

from pathlib import Path
import argparse

import h5py as h5

import core
core.prepare_module_import()
import cns
import cns.meas


def main(args):
    """!Run measurements from a file with saved configurations."""

    ensemble = setup(args)

    measurements = [
        (1, cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa), "/logdet"),
        (1, cns.meas.TotalPhi(), "/field"),
        (1, cns.meas.Action(), "/"),
        (1, cns.meas.ChiralCondensate(1234, 10, ensemble.nt, ensemble.kappaTilde,
                                      ensemble.mu, ensemble.sigmaKappa,
                                      cns.Species.PARTICLE),
         "/"),
        (10, cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                               ensemble.mu, ensemble.sigmaKappa,
                                               cns.Species.PARTICLE),
         "/correlation_functions/single_particle"),
        (10, cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                               ensemble.mu, ensemble.sigmaKappa,
                                               cns.Species.HOLE),
         "/correlation_functions/single_hole"),
        (10, cns.meas.Phase(), "/"),
    ]

    run(args, measurements)

def run(args, measurements):
    r"""!
    Run Measurements.
    \param args Parsed command line arguments. Encode which configurations are processed.
    \param measurements List of measurements to execute. Each element is a tuple of
                        - Measurement frequency
                        - Measurement object.
                        - Path to location where measurements are saved. Is passed to
                          save function of measurement.
    """

    # Keep configuration h5 file closed as much as possible during measurements
    # First find find out all the configurations.
    with h5.File(args.infile[0], "r") as cfgf:
        configNames = sorted(cfgf["/configuration"], key=int)

    print("Performing measurements...")
    for i, configName in enumerate(configNames[args.n]):
        # read config and action
        with h5.File(args.infile[0], "r") as cfgf:
            phi = cfgf["configuration"][configName]["phi"][()]
            action = cfgf["configuration"][configName]["action"][()]
        # measure
        for frequency, measurement, _ in measurements \
            +[(100, cns.meas.Progress("Measurement", len(configNames)), "")]:

            if i % frequency == 0:
                measurement(phi, act=action, itr=i)

    print("Saving measurements...")
    with h5.File(args.outfile[0], "a") as measFile:
        for _, meas, path in measurements:
            meas.save(measFile, path)

def setup(args):
    r"""!
    Setup the run; load input data and prepare output file.
    \param args Parsed command line arguments.
    \returns The ensemble module.
    """

    # setup environment
    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    ensemble, ensembleText = cns.ensemble.load(args.infile[0].stem, args.infile[0])

    if args.outfile is None:
        args.outfile = cns.fileio.pathAndType(str(args.infile[0]).rsplit(".", 1)[0]+"_meas.h5")

    # prepare output file
    # delete if necessary and save ensemble
    if args.outfile[0].exists():
        if args.overwrite:
            args.outfile[0].unlink()
            with h5.File(args.outfile[0], "w") as cfgf:
                cns.ensemble.saveH5(ensembleText, cfgf)
        # else: Try to write to existing file. Will fail if trying to overwrite a dataset.
    else:
        with h5.File(args.outfile[0], "w") as cfgf:
            cns.ensemble.saveH5(ensembleText, cfgf)

    return ensemble

def parseArgs(argv=None):
    r"""!
    Parse command line arguments.
    \param argv List of arguments to parse. Does not include the file name.
                I.e. use `sys.argv[1:]`.
    \returns Parsed arguments.
    """

    def _sliceArgType(arg):
        return slice(*map(lambda x: int(x) if x else None, arg.split(":")))

    parser = argparse.ArgumentParser(description="""
    Run common measurements.
    """)
    parser.add_argument("infile", help="Input HDF5 file",
                        type=cns.fileio.pathAndType)
    parser.add_argument("-o", "--output", help="Output file",
                        type=cns.fileio.pathAndType, dest="outfile")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file."
                        +" This will delete the entire file, not just the datasets that are overwritten!")
    parser.add_argument("-n", type=_sliceArgType,
                        help="Select which trajectories to process. In slice notation without spaces.")
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parseArgs())
