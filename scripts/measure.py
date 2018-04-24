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

    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    ensemble = cns.ensemble.importEnsemble(args.ensemble)
    cfgFile = ensemble.name+".h5"

    # Keep configuration h5 file closed as much as possible during measurements
    # First find find out all the configurations.
    with h5.File(cfgFile, "r") as cfgF:
        configurations = sorted(cfgF["/configuration"],key=lambda x: int(x))

    measurements = [
        (1, cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa), "/logdet"),
        (1, cns.meas.TotalPhi(), "/field"),
        (1, cns.meas.Action(),"/"),
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

    print("Performing measurements...")
    
    for configuration, iteration in zip(configurations, range(len(configurations))):
        # Then, only keep the file open during reading of the data from that configuration.
        with h5.File(cfgFile, "r") as cfgF:
            phi = cfgF["configuration"][configuration]["phi"][()]
            action = cfgF["configuration"][configuration]["action"][()]
        # Finally, measure:
        for frequency, measurement, _ in measurements + [(100, cns.meas.Progress("Measurement", len(configurations)), "/monte_carlo")]:
            if iteration % frequency == 0:
                measurement(phi, act=action, itr=iteration)

    print("Saving measurements...")
    with h5.File(ensemble.name+".measurements.h5",
                 "w" if args.overwrite else "w-") as measFile:
        for _, meas, path in measurements:
            meas.save(measFile, path)

def parseArgs():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(description="""
    Investigate the ergodicity problem.
    """)
    parser.add_argument("ensemble", help="Ensemble module")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
