#!/usr/bin/env python3
"""!
Investigate the ergodicity problem.
"""

from pathlib import Path
import argparse

import h5py as h5

import core
core.prepare_module_import()
import cns
import cns.meas

def main(args):
    """!Run HMC and measurements."""

    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    ensemble = cns.ensemble.importEnsemble(args.ensemble)
    cfgFile = ensemble.name+".h5"
    with h5.File(cfgFile, "w" if args.overwrite else "w-") as cfgf:
        cns.ensemble.writeH5(args.ensemble, cfgf)

    measurements = [
        (1, cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa), "/logdet"),
        (1, cns.meas.TotalPhi(), "/field"),
        (1, cns.meas.Action(),"/"),
        (1, cns.meas.AcceptanceRate(), "/monte_carlo"),
        (1, cns.meas.Timer(), "/monte_carlo"),
        (100, cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                ensemble.mu, ensemble.sigmaKappa,
                                                cns.Species.PARTICLE),
         "/correlation_functions/single_particle"),
        (100, cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                ensemble.mu, ensemble.sigmaKappa,
                                                cns.Species.HOLE),
         "/correlation_functions/single_hole"),
        (100, cns.meas.Phase(), "/")
    ]

    print("thermalizing")
    phi = cns.hmc.hmc(ensemble.initialConfig, ensemble.hamiltonian,
                      ensemble.thermalizer,
                      ensemble.nTherm,
                      ensemble.rng,
                      [
                          (ensemble.nTherm/10, cns.meas.Progress("Thermalization", ensemble.nTherm)),
                      ],
                      [(20, cns.checks.realityCheck)])

    print("running production")
    phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.proposer,
                      ensemble.nProduction, ensemble.rng,
                      [
                          ( 10, cns.meas.WriteConfiguration(ensemble.name+".h5",
                                                            "/configuration/{itr}")),
                          (500, cns.meas.Progress("Production", ensemble.nProduction)),
                      ] + ([(freq, meas) for freq, meas, _ in measurements] if args.measure else []),
                      [(20, cns.checks.realityCheck)])

    if args.measure:
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
    parser.add_argument("--measure", action="store_true",
                        help="Perform additional measurements inline.")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
