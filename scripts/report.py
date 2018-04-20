#!/usr/bin/env python3
"""!
Produce a report of measurement output.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os, sys
from pathlib import Path

import core
core.prepare_module_import()
import cns
import cns.meas


def main(argv):
    """!Analyze HMC results."""
    
    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    ensemble = cns.ensemble.importEnsemble(argv[1])
    
    acceptanceRate = cns.meas.AcceptanceRate()
    action = cns.meas.Action()
    logDet = cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa)
    totalPhi = cns.meas.TotalPhi()
    
    particleCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa, cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa, cns.Species.HOLE)


    saved_measurements = [
        (action, "/"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (totalPhi, "/field"),
        (logDet, "/logdet"),
    ]

    with h5.File(ensemble.name+".measurements.h5", "r") as measurementFile:
        for measurement, path in saved_measurements:
            measurement.read(measurementFile[path])

    print("Processing results...")

    np.random.seed(4386)
    additionalThermalizationCut = 0
    finalMeasurement = particleCorrelators.corr.shape[0]
    NBS = 100
    BSLENGTH=finalMeasurement-additionalThermalizationCut

    bootstrapIndices = np.random.randint(additionalThermalizationCut, finalMeasurement, [NBS, BSLENGTH])

    for species, label in zip([particleCorrelators, holeCorrelators], ("PARTICLE", "HOLE")):
        mean = np.mean(species.corr,axis=0)
        std  = np.std(species.corr,axis=0)
        mean_err = np.std(np.array([ np.mean(np.array([species.corr[cfg] for cfg in bootstrapIndices[sample]]), axis=0) for sample in range(NBS) ] ), axis=0)
    
        fig, ax = cns.meas.common.newAxes("Bootstrapped "+str(label)+" Correlator", r"t", r"C")
        time = [ t * ensemble.beta / ensemble.nt for t in range(ensemble.nt) ]
        ax.set_yscale("log")

        for i in range(ensemble.lattice.nx()):
            ax.errorbar(time, np.real(mean[i]), yerr=np.real(mean_err[i]))

        fig.tight_layout()
        
        # ax = species.report()

    ax = acceptanceRate.report(20)
    ax.axvline(ensemble.nTherm, c="k")  # mark thermalization - production border

    ax = action.report(20)
    ax.axvline(ensemble.nTherm, c="k")  # mark thermalization - production border

    ax = logDet.report(cns.Species.PARTICLE)
    ax = logDet.report(cns.Species.HOLE)

    ax = totalPhi.reportPhiHistogram()
    ax = totalPhi.reportPhi()

    plt.show()

if __name__ == "__main__":
    main(sys.argv)
