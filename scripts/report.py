#!/usr/bin/env python3
"""!
Produce a report of measurement output.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import tables as h5

import core
core.prepare_module_import()
import cns
import cns.meas

#LATFILE = "four_sites.yml"  # input lattice
LATFILE = "one_site.yml"  # input lattice
# LATFILE="c20.yml"

NT = 16  # number of time slices
NTHERM = 3000  # number of thermalization trajectories

# model parameters
U = 2
BETA = 3
MU = 0
SIGMA_KAPPA = 1

UTILDE = U*BETA/NT

def main():
    """!Analyze HMC results."""
    
    ensembleName = ".".join(LATFILE.split(".")[:-1])+".nt"+str(NT)
    
    # TODO: also read the lattice from there?
    # load lattice
    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)  # actually \tilde{kappa}
    nx = len(np.array(cns.Matrix(kappa)))

    acceptanceRate = cns.meas.AcceptanceRate()
    action = cns.meas.Action()
    logDet = cns.meas.LogDet(kappa, MU, SIGMA_KAPPA)

    particleCorrelators = cns.meas.SingleParticleCorrelator(NT, kappa, MU, SIGMA_KAPPA, cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(NT, kappa, MU, SIGMA_KAPPA, cns.Species.HOLE)


    saved_measurements = [
        (action, "/metropolis"),
        (acceptanceRate, "/metropolis"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (logDet, "/logDet"),
    ]

    with h5.open_file(ensembleName+".measurements.h5", "r") as measurementFile:
        for measurement, path in saved_measurements:
            measurement.read(measurementFile,path)

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
        time = [ t * BETA / NT for t in range(NT) ]
        ax.set_yscale("log")

        for i in range(nx):
            ax.errorbar(time, np.real(mean[i]), yerr=np.real(mean_err[i]))

        fig.tight_layout()
        
        ax = species.report()

    ax = acceptanceRate.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = action.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = logDet.report(cns.Species.PARTICLE)
    ax = logDet.report(cns.Species.HOLE)

    plt.show()

if __name__ == "__main__":
    main()
