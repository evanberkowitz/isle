#!/usr/bin/env python3
"""!
Investigate the ergodicity problem.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import tables as h5

import core
core.prepare_module_import()
import cns
import cns.meas

LATFILE = "two_sites.yml"  # input lattice
# LATFILE = "one_site.yml"  # input lattice
# LATFILE="c20.yml"

NT = 8  # number of time slices
NTHERM = 3000  # number of thermalization trajectories
NPROD = 3000 # number of production trajectories

N_LEAPFROG_THERM = 8
N_LEAPFROG = 3

# model parameters
U = 2
BETA = 5
MU=0
SIGMA_KAPPA = -1

UTILDE = U*BETA/NT


def main():
    """!Run HMC and analyze results."""

    # load lattice
    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)  # actually \tilde{kappa}

    # NB!! np.linalg.eig produces eigenvectors in COLUMNS
    noninteracting_energies, irreps = np.linalg.eig(cns.Matrix(lat.hopping()))
    irreps = np.transpose(irreps)

    print("Non-interacting Irreps...")
    print(irreps)
    print("and their corresponding energies")
    print(noninteracting_energies)

    acceptanceRate = cns.meas.AcceptanceRate()
    action = cns.meas.Action()
    thermalizationProgress = cns.meas.Progress("Thermalization", NTHERM)
    productionProgress = cns.meas.Progress("Production", NPROD)
    logDet = cns.meas.LogDet(kappa, MU, SIGMA_KAPPA)

    particleCorrelators = cns.meas.SingleParticleCorrelator(irreps, NT, kappa, MU, SIGMA_KAPPA, cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(irreps, NT, kappa, MU, SIGMA_KAPPA, cns.Species.HOLE)


    saved_measurements = [
        (action, "/metropolis"),
        (acceptanceRate, "/metropolis"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (logDet, "/logDet"),
    ]

    with h5.open_file("measurements.h5", "r") as measurementFile:
        for measurement, path in saved_measurements:
            measurement.read(measurementFile,path)

    print("Processing results...")
    ax = particleCorrelators.report()
    ax = holeCorrelators.report()

    ax = acceptanceRate.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = action.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = logDet.report(cns.Species.PARTICLE)
    ax = logDet.report(cns.Species.HOLE)

    plt.show()

if __name__ == "__main__":
    main()
