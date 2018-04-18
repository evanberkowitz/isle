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

ENSEMBLE_NAME = "two_sites"

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
MU = 0
SIGMA_KAPPA = -1

UTILDE = U*BETA/NT


def main():
    """!Run HMC and analyze results."""

    # load lattice
    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)  # actually \tilde{kappa}

    # Hamiltonian for the Hubbard model
    ham = cns.Hamiltonian(cns.HubbardGaugeAction(UTILDE),
                          cns.HubbardFermiAction(kappa, MU, SIGMA_KAPPA))

    # initial state
    phi = cns.Vector(np.random.normal(0, np.sqrt(UTILDE), lat.nx()*NT)+0j)

    acceptanceRate = cns.meas.AcceptanceRate()
    action = cns.meas.Action()
    thermalizationProgress = cns.meas.Progress("Thermalization", NTHERM)
    productionProgress = cns.meas.Progress("Production", NPROD)
    logDet = cns.meas.LogDet(kappa, MU, SIGMA_KAPPA)

    # NB!! np.linalg.eig produces eigenvectors in COLUMNS
    noninteracting_energies, irreps = np.linalg.eig(cns.Matrix(lat.hopping()))
    irreps = np.transpose(irreps)

    print("Non-interacting Irreps...")
    print(irreps)
    print("and their corresponding energies")
    print(noninteracting_energies)

    particleCorrelators = cns.meas.SingleParticleCorrelator(NT, kappa, MU, SIGMA_KAPPA, cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(NT, kappa, MU, SIGMA_KAPPA, cns.Species.HOLE)

    rng = cns.random.NumpyRNG(1075)
    print("thermalizing")
    phi = cns.hmc.hmc(phi, ham,
                      cns.hmc.LinearStepLeapfrog(ham, (1, 1), (N_LEAPFROG_THERM, N_LEAPFROG), NTHERM-1),
                      NTHERM,
                      rng,
                      [
                          (1, acceptanceRate),
                          (1, action),
                          (NTHERM/10, thermalizationProgress),
                      ],
                      [(20, cns.checks.realityCheck)])
    print("thermalized!")

    print("running production")
    write = cns.meas.WriteConfiguration(ENSEMBLE_NAME+".h5", "/")
    phi = cns.hmc.hmc(phi, ham, cns.hmc.ConstStepLeapfrog(ham, 1, N_LEAPFROG),
                          NPROD,
                          rng,
                          [
                              (1, acceptanceRate),
                              (1, action),
                              (500, productionProgress),
                              (1, logDet),
                              (100, particleCorrelators),
                              (100, holeCorrelators),
                              (1, write)
                          ])

    print("Saving measurements...")

    saved_measurements = [
        (action, "/metropolis"),
        (acceptanceRate, "/metropolis"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (logDet, "/logDet"),
    ]

    with h5.open_file("measurements.h5", "w") as measurementFile:
        for measurement, path in saved_measurements:
            measurement.save(measurementFile,path)

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
