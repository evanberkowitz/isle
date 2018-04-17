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
NPROD = 5000 # number of production trajectories

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

    np.random.seed(1075)

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

    corrs = cns.meas.Corr_1p(irreps, NT, kappa, MU, SIGMA_KAPPA)

    print("thermalizing")
    phi = cns.hmc.hmc(phi, ham,
                      cns.hmc.LinearStepLeapfrog(ham, (1, 1), (N_LEAPFROG_THERM, N_LEAPFROG), NTHERM-1), NTHERM,
                      [
                          (1, acceptanceRate),
                          (1, action),
                          (NTHERM/10, thermalizationProgress),
                      ],
                      [(20, cns.checks.realityCheck)])
    print("thermalized!")

    the_configurations = h5.open_file("ensemble_name.h5","w")
    
    write = cns.meas.WriteConfiguration(the_configurations,"/configurations")

    # print("running production")
    # detMeas = DetMeas(cns.HubbardFermiMatrix(kappa, 0, SIGMA_KAPPA))
    print("running production")
    phi = cns.hmc.hmc(phi, ham, cns.hmc.ConstStepLeapfrog(ham, 1, N_LEAPFROG), NPROD,
                      [
                          (1, acceptanceRate),
                          (1, action),
                          (500, productionProgress),
                          (1, logDet),
                          (1, corrs),
                          (100, write),
                      ])

    the_configurations.close()

    print("Saving measurements...")

    saved_measurements = [
        (action, "/metropolis"),
        (acceptanceRate, "/metropolis"),
        (corrs, "/correlation_functions/single_particle"),
        (logDet, "/logDet"),
    ]

    with h5.open_file("measurements.h5", "w") as the_measurements:
        for measurement, path in saved_measurements:
            measurement.save(the_measurements,path)

    print("Processing results...")
    ax = corrs.report()

    ax = acceptanceRate.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = action.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = logDet.report(cns.Species.PARTICLE)
    ax = logDet.report(cns.Species.HOLE)

    plt.show()

if __name__ == "__main__":
    main()
