#!/usr/bin/env python3
"""!
Investigate the ergodicity problem.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import tables as h5
import os, sys
from pathlib import Path

import core
core.prepare_module_import()
import cns
import cns.meas


def main():
    """!Run HMC and analyze results."""
    
    cns.env["latticeDirectory"] = str(Path(__file__).resolve().parent.parent)+"/lattices"
    
    ensembleFile = sys.argv[1]
    ensemble = cns.importEnsemble(ensembleFile)
    
    # initial state
    phi = cns.Vector(np.random.normal(0, np.sqrt(ensemble.UTilde), ensemble.spacetime)+0j)

    acceptanceRate = cns.meas.AcceptanceRate()
    action = cns.meas.Action()
    thermalizationProgress = cns.meas.Progress("Thermalization", ensemble.nTherm)
    productionProgress = cns.meas.Progress("Production", ensemble.nProduction)
    logDet = cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigma_kappa)
    totalPhi = cns.meas.TotalPhi()

    # NB!! np.linalg.eig produces eigenvectors in COLUMNS
    noninteracting_energies, irreps = np.linalg.eig(cns.Matrix(ensemble.lattice.hopping()))
    irreps = np.transpose(irreps)

    print("Non-interacting Irreps...")
    print(irreps)
    print("and their corresponding energies")
    print(noninteracting_energies)

    particleCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde, ensemble.mu, ensemble.sigma_kappa, cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde, ensemble.mu, ensemble.sigma_kappa, cns.Species.HOLE)

    rng = cns.random.NumpyRNG(1075)
    print("thermalizing")
    phi = cns.hmc.hmc(phi, ensemble.hamiltonian,
                      ensemble.thermalizer,
                      ensemble.nTherm,
                      rng,
                      [
                          (1, acceptanceRate),
                          (1, action),
                          (1, totalPhi),
                          (ensemble.nTherm/10, thermalizationProgress),
                      ],
                      [(20, cns.checks.realityCheck)])
    print("thermalized!")

    print("running production")
    configurationFile = ensemble.name+".h5"
    
    # Right now, throw away previously generated ensemble.
    # Here's where instead we might load a checkpoint file.
    try:
        os.remove(configurationFile)
    except:
        pass
    
    write = cns.meas.WriteConfiguration(configurationFile, "/cfg/cfg_{itr}")
    phi = cns.hmc.hmc(phi, ensemble.hamiltonian, ensemble.proposer,
                          ensemble.nProduction,
                          rng,
                          [
                              (1, acceptanceRate),
                              (1, action),
                              (1, logDet),
                              (1, totalPhi),
                              (100, particleCorrelators),
                              (100, holeCorrelators),
                              (100, write),
                              (500, productionProgress),
                          ])

    print("Saving measurements...")

    saved_measurements = [
        (action, "/metropolis"),
        (acceptanceRate, "/metropolis"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (totalPhi, "/field"),
        (logDet, "/logDet"),
    ]

    with h5.open_file(ensemble.name+".measurements.h5", "w") as measurementFile:
        for measurement, path in saved_measurements:
            measurement.save(measurementFile,path)

    

if __name__ == "__main__":
    main()
