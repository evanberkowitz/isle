#!/usr/bin/env python3
"""!
Produce a report of measurement output.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as h5
import os, sys
from pathlib import Path
import argparse

import core
core.prepare_module_import()
import cns
import cns.meas

def symmetrize(a):
    return 0.5*(a + a.conj().T)

def main(args):
    """!Analyze HMC results."""
    
    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    # ensemble = cns.ensemble.importEnsemble(args.ensemble)
    with h5.File(args.ensemble,'r') as measFile:
        ensemble, ensembleText = cns.ensemble.loadH5("ensemble", measFile)
    
    action = cns.meas.Action()
    totalPhi = cns.meas.TotalPhi()
    logDet = cns.meas.LogDet(ensemble.kappaTilde, ensemble.mu, ensemble.sigmaKappa)
    particleCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                            ensemble.mu, ensemble.sigmaKappa,
                                                            cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                        ensemble.mu, ensemble.sigmaKappa,
                                                        cns.Species.HOLE)
    chiralCondensate = cns.meas.ChiralCondensate(1234, 10, ensemble.nt, ensemble.kappaTilde,
                                            ensemble.mu, ensemble.sigmaKappa,
                                            cns.Species.PARTICLE)
    
    with_acceptance = False
    acceptanceRate = cns.meas.AcceptanceRate()
    with_timer = False
    timer = cns.meas.Timer()
    
    phase = cns.meas.Phase()

    saved_measurements = [
        (action, "/"),
        (phase, "/"),
        (totalPhi, "/field"),
        (logDet, "/logdet"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
        (chiralCondensate, "/"),
    ]

    with h5.File(args.ensemble,'r') as measurementFile:
        # "acceptance rate is measured only if measurements were created inline"
        if "/monte_carlo/acceptance" in measurementFile:
            with_acceptance = True
            acceptanceRate.accRate = measurementFile["/monte_carlo/acceptance"][()]
        if "/monte_carlo/time" in measurementFile:
            with_timer = True
            timer.times = measurementFile["/monte_carlo/time"][()]
        for measurement, path in saved_measurements:
            try:
                measurement.read(measurementFile[path])
            except:
                pass

    print("Processing results...")

    # if with_timer:
    #     ax = timer.report(1)
    #     ax = timer.histogram(100)
    #
    # np.random.seed(4386)
    # additionalThermalizationCut = 0
    # finalMeasurement = particleCorrelators.corr.shape[0]
    # NBS = 100
    # BSLENGTH=finalMeasurement-additionalThermalizationCut
    #
    # bootstrapIndices = np.random.randint(additionalThermalizationCut, finalMeasurement, [NBS, BSLENGTH])
    #
    # weight = np.exp(phase.theta*1j)
    #
    # for species, label in zip([particleCorrelators, holeCorrelators], ("PARTICLE", "HOLE")):
    #     reweighted = [ w * c for w,c in zip(weight,species.corr) ]
    #     mean = np.mean( reweighted, axis=0) / np.mean(weight)
    #     std  = np.std(reweighted ,axis=0)
    #     mean_err = np.std(np.array([ np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0)
    #                                  for sample in range(NBS) ] ), axis=0)
    #
    #     fig, ax = cns.meas.common.newAxes("Bootstrapped "+str(label)+" Correlator", r"t", r"C")
    #     time = [ t * ensemble.beta / ensemble.nt for t in range(ensemble.nt) ]
    #     ax.set_yscale("log")
    #
    #     for i in range(ensemble.lattice.nx()):
    #         ax.errorbar(time, np.real(mean[i,i]), yerr=np.real(mean_err[i,i]))
    #
    #     fig.tight_layout()
    #
    # for species, label in zip([particleCorrelators, holeCorrelators], ("PARTICLE", "HOLE")):
    #     reweighted = [ w * c for w,c in zip(weight,species.corr)]
    #     mean = np.mean( reweighted, axis=0) / np.mean(weight)
    #
    #     diagonalized = np.zeros(mean.shape[1:], dtype=complex)
    #     for time in range(ensemble.nt):
    #         diagonalized[:,time] = np.linalg.eigvals(symmetrize(mean[:,:,time]))
    #
    #     fig, ax = cns.meas.common.newAxes("Bootstrapped "+str(label)+" Diagonal", r"t", r"C")
    #     time = [ t * ensemble.beta / ensemble.nt for t in range(ensemble.nt) ]
    #     ax.set_yscale("log")
    #
    #     for i in range(ensemble.lattice.nx()):
    #         ax.plot(time, np.real(diagonalized[i]))
    #
    #     fig.tight_layout()
    #
    #
    # fig, ax = cns.meas.common.newAxes("Cross Correlation on middle timeslice", r"i", r"j")
    # mean = np.mean( [ w*c for w,c in zip(weight,particleCorrelators.corr)], axis=0) / np.mean(weight)
    # print(mean.shape)
    # im = ax.imshow(np.real(mean[:,:,mean.shape[2]//2]))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    #
    #
    # if with_acceptance:
    #     ax = acceptanceRate.report(20)
    #     if args.with_thermalization:
    #         args.withax.axvline(ensemble.nTherm, c="k")  # mark thermalization - production border
    #
    # ax = action.report(20)
    # if args.with_thermalization:
    #     ax.axvline(ensemble.nTherm, c="k")  # mark thermalization - production border
    #
    # ax = logDet.report(cns.Species.PARTICLE)
    # ax = logDet.report(cns.Species.HOLE)

    ax = phase.report()
    # ax = chiralCondensate.report()
    # ax = totalPhi.report()

    plt.show()

def parseArgs():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(description="""
    Produce a measurement report.
    """)
    parser.add_argument("ensemble", help="Ensemble module")
    parser.add_argument("--with-thermalization", action="store_true",
                        help="Measurements include thermalization.")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
