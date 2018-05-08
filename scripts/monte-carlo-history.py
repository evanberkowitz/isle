#!/usr/bin/env python3
"""!
Produce a report of ensemble diagnostics available immediately after HMC.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    
    with h5.File(args.ensemble,'r') as measFile:
        ensemble, ensembleText = cns.ensemble.loadH5("ensemble", measFile)
    
    fig = plt.figure(figsize=(11, 8))
    fig.canvas.set_window_title("Overview - "+ensemble.name)
    grid = gridspec.GridSpec(3, 4, height_ratios=[5,5,3])
    grid.update(hspace=0.6, wspace=0.4)
    
    monteCarloDiagnostics = {
        "acceptance":   cns.meas.AcceptanceRate(),
        "action":       cns.meas.Action(),
        "phi":          cns.meas.TotalPhi(),
        "phase":        cns.meas.Phase(),
    }
    
    
    with h5.File(args.ensemble,'r') as measurementFile:
        # h5 groups are unordered.  Sort them to get them into Monte Carlo time order.
        configurations = sorted([ cfg for cfg in measurementFile["configuration"]], key=int)
        for cfg in configurations:
            action = measurementFile["configuration"][cfg]["action"][()]
            field  = measurementFile["configuration"][cfg]["phi"][()]
            acceptance = measurementFile["configuration"][cfg]["acceptance"][()]
            for diagnostic in monteCarloDiagnostics.values():
                diagnostic(field, act=action, acc=acceptance, inline=True)
    
    
    acceptance_history = plt.subplot(grid[0,0:2])
    monteCarloDiagnostics["acceptance"].report(ax=acceptance_history)
    acceptance_history.set_title("Acceptance, total = {:3.2f}%".format(np.mean(monteCarloDiagnostics["acceptance"].accRate)*100))
    acceptance_history.set_xlabel(r"$N_{\mathrm{tr}}$")
    acceptance_history.set_ylabel("Accepted")
    
    action_history = plt.subplot(grid[0,2:])
    monteCarloDiagnostics["action"].report(ax=action_history)
    action_history.set_title("Action")
    action_history.set_xlabel(r"$N_{\mathrm{tr}}$")
    action_history.set_ylabel("S")
    
    
    phi_history = plt.subplot(grid[1,0])
    monteCarloDiagnostics["phi"].plotPhi(ax=phi_history)
    phi_history.set_title(r"$\Phi = \sum \varphi$")
    phi_history.set_xlabel(r"$N_{\mathrm{tr}}$")
    phi_history.set_ylabel(r"$\Phi$")

    phi_distribution = plt.subplot(grid[1,1])
    monteCarloDiagnostics["phi"].plotPhiDistribution(ax=phi_distribution, orientation="horizontal")
    phi_distribution.set_ylim(phi_history.get_ylim())
    phi_distribution.set_title(r"Histogram of $\Phi$")
    phi_distribution.set_xlabel(r"Frequency")
    
    
    phase_history = plt.subplot(grid[1,2])
    monteCarloDiagnostics["phase"].plotMCHistory(ax=phase_history)
    phase_history.set_ylim(1.05*np.array([-1,1])*np.pi)
    phase_history.set_title(r"$\theta$ = Im($S$)")
    phase_history.set_xlabel(r"$N_{\mathrm{tr}}$")
    phase_history.set_yticks((-np.pi,-np.pi/2,0,np.pi/2,np.pi))
    phase_history.set_yticklabels((r"$-\pi$", r"$-\pi/2$", r"0", r"$\pi/2$", r"$\pi$"))
    
    phase_histogram = plt.subplot(grid[1,3])
    theta = np.array(monteCarloDiagnostics["phase"].theta)
    weight = np.exp(-theta*1j)
    x = np.real(weight)
    y = np.imag(weight)
    
    view = [-1.05, 1.05]
    h = phase_histogram.hist2d(x, y, bins=51, range=[view, view], normed=1, label="histogram")
    phase_histogram.set_xlim(view)
    phase_histogram.set_ylim(view)
    phase_histogram.set_aspect(1)
    plt.colorbar(h[3], fraction=0.046, pad=0.04)
    
    # Indicate the average weight
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    phase_histogram.scatter([x_mean], [y_mean], c="w", )
    # monteCarloDiagnostics["phase"].plotHist2D(ax=phase_histogram)
    phase_histogram.set_title(r"Histogram of $e^{-i\theta}$")
    

    text = fig.add_subplot(grid[2, :])
    text.axis("off")
    
    text.text(0, -.5, fontsize=13, linespacing=1.5,
                    s=r"""Ensemble:   name = {name},     $N_{{\mathrm{{config}}}} = {nconfig}$
     Lattice:    file = {latfile},   $N_t = {nt}$,   $N_x = {nx}$
      Model:      $U = {U}$,  $\beta = {beta}$,  $\mu = {mu}$,  $\sigma_{{\kappa}} = {sigmaKappa}$
    """.format( name=ensemble.name, 
                nconfig=len(configurations),
                latfile=ensemble.latticeFile, 
                nt=ensemble.lattice.nt(),
                nx=ensemble.lattice.nx(),
                U=ensemble.U,
                beta=ensemble.beta,
                mu=ensemble.mu,
                sigmaKappa=ensemble.sigmaKappa))
    

    plt.show()
    
    return 0

def parseArgs():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(description="""
    Produce a measurement report.
    """)
    parser.add_argument("ensemble", help="Ensemble module")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
