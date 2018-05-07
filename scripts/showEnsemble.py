#!/usr/bin/env python3
"""!
Provides a quick overview of an ensemble.
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py as h5

import core
core.prepare_module_import()
import cns
import cns.meas

def readMeas(group):
    """!Read all default inline measurements."""
    acceptanceRate = cns.meas.AcceptanceRate()
    acceptanceRate.read(group, True)
    action = cns.meas.Action()
    action.read(group, True)

    return (acceptanceRate, action)

def main(args):
    """!Run the module"""

    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    with h5.File(args.file, "r") as cfgf:
        ensemble = cns.ensemble.load(args.file.replace(".", "_"), args.file)[0]
        nconfig = len(cfgf["configuration"])
        (acceptanceRate, action) = readMeas(cfgf["configuration"])

    fig = plt.figure(figsize=(11, 6))
    fig.canvas.set_window_title("Overview - "+ensemble.name)
    grid = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

    axAcc = fig.add_subplot(grid[0, 0])
    acceptanceRate.report(1, ax=axAcc)
    axAcc.set_title("Acceptance, total = {:3.2f}%".format(np.mean(acceptanceRate.accRate)*100))
    axAcc.set_xlabel(r"$N_{\mathrm{tr}}$")
    axAcc.set_ylabel("Accepted")

    axAct = fig.add_subplot(grid[0, 1])
    action.report(1, ax=axAct)
    axAct.set_title("Action")
    axAct.set_xlabel(r"$N_{\mathrm{tr}}$")
    axAct.set_ylabel("S")

    axText = fig.add_subplot(grid[1, :])
    axText.axis("off")
    axText.text(0, -.5, fontsize=13, linespacing=1.5,
                s=r"""Ensemble:   name = {name},     $N_{{\mathrm{{config}}}} = {nconfig}$
Lattice:    file = {latfile},   $N_t = {nt}$,   $N_x = {nx}$
Model:      $U = {U}$,  $beta = {beta}$,  $\mu = {mu}$,  $\sigma_{{\kappa}} = {sigmaKappa}$
""".format(name=ensemble.name, nconfig=nconfig,
           latfile=ensemble.latticeFile, nt=ensemble.lattice.nt(), nx=ensemble.lattice.nx(),
           U=ensemble.U, beta=ensemble.beta, mu=ensemble.mu, sigmaKappa=ensemble.sigmaKappa))

    fig.tight_layout()
    plt.show()

def parseArgs():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(description="Provides a quick overview of an ensemble.")
    parser.add_argument("file", help="Configurations datafile")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
