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
    
    with h5.File(args.ensemble,'r') as measFile:
        ensemble, ensembleText = cns.ensemble.loadH5("ensemble", measFile)
    
    
    monteCarloDiagnostics = [
        cns.meas.Action(),
        cns.meas.TotalPhi(),
        cns.meas.Phase()
    ]
    
    with h5.File(args.ensemble,'r') as measurementFile:
        for cfg in measurementFile["configuration"]:
            action = measurementFile["configuration"][cfg]["action"][()]
            field  = measurementFile["configuration"][cfg]["phi"][()]
            for diagnostic in monteCarloDiagnostics:
                diagnostic(field, act=action)
    
    
    for diagnostic in monteCarloDiagnostics:
        ax = diagnostic.report()
    
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
