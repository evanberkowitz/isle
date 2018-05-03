#!/usr/bin/env python3
"""!
Diagonalize correlation functions
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
    """!
    By symmetrize, I really mean Hermitian-ize.  Make the eigenvalues real.
    """
    mat = np.matrix(a)
    return 0.5*(mat + mat.H)
    
def matrix_plot(mat, **kwargs):
    M = np.real(np.matrix(mat))
    fig, ax = cns.meas.common.newAxes("Matrix", r"i", r"j")
    im = ax.imshow(M, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    
    
def orthogonalize(cross_correlator, starting_time, starting_eigenvectors=None):
    
    # Things will be easier in this routine if time is outer-most.  Reorder
    reordered = np.transpose(cross_correlator, [2,0,1])
    # and symmetrize.
    sym = np.array([ symmetrize(np.matrix(reordered[t])) for t in range(reordered.shape[0]) ])
    
    # Allocate space for the results.
    eigenvalues = np.zeros(sym.shape[0:2], dtype=complex)
    eigenvectors = np.zeros(sym.shape, dtype=complex)
    
    # And diagonalize the timeslice of choice.
    evals, evecs = np.linalg.eig(sym[starting_time])
    evecs = np.matrix(evecs)
    
    if starting_eigenvectors is None:
        eigenvalues[starting_time] = evals
        eigenvectors[starting_time] = evecs
    else:
        # See below for logic.  This is needed to ensure that each bootstrap sample is diagonalized the same way.
        inner = np.abs(evecs.H @ starting_eigenvectors)
        s, permutation = zip(*sorted([ [inner[i],i] for i in range(len(inner))], key=lambda x: np.argmax(x[0])))
        permutation = list(permutation)
        eigenvalues[starting_time] = evals[permutation]
        eigenvectors[starting_time] = evecs[:,permutation]

    for r, step in [ [range(starting_time)[::-1], -1], [range(starting_time+1,sym.shape[0]), +1]]:
        for t in r:
            
            previous = t-step
            evals, evecs = np.linalg.eig( sym[t] )
            evecs = np.matrix(evecs)
            
            # np.linalg.eig doesn't produce a consistent order from timeslice to timeslice.
            # To ensure that we're always tracking the right eigenvalue, 
            # we assume the eigenvectors change mildly with each time step.
            # We construct the matrix of |inner product|s.
            inner = np.abs(evecs.H @ eigenvectors[previous])
            # and find the permutation where the maximal |inner product| is put on the diagonal.
            s, permutation = zip(*sorted([ [inner[i],i] for i in range(len(inner))], key=lambda x: np.argmax(x[0])))
            permutation = list(permutation)
            # We can show the reordering is correct:
            # s = np.array(s)[:,0,:]
            # matrix_plot(s, vmin=0, vmax=1)
            
            # Finally, store the eigenvalues and eigenvectors so that we can go to 
            # the next timeslice and repeat this matching process.
            eigenvalues[t] = evals[permutation]
            eigenvectors[t] = evecs[:,permutation]
        
    # Make sure to re-reorder so that time is slowest again!
    return np.transpose(np.array(eigenvalues), [1,0]), np.transpose(np.array(eigenvectors), [1,2,0])

def main(args):
    """!Analyze correlation functions."""
    
    cns.env["latticeDirectory"] = Path(__file__).resolve().parent.parent/"lattices"

    with h5.File(args.ensemble, 'r') as measurementFile:
        ensemble = cns.ensemble.readH5(measurementFile)
    
    particleCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                            ensemble.mu, ensemble.sigmaKappa,
                                                            cns.Species.PARTICLE)
    holeCorrelators = cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                        ensemble.mu, ensemble.sigmaKappa,
                                                        cns.Species.HOLE)
    phase = cns.meas.Phase()

    saved_measurements = [
        (phase, "/"),
        (particleCorrelators, "/correlation_functions/single_particle"),
        (holeCorrelators, "/correlation_functions/single_hole"),
    ]

    with h5.File(ensemble.name+".measurements.h5", "r") as measurementFile:
        for measurement, path in saved_measurements:
            try:
                measurement.read(measurementFile[path])
            except:
                pass

    print("Processing results...")

    np.random.seed(4386)
    additionalThermalizationCut = 0
    finalMeasurement = particleCorrelators.corr.shape[0]
    NBS = 100
    BSLENGTH=finalMeasurement-additionalThermalizationCut

    print("# Measurements =",BSLENGTH)

    time = np.array([ t * ensemble.beta / ensemble.nt for t in range(ensemble.nt) ])
    pivot_time = int(args.pivot)

    weight = np.exp(phase.theta*1j)

    bootstrapIndices = np.random.randint(additionalThermalizationCut, finalMeasurement, [NBS, BSLENGTH])

    for species, label in zip([particleCorrelators, holeCorrelators], ("PARTICLE", "HOLE")):

        reweighted = [ w * c for w,c in zip(weight,species.corr) ]

        # Plot just the diagonal of the correlator.
        fig, ax = cns.meas.common.newAxes("Bootstrapped "+str(label)+" Diagonal Entries", r"t", r"C")
        mean = np.mean( reweighted, axis=0) / np.mean(weight)
        # Bootstrap an error bar on the diagonal
        mean_err = np.std(np.array([ np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0) /
                                     np.mean(np.array([weight[cfg] for cfg in bootstrapIndices[sample]]), axis=0)
                                     for sample in range(NBS) ] ), axis=0)
        
        ax.set_yscale("log")
        for i in range(ensemble.lattice.nx()):
            ax.errorbar(time, np.real(mean[i,i]), yerr=np.real(mean_err[i,i]))

        fig.tight_layout()

        # Now diagonalize the correlator.
        fig, ax = cns.meas.common.newAxes("Bootstrapped "+str(label)+" Diagonalized", r"t", r"C")
        # Get the mean and the corresponding eigenvectors.
        mean, evecs = orthogonalize(np.mean( reweighted, axis=0) / np.mean(weight), pivot_time)
        mean_err = np.zeros((NBS,)+mean.shape, dtype=complex)
        for sample in range(NBS):
            # Bootstrap the weights
            mean_phase = np.mean(weight[bootstrapIndices[sample]])
            # and the correlators
            m, e = orthogonalize(np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0) / mean_phase, 
                                 pivot_time,
                                 # Make sure that the bootstrapped eigenv*s come out sorted in the same was as the mean did.
                                 evecs[:,:,pivot_time])
            mean_err[sample] = m

        mean_err = np.std(mean_err, axis=0)

        ax.set_yscale("log")
        for i in range(ensemble.lattice.nx()):
            offset = (i - ensemble.lattice.nx() / 2)*(time[1]-time[0]) / ensemble.lattice.nx() / 2.0
            ax.errorbar(offset + time, np.real(mean[i]), yerr=np.real(mean_err[i]))
        ax.axvline(time[pivot_time], c="k")
        fig.tight_layout()

    ax = phase.report()

    plt.show()

def parseArgs():
    "Parse command line arguments."
    parser = argparse.ArgumentParser(description="""
    Produce a measurement report.
    """)
    parser.add_argument("ensemble", help="Ensemble module")
    parser.add_argument("pivot", help="Integer timeslice on which to pivot the analysis.")
    parser.add_argument("--with-thermalization", action="store_true",
                        help="Measurements include thermalization.")
    return parser.parse_args()

if __name__ == "__main__":
    main(parseArgs())
