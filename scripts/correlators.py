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
from cns.meas.common import newAxes

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
    reordered = np.array(np.transpose(cross_correlator, [2,0,1]))
    # and symmetrize.
    sym = np.array([ symmetrize(np.matrix(reordered[t])) for t in range(reordered.shape[0]) ])
    
    # Allocate space for the results.
    eigenvalues = np.zeros(sym.shape[0:2], dtype=complex)
    eigenvectors = np.zeros(sym.shape, dtype=complex)
    
    # And diagonalize the timeslice of choice.
    evals, evecs = np.linalg.eig(sym[starting_time])
    evecs = np.matrix(evecs)
    
    if starting_eigenvectors is None:
        # If there's no prior order given by starting_eigenvectors,
        # Sort them from greatest-to-least eigenvalue
        # so you can "go down the list" and also go down the axis of a figure.
        s, permutation = zip(*sorted(zip(evals,range(len(evals))), key=lambda x: -np.abs(x[0])))
        permutation = list(permutation)
        eigenvalues[starting_time] = evals[permutation]
        eigenvectors[starting_time] = evecs[:,permutation]
    else:
        # See below for logic.
        # This is needed to ensure that each bootstrap sample is diagonalized the same way.
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
        ensemble, ensembleText = cns.ensemble.loadH5("ensemble", measurementFile["/"])
        
        correlator = {
        cns.Species.PARTICLE: cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                                ensemble.mu, ensemble.sigmaKappa,
                                                                cns.Species.PARTICLE),
        cns.Species.HOLE:     cns.meas.SingleParticleCorrelator(ensemble.nt, ensemble.kappaTilde,
                                                                ensemble.mu, ensemble.sigmaKappa,
                                                                cns.Species.HOLE),
        }
        phase = cns.meas.Phase()
        
        saved_measurements = [
            (phase, "/"),
            (correlator[cns.Species.PARTICLE], "/correlation_functions/single_particle"),
            (correlator[cns.Species.HOLE], "/correlation_functions/single_hole"),
        ]
        
        for measurement, path in saved_measurements:
            try:
                measurement.read(measurementFile[path])
            except:
                pass
        
        time = np.array([ t * ensemble.beta / ensemble.nt for t in range(ensemble.nt) ])
        weight = np.exp(phase.theta*1j)

    print("Processing results...")
    pivot = int(args.pivot)
    print("Diagonalization pivot is {}".format(pivot))

    np.random.seed(4386)
    additionalThermalizationCut = 0
    finalMeasurement = correlator[cns.Species.PARTICLE].corr.shape[0]
    NBS = 100
    BSLENGTH=finalMeasurement-additionalThermalizationCut

    SPECIES = [cns.Species.PARTICLE, cns.Species.HOLE]

    print("There are {} measurements.".format(BSLENGTH))

    bootstrapIndices = np.random.randint(additionalThermalizationCut, finalMeasurement, [NBS, BSLENGTH])
    
    # First, just do the diagonal entries:
    bootstrapped={}
    for species in SPECIES:
        reweighted = [ w * c for w,c in zip(weight,correlator[species].corr) ]
        bootstrapped[species] = {
        "mean": np.mean( reweighted, axis=0) / np.mean(weight),
        "err":  np.std(np.array([ np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0) / 
                                  np.mean(np.array([weight[cfg]     for cfg in bootstrapIndices[sample]]), axis=0)
                                     for sample in range(NBS) ] ), axis=0)
        }
    for species, label in zip(SPECIES,["Particles", "Holes"]):
        fig, ax = newAxes(label, "t", "C")
        ax.set_yscale("log")
        for i in range(ensemble.lattice.nx()):
            ax.errorbar(time,
                       # Taking [i,i] components is taking the diagonal:
                       np.real(bootstrapped[species]["mean"][i,i]),
                       yerr=np.real(bootstrapped[species]["err"][i,i]))
            ax.set_ylabel(label)


    # Now do the diagonalization without any sophistication:
    diagonalized={}

    for species in SPECIES:
        reweighted = [ w * c for w,c in zip(weight,correlator[species].corr) ]
        mean, evecs = orthogonalize(np.mean( reweighted, axis=0) / np.mean(weight), pivot)
        err = np.zeros((NBS,)+mean.shape, dtype=complex)
        for sample in range(NBS):
            mean_phase = np.mean(weight[bootstrapIndices[sample]])
            m, e = orthogonalize(np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0) / mean_phase, 
                                     pivot,
                                     # Make sure that the bootstrapped eigenv*s come out sorted in the same was as the mean did.
                                     evecs[:,:,pivot])
            err[sample] = m
        err = np.std(err, axis=0)
        
        diagonalized[species] = {
            "mean": mean,
            "err":  err
        }
    
    # for species, label in zip(SPECIES,["Particles", "Holes"]):
    #     fig, ax = newAxes("Diagonalize "+label, "t", "C")
    #     ax.set_yscale("log")
    #     for i in range(ensemble.lattice.nx()):
    #         ax.errorbar(time,
    #                    np.real(diagonalized[species]["mean"][i]),
    #                    yerr=np.real(diagonalized[species]["err"][i]))
    #         ax.set_ylabel(label)


    # NOW do the chopping of the zeros.
    referenceTime = 0
    # Make sure you catch the lowest eigenvalue:
    cutoff=np.min(np.real(diagonalized[cns.Species.PARTICLE]["mean"][:,0]))/2

    # Or, take all the data:
    # cutoff=0

    magnitude = np.abs(np.mean(correlator[cns.Species.PARTICLE].corr, axis=0)[:,:,referenceTime])
    chopped_magnitude = np.where(magnitude > cutoff, magnitude,0)

    fig, ax = newAxes("Overlap Histogram", "Overlap", "#")

    ax.hist(np.reshape(magnitude,[-1]),100,range=[0,0.02])
    ax.axvline(cutoff, c="k")

    fig, ax = newAxes("Before Chopping","i","j")
    im = ax.imshow(magnitude)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    fig, ax = newAxes("After Chopping","i","j")
    im = ax.imshow(chopped_magnitude)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    refined={}
    
    for species in SPECIES:
        reweighted = np.array([ w * c for w,c in zip(weight,correlator[species].corr) ])
        for measurement in range(len(reweighted)):
            for t in range(ensemble.nt):
                reweighted[measurement,:,:,t] = np.where(magnitude > cutoff, reweighted[measurement,:,:,t],0) 
        mean, evecs = orthogonalize(np.mean( reweighted, axis=0) / np.mean(weight), pivot)
        err = np.zeros((NBS,)+mean.shape, dtype=complex)
        for sample in range(NBS):
            mean_phase = np.mean(weight[bootstrapIndices[sample]])
            m, e = orthogonalize(np.mean(np.array([reweighted[cfg] for cfg in bootstrapIndices[sample]]), axis=0) / mean_phase, 
                                     pivot,
                                     # Make sure that the bootstrapped eigenv*s come out sorted in the same was as the mean did.
                                     evecs[:,:,pivot])
            err[sample] = m
        err = np.std(err, axis=0)
        
        refined[species] = {
            "mean": mean,
            "err":  err
        }
        

    for species, ax, label in zip(SPECIES,[[0.05,0.05,1.1,1.1],[1.3,0.05,1.1,1.1]],["Particles", "Holes"]):
        fig, ax = newAxes("Diagonalize {} after chopping".format(label), "t", "C")
        ax.set_yscale("log")
        for i in range(ensemble.lattice.nx()):
            ax.errorbar(time,
                       np.real(refined[species]["mean"][i]), 
                       yerr=np.real(refined[species]["err"][i]))
            ax.set_ylabel(label)

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
