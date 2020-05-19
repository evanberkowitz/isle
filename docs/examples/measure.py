#!/usr/bin/env python3

"""!
Example script to perform measurements on configurations.

See doc/examples/hmc-evolution.py for an example how to generate configurations.
"""

# For nicer slice notation.
from numpy import s_

# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers
# Import built in measurements (not included in above).
import isle.meas

import numpy as np

def main():
    # Initialize Isle.
    # This sets up the command line interface, defines an argument parser for a measurement
    # command, and parses and returns arguments.
    args = isle.initialize("meas", prog="measure")

    # Set up a measurement driver to run the measurements.
    measState = isle.drivers.meas.init(args.infile, args.outfile, args.overwrite)
    # The driver has retrieved all previously stored parameters from the input file,
    params = measState.params
    # as well as the lattice including the number of time slices nt.
    lat = measState.lattice

    # For simplicity do not allow the spin basis.
    assert params.basis == isle.action.HFABasis.PARTICLE_HOLE

    # Get "tilde" parameters (xTilde = x*beta/Nt) needed to construct measurements.
    muTilde = params.tilde("mu", lat)
    kappaTilde = params.tilde(measState.lattice.hopping(), lat.nt())

    # This object is a lower level interface for the Hubbard fermion action
    # needed by some measurements. The discretization (hopping) needs
    # to be selected manually.
    if params.hopping == isle.action.HFAHopping.DIA:
        hfm = isle.HubbardFermiMatrixDia(kappaTilde, muTilde, params.sigmaKappa)
    else:
        hfm = isle.HubbardFermiMatrixExp(kappaTilde, muTilde, params.sigmaKappa)

    # Define measurements to run.
    species =   (isle.Species.PARTICLE, isle.Species.HOLE)
    allToAll = {s: isle.meas.propagator.AllToAll(hfm, s) for s in species}

    _, diagonalize = np.linalg.eigh(isle.Matrix(hfm.kappaTilde()))

    #
    # The measurements are run on each configuration in the slice passed to 'configSlice'.
    # It defaults to 'all configurations' in e.g. the Logdet measurement.
    # The two correlator measurements are called for every 10th configuration only
    # but across the entire range of configurations.
    #
    # The string parameter in each constructor call is the path in the output file
    # where the measurement shall be stored.
    # The driver ensures that the location can be written to and that nothing gets
    # overwritten by accident.
    measurements = [
        # log(det(M)) where M is the fermion matrix
        isle.meas.Logdet(hfm, "logdet"),
        # \sum_i phi_i
        isle.meas.TotalPhi("field"),
        # collect all weights and store them in consolidated datasets instead of
        # spread out over many HDF5 groups
        isle.meas.CollectWeights("weights"),
        # polyakov loop
        isle.meas.Polyakov(params.basis, lat.nt(), "polyakov", configSlice=s_[::10]),
        # one-point functions
        isle.meas.OnePointFunctions(allToAll[isle.Species.PARTICLE],
                                    allToAll[isle.Species.HOLE],
                                    "correlation_functions/one_point",
                                    configSlice=s_[::10],
                                    transform=diagonalize),
        # single particle correlator for particles / spin up
        isle.meas.SingleParticleCorrelator(allToAll[isle.Species.PARTICLE],
                                           "correlation_functions/single_particle",
                                           configSlice=s_[::10],
                                           transform=diagonalize),
        # single particle correlator for holes / spin down
        isle.meas.SingleParticleCorrelator(allToAll[isle.Species.HOLE],
                                           "correlation_functions/single_hole",
                                           configSlice=s_[::10],
                                           transform=diagonalize),
        isle.meas.SpinSpinCorrelator(allToAll[isle.Species.PARTICLE],
                                     allToAll[isle.Species.HOLE],
                                     "correlation_functions/spin_spin",
                                     configSlice=s_[::10],
                                     transform=diagonalize,
                                     sigmaKappa=params.sigmaKappa),
        isle.meas.DeterminantCorrelators(allToAll[isle.Species.PARTICLE],
                                         allToAll[isle.Species.HOLE],
                                         "correlation_functions/det",
                                         configSlice=s_[::10],
        )
    ]

    # Run the measurements on all configurations in the input file.
    # This automatically saves all results to the output file when done.
    measState(measurements)


if __name__ == "__main__":
    main()
