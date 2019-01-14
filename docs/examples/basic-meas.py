"""!
Example script to perform measurements on configurations generated with basic-hmc.py.
"""


# Import base functionality of Isle.
import isle
# Import drivers (not included in above).
import isle.drivers
# Import built in measurements (not included in above).
import isle.meas


def main():
    # Initialize Isle.
    # This sets up the command line interface, defines an argument parser for a measurement
    # command, and parses and returns arguments.
    args = isle.initialize("hmc", name="basic-hmc")

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

    # This object is a lover level interface for the Hubbard fermion action
    # needed by some measurements. The discretization (hopping) needs
    # to be selected manually.
    if params.hopping == isle.action.HFAHopping.DIA:
        hfm = isle.HubbardFermiMatrixDia(kappaTilde, muTilde, params.sigmaKappa)
    else:
        hfm = isle.HubbardFermiMatrixExp(kappaTilde, muTilde, params.sigmaKappa)

    # Define measurements to run.
    # Each list element is a tuple (freq, meas, store), where
    #   - freq is the measurement frequency. Freq=x means that the measurement
    #     is run every x configurations.
    #   - meas is the measurement callable.
    #   - store is a path inside the output file that the measurement results
    #     are to be written to.
    measurements = [
        # log(det(M)) where M is the fermion matrix
        (1, isle.meas.Logdet(hfm), "/logdet"),
        # \sum_i phi_i
        (1, isle.meas.TotalPhi(), "/field"),
        # copy the action into the output file
        (1, isle.meas.Action(), "/"),
        # single particle correlator for particles / spin up
        (10, isle.meas.SingleParticleCorrelator(hfm, isle.Species.PARTICLE),
         "/correlation_functions/single_particle"),
        # single particle correlator for holes / spin down
        (10, isle.meas.SingleParticleCorrelator(hfm, isle.Species.HOLE),
         "/correlation_functions/single_hole"),
    ]

    # Run the measurements on all configurations in the input file.
    # This automatically saves all results to the output file when done.
    measState(measurements)

if __name__ == "__main__":
    main()
