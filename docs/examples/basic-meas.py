"""!
Example script to perform measurements on configurations generated with basic-hmc.py.
"""

import isle
import isle.drivers
import isle.meas

def main():
    args = isle.initialize("meas", name="my-meas")

    measState = isle.drivers.meas.init(args.infile, args.outfile, args.overwrite)
    params = measState.params
    nt = measState.lattice.nt()

    muTilde = params.tilde("mu", nt)
    kappaTilde = params.tilde(measState.lattice.hopping(), nt)

    hfm = isle.HubbardFermiMatrixDia(kappaTilde, muTilde, params.sigmaKappa)

    measurements = [
        (1, isle.meas.Logdet(hfm), "/logdet"),
        (1, isle.meas.TotalPhi(), "/field"),
        (1, isle.meas.Action(), "/"),
        (1, isle.meas.ChiralCondensate(1234, 10, hfm, isle.Species.PARTICLE), "/"),
        (10, isle.meas.SingleParticleCorrelator(hfm, isle.Species.PARTICLE),
         "/correlation_functions/single_particle"),
        (10, isle.meas.SingleParticleCorrelator(hfm, isle.Species.HOLE),
         "/correlation_functions/single_hole"),
    ]

    measState(measurements)

if __name__ == "__main__":
    main()
