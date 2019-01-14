"""!
Example script to produce configurations with HMC.
"""

from logging import getLogger

import yaml

import isle
import isle.drivers


### Specify parameters.
# isle.util.parameters takes arbitrary keyword arguments, constructs a new dataclass,
# and stores the function arguments in an instance of it.
# The object is written to the output file and read back in by all subsequent processes.
# Use this to store all physical and model parameters and make them accessible later.
PARAMS = isle.util.parameters(
    beta=3,
    U=2,
    mu=0,
    sigmaKappa=-1,
    alpha=1,
    hopping=isle.action.HFAHopping.DIA,
    variant=isle.action.HFAVariant.ONE
)

# define an action
def makeAction(lat, params):
    import isle
    import isle.action

    return isle.Hamiltonian(isle.action.HubbardGaugeAction(params.tilde("U", lat)),
                            isle.action.makeHubbardFermiAction(lat,
                                                               params.beta,
                                                               params.tilde("mu", lat),
                                                               params.sigmaKappa,
                                                               params.alpha,
                                                               params.hopping,
                                                               params.variant))

def main():
    # initialize command line interface and parse command line arguments
    args = isle.initialize("hmc", name="basic-hmc")
    log = getLogger("HMC")

    # load lattice
    with open("resources/lattices/four_sites.yml", "r") as latfile:
        lat = yaml.safe_load(latfile.read())
    lat.nt(16)  # lattice files usually only contain information on spatial lattice

    # set up a random number generator
    rng = isle.random.NumpyRNG(1075)

    # set up a fresh HMC driver to control HMC evolution
    hmcState = isle.drivers.hmc.init(lat, PARAMS, rng, makeAction, args.outfile,
                                     args.overwrite, startIdx=0)

    # random initial condition
    phi = isle.Vector(rng.normal(0, (PARAMS.U * PARAMS.beta / lat.nt())**(1/2), lat.lattSize())+0j)

    log.info("Thermalizing")
    # a proposer to linearly decrease the number of MD steps
    proposer = isle.proposers.LinearStepLeapfrog(hmcState.ham, (1, 1), (20, 5), 99)
    # thermalize configuration without saving anything
    phi = hmcState(phi, proposer, 100, saveFreq=0, checkpointFreq=0)
    # reset the internal counter so we start saving configs at index 0
    hmcState.resetIndex()

    log.info("Producing")
    # new proposer with a constant number of steps
    proposer = isle.proposers.ConstStepLeapfrog(hmcState.ham, 1, 5)
    # produce configurations and save in intervals
    phi = hmcState(phi, proposer, 100, saveFreq=2, checkpointFreq=10)

if __name__ == "__main__":
    main()
