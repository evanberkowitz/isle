"""!
Investigate the ergodicity problem.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

import core
core.prepare_module_import()
import cns
import cns.meas

LATFILE = "two_sites.yml"  # input lattice

NT = 8  # number of time slices
NTHERM = 3000  # number of thermalization trajectories
NPROD = 10000  # number of production trajectories

# model parameters
U = 2
BETA = 5
SIGMA_KAPPA = 1

UTILDE = U*BETA/NT


def main():
    """!Run HMC and analyze results."""

    np.random.seed(1075)

    # load lattice
    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)  # actually \tilde{kappa}

    # Hamiltonian for the Hubbard model
    ham = cns.Hamiltonian(cns.HubbardGaugeAction(UTILDE),
                          cns.HubbardFermiAction(kappa, 0, SIGMA_KAPPA))

    # initial state
    phi = cns.Vector(np.random.normal(0, np.sqrt(UTILDE), lat.nx()*NT)+0j)

    accMeas = cns.meas.AcceptanceRate()
    actMeas = cns.meas.Action()

    print("thermalizing")
    phi = cns.hmc.hmc(phi, ham,
                      cns.hmc.LinearStepLeapfrog(ham, (1, 1), (20, 4), NTHERM-1), NTHERM,
                      [(1, accMeas), (1, actMeas)],
                      [(20, cns.checks.realityCheck)])

    # print("running production")
    # detMeas = DetMeas(cns.HubbardFermiMatrix(kappa, 0, SIGMA_KAPPA))
    # print("running production")
    # phi = cns.hmc.hmc(phi, ham, cns.hmc.ConstStepLeapfrog(ham, 1, 4), NPROD,
    #                   [(1, accMeas), (1, detMeas)])

    print("processing results")
    ax = accMeas.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = actMeas.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    plt.show()

if __name__ == "__main__":
    main()
