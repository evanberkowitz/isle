"""!
Investigate the ergodicity problem.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

import core
core.prepare_module_import()
import cns

LATFILE = "one_site.yml"  # input lattice

NT = 16  # number of time slices
NTHERM = 1000  # number of thermalization trajectories
NPROD = 1000  # number of production trajectories

# model parameters
U = 2
BETA = 5
SIGMA_KAPPA = 1

UTILDE = U*BETA/NT

def main():
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

    # thermalize
    accMeas = cns.meas.AcceptanceRate()
    actMeas = cns.meas.Action()
    phi = cns.hmc.hmc(phi, ham,
                      cns.hmc.LinearStepLeapfrog(ham, (1.5, 1), (30, 4), NTHERM-1), NTHERM,
                      [(1, accMeas), (1, actMeas)], [(20, cns.checks.realityCheck)])

    # production
    phi = cns.hmc.hmc(phi, ham, cns.hmc.ConstStepLeapfrog(ham, 1, 4), NPROD,
                      [(1, accMeas), (1, actMeas)])

    # show results
    ax = accMeas.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    ax = actMeas.report(20)
    ax.axvline(NTHERM, c="k")  # mark thermalization - production border

    plt.show()

if __name__ == "__main__":
    main()
