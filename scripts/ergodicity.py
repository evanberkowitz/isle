import sys
import os
import contextlib
import yaml
import numpy as np
import matplotlib.pyplot as plt

import core
core.prepare_module_import()
import cns

LATFILE = "one_site.yml"

NT = 16   # number of time slices
NTR = 64  # number of trajectories
NMD = 4  # number of MD steps per trajectory
MDSTEP = 1/NMD  # size of MD steps

U = 2
BETA = 3
SIGMA_KAPPA = -1

UTILDE = U*BETA/NT


class Hamiltonian:
    def __init__(self, kappa):
        self.gaugeAct = cns.HubbardGaugeAction(UTILDE)
        self.fermiAct = cns.HubbardFermiAction(kappa, 0, 0, SIGMA_KAPPA)

    def eval(self, phi, pi):
        return np.linalg.norm(pi)**2/2 + self.gaugeAct.eval(phi)+self.fermiAct.eval(phi)

    def force(self, phi):
        return self.gaugeAct.force(phi)+self.gaugeAct.force(phi)


# uses leapfrog
def moledyn(ham, phi, pi, direction=+1):
    eps = direction*MDSTEP

    # copy input to not overwrite it
    phi = cns.Vector(phi, dtype=complex)

    # initial half step
    pi = pi + ham.force(phi)*eps/2

    for _ in range(NMD-2):
        phi += pi*eps
        pi += ham.force(phi)*eps

    phi += pi*eps # final step (left out above)
    pi += ham.force(phi)*eps/2  # final half step

    return phi, pi


def main():
    np.random.seed(1)

    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = cns.SparseMatrix(lat.hopping(), dtype=float)

    ham = Hamiltonian(kappa)

    # initial state
    phi = cns.Vector(np.random.randn(lat.nx()*NT)+0j)
    pi = cns.Vector(np.random.normal(0, 1, len(phi)))
    oldS = ham.eval(phi, pi)

    cfgs = [phi]
    phases = [np.imag(oldS)]

    for i in range(NTR-1):  # -1 because we already have the initial 'trajectory'
        newPhi, newPi = moledyn(ham, phi, pi, +1)

        repPhi, repPi = moledyn(ham, newPhi, newPi, -1)
        if np.linalg.norm(repPhi-phi) > 1e-14:
            print("Repro check failed in traj {} with error in phi: {}".format(i, np.linalg.norm(repPhi-phi)))
            return
        if np.linalg.norm(repPi-pi) > 1e-14:
            print("Repro check failed in traj {} with error in pi: {}".format(i, np.linalg.norm(repPi-pi)))
            return

        if np.max(np.imag(newPhi)) > 1e-14:
            print("phi has acquired an imaginary part")
            return

        newS = ham.eval(newPhi, pi)

        if np.min((1, np.exp(np.real(-newS+oldS)))) > np.random.uniform(0, 1):
            print("accept: ", newS-oldS)
            oldS = newS
            phi = newPhi
        else:
            print("reject: ", newS-oldS)


        cfgs.append(phi)
        phases.append(np.imag(newS))

        pi = cns.Vector(np.random.normal(0, 1, len(phi)))

    print("phases: ", phases)

    dets = list(map(
        lambda cfg:
        np.exp(np.real(cns.logdet(cns.Matrix(cns.HubbardFermiMatrix(lat.hopping(), cfg, 0, 1, SIGMA_KAPPA).MMdag())))),
        cfgs))

    plt.figure()
    plt.hist(dets, bins=32)
    plt.show()

if __name__ == "__main__":
    main()
