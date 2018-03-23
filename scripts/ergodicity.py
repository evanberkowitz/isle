import sys
import os
import contextlib
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import core
core.prepare_module_import()
import cns

LATFILE = "two_sites.yml"

NT = 16   # number of time slices
NTR = 1000  # number of trajectories
NMD = 3  # number of MD steps per trajectory
MDSTEP = 1/NMD  # size of MD steps

U = 2
BETA = 3
SIGMA_KAPPA = 1

UTILDE = U*BETA/NT

class PhaseMeas:
    "Record and plot pahses of action."

    def __init__(self):
        self.phases = []

    def __call__(self, cfg, h):
        self.phases.append(np.imag(h))

    def report(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Phases")
        ax.plot(self.phases)
        plt.draw()

class DetMeas:
    def __init__(self, hfm):
        self._hfm = hfm
        self.detM = []
        self.detQ = []

    def __call__(self, cfg, h):
        self._hfm.updatePhi(cfg)
        self.detM.append(np.exp(cns.logdet(cns.Matrix(self._hfm.M(False)))))
        self.detQ.append(np.exp(cns.logdet(cns.Matrix(self._hfm.Q()))))

    def report(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("det(M)")
        plt.hist2d(np.real(self.detM), np.imag(self.detM), bins=40, norm=LogNorm())
        plt.draw()


# TODO how do repro check with generic proposer?
# TODO I don't want to have to pass ham in, compute hold using only the change in pi
def hmc(phi, ham, proposer, ntr, measurements, realCheckFreq=0):
    nacc = 0  # number of accepted trajectories
    for i in range(ntr):
        pi = cns.Vector(np.random.normal(0, 1, len(phi))+0j)
        hold = ham.eval(phi, pi)

        # do MD
        newPhi, newPi, hnew = proposer(phi, pi)

        # check that phi and pi are real
        if realCheckFreq != 0 and i % realCheckFreq == 0:
            if np.max(cns.imag(newPhi))/np.max(cns.real(newPhi)) > 1e-14:
                raise RuntimeError("phi has acquired an imaginary part at iteration {}: {}".format(i, newPhi))
            if np.max(cns.imag(newPi))/np.max(cns.real(newPi)) > 1e-14:
                raise RuntimeError("pi has acquired an imaginary part at iteration {}: {}".format(i, newPi))

        # accept-reject
        if np.exp(cns.real(hold-hnew)) > np.random.uniform(0, 1):
            phi = newPhi
            hold = hnew
            nacc += 1
            
        # perform measurements
        for meas in measurements:
            meas(phi, hold)

    return phi, nacc

def main():
    np.random.seed(1)

    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)

    ham = cns.Hamiltonian(cns.HubbardGaugeAction(UTILDE),
                          cns.HubbardFermiAction(kappa, 0, SIGMA_KAPPA))

    # initial state
    phi = cns.Vector(np.random.normal(0, np.sqrt(UTILDE), lat.nx()*NT)+0j)

    # thermalize
    phi, nacct = hmc(phi, ham, lambda phi, pi: cns.leapfrog(phi, pi, ham, 1, NMD),
                     100, [], realCheckFreq=1)
    print("Acceptance rate during thermalization: {}".format(nacct/100))

    # define measurements
    phaseMeas = PhaseMeas()
    detMeas = DetMeas(cns.HubbardFermiMatrix(kappa, phi, 0, SIGMA_KAPPA))

    # measure
    phi, nacct = hmc(phi, ham, lambda phi, pi: cns.leapfrog(phi, pi, ham, 1, NMD),
                     NTR, [phaseMeas, detMeas], realCheckFreq=1)
    print("Acceptance rate during measurements: {}".format(nacct/NTR))

    phaseMeas.report()
    detMeas.report()
    plt.show()

if __name__ == "__main__":
    main()
