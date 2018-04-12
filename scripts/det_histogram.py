import sys
import os
import contextlib
import yaml
import numpy as np
import tables as h5


import core
core.prepare_module_import()
import cns

LATFILE = "two_sites.yml"

NT = 12   # number of time slices
NTR = 400000  # number of trajectories
NMD = 3  # number of MD steps per trajectory
MDSTEP = 1/NMD  # size of MD steps
REPORT_FREQUENCY = 2000

U = 2
BETA = 3
SIGMA_KAPPA = 1

UTILDE = U*BETA/NT

# uses leapfrog
def moledyn(ham, phi, pi, direction=+1):
    eps = direction*MDSTEP

    # copy input to not overwrite it
    phi = cns.Vector(phi, dtype=complex)

    # initial half step
    pi = pi + cns.real(ham.force(phi))*eps/2

    for _ in range(NMD-1):
        phi += pi*eps
        pi += cns.real(ham.force(phi))*eps

    phi += pi*eps # final step (left out above)
    pi += cns.real(ham.force(phi))*eps/2  # final half step

    return phi, pi


def main():
    np.random.seed(1)

    with open(str(core.SCRIPT_PATH/"../lattices"/LATFILE), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    kappa = lat.hopping() * (BETA / NT)

    ham = cns.Hamiltonian(cns.HubbardGaugeAction(UTILDE),
                          cns.HubbardFermiAction(kappa, 0, SIGMA_KAPPA))

    # initial state
    phi = cns.Vector(np.random.normal(0, np.sqrt(UTILDE), lat.nx()*NT)+0j)

    # TODO store intitial traj?
    cfgs = []
    phases = []

    nacc = 0
    for i in range(NTR):
        pi = cns.Vector(np.random.normal(0, np.sqrt(UTILDE), len(phi))+0j)
        oldS = ham.eval(phi, pi)

        newPhi, newPi = moledyn(ham, phi, pi, +1)

        # check for imaginary parts
        if np.max(cns.imag(newPhi)) > 1e-14:
            print("phi has acquired an imaginary part: {}".format(newPhi))
            return
        if np.max(cns.imag(newPi)) > 1e-14:
            print("pi has acquired an imaginary part: {}".format(newPhi))
            return

        # new energy
        newS = ham.eval(newPhi, newPi)

        # if np.min((1, np.exp(cns.real(-newS+oldS)))) > np.random.uniform(0, 1):
        if np.exp(cns.real(oldS-newS)) > np.random.uniform(0, 1):
#            print("accept: ", newS-oldS)
            oldS = newS
            phi = newPhi
            nacc += 1
#        else:
#            print("reject: ", newS-oldS)

        cfgs.append(phi)
        phases.append(cns.imag(newS))

        if i%REPORT_FREQUENCY==0:
            print("Trajectory",i,"acceptance rate:",nacc/(i+1))

    print("max phases: ", np.max(phases))

    print("acceptance rate: ", nacc/NTR)

    M = cns.HubbardFermiMatrix(kappa, 0, SIGMA_KAPPA)

    dets = [ np.exp( cns.logdetM( M, cfg, False  ) ) for cfg in cfgs ]

    detsReal, detsImag = np.array(dets).real,np.array(dets).imag
    
    out_name = (LATFILE.split("."))[0]+f".nt{NT}nmd{NMD}"+".hdf5"
    
    with h5.open_file(out_name,"w") as out:
        out.create_group("/","det")
        out.create_array("/det","real",detsReal)
        out.create_array("/det","imag",detsImag)

if __name__ == "__main__":
    main()
