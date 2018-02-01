#!/usr/bin/env python3

"""
Showcase HubbardFermiMatrix and logdet.
"""

import yaml
import numpy as np

import core
core.prepare_module_import()
import cns

def main():
    with open("../lattices/c60_ipr.yml", "r") as yamlf:
        lat = yaml.safe_load(yamlf)

    # use this instead of nt read by lattice
    nt = 4

    # create a normal distributed phi
    phi = cns.Vector(np.random.randn(lat.nx()*nt)
                     + 1j*np.random.randn(lat.nx()*nt), dtype=complex)

    # make a fermion matrix
    hfm = cns.HubbardFermiMatrix(lat.hopping(), phi, 0, 1, -1)

    # compare logdets
    print("Dense logdet (LAPACK): ", cns.logdet(cns.Matrix(hfm.MMdag(), dtype=complex)))
    print("Sparse logdet (custom LU): ", cns.logdet(hfm))

if __name__ == "__main__":
    main()
