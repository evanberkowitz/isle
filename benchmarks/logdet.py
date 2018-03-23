#!/usr/bin/env python3

"""
Showcase and benchmark logdet.
"""

import timeit
import site
from pathlib import Path
import pickle

import yaml
import numpy as np

BENCH_PATH = Path(__file__).resolve().parent
site.addsitedir(str(BENCH_PATH/"../modules"))
import cns

def main():
    with open(str(BENCH_PATH/"../lattices/c60_ipr.yml"), "r") as yamlf:
        lat = yaml.safe_load(yamlf)

    nts = (4, 8, 12, 16, 20, 24, 28, 32)
    custom = []
    lapack = []

    logdet = cns.logdet
    for nt in nts:
        # create a normal distributed phi
        phi = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)

        # make a fermion matrix
        hfm = cns.HubbardFermiMatrix(lat.hopping(), phi, 0, -1)
        mmdag = cns.Matrix(hfm.Q(), dtype=complex)

        # do the benchmark
        custom.append(timeit.timeit("logdet(hfm)", globals=locals(), number=10)/10)
        lapack.append(timeit.timeit("logdet(mmdag)", globals=locals(), number=10)/10)

    # save benchmark to file
    pickle.dump({"xlabel": "Nt",
                 "ylabel": "time / s",
                 "xvalues": nts,
                 "results": {"custom": custom,
                             "LAPACK": lapack}}, open("logdet.ben", "wb"))

if __name__ == "__main__":
    main()
