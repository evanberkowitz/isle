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
    cbtdlu = []
    analytical = []
    lapack = []

    logdetQ = cns.logdetQ
    logdetM = cns.logdetM
    logdet = cns.logdet
    for nt in nts:
        print("doing nt = {}".format(nt))
        # create a normal distributed phi
        phi = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)

        # make a fermion matrix
        hfm = cns.HubbardFermiMatrix(lat.hopping()/nt, 0, -1)
        Q = cns.Matrix(hfm.Q(phi))

        # do the benchmark
        cbtdlu.append(timeit.timeit("logdetQ(hfm,phi)", globals=locals(), number=10)/10)
        analytical.append(timeit.timeit("logdetM(hfm, phi , False)+logdetM(hfm, phi, True)", globals=locals(), number=10)/10)
        lapack.append(timeit.timeit("logdet(Q)", globals=locals(), number=10)/10)

    # save benchmark to file
    pickle.dump({"xlabel": "Nt",
                 "ylabel": "time / s",
                 "xvalues": nts,
                 "results": {"CBTDLU": cbtdlu,
                             "analytical": analytical,
                             "LAPACK": lapack}}, open("logdet.ben", "wb"))

if __name__ == "__main__":
    main()
