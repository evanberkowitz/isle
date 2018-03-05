#!/usr/bin/env python3

"""
Demonstrate and benchmark custom solver for HFM based systems of equations.
"""

import timeit
import site
import pickle
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

BENCH_PATH = Path(__file__).resolve().parent
site.addsitedir(str(BENCH_PATH/"../modules"))
import cns

def main():
    np.random.seed(1)

    with open(str(BENCH_PATH/"../lattices/c60_ipr.yml"), "r") as yamlf:
        lat = yaml.safe_load(yamlf)

    nts = (4, 8, 16, 32)
    timeLU = []
    timeSolve = []
    timePardiso = []
    solve = cns.solve
    getLU = cns.getLU
    solvePardiso = cns.solvePardiso
    for nt in nts:
        # make random auxilliary field and HFM
        phi = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)
        hfm = cns.HubbardFermiMatrix(lat.hopping(), phi, 0, 1, -1)

        # make random right hand side
        rhs = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)

        # measure time for LU-decompositon
        timeLU.append(timeit.timeit("getLU(hfm)", globals=locals(), number=1)/1)
        # measure time for solver itself
        lu = getLU(hfm)
        timeSolve.append(timeit.timeit("solve(lu, rhs)", globals=locals(), number=1)/1)

        timePardiso.append(timeit.timeit("solvePardiso(hfm, rhs)", globals=locals(), number=1)/1)

    # save benchmark to file
    pickle.dump({"xlabel": "Nt",
                 "ylabel": "time / s",
                 "xvalues": nts,
                 "results": {"custom lu": timeLU,
                             "custom solve": timeSolve,
                             "custom total": [tlu+ts for tlu, ts in zip(timeLU, timeSolve)],
                             "PARDISO": timePardiso}},
                open("solver.ben", "wb"))

if __name__ == "__main__":
    main()
