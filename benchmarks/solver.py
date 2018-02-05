#!/usr/bin/env python3

"""
Demonstrate and benchmark custom solver for HFM based systems of equations.
"""

import timeit
import site
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
    solve = cns.solve
    getLU = cns.getLU
    for nt in nts:
        # make random auxilliary field and HFM
        phi = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)
        hfm = cns.HubbardFermiMatrix(lat.hopping(), phi, 0, 1, -1)

        # make random right hand side
        rhs = cns.Vector(np.random.randn(lat.nx()*nt)
                         + 1j*np.random.randn(lat.nx()*nt), dtype=complex)

        # measure time for LU-decompositon
        timeLU.append(timeit.timeit("getLU(hfm)", globals=locals(), number=20)/20)
        # measure time for solver itself
        lu = getLU(hfm)
        timeSolve.append(timeit.timeit("solve(lu, rhs)", globals=locals(), number=20)/20)

    print("timeLU:    ", timeLU)
    print("timeSolve: ", timeSolve)

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Benchmark HFM solver")
    ax.set_xlabel("Nt")
    ax.set_ylabel("time / s")
    ax.plot(nts, timeLU, label="LU-factorization")
    ax.plot(nts, timeSolve, label="Solver")
    ax.plot(nts, [tlu+ts for tlu, ts in zip(timeLU, timeSolve)], label="Total")
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
