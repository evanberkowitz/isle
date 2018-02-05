#!/usr/bin/env python3

"""
Showcase and benchmark logdet.
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
        hfm = cns.HubbardFermiMatrix(lat.hopping(), phi, 0, 1, -1)
        mmdag = cns.Matrix(hfm.MMdag(), dtype=complex)

        # do the benchmark
        custom.append(timeit.timeit("logdet(hfm)", globals=locals(), number=50))
        lapack.append(timeit.timeit("logdet(mmdag)", globals=locals(), number=50))

    # plot result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Benchmark logdet")
    ax.set_xlabel("Nt")
    ax.set_ylabel("time / s")
    ax.plot(nts, custom, label="custom")
    ax.plot(nts, lapack, label="LAPACK")
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
