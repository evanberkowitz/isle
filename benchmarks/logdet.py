"""
Showcase and benchmark logdet.
"""

import timeit
from pathlib import Path
import pickle

import yaml
import numpy as np

import isle

BENCH_PATH = Path(__file__).resolve().parent

NTS = (4, 8, 12, 16, 24, 32, 64, 96)
NREP = 500


def nt_scaling():
    "Benchmark scaling with Nt."

    lat = isle.LATTICES["c60_ipr"]
    nx = lat.nx()

    functions = {
        "dense": (isle.logdet, "fn(Q)", "Q = isle.Matrix(hfm.Q(phi))"),
        "logdetQ": (isle.logdetQ, "fn(hfm, phi)", ""),
        "logdetM": (isle.logdetM, "fn(hfm, phi, isle.Species.PARTICLE)+fn(hfm, phi, isle.Species.HOLE)", ""),
    }
    times = {key: [] for key in functions}

    for nt in NTS:
        print("nt = {}".format(nt))

        # make random auxilliary field and HFM
        phi = isle.Vector(np.random.randn(nx*nt)
                         + 1j*np.random.randn(nx*nt))
        hfm = isle.HubbardFermiMatrixDia(lat.hopping()/nt, 0, -1)

        # do the benchmarks
        for name, (function, execStr, setupStr) in functions.items():
            if nt > 12 and name == "dense":   # this is just too slow
                continue

            times[name].append(timeit.timeit(execStr,
                                             setup=setupStr,
                                             globals={"fn": function, "hfm": hfm,
                                                      "phi": phi, "isle": isle},
                                             number=NREP) / NREP)

    # save benchmark to file
    pickle.dump({"xlabel": "Nt",
                 "ylabel": "time / s",
                 "xvalues": NTS,
                 "results": times},
                open("logdet.ben", "wb"))


def nx_scaling():
    "Benchmark scaling with Nx."

    lattices = [isle.LATTICES[name]
                for name in isle.LATTICES.keys()]
    lattices = sorted(lattices, key=lambda lat: lat.nx())
    NT = 16

    times = {"logdetM": []}

    nxs = []
    for lat in lattices:
        print(f"lat = {lat.name}")
        nx = lat.nx()
        if nx in nxs:
            continue

        nxs.append(nx)
        lat.nt(NT)

        # make random auxilliary field and HFM
        phi = isle.Vector(np.random.randn(nx*NT)
                          + 1j*np.random.randn(nx*NT))
        hfm = isle.HubbardFermiMatrixDia(lat.hopping()/NT, 0, -1)

        times["logdetM"].append(np.min(timeit.repeat(
            "isle.logdetM(hfm, phi, isle.Species.PARTICLE)",
            globals={"hfm": hfm, "phi": phi, "isle": isle},
            number=NREP, repeat=5)) / NREP)

    # save benchmark to file
    pickle.dump({"xlabel": "Nx",
                 "ylabel": "time / s",
                 "xvalues": nxs,
                 "results": times},
                open("logdet.ben", "wb"))


def main():
    nx_scaling()

if __name__ == "__main__":
    main()
