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

NTS = (4, 8, 12, 16, 24, 32, 64, 96)
NREP = 5


def main():
    with open(str(BENCH_PATH/"../lattices/c60_ipr.yml"), "r") as yamlf:
        lat = yaml.safe_load(yamlf)
    nx = lat.nx()

    functions = {
        "dense": (cns.logdet, "fn(Q)", "Q = cns.Matrix(hfm.Q(phi))"),
        "logdetQ": (cns.logdetQ, "fn(hfm, phi)", ""),
        "logdetM": (cns.logdetM, "fn(hfm, phi, cns.PH.PARTICLE)+fn(hfm, phi, cns.PH.HOLE)", "")
    }
    times = {key: [] for key in functions}

    for nt in NTS:
        print("nt = {}".format(nt))

        # make random auxilliary field and HFM
        phi = cns.Vector(np.random.randn(nx*nt)
                         + 1j*np.random.randn(nx*nt))
        hfm = cns.HubbardFermiMatrix(lat.hopping()/nt, 0, -1)

        # do the benchmarks
        for name, (function, execStr, setupStr) in functions.items():
            if nt > 12 and name == "dense":   # this is just too slow
                continue

            times[name].append(timeit.timeit(execStr,
                                             setup=setupStr,
                                             globals={"fn": function, "hfm": hfm,
                                                      "phi": phi, "cns": cns},
                                             number=NREP) / NREP)

    # save benchmark to file
    pickle.dump({"xlabel": "Nt",
                 "ylabel": "time / s",
                 "xvalues": NTS,
                 "results": times},
                open("logdet.ben", "wb"))

if __name__ == "__main__":
    main()
