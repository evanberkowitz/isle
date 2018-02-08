#!/usr/bin/env python3

"""
Plot results of benchmarks.
"""

import sys
import pickle

import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Pass name of benchmark file as argument!")
        return

    ben = pickle.load(open(sys.argv[1], "rb"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Benchmark '{}'".format(sys.argv[1]))
    ax.set_xlabel(ben["xlabel"])
    ax.set_ylabel(ben["ylabel"])
    for name, values in ben["results"].items():
        ax.plot(ben["xvalues"], values, label=name)
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
