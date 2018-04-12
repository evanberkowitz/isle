import sys
import os
import contextlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tables as h5
import time


def main():
    
    filename = sys.argv[-1]
    
    
    with h5.open_file(filename,"r") as data:
        detsReal = data.get_node("/det","real").read()
        detsImag = data.get_node("/det","imag").read()

    plt.figure(figsize=(4,4))
    plt.title(r"$\mathrm{det}(M)$ "+filename+" "+str(len(detsReal))+"traj.")
    plt.hist2d(detsReal, detsImag, bins=40, norm=LogNorm())
    plt.xlim([-15,25])
    plt.ylim([-20,20])
    # plt.show()
    plt.savefig(filename.split("nmd")[0]+".png")


if __name__ == "__main__":
    main()
