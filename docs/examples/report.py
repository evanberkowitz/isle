"""!

"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

try:
    from sklearn.neighbors import KernelDensity
    DO_KDE = True
except ImportError:
    DO_KDE = False

import isle
import isle.meas
from isle.drivers.meas import Measure

ITR_LABEL = r"$i_{\mathrm{tr}}$"

# CREAL = "#cc071e"
CREAL = "#a11035"
CIMAG = "#00549f"

plt.rc("axes", edgecolor="0.35", labelcolor="0.35", linewidth=1.5)
plt.rc("text", color="0.35")
plt.rc("xtick", color="0.35")
plt.rc("ytick", color="0.35")
plt.rc("xtick.major", width=1.5)
plt.rc("ytick.major", width=1.5)


def oneDimKDE(dat, bandwidth=0.2, nsamples=1024, kernel="gaussian"):
    """!
    Perform a 1D kenrel density estimation on some data.
    \returns Tuple of sampling points and densities.
    """

    if DO_KDE:
        # make 2D array shape (len(totalPhi), 1)
        twoDDat = np.array(dat)[:, np.newaxis]
        # make 2D set of sampling points
        samplePts = np.linspace(np.min(dat)*1.1, np.max(dat)*1.1, nsamples)[:, np.newaxis]
        # estimate density
        dens = np.exp(KernelDensity(kernel=kernel, bandwidth=bandwidth)
                      .fit(twoDDat)
                      .score_samples(samplePts))
        return samplePts[:, 0], dens

    return None, None


def placeholder(ax):
    """!Mark an empty Axes by removing ticks and drawing a diagonal line."""
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax.plot(ax.get_xlim(), ax.get_ylim(), c="k", alpha=0.5)


def plotTotalPhi(measState, axPhi, axPhiHist):
    """!Plot MC history and histogram of total Phi."""

    # load data from previous measurement of compute from configurations
    with h5.File(str(measState.infname), "r") as h5f:
        if "field" in h5f:
            totalPhi = h5f["field"]["totalPhi"][()]
        elif "configuration" in h5f:
            meas = isle.meas.TotalPhi()
            measState.mapOverConfigs([(1, meas, None)])
            totalPhi = meas.Phi
        else:
            print("no configurations or total Phi found")
            totalPhi = None

    # need those in any case
    axPhi.set_title(r"$\Phi = \sum \varphi$")
    axPhiHist.set_title(r"Histogram of $\Phi$")

    if totalPhi is None:
        # no data - empty frames
        placeholder(axPhi)
        placeholder(axPhiHist)
        return

    # show history
    axPhi.plot(np.real(totalPhi), label="Re", c=CREAL, alpha=0.7)
    axPhi.plot(np.imag(totalPhi), label="Im", c=CIMAG, alpha=0.7)

    # show histograms + KDE
    axPhiHist.hist(np.real(totalPhi), label="totalPhi, real part, histogram",
                   orientation="horizontal", bins=len(totalPhi)//100, density=True,
                   facecolor=CREAL, alpha=0.7)
    samplePts, dens = oneDimKDE(np.real(totalPhi), bandwidth=3/5)
    if dens is not None:
        axPhiHist.plot(dens, samplePts, color=CREAL, label="totalPhi, real part, kde")
    if np.max(np.imag(totalPhi)) > 0:
        axPhiHist.hist(np.imag(totalPhi), label="totalPhi, imag part, histogram",
                       orientation="horizontal", bins=len(totalPhi)//100,
                       density=True, facecolor=CIMAG, alpha=0.7)
        samplePts, dens = oneDimKDE(np.imag(totalPhi), bandwidth=3/5)
        if dens is not None:
            axPhiHist.plot(dens, samplePts, color=CIMAG, label="totalPhi, imag part, kde")

    # prettify
    axPhi.set_xlabel(ITR_LABEL)
    axPhi.set_ylabel(r"$\Phi$")
    axPhi.legend()
    axPhiHist.set_ylim(axPhi.get_ylim())
    axPhiHist.tick_params(axis="y", which="both", labelleft=False, direction="inout")
    axPhiHist.set_xlabel("Frequency")

def runningDensityEstimate(data, binsize):
    data = np.array(data)
    indices = np.arange(binsize, len(data))
    density = np.empty(len(indices), dtype=float)
    for i, j in enumerate(indices):
        density[i] = np.mean(data[j-binsize:j])
    return indices, density

def plotTrajPoints(measState, ax):
    """!Plot MC history of accepted trajectory points."""

    # load data from configurations if possible
    with h5.File(str(measState.infname), "r") as h5f:
        trajPoints = []
        if "configuration" in h5f:
            for configName in sorted(h5f["configuration"], key=int):
                trajPoints.append(h5f["configuration"][configName]["trajPoint"][()])
        else:
            print("No traj points found")
            trajPoints = None

    if trajPoints is None:
        # no data - empty frame
        ax.set_title("Trajectory Points")
        placeholder(ax)
        return

    # plot raw and averaged data
    ax.set_title(f"Trajectory Points, average={np.mean(trajPoints)}")
    ax.plot(trajPoints, c=CIMAG, alpha=0.7, label="raw")
    ax.plot(*runningDensityEstimate(trajPoints, 20), c=CREAL, alpha=0.7, label="averaged")
    ax.set_xlabel(ITR_LABEL)
    ax.set_ylabel("Trajectory Point")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="lower center")

def _loadAction(measState):
    """!Load action from previous measurement of compute from configurations."""
    with h5.File(str(measState.infname), "r") as h5f:
        if "action" in h5f:
            return h5f["action"][()]
        if "configuration" in h5f:
            meas = isle.meas.Action()
            measState.mapOverConfigs([(1, meas, None)])
            return meas.action
    print("no configurations or action found")
    return None

def plotAction(measState, ax):
    """!Plot real and imaginary parts of the action."""
    action = _loadAction(measState)

    if action is None:
        # no data - empty frame
        ax.set_title(r"Action")
        placeholder(ax)
        return

    ax.set_title(f"Action, average = {np.mean(np.real(action)):1.3e} + {np.mean(np.imag(action)):1.3e} i")
    ax.plot(np.real(action), c=CREAL, alpha=0.7, label=r"$\mathrm{Re}(S)$")
    ax.plot(np.imag(action), c=CIMAG, alpha=0.7, label=r"$\mathrm{Im}(S)$")
    ax.set_xlabel(ITR_LABEL)
    ax.set_ylabel(r"$\mathrm{Re}(S(\phi))$")
    ax.legend()

def plotPhase(measState, axPhase, axPhase2D):
    """!Plot MC history and 2D histogram of the phase."""
    theta = np.imag(_loadAction(measState))

    if theta is None:
        # no data - empty frame
        placeholder(axPhase)
        placeholder(axPhase2D)
        return

    if np.max(np.abs(theta)) > 1e-13:
        # show 1D histogram + KDE
        axPhase.hist(theta, bins=len(theta)//100, density=True,
                     facecolor=CIMAG, alpha=0.7)
        samplePts, dens = oneDimKDE(theta, bandwidth=1/5)
        if dens is not None:
            axPhase.plot(samplePts, dens, color=CIMAG)

        axPhase.set_title(r"$\theta = \mathrm{Im}(S)$")
        axPhase.set_xlabel(r"$\theta$")
        axPhase.set_ylabel(r"Density")

        # show 2D histogram
        weight = np.exp(-1j*theta)
        x, y = np.real(weight), np.imag(weight)

        view = [-1.05, 1.05]
        hist = axPhase2D.hist2d(x, y, bins=51, range=[view, view], normed=1)
        axPhase2D.set_xlim(view)
        axPhase2D.set_ylim(view)
        axPhase2D.set_aspect(1)
        axPhase2D.get_figure().colorbar(hist[3], ax=axPhase2D)

        # indicate the average weight
        axPhase2D.scatter([np.mean(x)], [np.mean(y)], c="w", marker="x")

        axPhase2D.set_title(r"$w = e^{-i \theta}$")
        axPhase2D.set_xlabel(r"$\mathrm{Re}(w)$")
        axPhase2D.set_ylabel(r"$\mathrm{Im}(w)$")

    else:
        placeholder(axPhase)
        axPhase.text(0.5, 0.5, r"$\theta = 0$", transform=axPhase.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     size=20, bbox=dict(facecolor="white", alpha=0.5, edgecolor='0.35'))
        placeholder(axPhase2D)
        axPhase2D.text(0.5, 0.5, r"$\theta = 0$", transform=axPhase2D.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       size=20, bbox=dict(facecolor="white", alpha=0.5, edgecolor='0.35'))

def _figure():
    fig = plt.figure(figsize=(11, 7))
    gspec = gridspec.GridSpec(3, 4, height_ratios=[5,5,3])

    axTP = fig.add_subplot(gspec[0, 0:2])
    axAct = fig.add_subplot(gspec[0, 2:4])
    axPhase = fig.add_subplot(gspec[1, 2])
    axPhase2D = fig.add_subplot(gspec[1, 3])
    axText = fig.add_subplot(gspec[2, :])

    gspecPhi = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0,
                                                subplot_spec=gspec[1, 0:2])
    axPhi = fig.add_subplot(gspecPhi[0, 0])
    axPhiHist = fig.add_subplot(gspecPhi[0, 1])

    return fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText)

def _nconfig(fname):
    """!Attempt to get the number of configurations in a file."""
    with h5.File(str(fname), "r") as h5f:
        if "configuration" in h5f:
            return len(h5f["configuration"])
        if "action" in h5f:
            return len(h5f["action"])
    return None

def _formatParams(params):
    """!Format parameters as a multi line string."""
    lines = []
    line = ""
    for name, val in params.asdict().items():
        line += f"{name} = {val}, "
        if len(line) > 70:
            lines.append(line[:-2])
            line = "                 "
    if line.strip():
        lines.append(line)
    return "\n".join(lines)

def main():
    args = isle.cli.init("report", name="report")
    infname = args.input[0]

    lattice, params, makeActionSrc = isle.fileio.h5.readMetadata(args.input)

    measState = Measure(lattice, params,
                        isle.fileio.callFunctionFromSource(makeActionSrc, lattice, params),
                        infname, None)

    fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText) = _figure()
    fig.canvas.set_window_title(f"Isle Overview - {infname}")

    plotTotalPhi(measState, axPhi, axPhiHist)
    plotTrajPoints(measState, axTP)
    plotAction(measState, axAct)
    plotPhase(measState, axPhase, axPhase2D)

    axText.axis("off")
    axText.text(0, 0, fontsize=13, linespacing=2, verticalalignment="bottom",
                s=rf"""Ensemble: $N_{{\mathrm{{config}}}} = {_nconfig(infname)}$ in {infname}
     Lattice: {lattice.name}   $N_t = {measState.lattice.nt()}$, $N_x = {measState.lattice.nx()}$    '{lattice.comment}'
      Model: {_formatParams(measState.params)}""")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
