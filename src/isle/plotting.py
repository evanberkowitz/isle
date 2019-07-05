r"""! \file
Utilities for plotting.

This module requires matplotlib which is not needed for the core operation of isle.
Thus, plotting is not imported automatically when isle is imported.
"""

from logging import getLogger

## \cond DO_NOT_DOCUMENT
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

except ImportError:
    getLogger(__name__).error("Cannot import matplotlib, plotting functionality is not available.")
    raise

try:
    from sklearn.neighbors import KernelDensity
    _DO_KDE = True
except ImportError:
    _DO_KDE = False

import numpy as np
import h5py as h5

import isle.meas
from isle.collection import neighbors
## \endcond DO_NOT_DOCUMENT

def setupMPL():
    """!Modify matplotlib's plotting style for a uniform look of all Isle applications."""
    plt.rc("axes.formatter", useoffset=False)  # no one knows how to read it anyway
    plt.rc("errorbar", capsize=3)  # puts caps at the end of errorbars
    plt.rc("legend", fancybox=False)  # don't use round corners
    plt.rc("axes", edgecolor="0.35", labelcolor="0.35", linewidth=1.5,
           prop_cycle=mpl.cycler("color",
                                 ["#a11035", "#00549f", "#2ca02c", "#ff7f0e",
                                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                                  "#bcbd22", "#17becf"]))
    plt.rc("text", color="0.35")
    plt.rc("xtick", color="0.35", top=True, direction="in")
    plt.rc("ytick", color="0.35", right=True, direction="in")
    plt.rc("xtick.major", width=1.5)
    plt.rc("ytick.major", width=1.5)


def placeholder(ax):
    """!Mark an empty Axes by removing ticks and drawing a diagonal line."""
    ax.tick_params(axis="both", which="both",
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    if ax.name != "polar":
        ax.plot(ax.get_xlim(), ax.get_ylim(), c="k", alpha=0.5)

def oneDimKDE(dat, bandwidth=0.2, nsamples=1024, kernel="gaussian",
              sampleRange=None):
    r"""!
    Perform a 1D kenrel density estimation on some data.
    \returns Tuple of sampling points and densities.
             Or return (None, None) if scikit-learn is not available.
    """

    if _DO_KDE:
        # make 2D array shape (len(totalPhi), 1)
        twoDDat = np.array(dat)[:, np.newaxis]
        # make 2D set of sampling points
        sampleRange = sampleRange if sampleRange else (np.min(dat)*1.1, np.max(dat)*1.1)
        samplePts = np.linspace(*sampleRange, nsamples)[:, np.newaxis]
        # estimate density
        dens = np.exp(KernelDensity(kernel=kernel, bandwidth=bandwidth)
                      .fit(twoDDat)
                      .score_samples(samplePts))
        return samplePts[:, 0], dens

    return None, None

def polarDensity(data, innerRadius, outerRadius, kde,
                 bins=None, bandwidth=None, kernel=None):
    r"""!
    Calculate density of data for a plot in polar coordinates.
    \param data 1D array-like.
    \param innerRadius Radius of the base line for showing the density.
                       Corresponds to density=0.
    \param outerRadius Maximum radius, corresponds to density=1.
    \param kde If `True`, perfom a KDE to estimate the density, otherwise use a histogram.
    \param bins Number of histogram bins or number of KDE samples.
    \param bandwidth *KDE only*. Bandwidth of the kernel.
    \param kernel *KDE only*. The kind of kernel to use. See `sklearn.neighbors.KernelDensity`.
    \returns Two lists:
             - Angles of the points where the density was estimated. Range: `[-pi, pi]`
             - Radii computed from the density.
    """

    log = getLogger(__name__)

    if kde:
        bandwidth = 0.01 if not bandwidth else bandwidth
        bins = int(np.sqrt(len(data))) if not bins else bins
        kernel = "gaussian" if not kernel else kernel

        xlist, ytmp = oneDimKDE(np.concatenate((data, data+2*np.pi, data-2*np.pi)),
                                bandwidth, bins, kernel, (-np.pi, np.pi))
        # Multiply by 3 because twoDDat has 3 replicas of the data,
        # so the normalisation of dens is off by that factor.
        ylist = innerRadius + (outerRadius-innerRadius)*ytmp*3

    else:
        if bandwidth is not None:
            log.warning("polarDensity: Argument 'bandwidth' has no effect when kde==False")
        if kernel is not None:
            log.warning("polarDensity: Argument 'kernel' has no effect when kde==False")
        if not bins:
            bins = int(np.sqrt(len(data)))

        hist, bin_edges = np.histogram(data, bins, (-np.pi, np.pi), density=True)
        # angle, outer radius pairs
        points = [((high+low)/2,  # assumes that there are no bins on the -pi, pi boundary
                   innerRadius + (outerRadius-innerRadius)*val)
                  for val, (low, high) in zip(hist, neighbors(bin_edges))]
        xlist, ylist = zip(*points)

    return xlist, ylist

def runningAverage(data, binsize):
    """!
    Compute an average over the previous `binsize` points for each point in `data[binsize:]`.
    """
    data = np.array(data)
    indices = np.arange(binsize, len(data))
    density = np.empty(len(indices), dtype=float)
    for i, j in enumerate(indices):
        density[i] = np.mean(data[j-binsize:j])
    return indices, density

def polarHistogram(ax, data, kde=False, bins=None, bandwidth=None, kernel=None,
                   innerRadius=1, outerRadius=2,
                   ls=None, marker=".", fill=False,
                   edgecolor=None, facecolor=None, edgealpha=None, facealpha=None):
    r"""!
    Plot a polar histogram into an Axes.
    """

    xlist, ylist = polarDensity(data, innerRadius, outerRadius, kde,
                                bins, bandwidth, kernel)

    # draw base-line circle
    baselines, = ax.plot(list(xlist)+[xlist[0]], [innerRadius]*(len(xlist)+1), c=edgecolor)

    # the above is the first plot command, use it to set the color if not specified
    if edgecolor is None:
        edgecolor = baselines.get_color()
    if facecolor is None:
        facecolor = baselines.get_color()

    # lines connecting baseline to markers
    if ls != "":
        for x, y in zip(xlist, ylist):
            ax.plot((x, x), (innerRadius, y), ls=ls, c=edgecolor, alpha=edgealpha)

    # fill area between baseline and markers
    if fill:
        ax.fill_between(list(xlist)+[xlist[0]],
                        [innerRadius]*(len(xlist)+1),
                        list(ylist)+[ylist[0]],
                        facecolor=facecolor, alpha=facealpha)

    # show markers at the outer tips of the bins
    if marker:
        lines = ax.plot(xlist, ylist, ls="", marker=marker, c=edgecolor, alpha=edgealpha)
    else:
        lines = []

    return lines

def setPolarTicks(ax, which="both"):
    r"""!
    Set the ticks for a polar Axes.
    """

    allowedWhich = ("both", "x", "y", "none")
    if which not in allowedWhich:
        getLogger(__name__).error("Invalid value for argument 'which': '%s'\n"
                                  "Supported values are %s",
                                  which, allowedWhich)
        raise ValueError(f"Invalid value for argument 'which': '{which}'")

    if which in ("both", "x"):
        ax.set_xticks([0, np.pi/4, np.pi/2, np.pi*3/4, np.pi,
                       np.pi*5/4, 3*np.pi/2, np.pi*7/4])
        ax.xaxis.set_ticklabels([r"0",
                                 r"$\frac{1}{4}\pi$",
                                 r"$\frac{1}{2}\pi$",
                                 r"$\frac{3}{4}\pi$",
                                 r"$\pi$",
                                 r"$-\frac{3}{4}\pi$",
                                 r"$-\frac{1}{2}\pi$",
                                 r"$-\frac{1}{4}\pi$"])

    if which in ("both", "y"):
        ax.set_yticks([])


def plotTotalPhi(measState, axPhi, axPhiHist):
    """!Plot MC history and histogram of total Phi."""

    # load data from previous measurement or compute from configurations
    with h5.File(str(measState.infile), "r") as h5f:
        if "field" in h5f:
            totalPhi = h5f["field"]["totalPhi"][()]
        elif "configuration" in h5f:
            meas = isle.meas.TotalPhi(None)
            measState.mapOverConfigs([meas])
            totalPhi = meas.Phi
        else:
            getLogger(__name__).info("No configurations or total Phi found.")
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
    axPhi.plot(np.real(totalPhi), label="Re", c="C0", alpha=0.8)
    axPhi.plot(np.imag(totalPhi), label="Im", c="C1", alpha=0.8)

    # show histograms + KDE
    axPhiHist.hist(np.real(totalPhi), label="totalPhi, real part, histogram",
                   orientation="horizontal", bins=max(len(totalPhi), 1000)//100, density=True,
                   facecolor="C0", alpha=0.7)
    samplePts, dens = oneDimKDE(np.real(totalPhi), bandwidth=3/5)
    if dens is not None:
        axPhiHist.plot(dens, samplePts, color="C0", label="totalPhi, real part, kde")
    if np.max(np.imag(totalPhi)) > 0:
        axPhiHist.hist(np.imag(totalPhi), label="totalPhi, imag part, histogram",
                       orientation="horizontal", bins=len(totalPhi)//100,
                       density=True, facecolor="C1", alpha=0.7)
        samplePts, dens = oneDimKDE(np.imag(totalPhi), bandwidth=3/5)
        if dens is not None:
            axPhiHist.plot(dens, samplePts, color="C1", label="totalPhi, imag part, kde")

    # prettify
    axPhi.set_xlabel(r"$i_{\mathrm{tr}}$")
    axPhi.set_ylabel(r"$\Phi$")
    axPhi.legend()
    axPhiHist.set_ylim(axPhi.get_ylim())
    axPhiHist.tick_params(axis="y", which="both", labelleft=False)
    axPhiHist.set_xlabel("Frequency")

def plotTrajPoints(measState, ax):
    """!Plot MC history of accepted trajectory points."""

    # load data from configurations if possible
    with h5.File(str(measState.infile), "r") as h5f:
        trajPoints = []
        if "configuration" in h5f:
            for configName in sorted(h5f["configuration"], key=int):
                trajPoints.append(h5f["configuration"][configName]["trajPoint"][()])
        else:
            getLogger(__name__).info("No traj points found.")
            trajPoints = None

    if trajPoints is None:
        # no data - empty frame
        ax.set_title("Trajectory Points")
        placeholder(ax)
        return

    # plot raw and averaged data
    mean = np.mean(trajPoints)
    ax.set_title(f"Trajectory Points, average={mean}")
    ax.plot(trajPoints, c="C1", alpha=0.8, label="Raw")
    ax.plot(*runningAverage(trajPoints, 20), c="C0", alpha=0.8, label="Running Average")
    ax.axhline(mean, linestyle="--", c="k", alpha=0.8, label="Average")
    ax.set_xlabel(r"$i_{\mathrm{tr}}$")
    ax.set_ylabel("Trajectory Point")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc="lower center")


def plotAction(action, ax):
    """!Plot real and imaginary parts of the action."""
    if action is None:
        # no data - empty frame
        ax.set_title(r"Action")
        placeholder(ax)
        return

    ax.set_title(f"Action, average = {np.mean(np.real(action)):1.3e} + {np.mean(np.imag(action)):1.3e} i")
    ax.plot(np.real(action), c="C0", alpha=0.8, label=r"$\mathrm{Re}(S)$")
    ax.plot(np.imag(action), c="C1", alpha=0.8, label=r"$\mathrm{Im}(S)$")
    ax.set_xlabel(r"$i_{\mathrm{tr}}$")
    ax.set_ylabel(r"$\mathrm{Re}(S(\phi))$")
    ax.legend()

# TODO include all weights including jacobian
def plotPhase(action, axPhase, axPhase2D):
    """!Plot MC history and 2D histogram of the phase."""
    if action is None:
        # no data - empty frame
        placeholder(axPhase)
        placeholder(axPhase2D)
        return

    theta = np.imag(action)

    if np.max(np.abs(theta)) > 1e-13:
        # show 1D histogram + KDE
        axPhase.hist(theta, bins=max(len(theta)//100, 10), density=True,
                     facecolor="C1", alpha=0.7)
        samplePts, dens = oneDimKDE(theta, bandwidth=1/5)
        if dens is not None:
            axPhase.plot(samplePts, dens, color="C1")

        axPhase.set_title(r"$\theta = \mathrm{Im}(S)$")
        axPhase.set_xlabel(r"$\theta$")
        axPhase.set_ylabel(r"Density")

        # show 2D histogram
        polarHistogram(axPhase2D, theta, _DO_KDE, bins=50, ls="-", marker=".", fill=True,
                       edgecolor="C1", facealpha=0.3)
        average = np.mean(np.exp(-1j*theta))
        axPhase2D.plot((np.angle(average),), (np.abs(average),), ls="", marker="x",
                       c=mpl.rcParams["text.color"], markersize=10)
        setPolarTicks(axPhase2D)
        axPhase2D.set_ylim((0, 2.1))  # the outerRadius of the plot is 2
        axPhase2D.set_ylabel(r"$\theta$", labelpad=20)
        axPhase2D.yaxis.set_label_position("right")

    else:
        placeholder(axPhase)
        axPhase.text(0.5, 0.5, r"$\theta = 0$", transform=axPhase.transAxes,
                     horizontalalignment='center', verticalalignment='center',
                     size=20, bbox=dict(facecolor="white", alpha=0.5, edgecolor='0.35'))
        placeholder(axPhase2D)
        axPhase2D.text(0.5, 0.5, r"$\theta = 0$", transform=axPhase2D.transAxes,
                       horizontalalignment='center', verticalalignment='center',
                       size=20, bbox=dict(facecolor="white", alpha=0.5, edgecolor='0.35'))

def plotCorrelators(measState, axP, axH):
    r"""!
    Plot particle and hole Correlators.
    The correlators are reweighted with the imaginary part of the action if possible.
    \returns True if successful, False if no data was found.
    """

    # load data from previous measurement
    with h5.File(str(measState.infile), "r") as h5f:
        if "correlation_functions" in h5f:
            dsetP = h5f["correlation_functions/single_particle/correlators"]
            corrP = dsetP[()]
            dsetH = h5f["correlation_functions/single_hole/correlators"]
            corrH = dsetH[()]

            try:
                weightP = isle.h5io.loadActionWeightsFor(dsetP)
                weightH = isle.h5io.loadActionWeightsFor(dsetH)
            except KeyError:
                getLogger(__name__).warning("Unable to load action to do reweighting for correlators."
                                            "Showing correlators without applying weights.")
                weightP = np.ones(dsetP.shape[0])
                weightH = np.ones(dsetH.shape[0])
        else:
            getLogger(__name__).error("No correlation functions found.")
            return False

    # ensemble averages
    corrP = np.sum(corrP*weightP.reshape(-1, 1, 1, 1), axis=0) / np.sum(weightP)
    corrH = np.sum(corrH*weightH.reshape(-1, 1, 1, 1), axis=0) / np.sum(weightH)

    # plot all correlators
    nx = corrP.shape[0]
    for i, j in zip(range(nx), range(nx)):
        axP.plot(np.real(corrP[i, j, :]), color="C0", alpha=0.6)
    for i, j in zip(range(nx), range(nx)):
        axH.plot(np.real(corrH[i, j, :]), color="C0", alpha=0.6)

    axP.set_title("Particles")
    axH.set_title("Holes")
    axP.set_xlabel(r"$N_t$")
    axH.set_xlabel(r"$N_t$")
    axP.set_yscale("log")
    axH.set_yscale("log")

    return True

def plotTunerFit(ax, probabilityPoints, trajPointPoints, fitResult, verification):
    r"""!
    \todo document
    """

    x, y, err = zip(*probabilityPoints)
    ax.errorbar(np.asarray(x)+0.05, y, err, ls="", marker=".", label=r"min(1, $\exp{(dH)}$)")
    x, y, err = zip(*trajPointPoints)
    ax.errorbar(np.asarray(x)-0.05, y, err, ls="", marker=".", label="trajPoint")

    if fitResult is not None:
        x = np.linspace(0, ax.get_xlim()[1]*1.1, 1000)
        bestFit, otherFits = fitResult.evalOn(x)
        for y in otherFits:
            ax.plot(x, y, c="k", alpha=0.5)
        ax.plot(x, bestFit, c="k")

    if verification:
        # draw a rectangle just inside of the axes borders
        ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, linewidth=2, transform=ax.transAxes,
                               edgecolor="#E3D514", facecolor="none"))

def plotTunerTrace(ax, records):
    r"""!
    \todo document
    """

    axNstep = ax
    axProbTP = ax.twiny()

    lastYMax = 0
    axNstep.axhline(-0.5, ls=":", c="k", alpha=0.5)
    for record in records:
        if len(record) == 0:
            continue

        # y values for this record (~ trajectory index)
        yMax = lastYMax + len(record)
        y = np.arange(lastYMax, yMax)
        lastYMax = yMax

        lineProb, = axProbTP.plot(record.probabilities, y, ls="", marker="v", c="C0")
        lineTP, = axProbTP.plot(record.trajPoints, y, ls="", marker="x", c="C1")
        lineNstep, = axNstep.plot([record.nstep]*2, (y[0], y[-1]),
                                  ls="-", marker="", linewidth=2, c="k")
        axNstep.axhline(yMax-0.5, ls=":", c="k", alpha=0.5)

    axNstep.xaxis.set_major_locator(MaxNLocator(integer=True))
    axNstep.yaxis.set_major_locator(MaxNLocator(integer=True))
    axNstep.set_xlabel(r"$N_{\mathrm{MD}}$")
    axNstep.set_ylabel("trajectory")
    axProbTP.set_xlabel(r"min(1, $\exp{(dH)}$) | trajPoint")

    # manual legend so that all lines are shown
    axNstep.legend([lineNstep, lineProb, lineTP],
                   [r"$N_{\mathrm{MD}}$", r"min(1, $\exp{(dH)}$)", r"trajPoint"],
                   bbox_to_anchor=(0, -0.08, 1, 0), loc="lower left", mode="expand",
                   ncol=3, borderaxespad=0, handletextpad=0.05)
