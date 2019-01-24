r"""! \file
Script to report on input and output files of Isle.

Run the script via show.main().
"""

from logging import getLogger

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # (unused import) pylint: disable=W0611
    from matplotlib import cm

except ImportError:
    getLogger("isle.show").error("Cannot import matplotlib, show command is not available.")
    raise

import numpy as np
import h5py as h5

import isle
import isle.plotting
from isle.drivers.meas import Measure
from isle.meta import callFunctionFromSource

def _overviewFigure():
    """!Open a new figure and construct a GridSpec to lay out subplots."""

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

def _loadAction(measState):
    """!Load action from previous measurement or compute from configurations."""
    with h5.File(str(measState.infname), "r") as h5f:
        if "action" in h5f:
            return h5f["action"][()]
        if "configuration" not in h5f:
            getLogger("isle.show").info("No configurations or action found.")
            return None

    # do this here so the file is closed when the meas driver reads from it
    meas = isle.meas.Action()
    measState.mapOverConfigs([(1, meas, None)])
    return meas.action

def _formatParams(params):
    """!Format parameters as a multi line string."""
    lines = []
    line = ""
    for name, val in params.asdict().items():
        line += f"{name}={val}, "
        if len(line) > 70:
            lines.append(line[:-2])
            line = "                 "
    if line.strip():
        lines.append(line)
    return "\n".join(lines)

def _overview(infname, lattice, params, makeActionSrc):
    """!
    Show an overview of a HDF5 file.
    """

    log = getLogger("isle.show")
    log.info("Showing overview of file %s", infname)

    if lattice is None or params is None or makeActionSrc is None:
        log.error("Could not find all required information in the input file to generate an overview."
                  "Need HDF5 files.")
        return

    # use this to bundle information and perform simple measurements if needed
    measState = Measure(lattice, params,
                        callFunctionFromSource(makeActionSrc, lattice, params),
                        infname, None)

    # set up the figure
    fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText) = _overviewFigure()
    fig.canvas.set_window_title(f"Isle Overview - {infname}")

    # plot a bunch of stuff
    isle.plotting.plotTotalPhi(measState, axPhi, axPhiHist)
    isle.plotting.plotTrajPoints(measState, axTP)
    action = _loadAction(measState)
    isle.plotting.plotAction(action, axAct)
    isle.plotting.plotPhase(action, axPhase, axPhase2D)

    # show metadata at bottom of figure
    axText.axis("off")
    axText.text(0, 0, fontsize=13, linespacing=2, verticalalignment="bottom",
                s=rf"""Ensemble: $N_{{\mathrm{{config}}}}={_nconfig(infname)}$ in {infname}
     Lattice: {lattice.name}   $N_t={measState.lattice.nt()}$, $N_x={measState.lattice.nx()}$    '{lattice.comment}'
      Model: {_formatParams(measState.params)}""")

    fig.tight_layout()

def _lattice(infname, lattice):
    """!
    Show the hopping matrix as a 3D grid.
    """

    getLogger("isle.show").info("Showing lattice in file %s", infname)

    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(f"Isle Lattice - {infname}")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(lattice.name)
    ax.axis("equal")

    # draw edges
    hopping = lattice.hopping()
    maxHopping = np.max(isle.Matrix(hopping))
    for i in range(lattice.nx()-1):
        for j in range(i+1, lattice.nx()):
            if lattice.areNeighbors(i, j):
                ax.plot(*zip(lattice.position(i), lattice.position(j)),
                        color=cm.viridis_r(hopping[i, j]/maxHopping))

    # an x marks the center
    center = sum(np.array(lattice.position(i)) for i in range(lattice.nx()))/lattice.nx()
    ax.scatter((center[0], ), (center[1], ), marker="x", c="k")

    # make background white
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def _correlator(infname, lattice, params, makeActionSrc):
    """!
    Show all-to-all correlators.
    """

    log = getLogger("isle.show")
    log.info("Showing correlators in file %s", infname)

    if lattice is None or params is None or makeActionSrc is None:
        log.error("Could not find all required information in the input file to show correlators."
                  "Need HDF5 files.")
        return

    # use this to bundle information and perform simple measurements if needed
    measState = Measure(lattice, params,
                        callFunctionFromSource(makeActionSrc, lattice, params),
                        infname, None)

    fig = plt.figure(figsize=(11, 5))
    fig.canvas.set_window_title(f"Isle Correlators - {infname}")

    if isle.plotting.plotCorrelators(measState, fig.add_subplot(121), fig.add_subplot(122)):
        fig.tight_layout()
    else:
        # failed -> do not show the figure
        plt.close(fig)

def _verifyIsleVersion(version, fname):
    """!
    Check version of Isle, warn if it does not match.
    """

    this = isle.isleVersion
    comp = isle.util.compareVersions(this, version)
    if comp == "none":
        getLogger("isle.show").warning("Extra version string of Isle (%s) is different from "
                                       "version in file %s (%s)",
                                       this, fname, version)
    if comp != "equal":
        getLogger("isle.show").warning("Version of Isle (%s), is %s than in file %s (%s).",
                                       this, comp, fname, version)

def _loadMetadata(fname, ftype):
    """!
    Load all available metadata from a file.
    Abstracts away the file type.
    """

    if ftype == isle.fileio.FileType.HDF5:
        lattice, params, makeActionSrc, versions = isle.fileio.h5.readMetadata((fname, ftype))
        _verifyIsleVersion(versions["isle"], fname)

    elif ftype == isle.fileio.FileType.YAML:
        lattice = isle.fileio.yaml.loadLattice(fname)
        params = None
        makeActionSrc = None

    else:
        getLogger("isle.show").error("Cannot load file '%s', unsupported file type.",
                                     fname)
        raise ValueError("Invalid file type")

    return lattice, params, makeActionSrc


def main(args):
    r"""!
    Run the show script to report on contents of Isle files.

    \param args Parsed command line arguments.
    """

    # set up matplotlib once for all reporters
    isle.plotting.setupMPL()

    for fname in args.input:
        try:
            lattice, params, makeActionSrc = _loadMetadata(fname, isle.fileio.fileType(fname))
        except ValueError:
            continue  # skip this file

        try:
            # call individual reporters
            if "overview" in args.report:
                _overview(fname, lattice, params, makeActionSrc)
            if "lattice" in args.report:
                _lattice(fname, lattice)
            if "correlator" in args.report:
                _correlator(fname, lattice, params, makeActionSrc)
        except:
            getLogger("isle.show").exception("Show command failed with file %s.", fname)

    plt.show()
