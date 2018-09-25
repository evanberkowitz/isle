import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D  # (unused import) pylint: disable=W0611
    from matplotlib import cm

except ImportError:
    print("Error: Cannot import matplotlib, show command is not available.")
    raise

import numpy as np
import h5py as h5

import isle
import isle.plotting
from isle.drivers.meas import Measure

def _overview_figure():
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
        if "configuration" in h5f:
            meas = isle.meas.Action()
            measState.mapOverConfigs([(1, meas, None)])
            return meas.action
    print("no configurations or action found")
    return None

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

    if lattice is None or params is None or makeActionSrc is None:
        print("Error: Could not find all required information in the input file to generate an overview.")
        sys.exit(1)

    # use this to bundle information and perform simple measurements if needed
    measState = Measure(lattice, params,
                        isle.fileio.callFunctionFromSource(makeActionSrc, lattice, params),
                        infname, None)

    # set up the figure
    fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText) = _overview_figure()
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
    fig.tight_layout()



def main(args):
    if args.input[1] == isle.fileio.FileType.HDF5:
        lattice, params, makeActionSrc = isle.fileio.h5.readMetadata(args.input)
    elif args.input[1] == isle.fileio.FileType.YAML:
        lattice = isle.fileio.yaml.loadLattice(args.input[0])
        params = None
        makeActionSrc = None
    else:
        print(f"Error: Cannot load file '{args.input[0]}', unsupported file type.")
        sys.exit(1)

    # set up matplotlib once for all reporters
    isle.plotting.setupMPL()

    if "overview" in args.report:
        _overview(args.input[0], lattice, params, makeActionSrc)
    if "lattice" in args.report:
        _lattice(args.input[0], lattice)

    plt.show()
