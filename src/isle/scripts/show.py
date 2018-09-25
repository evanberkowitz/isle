import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

except ImportError:
    print("Error: Cannot import matplotlib, show command is not available.")
    raise

import h5py as h5

import isle
import isle.plotting
from isle.drivers.meas import Measure

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
    if lattice is None or params is None or makeActionSrc is None:
        print("Error: Could not find all required information in the input file to generate an overview.")
        sys.exit(1)

    measState = Measure(lattice, params,
                        isle.fileio.callFunctionFromSource(makeActionSrc, lattice, params),
                        infname, None)

    isle.plotting.setupMPL()
    fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText) = _figure()
    fig.canvas.set_window_title(f"Isle Overview - {infname}")

    isle.plotting.plotTotalPhi(measState, axPhi, axPhiHist)
    isle.plotting.plotTrajPoints(measState, axTP)
    action = _loadAction(measState)
    isle.plotting.plotAction(action, axAct)
    isle.plotting.plotPhase(action, axPhase, axPhase2D)

    axText.axis("off")
    axText.text(0, 0, fontsize=13, linespacing=2, verticalalignment="bottom",
                s=rf"""Ensemble: $N_{{\mathrm{{config}}}}={_nconfig(infname)}$ in {infname}
     Lattice: {lattice.name}   $N_t={measState.lattice.nt()}$, $N_x={measState.lattice.nx()}$    '{lattice.comment}'
      Model: {_formatParams(measState.params)}""")

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

    if "overview" in args.report:
        _overview(args.input[0], lattice, params, makeActionSrc)

    plt.show()
