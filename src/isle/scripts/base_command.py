import isle

def _run_show(args):
    # import on demand in case matplotlib is not installed
    from . import show
    show.main(args)

def _run_hmc(args):
    raise NotImplementedError()

def _run_meas(args):
    raise NotImplementedError()

def main():
    """!Run Isle's base script. Dispatches to other scripts based on command line arguments."""

    commands = {"show": _run_show,
                "hmc": _run_hmc,
                "meas": _run_meas}

    args = isle.cli.init(list(commands.keys()), name="isle",
                         description="Base utility program of Isle. Dispatches to sub-commands. "
                         "Use -h for a sub-command to get more information.",
                         epilog="See https://github.com/jl-wynen/isle",
                         subdescriptions=[
                             "Pulles all data in can from a file in a format supported by Isle, "
                             "prints, and visualizes that data. Supported file types are HDF5 and YAML."
                             "Select a reporter via -r to choose which information to show.",
                             "",
                             ""
                         ])
    commands[args.cmd](args)
