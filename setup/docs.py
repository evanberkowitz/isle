"""
Configure CMake based extensions.
"""

import sys
import subprocess
from pathlib import Path
import tempfile
import distutils.cmd

def _read_style(docsdir):
    """
    Read doxygen style information from 'that_style' submodule.
    Downloads the git submodule if needed.
    """

    stylefile = docsdir/"that_style"/"doxyfile.conf"
    if not stylefile.exists():
        # download the submodule
        try:
            subprocess.check_call(["git", "submodule", "update"],
                                  cwd=stylefile.parent)
        except subprocess.CalledProcessError:
            print("error: Could not load git submodule 'that_style'")
            sys.exit(1)

    # read file
    with open(stylefile, "r") as stf:
        return stf.read()

def doxygen_command(doxyfile, version):
    "Return a command class to run doxygen."
    class _Doxygen(distutils.cmd.Command):
        description = "run Doxygen"
        user_options = [("doxygen=", None, "Doxygen executable to generate documentation")]

        def initialize_options(self):
            self.doxygen = "doxygen"

        def finalize_options(self):
            pass

        def run(self):
            "Execute the command."

            doxypath = Path(doxyfile)
            docsdir = doxypath.resolve().parent

            # read config
            with open(doxyfile, "r") as doxf:
                config = doxf.read()
            # append version number and style
            config = config + f"\nPROJECT_NUMBER = {version}\n" + _read_style(docsdir)

            with tempfile.NamedTemporaryFile(dir=str(docsdir), mode="w") as inputfile:
                # write complete config
                inputfile.write(config)

                # run doxygen
                try:
                    subprocess.check_call([self.doxygen, inputfile.name],
                                          cwd=docsdir)
                except subprocess.CalledProcessError:
                    print("error: Could not run doxygen")

                    sys.exit(1)

    return _Doxygen
