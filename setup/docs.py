"""
Configure CMake based extensions.
"""

import sys
import subprocess
from pathlib import Path
import distutils.cmd

def doxygen_command(doxyfile):
    "Return a command class to run doxygen."
    class _Doxygen(distutils.cmd.Command):
        description = "run Doxygen"
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            "Execute the command."

            doxypath = Path(doxyfile)
            docs_dir = doxypath.resolve().parent
            try:
                subprocess.check_call(["doxygen", doxypath.name],
                                      cwd=docs_dir)
            except subprocess.CalledProcessError:
                print("error: Could not run doxygen")
                sys.exit(1)
    return _Doxygen
