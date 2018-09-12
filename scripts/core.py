r"""!
\file
\brief Basic helpers for scripts using isle.
"""

## \defgroup scripts Scripts
# Python scripts based in the `cns` module.

from pathlib import Path
#: Absolute path to script directory.
SCRIPT_PATH = Path(__file__).resolve().parent

def prepare_module_import(path=Path("../modules")):
    r"""!
    Prepare site to import modules from ../modules.
    \param path Path to directory that contains the modules. Relative to SCRIPT_PATH.
    """

    import site
    site.addsitedir(str(SCRIPT_PATH/path))
