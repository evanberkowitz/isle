"""!
Core functionality for cns module.
"""

from pathlib import Path
## Absolute path to module directory.
MODULE_PATH = Path(__file__).resolve().parent

def prepare_cnxx_import(path=Path("../")):
    r"""!
    Prepare site for scripts based on cnxx. Call before importing cnxx.
    \param path Path to directory that contains cnxx library; relative to MODULE_PATH.
    """
    import site
    site.addsitedir(str(MODULE_PATH/path))
