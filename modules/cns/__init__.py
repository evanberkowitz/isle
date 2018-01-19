
from pathlib import Path
#: Absolute path to script directory.
SCRIPT_PATH = Path(__file__).resolve().parent

def prepare_cnxx_import(path=Path("../cnxx/build")):
    """
    Prepare site for scripts based on cnxx. Call before importing cnxx.
    Arguments:
        path: Path to directory that contains cnxx library. Relative to TEST_PATH.
    """

    import site
    site.addsitedir(str(SCRIPT_PATH/path))


prepare_cnxx_import("../../cnxx/build")

from cnxx import *

import cns.yaml_io as yaml
