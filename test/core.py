"""
Basic helpers for test scripts.
"""

from pathlib import Path
#: Absolute path to test directory.
TEST_PATH = Path(__file__).resolve().parent

def prepare_cnxx_import(path=Path("../cnxx/build")):
    """
    Prepare site for scripts based on cnxx. Call before importing cnxx.
    Arguments:
        path: Path to directory that contains cnxx library. Relative to TEST_PATH.
    """

    import site
    site.addsitedir(str(TEST_PATH/path))
