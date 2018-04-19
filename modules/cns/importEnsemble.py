"""!
Import an ensemble
"""

import importlib.util

from . import *

def importEnsemble(ensembleFile):
    moduleName = ensembleFile.replace("/", ".")[:-3] # replace / with . and chop off ".py"
    # As explained in
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(moduleName, ensembleFile)
    ensemble = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ensemble)
    return ensemble
