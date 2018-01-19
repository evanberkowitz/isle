"""
Import base functions and classes.

Imports everything from cnxx into namespace cns.
Requires that the cnxx module is installed in the parent directory.
"""

from cns.core import prepare_cnxx_import

prepare_cnxx_import()
from cnxx import *

import cns.yaml_io as yaml
