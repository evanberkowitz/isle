"""!
Measurement of logDet.
"""

import numpy as np

from .common import ensureH5GroupExists

class WriteConfiguration:
    r"""!
    \ingroup meas
    Write the field configuration to an HDF5 file.
    """

    def __init__(self, theFile, path):
        self.theFile = theFile
        self.path = path
        ensureH5GroupExists(self.theFile, path)

    def __call__(self, phi, inline=True, **kwargs):
        """!Record configuration and associated information."""
        path = self.path + "/cfg_" + str(kwargs["itr"])
        ensureH5GroupExists(self.theFile, path)
        self.theFile.create_array(path, "field", np.array(phi))
        if "act" in kwargs:
            self.theFile.create_array(path, "action", kwargs["act"])
        if "acc" in kwargs:
            self.theFile.create_array(path, "acceptance", kwargs["acc"])
