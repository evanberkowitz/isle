r"""!\file
\ingroup meas
Measurement of total phi and norm of phi.
"""

import numpy as np

from .measurement import Measurement
from ..h5io import createH5Group

class TotalPhi(Measurement):
    r"""!
    \ingroup meas
    Tabulate phi and mean value of phi^2.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath, configSlice)

        self.Phi = []
        self.phiSq = []

    def __call__(self, stage, itr):
        """!Record the total phi and mean value of phi^2."""
        self.Phi.append(np.sum(stage.phi))
        self.phiSq.append(np.linalg.norm(stage.phi)**2 / len(stage.phi))

    def save(self, h5group):
        r"""!
        Write both Phi and phiSquared.
        \param base HDF5 group in which to store data.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["totalPhi"] = self.Phi
        subGroup["phiSquared"] = self.phiSq


def read(h5group):
    r"""!
    Read Phi and phiSquared from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["totalPhi"][()], h5group["phiSquared"][()]
