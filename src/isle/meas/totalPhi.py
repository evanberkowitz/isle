r"""!\file
\ingroup meas
Measurement of total phi and norm of phi.
"""

import numpy as np

from .measurement import Measurement, BufferSpec


class TotalPhi(Measurement):
    r"""!
    \ingroup meas
    Tabulate phi and mean value of phi^2.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath,
                         (BufferSpec("Phi", (), np.complex128, "totalPhi"),
                          BufferSpec("phiSquared", (), np.float64, "phiSquared")),
                         configSlice)

    def __call__(self, stage, itr):
        """!Record the total phi and mean value of phi^2."""
        self.nextItem("Phi")[...] = np.sum(stage.phi)
        self.nextItem("phiSquared")[...] = np.linalg.norm(stage.phi)**2 / len(stage.phi)


def read(h5group):
    r"""!
    Read Phi and phiSquared from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["totalPhi"][()], h5group["phiSquared"][()]
