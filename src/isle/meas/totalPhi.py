r"""!\file
\ingroup meas
Measurement of total phi and norm of phi.
"""

from pathlib import Path

import numpy as np

from .measurement import Measurement


class TotalPhi(Measurement):
    r"""!
    \ingroup meas
    Tabulate phi and mean value of phi^2.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath, configSlice)

    def setup(self, memoryAllowance, expectedNConfigs, file):
        if self.isSetUp():
            raise RuntimeError("Cannot set up measurement, buffers are already set.")

        self._allocateBuffers((("Phi", np.complex128, (), Path(self.savePath)/"totalPhi"),
                               ("phiSquared", np.float64, (), Path(self.savePath)/"phiSquared")),
                              memoryAllowance,
                              expectedNConfigs,
                              file)

    def __call__(self, stage, itr):
        """!Record the total phi and mean value of phi^2."""
        next(self.PhiIterator)[1][...] = np.sum(stage.phi)
        next(self.phiSquaredIterator)[1][...] = np.linalg.norm(stage.phi)**2 / len(stage.phi)

    def save(self, h5group):
        r"""!
        Write both Phi and phiSquared.
        \param base HDF5 group in which to store data.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        raise NotImplementedError()
        # subGroup = createH5Group(h5group, self.savePath)
        # subGroup["totalPhi"] = self.Phi
        # subGroup["phiSquared"] = self.phiSq


def read(h5group):
    r"""!
    Read Phi and phiSquared from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return h5group["totalPhi"][()], h5group["phiSquared"][()]
