r"""!\file
\ingroup meas
Measurement of logdet.
"""

import numpy as np
from pentinsula.h5utils import open_or_pass_file

import isle
from .measurement import Measurement, BufferSpec

class Logdet(Measurement):
    r"""!
    \ingroup meas
    Measure the log determinant of particles and holes.
    """

    def __init__(self, hfm, savePath, configSlice=slice(None, None, None), alpha=1):
        assert alpha in [0, 1]

        super().__init__(savePath,
                         (BufferSpec("particles", (), np.complex128, "particles"),
                          BufferSpec("holes", (), np.complex128, "holes")),
                         configSlice)

        self.hfm = hfm
        self.alpha = alpha

    def __call__(self, stage, itr):
        """!Record logdet."""
        if self.alpha == 1:
            self.nextItem("particles")[...] = isle.logdetM(self.hfm, stage.phi,
                                                           isle.Species.PARTICLE)
            self.nextItem("holes")[...] = isle.logdetM(self.hfm, stage.phi,
                                                       isle.Species.HOLE)
        else:
            # use dense, slow numpy routine to get stable result
            ld = np.linalg.slogdet(isle.Matrix(self.hfm.M(-1j*stage.phi, isle.Species.PARTICLE)))
            self.nextItem("particles")[...] = np.log(ld[0]) + ld[1]
            ld = np.linalg.slogdet(isle.Matrix(self.hfm.M(-1j*stage.phi, isle.Species.HOLE)))
            self.nextItem("holes")[...] = np.log(ld[0]) + ld[1]

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        res = super().setup(memoryAllowance, expectedNConfigs, file, maxBufferSize)
        with open_or_pass_file(file, None, "a") as h5f:
            h5f[self.savePath].attrs["alpha"] = self.alpha
        return res
