r"""!\file
\ingroup meas
Measurement of logdet.
"""

import numpy as np

import isle
from .measurement import Measurement

class Logdet(Measurement):
    r"""!
    \ingroup meas
    Measure the log determinant of particles and holes.
    """

    def __init__(self, hfm, savePath, configSlice=slice(None, None, None), alpha=1):
        assert alpha in [0, 1]

        super().__init__(savePath, configSlice)

        self.hfm = hfm
        self.logdet = {isle.Species.PARTICLE: [],
                       isle.Species.HOLE: []}
        self.alpha = alpha

    def __call__(self, phi, action, itr):
        """!Record logdet."""
        if self.alpha == 1:
            for species, logdet in self.logdet.items():
                logdet.append(isle.logdetM(self.hfm, phi, species))
        else:
            for species, logdet in self.logdet.items():
                # use dense, slow numpy routine to get stable result
                ld = np.linalg.slogdet(isle.Matrix(self.hfm.M(-1j*phi, species)))
                logdet.append(np.log(ld[0]) + ld[1])

    def save(self, h5group):
        r"""!
        Write both the particle and hole logdet to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = isle.h5io.createH5Group(h5group, self.savePath)
        subGroup["particles"] = self.logdet[isle.Species.PARTICLE]
        subGroup["holes"] = self.logdet[isle.Species.HOLE]

    def saveAll(self, h5group):
        r"""!
        Save results of measurement as well as relevant metadata.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """

        self.save(h5group)
        # create the group here to make sure it really exists
        subGroup = isle.h5io.createH5Group(h5group, self.savePath)
        self.saveConfigSlice(subGroup)
        subGroup.attrs["alpha"] = self.alpha


def read(h5group):
    r"""!
    Read particle and hole logdet from a file.
    \param h5group HDF5 group which contains the data of this measurement.
    """
    return {isle.Species.PARTICLE: h5group["particles"][()],
            isle.Species.HOLE: h5group["holes"][()]}, \
            h5group.attrs["alpha"] if "alpha" in h5group.attrs else None
