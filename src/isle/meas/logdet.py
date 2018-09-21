"""!
Measurement of logdet.
"""

import isle
from ..h5io import createH5Group

class Logdet:
    r"""!
    \ingroup meas
    Measure the log determinant of particles and holes.
    """

    def __init__(self, hfm):
        self.hfm = hfm
        self.logdet = {isle.Species.PARTICLE: [],
                       isle.Species.HOLE: []}

    def __call__(self, phi, action, itr):
        """!Record logdet."""
        for species, logdet in self.logdet.items():
            logdet.append(isle.logdetM(self.hfm, phi, species))

    def save(self, base, name):
        r"""!
        Write both the particle and hole logdet to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup of base for this measurement.
        """
        group = createH5Group(base, name)
        group["particles"] = self.logdet[isle.Species.PARTICLE]
        group["holes"] = self.logdet[isle.Species.HOLE]

    def read(self, group):
        r"""!
        Read particle and hole logdet from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.logdet[isle.Species.PARTICLE] = group["particles"][()]
        self.logdet[isle.Species.HOLE] = group["holes"][()]
