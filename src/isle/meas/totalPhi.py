"""!
Measurement of total phi and norm of phi.
"""

import numpy as np

from ..h5io import createH5Group

class TotalPhi:
    r"""!
    \ingroup meas
    Tabulate phi and mean value of phi^2.
    """

    def __init__(self):
        self.Phi = []
        self.phiSq = []

    def __call__(self, phi, action, itr):
        """!Record the total phi and mean value of phi^2."""
        self.Phi.append(np.sum(phi))
        self.phiSq.append(np.linalg.norm(phi)**2 / len(phi))

    def save(self, base, name):
        r"""!
        Write both Phi and phiSquared.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup of base for this measurement.
        """
        group = createH5Group(base, name)
        group["totalPhi"] = self.Phi
        group["phiSquared"] = self.phiSq

    def read(self, group):
        r"""!
        Read Phi and phiSquared from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        self.Phi = group["totalPhi"][()]
        self.phiSq = group["phiSquared"][()]
