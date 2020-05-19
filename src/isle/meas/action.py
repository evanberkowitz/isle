r"""!\file
\ingroup meas
Measurement of action.
"""

from logging import getLogger

from ..h5io import createH5Group
from .measurement import Measurement

class Action(Measurement):
    r"""!
    \ingroup meas
    Transfer the action from a configuration file into a measurement file.
    Does nothing if the measurement file already contains the action.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        r"""!
        \param savePath Path in an HDF5 file under which results are stored.
        \param configSlice Indicates which configurations the measurement is taken on.
        """
        super().__init__(savePath, configSlice)
        self.action = []

    def __call__(self, stage, itr):
        """!Record action."""
        self.action.append(stage.actVal)

    def save(self, h5group):
        r"""!
        Write the action to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["action"] = self.action
