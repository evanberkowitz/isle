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

    def __call__(self, phi, action, itr):
        """!Record action."""
        self.action.append(action)

    def save(self, h5group):
        r"""!
        Write the action to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        subGroup["action"] = self.action

    def read(self, h5group):
        r"""!
        Read the action from a file.
        \param h5group HDF5 group which contains the data of this measurement.
        """
        # TODO add centralized load function

        if "action" in h5group:
            self.action = h5group["action"][()]
        elif "configuration" in h5group:
            cfgGrp = h5group["configuration"]
            self.action = [cfgGrp[cfg]["action"][()] for cfg in cfgGrp]
        else:
            getLogger(__name__).error("No action found in HDF5 group '%s'. "
                                      "Did not find subgroups 'action' or 'configuration'.",
                                      h5group)
            raise RuntimeError("No action found in HDF5")
