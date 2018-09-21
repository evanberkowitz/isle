"""!
Measurement of action.
"""

from ..h5io import createH5Group

class Action:
    r"""!
    \ingroup meas
    Transfer the action from a configuration file into a measurement file.
    Does nothing if the measurement file already contains the action.
    """

    def __init__(self):
        self.action = []

    def __call__(self, phi, action, itr):
        """!Record action."""
        self.action.append(action)

    def save(self, base, name):
        r"""!
        Write the action to a file.
        \param base HDF5 group in which to store data.
        \param name Name of the subgroup of base for this measurement.
        """
        group = createH5Group(base, name)
        if not "action" in group and not "configuration" in group \
           and not "configuration" in base:
            group["action"] = self.action

    def read(self, group):
        r"""!
        Read the action from a file.
        \param group HDF5 group which contains the data of this measurement.
        """
        if "action" in group:
            self.action = group["action"][()]
        elif "configuration" in group:
            cfgGrp = group["configuration"]
            self.action = [cfgGrp[cfg]["action"][()] for cfg in cfgGrp]
        else:
            print(f"Error: no action found in HDF5 group '{group}'."
                  " Did not find subgroups 'action' or 'configuration'.")
