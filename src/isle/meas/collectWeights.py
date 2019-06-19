r"""!\file
\ingroup meas
Measurement to collect weights of configurations.
"""

from logging import getLogger

from ..h5io import createH5Group
from .measurement import Measurement

class CollectWeights(Measurement):
    r"""!
    \ingroup meas
    Collect all log weights from individual trajectories and store them 
    in a consolidated way.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        r"""!
        \param savePath Path in an HDF5 file under which results are stored.
        \param configSlice Indicates which configurations the measurement is taken on.
        """
        super().__init__(savePath, configSlice)
        ## `dict` of log weights (`str` -> `list`)
        self.logWeights = {}

    def __call__(self, stage, itr):
        r"""!
        Collect weights.
        \param stage Instance of `isle.evolver.EvolutionStage` containing the
                     configuration to measure on and associated data.
        \param itr Index of the current trajectory.
        """
        if not self.logWeights:
            weightNames = list(stage.logWeights.keys())
            getLogger(__name__).info("Start collecting log weights. "
                                     "Found %s at trajectory %d.", weightNames, itr)
            self.logWeights = {name: [] for name in weightNames}
        for key, val in stage.logWeights.items():
            self.logWeights[key].append(val)

    def save(self, h5group):
        r"""!
        Write the action to a file.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """
        subGroup = createH5Group(h5group, self.savePath)
        for key, vals in self.logWeights.items():
            subGroup[key] = vals
