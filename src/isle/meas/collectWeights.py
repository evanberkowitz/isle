r"""!\file
\ingroup meas
Measurement to collect weights of configurations.
"""

import h5py as h5

from .measurement import Measurement, BufferSpec

class CollectWeights(Measurement):
    r"""!
    \ingroup meas
    Collect all log weights from individual trajectories and store them
    in a consolidated way.

    \note This measurement cannot be set up when calling setup.
          Instead, buffers are allocated only when the measurement is called
          for the first time because it is not known beforehand, which weights exist.
    """

    def __init__(self, savePath, configSlice=slice(None, None, None)):
        r"""!
        \param savePath Path in an HDF5 file under which results are stored.
        \param configSlice Indicates which configurations the measurement is taken on.
        """
        super().__init__(savePath, (), configSlice)
        ## `dict` of log weights (`str` -> `list`)
        self.logWeights = None
        ## The output file.
        self.outfile = None
        ##
        self._allocateBuffersArgs = None

    def __call__(self, stage, itr):
        r"""!
        Collect weights.
        \param stage Instance of `isle.evolver.EvolutionStage` containing the
                     configuration to measure on and associated data.
        \param itr Index of the current trajectory.
        """
        if not self._isSetUp:
            self._deferredSetup(stage)
        for key, val in stage.logWeights.items():
            self.nextItem(key)[...] = complex(val)

    def _deferredSetup(self, stage):
        """!
        Allocate buffers for all weights in given stage.
        """
        self._bufferSpecs = tuple(BufferSpec(name, (), complex, name)
                                  for name in stage.logWeights.keys())
        self._allocateBuffers(*self._allocateBuffersArgs)
        self._isSetUp = True

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        """!
        Override to only store arguments for later setup.
        """
        self._allocateBuffersArgs = (memoryAllowance, expectedNConfigs,
                                     file.filename if isinstance(file, h5.File) else file,
                                     maxBufferSize)
        return 0  # don't know how much memory is needed => assume all of it
