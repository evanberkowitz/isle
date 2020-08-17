r"""!\file
\ingroup meas
Base class for measurements.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Union, Tuple, Type

import numpy as np
from pentinsula import TimeSeries
from pentinsula.h5utils import open_or_pass_file

from ..h5io import createH5Group


class Measurement(metaclass=ABCMeta):
    r"""!
    \ingroup meas
    Base class for measurements.

    Stores general measurement parameters:

    - An HDF5 path indicating where the measurement results shall be saved.
      That path is relative to the HDF5 group passed to Measurement.save() and
      Measurement.saveAll().

    - A `slice` object showing which configurations the measurement
      has to be / has been run on.
      This slice is treated with respect to the labels of configurations in a file
      as controlled through the `itr` parameter of `Measurement.__call__`.
      `configSlice.end` may be `None` in which case no upper limit is set.
      It must be set to a number when saving the slice however!
    """

    def __init__(self, savePath, buffers, configSlice=slice(None, None, None)):
        r"""!
        Store common parameters.
        \param savePath Path in an HDF5 file under which results are stored.
        \param configSlice Indicates which configurations the measurement is taken on.
        """
        ## Path in an HDF5 file under which results are stored.
        self.savePath = savePath
        ## Indicates which configurations the measurement is taken on.
        self.configSlice = configSlice
        ##
        self._buffers = {}
        ##
        self._bufferIterators = {}
        ##
        self._bufferSpecs = buffers
        ##
        self._isSetUp = False

    def _allocateBuffers(self, memoryAllowance, expectedNConfigs, file):
        nremaining = len(self._bufferSpecs)
        residualMemory = memoryAllowance  # in case no buffers are allocated
        for spec in self._bufferSpecs:
            try:
                bufferLength, residualMemory = calculateBufferLength(memoryAllowance // nremaining,
                                                                     expectedNConfigs,
                                                                     spec.shape,
                                                                     spec.dtype)
            except RuntimeError:
                getLogger(__name__).error("Failed to allocate buffer %s", spec.name)
                raise
            memoryAllowance -= memoryAllowance // nremaining - residualMemory
            nremaining -= 1

            getLogger(__name__).info("Allocating buffer '%s' in measurement %s with %d time steps",
                                     spec.name, type(self).__name__, bufferLength)
            self._buffers[spec.name] = TimeSeries(file, Path(self.savePath) / spec.path,
                                                  bufferLength, spec.shape, spec.dtype)

        if self._bufferSpecs:
            with open_or_pass_file(file, None, "a") as h5f:
                group = createH5Group(h5f, self.savePath)
                self.saveConfigSlice(group)
                for buffer in self._buffers.values():
                    buffer.create_dataset(h5f, write=False)

        self._bufferIterators = {name: iter(buffer.write_iter(flush=True))
                                 for name, buffer in self._buffers.items()}

        return residualMemory

    def setup(self, memoryAllowance, expectedNConfigs, file):
        if self._isSetUp:
            raise RuntimeError("Cannot set up measurement, buffers are already set.")

        residualMemory = self._allocateBuffers(memoryAllowance, expectedNConfigs, file)
        self._isSetUp = True
        return residualMemory

    def nextItem(self, name):
        return next(self._bufferIterators[name])[1]

    @abstractmethod
    def __call__(self, stage, itr):
        r"""!
        Execute the measurement.
        \param stage Instance of `isle.evolver.EvolutionStage` containing the
                     configuration to measure on and associated data.
        \param itr Index of the current trajectory.
        """

    def save(self, file=None):
        r"""!
        Save current results of the measurement to HDF5.

        This is function redundant as results saved automatically.
        But it can be used to write data before the result buffer is full.
        Flushing the buffer (happens automatically) writes anything again that
        has been written by this function.
        """
        for buffer in self._buffers:
            buffer.write(file=file)

    def saveConfigSlice(self, h5obj, name="configurations"):
        r"""!
        Save the configuration slice as an attribute to an HDF5 object.
        \param h5obj HDF5 in whose attribute the slice is stored.
        \param name Name of the attribute.
                    <B>Warning</B>: changing this breaks compatibility with other parts of Isle.
        \attention <B>No</B> part of `self.configSlice` may be `None` when calling this function.
        """

        sliceElems = (self.configSlice.start,
                      self.configSlice.stop,
                      self.configSlice.step)
        if None in sliceElems:
            getLogger(__name__).error("Tried to save config slice which contains None "
                                      "in measurement %s.\n    configSlice=%s",
                                      type(self), self.configSlice)
            raise ValueError(f"configRange contains None: {self.configSlice}")

        h5obj.attrs[name] = ":".join(map(str, sliceElems))


@dataclass
class BufferSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: Union[np.dtype, Type[int], Type[float], Type[complex]]
    path: str


def calculateBufferLength(maxMemory, expectedNTimePoints, shape, dtype):
    timePointSize = np.dtype(dtype).itemsize * int(np.prod(shape))
    if timePointSize > maxMemory:
        getLogger(__name__).error(
            f"""Cannot allocate memory for a buffer of shape {shape} and dtype {dtype}, not enough memory available.
    Needed:    {timePointSize:12,} B
    Available: {maxMemory:12,} B""")
        raise RuntimeError("Insufficient memory to allocate buffer.")

    bufferLength = min(maxMemory // timePointSize, expectedNTimePoints)
    residual = maxMemory - bufferLength * timePointSize
    return bufferLength, residual
