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

    Stores general measurement parameters and result buffers:

    - An HDF5 path indicating where the measurement results shall be saved.
      That path is relative to the HDF5 group passed to Measurement.save() and
      Measurement.saveAll().

    - A `slice` object showing which configurations the measurement
      has to be / has been run on.
      This slice is treated with respect to the labels of configurations in a file
      as controlled through the `itr` parameter of `Measurement.__call__`.
      `configSlice.end` may be `None` in which case no upper limit is set.
      It must be set to a number when saving the slice however!

    - A map from buffer names to pentinsula.TimeSeries objects and a map from names
      to write iterators of those time series.
      Those buffers are meant to be used for storing the results of measurements.
      They are only allocated when calling the setup method.

    - A flag indicating whether the measurement has been set up.
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
        ## Maps buffer names to TimeSeries objects.
        self._buffers = {}
        ## Maps buffer names to TimeSeries write iterators.
        self._bufferIterators = {}
        ## List of BufferSpec objects
        self._bufferSpecs = buffers
        ## True if setup has been called.
        self._isSetUp = False

    @abstractmethod
    def __call__(self, stage, itr):
        r"""!
        Execute the measurement.
        \param stage Instance of `isle.evolver.EvolutionStage` containing the
                     configuration to measure on and associated data.
        \param itr Index of the current trajectory.
        """

    def _allocateBuffers(self, memoryAllowance, expectedNConfigs, file, maxBufferSize):
        """!
        Construct TimeSeries objects for all buffers and create datasets
        in the file.
        """

        nremaining = len(self._bufferSpecs)
        residualMemory = memoryAllowance  # in case no buffers are allocated
        for spec in self._bufferSpecs:
            try:
                # Partition remaining allowance evenly between remaining measurements
                # but restrict each buffer to at most 4GB (limit on HDF5 chunks).
                thisAllowance = min(memoryAllowance // nremaining, 4*10**9-1)
                if maxBufferSize:
                    thisAllowance = min(thisAllowance, maxBufferSize)
                bufferLength, residualMemory = calculateBufferLength(thisAllowance,
                                                                     expectedNConfigs,
                                                                     spec.shape,
                                                                     spec.dtype)
            except RuntimeError:
                getLogger(__name__).error("Failed to allocate buffer %s", spec.name)
                raise
            usedMemory = thisAllowance - residualMemory
            memoryAllowance -= usedMemory
            nremaining -= 1

            getLogger(__name__).info(f"Allocating buffer '{spec.name}' in measurement "
                                     f"{type(self).__name__} with {bufferLength} time steps "
                                     f"({usedMemory:,} B)")
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

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        r"""!
        Setup a measurement for processing a given number of configurations.
        \param memoryAllowance Rough amount memory in bytes that the measurement
                               is allowed to use.
        \param expectedNConfigs Expected number of configurations that will be processed.
                                The actual number may be different.
                                This argument is only used for computing buffer sizes.
        \param file Name or handle of output file.
        \param maxBufferSize Maximum size in bytes of buffers.
        \returns The unused amount of memory out of `memoryAllowance` in bytes.
        """

        if self._isSetUp:
            raise RuntimeError("Cannot set up measurement, buffers are already set.")

        residualMemory = self._allocateBuffers(memoryAllowance, expectedNConfigs, file,
                                               maxBufferSize)
        self._isSetUp = True
        return residualMemory

    def nextItem(self, name):
        r"""!
        Return the next item in buffer `name`.
        This item can be modified to and will be written to file at an appropriate time.
        """
        if not self._isSetUp:
            raise RuntimeError(f"Cannot get next item {name}, the measurement is not set up.")
        return next(self._bufferIterators[name])[1]

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
    """!
    \ingroup meas
    Specification of a result buffer.
    """
    ## Name of the buffer.
    name: str
    ## Shape of items in the buffer (excluding time dimension).
    shape: Tuple[int, ...]
    ## Data type of the buffer.
    dtype: Union[np.dtype, Type[int], Type[float], Type[complex]]
    ## Path relative to measurement's savePath to the dataset associated with this buffer.
    path: str


def calculateBufferLength(maxMemory, expectedNTimePoints, shape, dtype):
    r"""!
    \ingroup meas
    Calculate the number of time steps to use in a TimeSeries buffer.

    \param maxMemory Maximum amount of memory in bytes that the buffer
                     is allowed to use.
    \param expectedNTimePoints Expected number of time points that
                               will be stored in the time series.
    \param shape Shape of items in the time series.
    \param dtype Datatype of the buffer / dataset.
    \returns Tuple of
             - The Number of time points to use.
             - The amount of unused memory out of `maxMemory` in bytes.
    """

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
