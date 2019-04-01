r"""!\file
\ingroup meas
Base class for measurements.
"""

from abc import ABCMeta, abstractmethod
from logging import getLogger

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

    def __init__(self, savePath, configSlice=slice(0, None, 1)):
        r"""!
        Store common parameters.
        \param savePath Path in an HDF5 file under which results are stored.
        \param configSlice Indicates which configurations the measurement is taken on.
        """
        ## Path in an HDF5 file under which results are stored.
        self.savePath = savePath
        ## Indicates which configurations the measurement is taken on.
        self.configSlice = configSlice

    @abstractmethod
    def __call__(self, phi, actVal, itr):
        r"""!
        Execute the measurement.
        \param phi Field configuration.
        \param actVal Value of the action at field phi.
        \param itr Index of the current trajectory.
        """

    @abstractmethod
    def save(self, h5group):
        r"""!
        Save results of the measurement to HDF5.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """

    def saveAll(self, h5group):
        r"""!
        Save results of measurement as well as relevant metadata.
        \param h5group Base HDF5 group. Data is stored in subgroup `h5group/self.savePath`.
        """

        self.save(h5group)
        # create the group here to make sure it really exists
        createH5Group(h5group, self.savePath)
        self.saveConfigSlice(h5group[self.savePath])

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
