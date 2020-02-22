"""!
Utilities for random number generators.

See isle.random.RNGWrapper for the common interface of wrappers.
"""

from abc import ABC, abstractmethod
import numpy as np

class RNGWrapper(ABC):
    """!
    Base for all RNG wrappers.
    """

    @property
    def NAME(self):
        """!Unique name of the RNG."""
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed):
        """!Reseed the RNG."""
        pass

    @abstractmethod
    def uniform(self, low=0.0, high=1.0, size=None, cmplx=False):
        r"""!
        Return uniformly distributed random numbers.
        \param low Minimum value (inclusive).
        \param high Maximum value (exclusive).
        \param size Int or tuple of ints encoding the shape of the returned array.
                    If not given, a single number is returned.
        \param cmplx Select whether to return real or complex numbers.
        """
        pass

    @abstractmethod
    def normal(self, mean=0.0, std=1.0, size=None, cmplx=False):
        r"""!
        Return normally distributed random numbers.
        \param mean Mean value of the distribution.
        \param std Width of the destribution.
        \param size Int or tuple of ints encoding the shape of the returned array.
                    If not given, a single number is returned.
        \param cmplx Select whether to return real or complex numbers.
        """
        pass

    @abstractmethod
    def choice(self, a, size=None, replace=True):
        r"""!
        Generate a random sample from a given 1-D array
        \param a If an np.ndarray or equivalent, a random sample is generated from
                 its elements. If an int, the random sample is generated as if a were np.arange(a).
        \param size Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
                    samples are drawn. Default is None, in which case a single value is returned.
        \param replace Whether the sample is with or without replacement.
        \param p The probabilities associated with each entry in a. If not given the sample
                 assumes a uniform distribution over all entries in a.
        """
        pass

    @abstractmethod
    def writeH5(self, group):
        r"""!
        Write the current state of the RNG into a HDF5 group.
        \param group ´h5.Group` instance to write into. No new group is created inside of it.
        """
        pass

    @abstractmethod
    def readH5(self, group):
        r"""!
        Read the RNG state from HDF5.
        \param group `h5.Group` instance which contains all needed data.
        """
        pass


class NumpyRNG(RNGWrapper):
    """!
    Wrapper around numpy's Mersenne Twister random number generator.
    """

    ## Unique name of this RNG.
    NAME = "np.MT19937"

    def __init__(self, seed):
        """!Initialize from a seed."""
        self._state = np.random.RandomState(seed)

    def seed(self, seed):
        """!Reseed the RNG."""
        self._state.seed(seed)

    def uniform(self, low=0.0, high=1.0, size=None, cmplx=False):
        r"""!
        Return uniformly distributed random numbers.
        \param low Minimum value (inclusive).
        \param high Maximum value (exclusive).
        \param size Int or tuple of ints encoding the shape of the returned array.
                    If not given, a single number is returned.
        \param cmplx Select whether to return real or complex numbers.
        """

        if cmplx:
            return self._state.uniform(low, high, size) \
                + 1j*self._state.uniform(low, high, size)
        return self._state.uniform(low, high, size)

    def normal(self, mean=0.0, std=1.0, size=None, cmplx=False):
        r"""!
        Return normally distributed random numbers.
        \param mean Mean value of the distribution.
        \param std Width of the destribution.
        \param size Int or tuple of ints encoding the shape of the returned array.
                    If not given, a single number is returned.
        \param cmplx Select whether to return real or complex numbers.
        """

        if cmplx:
            return self._state.normal(mean, std, size) \
                + 1j*self._state.normal(mean, std, size)
        return self._state.normal(mean, std, size)

    def choice(self, a, size=None, replace=True, p=None):
        r"""!
        Generate a random sample from a given 1-D array
        \param a If an np.ndarray or equivalent, a random sample is generated from
                 its elements. If an int, the random sample is generated as if a were np.arange(a).
        \param size Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
                    samples are drawn. Default is None, in which case a single value is returned.
        \param replace Whether the sample is with or without replacement.
        \param p The probabilities associated with each entry in a. If not given the sample
                 assumes a uniform distribution over all entries in a.
        """

        return self._state.choice(a, size=size, replace=replace, p=p)

    def writeH5(self, group):
        r"""!
        Write the current state of the RNG into a HDF5 group.
        \param group ´h5.Group` instance to write into. No new group is created inside of it.
        """

        stDat = self._state.get_state()
        assert stDat[0] == self.NAME.rsplit(".")[-1]

        # write state
        group["name"] = self.NAME
        group["keys"] = stDat[1]
        group.attrs["pos"] = stDat[2]
        group.attrs["has_gauss"] = stDat[3]
        group.attrs["cached_gaussian"] = stDat[4]

    def readH5(self, group):
        r"""!
        Read the RNG state from HDF5.
        \param group `h5.Group` instance which contains all needed data.
        """

        try:
            # make sure this is the correct RNG
            if group["name"][()] != self.NAME:
                raise RuntimeError("Wrong kind of RNG. Expected {}, got {}"
                                   .format(self.NAME, group["name"]))

            # load state
            self._state.set_state((self.NAME.rsplit(".")[-1],
                                   np.array(group["keys"]),
                                   group.attrs["pos"],
                                   group.attrs["has_gauss"],
                                   group.attrs["cached_gaussian"]))
        except KeyError as err:
            # add some info and throw it back at my face
            err.args = ("Malformatted RNG group: {}".format(err), )
            raise err


## Dictionary of all RNGs addressable through their names.
RNGS = {NumpyRNG.NAME: NumpyRNG}

def writeStateH5(rng, group):
    r"""!
    Write an RNG state to HDF5.
    \param rng State to write.
    \param group HDF5 group to write into.
    """
    rng.writeH5(group)

def readStateH5(group):
    r"""!
    Construct a new RNG state from HDF5.
    \param group HDF5 group that contains data for an RNG state. Must contain a dataset
                 called 'name' that indicates which RNG is stored. Other fields are
                 specific to the RNG that was stored.
    \returns A new RNG state instance. The type is determined from HDF5.
    """
    if "name" not in group:
        raise ValueError("HDF5 group does not contain an RNG")
    rng = RNGS[group["name"][()]](0)  # instantiate new RNG
    rng.readH5(group)  # read state
    return rng
