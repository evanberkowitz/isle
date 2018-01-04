import numpy as np

def setup(seed):
    "Setup the RNG with a given seed."
    np.random.seed(seed)

def randn(low, high, size, typ):
    "Generate size N random numbers in [low, high] with type typ."
    if typ == int:
        return np.random.randint(low, high, size)
    if typ == float:
        return np.random.uniform(low, high, size)
    if typ == complex:
        return randn(low, high, size, float) \
            + 1j * randn(low, high, size, float)
    raise TypeError("Unknown type for RNG: {}".format(typ))

def randScalar(low, high, typ):
    "Generates random number in [low, high] with type typ."
    return randn(low, high, 1, typ)[0]