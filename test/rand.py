import numpy as np

def setup(seed):
    "Setup the RNG with a given seed."
    np.random.seed(seed)

def randn(low, high, size, typ):
    "Generate size N random numbers in [low, high] with type typ.  Zero is excluded."
    if typ == int:
        res = np.random.randint(low, high, size)
    elif typ == float:
        res =  np.random.uniform(low, high, size)
    elif typ == complex:
        res = randn(low, high, size, float) \
            + 1j * randn(low, high, size, float)
    else:
        raise TypeError("Unknown type for RNG: {}".format(typ))

    for nel, el in enumerate(res):
      if abs(el) < 1.e-7:
        res[nel] = randScalar(low, high, typ)

    return res
    

def randScalar(low, high, typ, nRec=0):
    "Generates random number in [low, high] with type typ. Zero is excluded."
    if nRec > 10:
        raise ValueError("Was not able to generate random number != 0. Check boundaries...")

    if typ == complex:
        res = randn(low, high, 1, float)[0] \
              + 1j * randn(low, high, 1, float)[0]
    elif typ == int or typ == float:
        res = randn(low, high, 1, typ)[0]
    else:
      raise TypeError("Unknown type for RNG: {}".format(typ))

    if abs(res) < 1.e-7:
        return randScalar(low, high, typ, nRec=nRec+1)
    else:
        return res