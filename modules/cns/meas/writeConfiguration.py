"""!
Measurement of logDet.
"""

import numpy as np
import h5py as h5

from ..util import createH5Group

class WriteConfiguration:
    r"""!
    \ingroup meas
    Write the gauge configuration to an HDF5 file.

    Must be called with all inline measurement parameters,
    i.e. `itr`, `act`, `acc`, and `rng`.
    """

    def __init__(self, filename, groupname, recordnameFmt="cfg_{itr}"):
        self.filename = filename
        self.groupname = groupname
        self.recordnameFmt = recordnameFmt

    def __call__(self, phi, inline=True, **kwargs):
        """!Write a configuration to HDF5."""
        try:
            with h5.File(self.filename, "a") as h5f:

                grp = createH5Group(h5f, self.groupname)  # base group for all configs
                grp = grp.create_group(self.recordnameFmt.format(itr=kwargs["itr"])) # this cfg

                grp["config"] = np.array(phi, copy=False)
                grp["action"] = kwargs["act"]
                grp["acceptance"] = kwargs["acc"]
                kwargs["rng"].writeH5(createH5Group(grp, "rng_state"))
        except KeyError:
            raise KeyError("Need to pass all inline meas arguments when calling WriteConfiguration") from None
        except (ValueError, RuntimeError) as err:
            if "name already exists" in err.args[0]:
                raise RuntimeError("Cannot write config {} to file {}. A dataset with the same name already exists."
                                   .format(kwargs["itr"], self.filename)) from None
            raise
