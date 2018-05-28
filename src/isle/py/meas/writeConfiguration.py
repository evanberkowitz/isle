"""!
Write a configuration do HDF5.
"""

import numpy as np
import h5py as h5

def writeCfg(base, groupname, phi, act, acc):
    """Write a configuration to a HDF5 group."""
    grp = base.create_group(groupname)
    grp["phi"] = np.array(phi, copy=False)
    grp["action"] = act
    grp["acceptance"] = acc
    return grp

def writeCheckpoint(group, groupname, rng, cfgGrp):
    """Write a configuration to a HDF5 group."""
    grp = group.create_group(groupname)
    rng.writeH5(grp.create_group("rng_state"))
    grp["cfg"] = h5.SoftLink(cfgGrp.name)


class WriteConfiguration:
    r"""!
    \ingroup meas
    Write gauge configurations and checkpoints to a HDF5 file.

    Configurations are stored every time this measurement gets called.
    Each configuration contains
    - <B>config</B>: The gauge field as a spacetime vector represented as
       an array of complex numbers in time-major layout.
    - <B>action</B>: The action as a complex number.
    - <B>acceptance</B>: A boolean indicating whether the trajectory was accepted or not.
                         If `False`, `config` holds the previous configuration, not the
                         rejected one.

    Checkpoints can be customized separately. A checkpoint comprises of
    - <B>rng_state</B>: The state of the random number generator after the
                        accept-reject step.
    - <B>cfg</B>: A link to the corresponding configuration.

    \attention Must be called with all inline measurement parameters,
    i.e. `itr`, `act`, `acc`, and `rng`.
    """

    def __init__(self, filename, cfgGroupnameFmt="/configuration/{itr}",
                 checkpointFreq=0, checkpointGroupnameFmt="/checkpoint/{itr}"):
        r"""!
        Configurations are stored under a path determined from `cfgGroupnameFmt`.
        This format specifier is processed for each configuration by replacing
        the strings `'{imeas}'` and `'{itr}'`.
        - `{imeas}` starts at 0 and increases each time this measurement gets called.
        - `{itr}` is the number of the trajectory that this measurement is called on.

        The same holds for checkpoint which and `checkpointGroupnameFmt`.
        Checkpoints have their own counter `imeas`.

        \param filename Name of the HDF5 file.
        \param cfgGroupnameFmt Format string for groups containing configurations;
                               relative to root of file.
        \param checkpointFreq Frequency with which to write checkpoints; set to 0 to
                              disable checkpointing. The value is relative
                              to the measurement frequency. I.e. if the measurement is
                              perfomed every 10th trajectory and ´checkpointFreq=2`,
                              checkpoints are written every 20 trajectories.
        \param cfgGroupnameFmt Format string for groups containing checkpoints;
                               relative to root of file.
        """
        self.filename = filename
        self.cfgGroupnameFmt = cfgGroupnameFmt
        self.checkpointFreq = checkpointFreq
        self.checkpointGroupnameFmt = checkpointGroupnameFmt
        self._cfgCnt = 0
        self._chkptCnt = 0

    def __call__(self, phi, inline, itr, act, acc, rng, **kwargs):
        """!Write a configuration to HDF5."""
        with h5.File(self.filename, "a") as h5f:
            # write configuration
            try:
                cfgGrp = writeCfg(h5f,
                                  self.cfgGroupnameFmt.format(itr=itr, imeas=self._cfgCnt),
                                  phi, act, acc)
                self._cfgCnt += 1
            except (ValueError, RuntimeError) as err:
                if "name already exists" in err.args[0]:
                    raise RuntimeError("Cannot write config {} (trajectory {}) to file {}. A dataset with the same name already exists."
                                       .format(self._cfgCnt, itr, self.filename)) from None
                raise

            # write checkpoint
            if self.checkpointFreq != 0 \
               and itr % self.checkpointFreq == 0:
                try:
                    writeCheckpoint(h5f,
                                    self.checkpointGroupnameFmt.format(itr=itr,
                                                                       imeas=self._chkptCnt),
                                    rng, cfgGrp)
                    self._chkptCnt += 1
                except (ValueError, RuntimeError) as err:
                    if "name already exists" in err.args[0]:
                        raise RuntimeError("Cannot write checkpoint {} (trajectory {}) to file {}. A dataset with the same name already exists."
                                           .format(self._chkptCnt, itr, self.filename)) from None
                    raise
