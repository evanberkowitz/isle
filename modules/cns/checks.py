"""!
Various basic measurements.
"""

## \defgroup check Consistency checks
# Perform consistency checks on HMC trajectories.
#
# All callables in this module satisfy the requirements of hmc.hmc, i.e.
# they have arguments
#  - `startPhi`/`endPhi`: Configuration before and after the proposer.
#  - `startPi`/`endPi`: Momentum before and after the proposer.
#  - `startEnergy`/`endEnergy`: Energy before and after the proposer.
#
# and raise a `ConsistencyCheckFailure` in case of failure.
#

import numpy as np

class ConsistencyCheckFailure(Exception):
    """!
    \ingroup check
    Indicate failure of a consistency check during HMC.
    """

def realityCheck(startPhi, startPi, startEnergy, endPhi, endPi, endEnergy):
    r"""!
    \ingroup check
    Check whether endPhi and endPi are real.
    """
    if np.max(np.imag(endPhi)/np.real(endPhi)) > 1e-15:
        raise ConsistencyCheckFailure("phi is not real")
    if np.max(np.imag(endPi)/np.real(endPi)) > 1e-15:
        raise ConsistencyCheckFailure("pi is not real")
