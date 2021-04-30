r"""!\file
\ingroup evolvers
Selectors to pick proposed trajectories (accept/reject).
"""

import numpy as np


class BinarySelector:
    r"""! \ingroup evolvers
    Select one of two trajectories based on Metropolis accept/reject.
    """

    def __init__(self, rng):
        r"""!
        \param rng Central random number generator of the run used for accept/reject.
        """
        self.rng = rng

    def selectTrajPoint(self, energy0, energy1):
        r"""!
        Select a trajectory point using Metropolis accept/reject.
        \param energy0 Energy at point 0 including the artificial kinetic term .
        \param energy1 Energy at point 1 including the artificial kinetic term.
        \return `0` if `energy0` was selected, `1` otherwise.
        """

        deltaE = np.real(energy1 - energy0)
        return 1 if deltaE < 0 or np.exp(-deltaE) > self.rng.uniform(0, 1) \
            else 0

    def selectTrajectory(self, energy0, data0, energy1, data1):
        r"""!
        Select a trajectory point and pass along extra data.
        \param energy0 Energy at point 0 including the artificial kinetic term .
        \param data0 Arbitrary data assiciated with point 0.
        \param energy1 Energy at point 1 including the artificial kinetic term.
        \param data1 Arbitrary data assiciated with point 1.
        \return `(energy0, data0, 0)` if `energy0` was selected, otherwise
                `(energy1, data1, 1)`.
        """

        return (energy1, data1, 1) if self.selectTrajPoint(energy0, energy1) == 1 \
            else (energy0, data0, 0)
