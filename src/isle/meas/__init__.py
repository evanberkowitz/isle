"""!
General measurements that can be useful in various contexts.

Canonically, a measurement is a class with a name in UpperCamelCase in a module
of the same name but in lowerCamelCase.

More details can be found under \ref measdoc "Measurements".
"""

## \defgroup meas Measurements
# Perform measurements on configurations.

from .measurement import Measurement  # (unused import) pylint: disable=W0611
from .action import Action  # (unused import) pylint: disable=W0611
from .chiralCondensate import ChiralCondensate  # (unused import) pylint: disable=W0611
from .logdet import Logdet  # (unused import) pylint: disable=W0611
from .singleParticleCorrelator import SingleParticleCorrelator  # (unused import) pylint: disable=W0611
from .totalPhi import TotalPhi  # (unused import) pylint: disable=W0611
