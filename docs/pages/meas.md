\page measdoc Measurements

Measurements are a uniform mechanism to extract information from gauge configurations.
They can be called both during HMC as 'inline' measurements or afterwards as
'out-of-line' measurements.
A number of fairly general purpose measurements are defined in namespace cns.meas.
See section Canonical Implementation for details on how to write those measurements.

Inline measurements represent a general hook into the Hybrid Monte Carlo function provided
by cns.hmc. Measurements are run after each Metropolis accept-reject step so they can be
used for a number of tasks. For instance, cns.hmc.hmc does not store or output intermediate
results by itself. Instead, a measurement can be used to save configurations to disk.
Other possible usage cases would be checkpointing or logging information to the terminal.

## Interface
An inline measurement must be a callable with parameters `phi, inline, itr, act, acc`, where
- <B>`phi`</B> - configuration,
- <B>`inline`</B> - `True` when called inline of HMC,
- <B>`itr`</B> - trajectory index,
- <B>`act`</B> - action given configuration `phi`,
- <B>`acc`</B> - `True` if trajectory was accepted, `False` otherwise.
- <B>`rng`</B> - A random number generator that implements cns.random.RNGWrapper.

`phi` and `inline` are called by position whereas `itr`, `act`, `acc`, and `rng` are called
by name. The return value is ignored when the measurement is called in-line.

Out-of-line measurements can have an arbitrary interface because they are only called
by the user. However, in order to provide a uniform way of calling both inline and
out-of-line measurements, the recommended implementation is
```.py
def myMeas(phi, inline=False, **kwargs):
    # ...
```
for functions and
```.py
class MyMeas:
    # ...

    def __call__(self, phi, inline=False, **kwargs):
        # ...
```
for callable classes. This can accommodate the special arguments passed to inline
measurements which might not be available out-of-line as well as possible extra
arguments used with out-of-line measurements not available inline.

## Canonical Implementation
All measurements in package cns.meas follow the same implementation scheme.
Each measurement is a class with a name in UpperCamelCase defined in its own file of the same
name but in lowerCamelCase. Measurements not defined in cns.meas can have any format as long
as the satisfy the interface described above.

As an example, look at the following:
```.py
# file acceptanceRate.py

class AcceptanceRate:
    def __call__(self, phi, inline=False, **kwargs):
        # ...
        
    # ...
```
This allows for the class to be found by the import script and it can be used as
```.py
# some other file

import cns
import cns.meas  # meas is not imported by cns base package

accMeas = cns.meas.AcceptanceRate()
```
If a different naming scheme is used or there are multiple measurements in one module,
the name of the module must be used:
```.py
import cns
import cns.meas  # still need this

nonCanMeas = cns.meas.non_canonocal_module.NonCanonicalMeasurement()
```
If a module contains extra functions or classes besides the central measurement, those can
only be addressed using the full module name as well.
