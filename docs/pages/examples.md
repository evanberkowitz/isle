\page examplesdoc Examples

Here are some example script showing how Isle can be used to perform HMC evolution and measurements.

- \subpage fullHMCExample - HMC evolution with thermalization and production with hard coded parameters.
- \subpage thermalizationExample - Thermalize a configuration controlled through command line arguments.
- \subpage changeExample - Continue an existing HMC run but swap out the evolver.
- \subpage tuningExample - Automatically tune a leapfrog integrator for an optimum acceptance rate.
- \subpage mpiTuningExample - Automatically tune a leapfrog integrator on multiple MPI ranks.
- \subpage measureExample - Perform measurements on an ensemble of configurations.

All scripts can be found in the directory `docs/examples`.

They can all process basic command line arguments.
Use  `py <scriptname> --help`  to get an overview of the supported arguments.


\page fullHMCExample HMC Evolution

This example shows how an HMC ensemble can be constructed from start to finish.
All parameters are hard coded, including the output file.
This serves only as a basic introduction into Isle and the other examples show more
sophisticated programs, see e.g. \ref thermalizationExample and \ref changeExample.

The following code can be found in `docs/examples/fullHMCEvolution.py`.
\include fullHMCEvolution.py



\page thermalizationExample Thermalization

This example shows how to set up an HMC run and start producing configurations.
The chosen evolver and parameters are meant for thermalization.
The transition to production can for instance be done with \ref changeExample.

This example also serves to show how to define a custom command line argument
parser which ties in with Isle's initialization routine.

The following code can be found in `docs/examples/hmcThermalization.py`.
\include hmcThermalization.py



\page changeExample Change Evolver

This example demonstrates how to continue HMC production from an existing file
and change the evolver in the process.
It uses leapfrog integration together with shifts by 2*pi in the field variables to
jump over barriers in configuration space which would otherwise hamper ergodicity.

This script requires the input file to contain checkpoints, not just saved configurations.
If this is not the case, more manual work is required to load / define the necessary objects
and set up the HMC driver.

The following code can be found in `docs/examples/changeEvolver.py`.
\include changeEvolver.py



\page tuningExample Automatic Leapfrog Tuning

This example demonstrates how a leapfrog integrator can be automatically tuned using the
isle.evolver.autotuner.LeapfrogTuner evolver.
First, a thermalized configuration is produced as in \ref fullHMCExample and \ref thermalizationExample.
Then, the leapfrog tuner is used to estimate parameters to achieve a 61% acceptance rate.
Finally, a few configurations are produced using the tuned evolver.

The tuning evolver can not write checkpoints due to a restriction in its implementation.
Thus continuing a run after tuning is not as easy as in other cases (see for instance \ref changeExample).
The function `produce` demonstrates how to switch to production given only the output of
a tuning run, not any of the parameters defined elsewhere in the script.
Note in particular that the RNG state can not be read from the file.

The following code can be found in `docs/examples/tuning.py`.
\include tuning.py



\page mpiTuningExample Automatic Leapfrog Tuning with MPI

This example extends the capabilities of \ref tuningExample by adding support to MPI.
Most Isle-related code should be familiar from other examples.
The script uses mpi4py.futures to dynamically assign ensembles to MPI ranks
and tune the integrator for each ensemble separately.

See the initial docstring in the code for usage information.

The following code can be found in `docs/examples/mpiTuning.py`.
\include mpiTuning.py



\page measureExample Measure

This example shows how to perform measurements on an existing ensemble of configurations.

\attention
    This script is not fully general, it requires a certain choice of action in order to work.
    In particular, it supports ensembles written with \ref fullHMCExample
    or \ref thermalizationExample.

The following code can be found in `docs/examples/measure.py`.
\include measure.py
