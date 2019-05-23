\page evolversdoc Evolvers

\todo documen transforms

Evolvers are the main component of HMC evolution.
Their task is transforming one configuration into another one along the Markov chain.

All evolvers must inherit from isle.evolver.Evolver and implement the abstract interface.
- The <B>`evolve`</B> method takes an input configuration `phi` and returns
  a new configuration both stored in an instance of isle.evolver.stage.EvolutionStage.
- <B>`save`</B> stores the current state of the evolver in an HDF5 group so the evolver can
  be reconstructed by `fromH5` later and resume evolution.
- The classmethod <B>`fromH5`</B> constructs a new instance of the evolver from HDF5 based
  on a state previously written with `save`.


### Evolution

Evolvers are responsible for almost all of HMC evolution.
They have to generate suitable proposals for new configurations and
select one all in such a way that reversibility and detailed balance are maintained.
Only the generation of the conjugate momenta is handled by the user.

An examples of this is isle.evolver.leapfrog.ConstStepLeapfrog which uses leapfrog integration
to generate a new `phi` and `pi` out of a starting configuration and momentum.
Those new fields are either accepted or rejected by isle.evolver.selector.BinarySelector which
uses Metropolis accept/reject.

Another example is isle.evolver.hubbard.TwoPiJumps which generates discrete shifts in the
configuration and uses knowledge of the action of the Hubbard model to compute the change in energy.
It can do multiple updates and accept or reject each one individually all encapsulated
in one call to `evolve`.

Evolvers have to operate as mostly closed off units and can only communicate by means of
the interface of the `evolve` method as mediated by the HMC driver.
The argument and return value of `evolve`, isle.evolver.stage.EvolutionStage, allows
storing custom MC weights in addition to the action.
On top of this, there is a facility to store arbitrary data (as long as it can be written to HDF5)
in the '`extra`' attribute.
For both special cases it is important to keep in mind that users might use evolvers in ways
not expected by the author.
It is therefore advised to guard any usage of `EvolutionStage.logWeights` and
`EvolutionStage.extra` to make sure MCMC proceeds correctly.
As a good practice, all evolvers should use the `accept` and `reject` methods of
`EvolutionStage` to clear out any unused extra attributes and help with detecting errors.


### Save / Load

Since evolvers drive HMC evolution, they can in some cases hold state which needs to be
handled consistently in order to construct a proper Markov chain.
Such state must be recovered when an HMC run is resumed from a checkpoint stored in a file.
Evolvers must therefore be able to save their state, or the relevant portions thereof, to HDF5
and retrieve it later.
In addition, information on the type of the evolver is stored in a separate location in the file.
This allows a run to be continued from just the HDF5 file without any need for
additional, user written files.

#### Saving Types

Saving and loading the type of an evolver is handled by isle.evolver.EvolverManager and the
individual evolvers do not have to do anything special to enable this
(except for the third point below).
An exception to this are collective evolvers such as isle.evolver.Alternator which uses an
evolver manager to save/load the types of its sub evolvers as part of its state.
This is necessary since the manager is unaware of the sub evolvers and cannot handle them when
saving/loading the %Alternator type.

isle.evolver.EvolverManager supports different ways of saving types in order to accommodate
custom evolvers as well as built in ones.
There are three scenarios for saving/loading evolver types
(see \ref filelayout for more details on how this looks in the file):
- The evolver is built into Isle (i.e. defined in package isle.evolver).
  In this case, reconstructing it is trivial and only its classname is stored in the file
  (dataset `__name__`).<br>
  Note to *developers*: All evolvers must be imported into the package isle.evolver
  in order for `saveEvolverType` to find and store them correctly.

- The evolver is custom defined but is provided as part of the `definitions` argument
  to the constructor of EvolverManager.
  Again, only the name of the class is stored and there is no visible difference
  in the HDF5 file.<br>
  The evolver can only be reconstructed if the type is provided to the EvolverManager
  which reads the file.
  This can potentially be in a different session and thus requires extra work and care by the user.

- The evolver is custom defined and not provided to the manager.
  In this case, the full source code of the evolver class is saved to the file in the dataset
  `__source__` and `__name__` is set to `"__as_source__"`.
  This allows custom evolvers to be reconstructed from file without any additional action by
  the user.
  This is used for instance by the program `isle continue`.
  \attention
      Evolvers saved this way must be fully self contained, i.e. import
      any required modules (modules `isle`, `evolvers`, and class `Evolver`
      are provided and don't have to be imported).

  &nbsp;


#### Saving State

Saving and loading state is the responsibility of the evolvers themselves.
The `save` method is given an HDF5 group for the evolver to do with as it pleases.
It must however ensure that calling `fromH5` with an HDF5 group constructs an evolver
that is completely equivalent to the one that was saved, i.e. any observable state must
be the same.
This is necessary to allow Isle to resume an HMC run.

Evolvers are allowed to issue additional save/load commands through the EvolverManager
that is provided to `save` and `fromH5`.
This can be used to save/load sub evolvers which would otherwise not be visible to
the manager.
See for example isle.evolver.Alternator.

`fromH5` has some additional parameters that contain some global parameters / state
of the simulation.
Those are the action, lattice, and random number generator.
The evolver is allowed to store references to them and use them for evolution.
Since those objects are provided, they do not have to be saved by the evolver,
thus avoiding duplication and reducing file size.
