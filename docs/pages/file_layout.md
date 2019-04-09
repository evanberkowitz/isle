\page filelayout File Layout

Almost all components of Isle use HDF5 files with a common layout to store data.
Many functions are flexible though and can read from files that have slightly different layouts.
In the interest of compatibility however, the default layout should be used whenever possible.


## HDF5 Files

With the exception of lattices (see below), all data is stored in HDF5 files.
Isle uses two kinds of such files, *configuration* and *measurement* files.
Those kinds are compatible and it is possible to store all data in a single file.

### Configuration Files

Configuration files always contain three groups at the root level:
- <B>`configuration`</B>: Stores field configurations obtained from HMC.
                          The group contains additional groups labeled by the trajectory index.
                          Each of those groups contains three fields
                          * `phi`: The field configuration. (Complex lattice vector)
                          * `action`: Value of the action for that configuration. (complex number)
                          * `trajPoint`: Point along the trajectory that was accepted. (integer)
- <B>`checkpoint`</B>: Stores checkpoints along the Markov chain to resume HMC runs.
                       The group contains additional groups labeled by the trajectory index.
                       Note that there can be fewer checkpoints than configurations.
                       Each of those groups contains three fields
                       * `cfg`: The field configuration. (Link to sub group `/configuration`)
                       * `rng_state`: State of the RNG at the end of the trajectory.
                                      The content depends on the specific RNG but there is
                                      always a dataset called `name` which identifies the RNG.
                                      (HDF5 group)
                       * `evolver`: A group containing the state of the evolver at the end of
                                    the trajectory. See below for more information. (HDF5 group)
- <B>`meta`</B>: Stores information on the physical system and algorithmic parameters.
                 Contains the following objects:
                 * `action`: Source code of a function that instantiates the action which was
                             used to generate the configurations in the file. (String)
                 * `lattice`: YAML representation of the lattice, see below. (String)
                 * `params`: YAML representation of the parameters dataclass object.
                             Stores user defined parameters, physical and otherwise. (String)
                 * `evolvers`: General information on evolvers, see below. (Group)
                 * `version`: Group containing various versions of the software that was used
                              to make the file, most notably the version of Isle.
                              All versions are stored as strings.

#### Evolvers

Evolvers are stored in two different places in HDF5 files.
The `/meta/evolvers` group contains an enumerated list of the types of evolvers
and `/checkpoint/<itr>/evolver` contains the state of the evolver after the
corresponding HMC trajectory.

The types of evolvers are stored so they can be reconstructed at a later point in time
to, for instance, continue an HMC run.
Types are saved on demand only; as long as no checkpoints are written, no types are saved.<br>
There are two ways, the types can be stored:
- If the evolver is built into Isle or passed to the EvolverManager instance via the
  `definitions` parameter (e.g. through the parameter `definitions` of `isle.drivers.meas.newRun`),
  only its name is stored in a dataset called `__name__`.
- If the evolver is custom defined and not included in `definitions`, the full source code of the
  evolver's definition is stored and `__name__` is set to `"__as_source__"`.
  The dataset `__source__` contains the full source code of the evolver.

The group of the evolver state contains an attribute `__index__` which holds an integer
indicating which entry in the types group contains the evolver's type.
Any other content is determined by the evolver itself.
Evolvers are allowed to refer to the type definitions.

\see Additional information on %evolvers, in particular handling of types,
can be found in \ref evolversdoc.

### Measurement Files

Measurement files contain the same metadata at the root level as configuration files.
In addition, the results of measurements are written to the root level but each measurement
writes into its own separate group.
All group names depend on the user and can be chosen when calling the measurement driver.
The group contents are determined by the measurements.

The only common element is an attribute to the group called `configurations`.
This attribute contains a string in slice notation (`start:stop:step`) indicating which
configurations the measurement was taken on.
The indices in those slices are interpreted relative to the trajectory indices (group names)
of configurations in the `/configuration` group.

### Example

Here is an example of a combined configuration and measurement file:
```
╿
├─ configuration
│  ├─ 2
│  │  ├─ phi (complex lattice vector)
│  │  ├─ action (complex number)
│  │  └─ trajPoint (integer)
│  │
│  ├─ 4
│  │  ├─ phi (complex lattice vector)
│  │  ├─ action (complex number)
│  │  └─ trajPoint (integer)
│  ┆
│
├─ checkpoint
│  ├─ 2
│  │  ├─ cfg (link to /configuration/0)
│  │  ├─ rng_state (group, RNG dependent)
│  │  └─ evolver (group, evolver dependent)
│  ┆
│
├─ meta
│  ├─ action (source of function)
│  ├─ lattice (YAML)
│  ├─ params (YAML)
│  ├─ evolvers (group)
│  └─ versions (group)
│
│  # measurements:
│
├─ action
│  ├─ [configurations] (string attribute)
│  └─ action (dataset)
│
├─ correlation_functions
│  ├─ single_particle
│  │  ├─ [configurations] (string attribute)
│  │  ├─ correlators (dataset)
│  │  └─ irreps (dataset)
│  └─ single_hole
│     ├─ [configurations] (string attribute)
│     ├─ correlators (dataset)
│     └─ irreps (dataset)
┆
```


## Lattice Files

Lattices are stored as [YAML](https://yaml.org/).
Those YAML strings are by default stored in stand alone files (/resources/lattices) or as part
of the metadata in HDF5 files (see above).

Lattices are encoded via custom YAML nodes denoted by `!lattice`.
These nodes contain a mapping with the following keys:
- <B>`name`</B>: A string identifying the lattice. Should be unique to avoid conflicts.
- <B>`comment`</B>: An arbitrary string to give detailed information on the lattice.
- <B>`adjacency`</B>: A list of lists describing the adjacency matrix.
                      Each element of the outer list is a list of two integers indicating that
                      those two sites are connected.
- <B>`hopping`</B>: Can be either
  * A float: The hopping strength for all connections in the lattice.
  * A list of floats: Hopping strength for each connection. Must be the same length
                      as `adjacency` and is specified in the same order.
- <B>`positions`</B>: List of lists containing the 3D positions of all lattice sites.
                      Each element of the outer list is a list of three floats, the 3D
                      coordinates of that site.<br>
                      This field determines how many lattice sites there are and which order
                      they are indexed in.
                      Both `adjacency` and `hopping` have to be consistent with that.
- <B>`nt`</B>: The number of time slices.
               This field is *optional* and defaults to 0.
               Stand alone files should not set `nt` but only describe the
               spatial properties of the lattice.
               This allows those files to be reused easily but users have to make sure
               to set `nt` manually after reading a lattice from YAML.

Here is an example of a lattice with four sites in a hexagonal alignment.
This is taken from /resources/lattices/four_sites.yml.<br>
In this lattice, site 0 is connected to site 1 with hopping strength 1 and to
site 3 with strength 2, etc.
The position of site 0 is `(0, 0, 0)`, of site 1 is `(1, 0, 0)`, etc.

```{.yml}
!lattice
name: 'four sites'
comment: 'Four sites (hexagonal)'
adjacency:
  - [0, 1]
  - [0, 3]
  - [1, 2]
  - [2, 3]
hopping: [1, 2, 2, 1]
positions:
  - [0, 0, 0]
  - [1, 0, 0]
  - [1.5, 0.866025403784439, 0]
  - [2.5, 0.866025403784439, 0]
nt: 32
```

\see isle.yamlio.loadLattice for a convenient way to read lattices from YAML files.<br>
     Thanks to the custom node, lattices can be loaded via `yaml.safe_load(lattice_string)`
     and saved via `yaml.dump(lattice)`.
