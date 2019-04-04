\page filelayout File Layout

All components if Isle use HDF5 files and a common layout for data files.
A small exception to this are lattices, see below.
Many functions are flexible though and can read from files that have slightly different layouts.
In the interest of compatibility however, the default layout should be used whenever possible.

## HDF5 Files

With the exception of lattices, all


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
