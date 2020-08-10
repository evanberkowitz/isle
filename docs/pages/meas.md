\page measdoc Measurements

\todo Paths are not realtive anymore

Isle provides the high level measurement driver isle.drivers.meas.Measure.
This driver can perform measurements on configurations stored in files.
It requires the file layout to be compatible with "Configuration Files" as described in \ref filelayout.
See \ref measureExample for an example usage.

The measurement driver currently has some strong limitations.
Once a measurement was taken on a set of configurations and saved, there is no built in way to
extend it by measurements on additional configurations.
Furthermore, results are only saved after all configurations have been processed.
This can be a limitation in case not all results fit into memory simultaneously.
Both of those issues can be worked around by calling the driver multiple times and using the
`configSlices` of the measurements (see below).

All measurements should inherit from \ref isle.meas.measurement.Measurement "isle.meas.Measurement"
which provides a standard interface for the driver to use.
There are two attributes all measurements must have:
- <B>`savePath`</B> - The measurement promises to save its result in a group at this location in the
  output HDF5 file. The drives ensures ahead of time that those locations are available in order to avoid
  wasting time by measurements failing to save at the end of the process.
- <B>`configSlice`</B> - A `slice` object that indicates which configurations the measurement is to be
  taken on. This gets processed by the driver and no steps other than saving it need to be done
  in the measurement itself. Note that this attribute gets modified by the driver by replacing
  any `None`'s with the actual integers.

The base class also provides utilities for saving the `configSlice` in the standard fashion.
This can be overridden if need be.
That would however disable, among others, the weight loading facilities provided by Isle.
Further information on the file layout of measurements can be found in \ref filelayout.


## Custom Measurements

It is possible to implement custom measurements by inheriting from
\ref isle.meas.measurement.Measurement "isle.meas.Measurement" and implementing the abstract interface.
It is important to initialize the base class in order to set the save path and configuration
slice properly as they are needed by the measurement driver.

The `savePath` is a promise to the measurement driver that the measurement object only writes to that
location in the file.
The measurement is expected to create a new HDF5 group under `savePath` and write all its
results into that group.
The `configSlice` gets written automatically by `saveAll` unless overridden.
If `save` writes anywhere else, the operation might fail which would happen at the end of the measurement
process and thus waste a lot of time.
