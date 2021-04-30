\page measdoc Measurements

Measurements are implemented as classes inheriting from \ref isle.meas.measurement.Measurement "isle.meas.Measurement".
You can instantiate a set of measurement objects and either apply them to configurations manually,
or ideally pass them to the high level measurement driver isle.drivers.meas.Measure.
This driver can perform measurements on configurations stored in files.
It requires the file layout to be compatible with "Configuration Files" as described in \ref filelayout.


## Performing Measurements

The easiest way to perform measurements is to use the built in measurement driver isle.drivers.meas.Measure
which can apply measurements to all configurations stored in a file and manage I/O.
See the example script \ref measureExample for the basic use case.

Each measurement stores a `configSlice` as an attribute which is a Python `slice` object specifying
which configurations the measurement is applied to.
Indices are with respect to the configuration indices from evolution.
That is, if you saved every 2nd configuration in an HMC run, a configuration slice of `slice(None, None, 2)`
means 'measure on _every stored_ configuration'.
A slice of `slice(None, None, 4)` means 'measure on every 2nd _stored_ configuration', while
`slice(None, None, 1)` is invalid because it refers to configurations that have not been saved to file.
Note that the driver may modify the slice in order to refer to replace `None` by the values for the concrete case.

The driver is not able to continue or extend existing measurements.
Once a certain measurement has been done and saved, you cannot apply it to additional configurations
and append the results to the same output dataset.
You can work around this issue by setting the `configSlices` of the measurements and writing the output
to a different location in the HDF5 file or a different file altogether.

Alternatively, you can call measurements yourself.
But note that measurements use a two-phase initialization, you must call the `setup` method before
applying a measurement.
(See below.)


## Output Format

Each measurement is given a `savePath` in the constructor which specifies where in the output HDF5 file
the results are saved.
An HDF5 group is created at that location and the measurement object has complete control over how and
where it stores its results in that group.
But all measurements should write their configuration slice as an attribute of their group.
Typically, results are stored as chunked HDF5 datasets through [pentinsula](https://github.com/jl-wynen/pentinsula).TimeSeries.


## Implementing Measurements

All measurements should inherit from \ref isle.meas.measurement.Measurement "isle.meas.Measurement"
which provides a standard interface for the driver to use.

Measurements are constructed in a two-phase initialization.
The init method is called by the user and should receive and store all relevant physical and algorithmic
parameters.
Make sure that you call the `__init__` method of the base class in order to save the common attributes
`savePath` and `configSlice` and also to prepare output buffers.
It is, however, not possible to estimate the available and required amounts of memory for the results
at this point because it is not know how many measurements there are or how many configurations need to
be processed.
For this reason, each measurement comes with a separate `setup` method which is normally called by the
driver which provides the necessary information for allocating output buffers.
For most cases, the default `setup` method provided by the base class is sufficient.

### Simple Example

Here is an example of a simple measurement:
```{.py}
from isle.meas.measurement import Measurement, BufferSpec

class MyMeas(Measurement):
    def __init__(self, lattice, savePath, configSlice=slice(None, None, None)):
        super().__init__(savePath,
                         (BufferSpec("Phi", (), complex, "totalPhi"),
                          BufferSpec("Phi_t", (lattice.nt(),), float, "Phi_t")),
                         configSlice)
        self.nt = lattice.nt()
        self.nx = lattice.nx()
```
It creates a new measurement class.
The init method passes the two usual arguments `savePath` and `configSlice` on to the base class constructor.
Also, in the call to `super().__init__`, it lists two buffer specifications which inform the base class
which output buffers will be needed by this measurement.
The arguments to `BufferSpec` are
- `name` - A name for querying the buffer in the measurement itself.
- `shape` - The shape of the result of the measurement for a single configuration.
            The 'Phi' buffer stores scalars, while 'Phi_t' stores vectors of length `nt`.
- `dtype` - The datatype of the buffer. 'Phi' stores complex numbers (equivalent to `np.complex128`)
            and 'Phi_t' stores real numbers (equivalent to `np.float64`).
- `path` - The location of the HDF5 dataset in the output file.
           This path is relative to the measurement's `savePath`.

At this point, no buffers are allocated.
This happens when the driver calls the `setup` method which uses the `BufferSpec`s.
We do not need to override the default setup here as we don't need any special behavior.

Measurements are performed by passing a configuration to the call operator.
In the example, we can implement it as
```{.py}
    def __call__(self, stage, itr):
        self.nextItem("Phi")[...] = np.sum(stage.phi)
        self.nextItem("Phi_t")[...] = np.sum(np.abs(np.reshape(stage.phi, (self.nt, self.nx))),
                                             axis=1)
```
The call method takes an \ref isle.evolver.stage.EvolutionStage "EvolutionStage"
and the index of the configuration as arguments.
In this case, the measurement simply calculates some sums over the field configuration and stores
the results in the output buffers.

The buffers can be accessed directly through the `_buffers` member of \ref isle.meas.measurement.Measurement "Measurement".
But it is easier to use the provided iterators through the \ref isle.meas.measurement.Measurement.nextItem "nextItem" method as shown in the example.
The method takes the name of a buffer and returns a numpy array view to the next item in the sequence.
This item has the shape given in the constructor and is meant to be written to.
(It is not safe to read from it!)
You can use any form of numpy indexing to set the item but it is important that you set all of it in order
to ensure that no uninitialized bytes are written to disk.

Note that each call to `nextItem` produces a new item that is appended to the sequence of results and
eventually written to file.
If you need to access an item multiple times, you need to store the return value of `nextItem`:
```{.py}
item = self.nextItem("myItem")
item[::2] = 1
item[1::2] = -1
```

Writing is done automatically when the internal buffer is full.
This happens _when_ calling `nextItem` and not after the `__call__` operator has finished.
The buffer is flushed when the measurement object is destroyed.

### Advanced Example

Here is an example that shows how to implement a custom setup for your measurement.
It is taken from isle.meas.logdet.Logdet.
```{.py}
from isle.meas.measurement import Measurement, BufferSpec
from pentinsula.h5utils import open_or_pass_file

class Logdet(Measurement):
    def __init__(self, hfm, savePath, configSlice=slice(None, None, None), alpha=1):
        super().__init__(savePath,
                         (BufferSpec("particles", (), np.complex128, "particles"),
                          BufferSpec("holes", (), np.complex128, "holes")),
                         configSlice)

        self.hfm = hfm
        self.alpha = alpha

    def __call__(self, stage, itr):
        if self.alpha == 1:
            self.nextItem("particles")[...] = isle.logdetM(self.hfm, stage.phi,
                                                           isle.Species.PARTICLE)
            self.nextItem("holes")[...] = isle.logdetM(self.hfm, stage.phi,
                                                       isle.Species.HOLE)
        else:
            # use dense, slow numpy routine to get stable result
            ld = np.linalg.slogdet(isle.Matrix(self.hfm.M(-1j*stage.phi, isle.Species.PARTICLE)))
            self.nextItem("particles")[...] = np.log(ld[0]) + ld[1]
            ld = np.linalg.slogdet(isle.Matrix(self.hfm.M(-1j*stage.phi, isle.Species.HOLE)))
            self.nextItem("holes")[...] = np.log(ld[0]) + ld[1]

    def setup(self, memoryAllowance, expectedNConfigs, file, maxBufferSize=None):
        res = super().setup(memoryAllowance, expectedNConfigs, file, maxBufferSize)
        with open_or_pass_file(file, None, "a") as h5f:
            h5f[self.savePath].attrs["alpha"] = self.alpha
        return res
```
This measurement computes `log det M` in either the spin basis (`alpha==0`) or the particle-hole basis (`alpha==1`).

The main point here is the custom `setup` method.
It extends the default setup in order to store the parameter `alpha` in the output file.
Note that the arguments of the method are forwarded unchanged to `super().setup`.
This is do ensure that all buffers are allocated properly and the group in the output file is created
and initialized.

See isle.meas.collectWeights.CollectWeights for an example on how to defer setup until
the first call of the the measurement.
