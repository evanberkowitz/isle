from .. import fileio

def verifyMetadataByException(outfname, lattice, params):
    "!Make sure that metadata in file agrees with function parameters."

    storedLattice, storedParams, _ = fileio.h5.readMetadata(outfname)

    if storedLattice.name != lattice.name:
        print(f"Error: Name of lattice in output file is {storedLattice.name} but new lattice has name {lattice.name}. Cannot write into existing output file.")
        raise RuntimeError("Lattice name inconsistent")

    if storedParams.asdict() != params.asdict():
        print(f"Error: Stored parameters do not match new parameters. Cannot write into existing output file.")
        raise RuntimeError("Parameters inconsistent")
