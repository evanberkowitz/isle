import yaml
import h5py as h5

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

def prepareOutfile(outfname, lattice, params, makeActionSrc,
                    extraGroups=[]):
    "!Prepare the output file by storing program versions, metadata, and creating groups."

    # TODO write Version(s)  -  write function in h5io

    with h5.File(str(outfname), "w") as outf:
        metaGrp = fileio.h5.createH5Group(outf, "meta")
        metaGrp["lattice"] = yaml.dump(lattice)
        metaGrp["params"] = yaml.dump(params)
        metaGrp["action"] = makeActionSrc
        for group in extraGroups:
            fileio.h5.createH5Group(outf, group)
