from . import Lattice
import yaml

def lattice(L):
    if type(L) is not Lattice:
        print("Wrong data type.", type(L))
        #// TODO: issue error?
        return None
    
    lat = {}
    lat["nt"] = L.nt()
    hopping = {}
    # // TODO: reconstruct hopping dictionary.
    # hopping["connections"] = ...
    # hopping["strengths"]   = ...
    # hopping["coordinates"] = ...
    lat["hopping"] = hopping
    return lat