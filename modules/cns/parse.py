from . import Lattice
import yaml

def lattice(lat):
    nt          = lat["nt"]
    hopping     = lat["hopping"]
    connections = hopping["connections"]
    strengths   = hopping["strengths"]
    coordinates = hopping["coordinates"]
    
    # If the user passed one number, use it for every connection.
    if type(strengths) in (int, float):
        strengths = [ strengths for c in connections ]
    
    # If the user passed an array of the wrong length, return None
    if len(connections) != len(strengths):
        print("{} connections but {} strengths.".format(len(connections),len(strengths)))
        #// TODO: error, rather than return None?
        return None
    
    # The number of sites is one more than the largest ID number, because the IDs start at 0.
    nx=max(max(row) for row in connections)+1
    
    # Initialize a lattice
    L = Lattice(nt, nx)
    # and loop over all the connections, setting their strengths.
    [ L.setNeighbor(i,j,s) for [[i,j], s] in zip(connections,strengths) ]
    
    #// TODO: set coordinates.

    return L
