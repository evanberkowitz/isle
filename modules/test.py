#!/usr/bin/env python3

import cns
import yaml

print("Imported cns!")

with open('test.yaml','r') as f:
    y = yaml.load(f)

L=cns.parse.lattice(y["Lattice"])

print(L)

l=cns.export.lattice(L)
print(l)
