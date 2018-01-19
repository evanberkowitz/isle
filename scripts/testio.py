#!/usr/bin/env python3

"""
Demonstrate IO routines.
"""

import yaml

import core
core.prepare_module_import()
import cns

with open("../lattices/c60_ipr.yml", "r") as yamlf:
    lat = yaml.safe_load(yamlf)
with open("out.yml", "w") as yamlf:
    yaml.dump(lat, yamlf)
