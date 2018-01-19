#!/usr/bin/env python3

import yaml
import cns

with open("c60_ipr.yml", "r") as yamlf:
    lat = yaml.safe_load(yamlf)
with open("out.yml", "w") as yamlf:
    yaml.dump(lat, yamlf)
