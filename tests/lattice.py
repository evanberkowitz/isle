
"""
Unittest for Lattice.
"""

import unittest
import yaml

import isle
from . import core

# lattices to test on
LAT_DIR = core.TEST_PATH/"../resources/lattices"
# second entry in tuples is True for bipartite lattices
LATTICES = [(LAT_DIR/"one_site.yml", True),
            (LAT_DIR/"two_sites.yml", True),
            (LAT_DIR/"triangle.yml", False),
            (LAT_DIR/"four_sites.yml", True),
            (LAT_DIR/"tetrahedron.yml", False),
            (LAT_DIR/"c20.yml", False),
            (LAT_DIR/"c60_ipr.yml", False),
            (LAT_DIR/"tube_3-3_1.yml", True),
            (LAT_DIR/"tube_4-2_2.yml", True),
            (LAT_DIR/"graphene_7_5.yml", True)]

class TestLattice(unittest.TestCase):
    def test_1_isBipartite(self):
        "Test function isBipartite."
        for latfile, isBP in LATTICES:
            with open(latfile, "r") as f:
                lat = yaml.safe_load(f.read())
            self.assertEqual(isle.isBipartite(lat), isBP)
