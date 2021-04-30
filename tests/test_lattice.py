
"""
Unittest for Lattice.
"""

import unittest
import yaml

import isle
from . import core

# second entry in tuples is True for bipartite lattices
LATTICES = [("one_site", True),
            ("two_sites", True),
            ("triangle", False),
            ("four_sites", True),
            ("tetrahedron", False),
            ("c20", False),
            ("c60_ipr", False),
            ("tube_3-3_1", True),
            ("tube_4-2_2", True),
            ("graphene_7_5", True)]

ILL_LABELED_LATTICE = """
!lattice
adjacency:
- [0, 1]
- [0, 11]
- [0, 11]
- [1, 3]
- [1, 3]
- [3, 2]
- [2, 4]
- [2, 4]
- [4, 5]
- [5, 6]
- [5, 6]
- [6, 7]
- [7, 8]
- [7, 8]
- [8, 9]
- [9, 10]
- [9, 10]
- [10, 11]
comment: 'Tube with non-alternating site labels, sites 2 and 3 are exchanged'
hopping: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0]
name: Tube (3, 3) <bad labels>
nt: 0
positions:
- [1.432394487827058, 0.0, 0.0]
- [1.0972778377541723, 0.9207254289585293, 0.0]
- [-0.2487326925113246, 1.4106331969840376, 0.8660254037844386]
- [0.7161972439135291, 1.240490014699032, 0.8660254037844386]
- [-0.7161972439135287, 1.2404900146990323, 0.0]
- [-1.346010530265497, 0.48990776802550845, 0.0]
- [-1.432394487827058, 1.754177324633719e-16, 0.8660254037844388]
- [-1.0972778377541723, -0.9207254289585293, 0.8660254037844388]
- [-0.7161972439135297, -1.2404900146990316, 0.0]
- [0.24873269251132413, -1.4106331969840378, 0.0]
- [0.7161972439135291, -1.240490014699032, 0.8660254037844384]
- [1.3460105302654968, -0.4899077680255092, 0.8660254037844384]
"""


class TestLattice(unittest.TestCase):
    def test_1_isBipartite(self):
        "Test function isBipartite."

        # test files with correctly alternating labels
        for latname, isBP in LATTICES:
            lat = isle.LATTICES[latname]
            self.assertEqual(isle.isBipartite(lat), isBP)

        # test ill-labeled
        lat = yaml.safe_load(ILL_LABELED_LATTICE)
        self.assertEqual(isle.isBipartite(lat), False)
