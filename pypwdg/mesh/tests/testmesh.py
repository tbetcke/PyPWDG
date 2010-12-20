'''
Created on Aug 6, 2010

@author: joel
'''
import unittest

import test.utils.mesh as tum
import scipy.sparse as ss

class TestMesh(unittest.TestCase):
    def test2DMesh(self):
        for n in range(1,6,2):
            mesh = tum.regularsquaremesh(n, "TAG")
            # Count the number of faces.  There are 2 * n^2 elements, 3 faces per element
            self.assertEqual(mesh.nfaces, 6 * n**2) #
            # Count the number of elements.  There are 2 * n^2.
            self.assertEqual(mesh.nelements, 2 * n**2) 
            # Check that every face is either a boundary or an internal face 
            self.assertEqual((mesh.boundary + mesh.internal - ss.eye(mesh.nfaces, mesh.nfaces)).nnz, 0)
            # Check that the 2 ways of determining the boundary give the same faces
            self.assertEqual((mesh.boundary - mesh.entityfaces["TAG"]).nnz, 0) 
            
        






if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()