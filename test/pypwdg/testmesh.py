'''
Created on Aug 6, 2010

@author: joel
'''
import unittest

import numpy as np
from test.pypwdg import __path__ as path

class TestMesh(unittest.TestCase):


    def setUp(self):
        from pypwdg.mesh.gmsh_reader import gmsh_reader
        from pypwdg.mesh.mesh import Mesh
        
        mesh_dict=gmsh_reader(path[0] + '/../../examples/2D/square.msh')
        self.squaremesh=Mesh(mesh_dict,dim=2)
        mesh_dict=gmsh_reader(path[0] + '/../../examples/3D/cube.msh')
        self.cubemesh=Mesh(mesh_dict,dim=3)

    def testQuadratures(self):
        """ This test ensures that the quadrature points and weights on neighbouring faces are the same """
        
        from pypwdg.mesh.meshutils import MeshQuadratures
        from pypwdg.utils.quadrature import legendrequadrature, trianglequadrature
        mq2 = MeshQuadratures(self.squaremesh, legendrequadrature(5))
        mq3 = MeshQuadratures(self.cubemesh, trianglequadrature(5))
        
        for mesh, mq in [(self.squaremesh, mq2), (self.cubemesh, mq3)]:
            for f1,f2 in enumerate(mesh.facemap):
                self.assertTrue(np.array_equal(mq.quadpoints(f1), mq.quadpoints(f2)))
                self.assertTrue(np.array_equal(mq.quadweights(f1), mq.quadweights(f2)))

    def testStructure(self):
        from pypwdg.mesh.structure import sparseindex
        rows = np.array([1,2])
        cols = np.array([2,1])
        id = sparseindex(rows, cols, 3)
        m = np.array([[0,0,0],[0,0,1],[0,1,0]])
        self.assertTrue(np.array_equal(m, id.todense()))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()