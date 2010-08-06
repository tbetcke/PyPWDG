'''
Created on Aug 6, 2010

@author: joel
'''
import unittest

import numpy as np

class TestMesh(unittest.TestCase):


    def setUp(self):
        from PyPWDG.mesh.gmsh_reader import gmsh_reader
        from PyPWDG.mesh.mesh import Mesh
        
        mesh_dict=gmsh_reader('../../examples/2D/square.msh')
        self.squaremesh=Mesh(mesh_dict,dim=2)
        mesh_dict=gmsh_reader('../../examples/3D/cube.msh')
        self.cubemesh=Mesh(mesh_dict,dim=3)

    def testQuadratures(self):
        """ This test ensures that the quadrature points and weights on neighbouring faces are the same """
        
        from PyPWDG.mesh.meshutils import MeshQuadratures
        from PyPWDG.utils.quadrature import legendrequadrature, trianglequadrature
        mq2 = MeshQuadratures(self.squaremesh, legendrequadrature(5))
        mq3 = MeshQuadratures(self.cubemesh, trianglequadrature(5))
        
        for mesh, mq in [(self.squaremesh, mq2), (self.cubemesh, mq3)]:
            for f1,f2 in enumerate(mesh.facemap):
                self.assertTrue(np.array_equal(mq.quadpoints(f1), mq.quadpoints(f2)))
                self.assertTrue(np.array_equal(mq.quadweights(f1), mq.quadweights(f2)))

        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()