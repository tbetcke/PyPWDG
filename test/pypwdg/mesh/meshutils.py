'''
Created on Dec 19, 2010

@author: joel
'''
import unittest

import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.quadrature as puq
import test.utils.mesh as tum

import numpy as np

class TestMeshUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test2DQuadratures(self):
        meshes = [tum.regularsquaremesh(1), tum.regularsquaremesh(3), tum.squaremesh()]
        self.doQuadratures(meshes, puq.legendrequadrature(6))
        
    def doQuadratures(self, meshes, quadrature):
        """ This test ensures that the quadrature points and weights on neighbouring faces are the same """
                                
        for mesh in [meshes]:
            mq = pmmu.MeshQuadratures(mesh, quadrature)
            internalfaces = np.nonzero(mesh.internal.diagonal())
            facemap = (mesh.connectivity * np.arange(mesh.nfaces))[internalfaces]
            for f1,f2 in zip(internalfaces, facemap):
                self.assertTrue(np.array_equal(mq.quadpoints(f1), mq.quadpoints(f2)))
                self.assertTrue(np.array_equal(mq.quadweights(f1), mq.quadweights(f2)))

