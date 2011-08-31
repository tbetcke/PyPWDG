'''
Created on Dec 19, 2010

@author: joel
'''
import unittest

import pypwdg.mesh.meshutils as pmmu
import pypwdg.utils.quadrature as puq
import pypwdg.test.utils.mesh as tum

import numpy as np
import math

class TestMeshUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    def test3DQuadratures(self):
        """ Ensure quadratures on neighbouring faces of various 3D meshes match """
        self.compareQuadratures(tum.meshes3d(), puq.trianglequadrature(4))

    def test2DQuadratures(self):        
        """ Ensure quadratures on neighbouring faces of various 2D meshes match """
        self.compareQuadratures(tum.meshes2d(), puq.legendrequadrature(6))
        
    def compareQuadratures(self, meshes, quadrature):
        """ Test that the quadrature points and weights on neighbouring faces of a list of meshes are the same""" 
        for mesh in meshes:
            mq = pmmu.MeshQuadratures(mesh, quadrature)
            internalfaces = np.nonzero(mesh.internal.diagonal())[0]
            facemap = (mesh.connectivity * np.arange(mesh.nfaces))[internalfaces]
    
            for f1,f2 in zip(internalfaces, facemap):
                self.assertTrue(np.array_equal(mq.quadpoints(f1), mq.quadpoints(f2)))
                self.assertTrue(np.array_equal(mq.quadweights(f1), mq.quadweights(f2)))

class TestElementQuadratures(unittest.TestCase):
    
    def testMEQ(self):
        for n in range(1,6):
            mesh = tum.regularsquaremesh(n)
            mq = pmmu.MeshElementQuadratures(mesh, puq.trianglequadrature(5))
            for i in range(2*n**2):
                self.assertAlmostEquals(sum(mq.quadweights(i)), 1.0/(2 * n**2))

    def testEQ(self):
        for n in range(1,6):
            mesh = tum.regularsquaremesh(n)
            mq = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(6))
            for i in range(6*n**2):
                if i % 3==1: self.assertAlmostEquals(sum(mq.quadweights(i)), math.sqrt(2)/n)
                else: self.assertAlmostEquals(sum(mq.quadweights(i)), 1.0/n)