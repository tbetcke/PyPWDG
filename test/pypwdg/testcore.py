'''
Created on Aug 5, 2010

@author: joel
'''
import unittest

import pypwdg.core.bases as pcb
import pypwdg.core.adaptivity as pca
import pypwdg.utils.quadrature as puq
import numpy
from test.pypwdg import __path__ as path


class TestBases(unittest.TestCase):


    def testPlaneWave(self):
        from pypwdg.core.bases import PlaneWaves
        dirs = numpy.array([[1,0]])
        k = 3
        pw = PlaneWaves(dirs, k)
        # check that there's just one function
        self.assertEqual(pw.n, 1)
        # evaluate it at 0
        fd0 = pw.values(numpy.zeros((1,2)), None)
        # answer should be 1
        self.assertAlmostEqual(fd0[0,0], 1.0)
        # now evaluate some directional derivatives
        fn0 = pw.derivs(numpy.zeros((1,2)), numpy.array([[1,0],[0,1]]))
        self.assertAlmostEqual(fn0[0,0], k*1j)
        self.assertAlmostEqual(fn0[1,0], 0)
    
    def testDirections(self):
        from pypwdg.core.bases import cubeDirections, circleDirections, cubeRotations
        n = 4
        d1 = cubeDirections(n)
        self.assertEqual(len(d1), n*n)
        d1r = cubeRotations(d1)
        self.assertEqual(len(d1r), 6 * n * n)
        d2 = circleDirections(n)
        self.assertEqual(len(d2), n)
        
    def testFourierBessel(self):
        from pypwdg.core.bases import FourierBessel
        orders = numpy.array([1,2,3,4])
        origin = numpy.zeros(2)
        k = 5
        fb = FourierBessel(origin, orders, k)
        points = numpy.array([[1,2],[1,0],[0,3]])
        n = numpy.array([[1,0],[0,1],[0.6,0.8]])
        h = 1E-10
        v0 = fb.values(points)
        vh = fb.values(points + n * h)
        d = fb.derivs(points, n)
        numpy.testing.assert_array_almost_equal(d, (vh - v0)/h, decimal=4)
        
        
        
class TestVandermondes(unittest.TestCase):
    
    def setUp(self):
        from pypwdg.mesh.gmsh_reader import gmsh_reader
        from pypwdg.mesh.mesh import Mesh
        
        mesh_dict=gmsh_reader(path[0] + '/../../examples/2D/square.msh')
        self.squaremesh=Mesh(mesh_dict,dim=2)

    
    def testLocalVandermondes(self):
        # This test is more about exercising the code than testing any results.  We need a know simple mesh to do that
        from pypwdg.core.bases import PlaneWaves
        from pypwdg.core.vandermonde import LocalVandermondes
        from pypwdg.utils.quadrature import legendrequadrature
        from pypwdg.mesh.meshutils import MeshQuadratures
        dirs = numpy.array([[1,0]])
        k = 3
        pw = PlaneWaves(dirs, k)
        faceid = 0
        numquads = 3
        mq = MeshQuadratures(self.squaremesh, legendrequadrature(numquads))
        LV = LocalVandermondes(self.squaremesh, [[pw]] * self.squaremesh.nelements, mq.quadpoints)
        for faceid in range(self.squaremesh.nfaces):
            v = LV.getValues(faceid)
            d = LV.getDerivs(faceid)
            self.assertEqual(v.shape, (numquads, 1))
            self.assertEqual(d.shape, (numquads, 1))
            self.assertEqual(LV.numbases[faceid], 1)
        
class TestAdaptivity(unittest.TestCase):
    def testOptimalBasis(self):
        """ Can we find the right direction to approximate a plane wave?"""
        k = 10
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        gen, ini = pca.pwbasisgeneration(k, npw)
        triquad = puq.trianglequadrature(nq)
        basis, coeffs, l2err = pca.optimalbasis(g, gen, ini, triquad, True)
        self.assertAlmostEqual(l2err,0)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()