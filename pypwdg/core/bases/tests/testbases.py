'''
Created on Dec 19, 2010

@author: joel
'''
import unittest

import pypwdg.core.bases as pcb
import test.utils.mesh as tum
import pypwdg.core.bases.utilities as pcbu
import pypwdg.core.bases.reference as pcbr
import pypwdg.utils.geometry as pug

import numpy
import numpy as np
import numpy.random as nr


class TestBases(unittest.TestCase):


    def testPlaneWave(self):
        dirs = numpy.array([[1,0]])
        k = 3
        pw = pcb.PlaneWaves(dirs, k)
        # check that there's just one function
        self.assertEqual(pw.n, 1)
        # evaluate it at 0
        fd0 = pw.values(numpy.zeros((1,2)))
        # answer should be 1
        self.assertAlmostEqual(fd0[0,0], 1.0)
        # now evaluate some directional derivatives
        fn0 = pw.derivs(numpy.zeros((1,2)), numpy.array([[1,0],[0,1]]))
        self.assertAlmostEqual(fn0[0,0], k*1j)
        self.assertAlmostEqual(fn0[1,0], 0)
    
    def testDirections(self):
        n = 4
        d1 = pcb.cubeDirections(n)
        self.assertEqual(len(d1), n*n)
        d1r = pcb.cubeRotations(d1)
        self.assertEqual(len(d1r), 6 * n * n)
        d2 = pcb.circleDirections(n)
        self.assertEqual(len(d2), n)
        
    def testFourierBessel(self):
        orders = numpy.array([1,2,3,4])
        k = 5
        for origin in [numpy.zeros(2), numpy.array([4.0,5.0])]:
            for fb in [pcb.FourierBessel(origin, orders, k), pcb.FourierHankel(origin,orders,k)]:
                points = numpy.array([[1,2],[1,0],[0,3]])
                n = numpy.array([[1,0],[0,1],[0.6,0.8]])
                h = 1E-10
                v0 = fb.values(points)
                vh = fb.values(points + n * h)
                d = fb.derivs(points, n)
                numpy.testing.assert_array_almost_equal(d, (vh - v0)/h, decimal=4)
                
    def testDerivatives(self):
        k = 10
#        meshes = [tum.regularsquaremesh(2)]
        meshes = tum.meshes2d()
        structuredpoints = pug.StructuredPoints([[0.01,0.01],[0.99,0.99]], [20,30])
        rules = [pcbu.planeWaveBases(2, k),
                 pcbu.FourierBesselBasisRule(range(-4,5)),
                 pcbu.FourierHankelBasisRule([[-1,-1]], range(-4,5)),
                 pcbu.ProductBasisRule(pcbu.planeWaveBases(2, k, 3),pcbr.ReferenceBasisRule(pcbr.Dubiner(0)))]
        basisDerivatives(rules, meshes, structuredpoints, k)

def basisDerivatives(basisrules, meshes, structuredpoints, k):
    # Tests that we get the correct directional derivatives 
    N = 5
    h = 1E-5
    for mesh in meshes:
        ei = pcbu.ElementInfo(mesh, k)
        for e in range(mesh.nelements):
            _, points = pug.elementToStructuredPoints(structuredpoints, mesh, e)
            # generate some random normals
            if points is not None and len(points):
                nn = nr.random((len(points),2))
                n = nn / np.sqrt(np.sum(nn**2, axis=1)).reshape(-1,1)
                for basisrule in basisrules:
                    basis = basisrule.populate(ei.info(e))[0]
                    
                    vd = basis.derivs(points)
                    vl = basis.laplacian(points)
                    vp = basis.values(points)
                    
                    vph = [basis.values(points + dx) for dx in [[h,0],[0,h], [-h,0], [0,-h]]]
                    vdh = np.dstack([vph[0] - vph[2], vph[1] - vph[3]])/(2*h)
                    vlh = (sum(vph) - 4*vp) / (h**2)
                    
                    vdn = basis.derivs(points, n)
                    vdnh = (basis.values(points + n * h) - basis.values(points - n * h)) / (2*h) # compute the derivatives in mesh coordinates using finite differences
                    
                    np.testing.assert_allclose(vdnh, vdn, rtol=1E-5, atol=1E-5)
                    np.testing.assert_allclose(vdh, vd, rtol=1E-5, atol=1E-5)
                    np.testing.assert_allclose(vlh, vl, rtol=1E-3, atol=1E-4)
        

#class TestBasisDerivatives(unittest.TestCase):
                 
                 
        
     
        