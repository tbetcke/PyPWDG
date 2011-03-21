'''
Created on Dec 19, 2010

@author: joel
'''
import unittest

import pypwdg.core.bases as pcb
import numpy


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
        
        