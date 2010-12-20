'''
Created on Dec 19, 2010

@author: joel
'''
import pypwdg.core.adaptivity as pca
import pypwdg.core.bases as pcb
import pypwdg.utils.quadrature as puq

import unittest

import numpy
        
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
        
