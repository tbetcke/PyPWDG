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
        k = 4
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        gen, ini = pca.pwbasisgeneration(k, npw)
        triquad = puq.trianglequadrature(nq)
        basis, coeffs, l2err = pca.optimalbasis(g, gen, ini, triquad, True)
        self.assertAlmostEqual(l2err,0)
        
    def testOptimalBasis2(self):
        """ Can we find the right direction to approximate a plane wave?"""
        k = 4
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        gen, ini = pca.pwbasisgeneration(k, npw)
        triquad = puq.trianglequadrature(nq)
        basis, (coeffs, l2err) = pca.optimalbasis2(g, gen, ini, triquad)
        self.assertAlmostEqual(sum(l2err),0)
        
    def testOptimalBasis3(self):
        """ Can we find the right direction to approximate a plane wave?"""
        k = 4
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        gen, ini = pca.pwbasisgeneration(k, npw)
        triquad = puq.trianglequadrature(nq)
        linearopt = pca.LeastSquaresFit(g, triquad)
        basis, (coeffs, l2err) = pca.optimalbasis3(linearopt.optimise, gen, ini)
        self.assertAlmostEqual(sum(l2err),0)
    
    def testPenalisedOptimisation(self):
        k = 4
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        alpha = 100
        ini = pcb.circleDirections(npw)
        triquad = puq.trianglequadrature(nq)
        linearopt = pca.LeastSquaresFit(g, triquad)
        pwpg = pca.PWPenaltyBasisGenerator(k, alpha, 2)
        basis, (coeffs, l2err) = pca.optimalbasis3(linearopt.optimise, pwpg.genbasis, ini, pwpg.penalty, pwpg.finalbasis)
        self.assertAlmostEqual(sum(l2err),0)        
        
    def testConstrainedOptimisation(self):
        k = 4
        g = pcb.PlaneWaves(numpy.array([[3.0/5,4.0/5]]), k).values
#        g = pcb.FourierBessel(numpy.array([-2,-1]), numpy.array([5]),k)
        npw = 3
        nq = 8
        ini = pcb.circleDirections(npw)
        triquad = puq.trianglequadrature(nq)
        linearopt = pca.LeastSquaresFit(g, triquad)
        pwpg = pca.PWPenaltyBasisGenerator(k, 1, 2)
        basis, (coeffs, l2err) = pca.optimalbasis3(linearopt.optimise, pwpg.finalbasis, ini)
        self.assertAlmostEqual(sum(l2err),0)        
        
        
        
