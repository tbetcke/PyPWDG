'''
Created on Dec 19, 2010

@author: joel
'''
import pypwdg.utils.quadrature as puq

import unittest

import numpy as np

class testQuadrature(unittest.TestCase):
    
    def testTriangle(self):
        for n in range(1,10):
            x,w = puq.trianglequadrature(n)
            # test integration of constants
            self.assertAlmostEqual(sum(w), 0.5)
            # integrate x
            self.assertAlmostEqual(np.dot(x[:,0],w), 1.0/6)
            # integrate y
            self.assertAlmostEqual(np.dot(x[:,1],w), 1.0/6)

    def testLegendre(self):
        for n in range(1,10):
            x,w = puq.legendrequadrature(n)
            for m in range(0,2*n):
                # test integration of x^m
                self.assertAlmostEqual(np.dot(x.transpose() ** m, w), 1.0/(m+1) )
                