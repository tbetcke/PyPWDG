'''
Created on Apr 2, 2011

@author: joel
'''
import unittest

import numpy as np
import numpy.random as nr

import pypwdg.utils.mappings as pum

#
#class TestAffineMap(unittest.TestCase):
#    
#    N = 10 
#    def test3to3(self):
#        """ build an affine map to map 4 points in 3-space to 4 other points in 3-space """
#        for _ in range(self.N):
#            pf = nr.random((4,3))
#            pt = nr.random((4,3))
#            A = pum.buildaffine(pf, pt)
#            np.testing.assert_array_almost_equal(A.apply(pf), pt)
#            np.testing.assert_array_almost_equal(A.applyinv(A.apply(pf)), pf)
#        
#    def test2to3(self):
#        """ build an affine map to map 3 points in 2-space to 3 points in 3-space """
#        for _ in range(self.N):
#            pf = nr.random((3,2))
#            pt = nr.random((3,3))
#            A = pum.buildaffine(pf, pt)
#            np.testing.assert_array_almost_equal(A.apply(pf), pt)
#    
#    def test3to3missing(self):
#        """ build an affine map to map 3 points in 3-space to 3 points in 3-space, preserving orthogonal vectors"""
#        for _ in range(self.N):
#            pf = nr.random((3,3))
#            pt = nr.random((3,3))
#            A = pum.buildaffine(pf, pt)
#            np.testing.assert_array_almost_equal(A.apply(pf), pt)
#            # construct the orthogonal vector
#            fognal = np.cross(pf[1] - pf[0], pf[2] - pf[0]) + pf[0]
#            np.testing.assert_array_almost_equal(np.dot(A.apply(fognal) - pt[0], (pt[1:] - pt[0]).transpose()), [[0,0]])
#            np.testing.assert_array_almost_equal(A.applyinv(A.apply(pf)), pf)
