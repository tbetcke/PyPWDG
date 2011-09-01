'''
Created on Sep 1, 2011

@author: joel
'''
import unittest
import numpy as np
import pypwdg.utils.curved as puc

class Test(unittest.TestCase):


    def testCurve1(self):
        n = 2
        vertices = np.vstack((np.eye(n), np.zeros((1,n))))
        subsimplexids = np.array([0,1]) 
        surface = lambda t: np.hstack((t*(1-t)/4, t))
        fm = puc.FaceMapping(vertices, subsimplexids, surface)
        np.testing.assert_almost_equal(vertices, fm.apply(vertices))
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCurve1']
    unittest.main()