'''
Created on Mar 17, 2011

@author: joel
'''
import unittest
import numpy as np
import pypwdg.raytrace.homogeneous as prh

class TestHomogeneous(unittest.TestCase):


    def testIntersect(self):
        for dim in [2,3]:
            for _ in range(10):
                linepoint = np.random.rand(1,dim)
                linedir = np.random.rand(1,dim)
                planepoint = np.random.rand(1,dim)
                planedirs = np.random.rand(dim-1,dim)
                l,m = prh.intersect(linepoint, linedir, planepoint, planedirs)
                # Check that the lambda (l) and mu (m) give the coordinates of an intersecting point
                np.testing.assert_almost_equal(linepoint + m * linedir, planepoint + np.dot(l, planedirs))
            
    def testReflect(self):
        for dim in [2,3]:
            for _ in range(10):
                linedir = np.random.rand(1,dim)
                planedirs = np.random.rand(dim-1,dim)
                refdir = prh.reflect(linedir, planedirs)
                # Check that the centre line is normal to the plane
                np.testing.assert_almost_equal(np.dot(planedirs, (refdir - linedir).transpose()), np.zeros((dim-1, 1)))
                # Check that the rest lies in the plane
                self.assertAlmostEqual(0, np.linalg.det(np.vstack((planedirs, refdir + linedir))))
                
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()