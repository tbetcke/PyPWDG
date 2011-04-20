'''
Created on Mar 17, 2011

@author: joel
'''
import unittest
import numpy as np
import math
import pypwdg.raytrace.element as pre
import pypwdg.raytrace.control as prc

import test.utils.mesh as tum

class TestHomogeneous(unittest.TestCase):


    def testIntersect(self):
        for dim in [2,3]:
            for _ in range(10):
                linepoint = np.random.rand(1,dim)
                linedir = np.random.rand(1,dim)
                planepoint = np.random.rand(1,dim)
                planedirs = np.random.rand(dim-1,dim)
                l,m = pre.intersect(linepoint, linedir, planepoint, planedirs)
                # Check that the lambda (l) and mu (m) give the coordinates of an intersecting point
                np.testing.assert_almost_equal(linepoint + m * linedir, planepoint + np.dot(l, planedirs))
            
    def testReflect(self):
        for dim in [2,3]:
            for _ in range(10):
                linedir = np.random.rand(1,dim)
                planedirs = np.random.rand(dim-1,dim)
                refdir = pre.reflect(linedir, planedirs)
                # Check that the centre line is normal to the plane
                np.testing.assert_almost_equal(np.dot(planedirs, (refdir - linedir).transpose()), np.zeros((dim-1, 1)))
                # Check that the rest lies in the plane
                self.assertAlmostEqual(0, np.linalg.det(np.vstack((planedirs, refdir + linedir))))
                
        
class TestRayTrace(unittest.TestCase):
    
    def test2D(self):
        for n in range(1,10):
            mesh = tum.regularsquaremesh(n, "BDY")
            faces = mesh.entityfaces["BDY"]
            tracer = pre.HomogenousTrace(mesh, [])
            
            for f in faces.tocsr().indices:
                if np.dot(mesh.normals[f], (-1,0)) > 0.5: # look for a face on the left side of the cube
                    point = mesh.directions[f][0] + np.sum(mesh.directions[f][1:-1], axis=0)/2.0 # pick a point in the middle
                    etods = prc.trace(f, point, [1,0], tracer, 6, -1)
                    self.assertEquals(len(etods), 2*n) # one strip contains 2n triangles
                    self.assertEquals(sum([len(ds) for ds in etods.values()]), 2*n*6+1) # each triangle should be painted 6 times, +1 for final reflection
                    
                    etods = prc.trace(f, point, [math.sqrt(2),1], tracer, -1, 100)
                    self.assertEquals(sum([len(ds) for ds in etods.values()]), 100) # we should have managed to paint 100 elements
                    break
            
            
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()