'''
Created on Mar 17, 2011

@author: joel
'''
import unittest
import numpy as np
import math
import pypwdg.raytrace.homogeneous as prh
import pypwdg.raytrace.boundary as prb
import pypwdg.core.bases as pcb
import pypwdg.core.boundary_data as pcbd
import pypwdg.utils.quadrature as puq
import pypwdg.mesh.meshutils as pmmu

import test.utils.mesh as tum

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
                
        
class TestRayTrace(unittest.TestCase):
    
    def test2D(self):
        for n in range(1,10):
            mesh = tum.regularsquaremesh(n, "BDY")
            faces = mesh.entityfaces["BDY"]
            tracer = prh.HomogenousTrace(mesh, ["BDY"])
            
            for f in faces.tocsr().indices:
                if np.dot(mesh.normals[f], (-1,0)) > 0.5: # look for a face on the left side of the cube
                    point = mesh.directions[f][0] + np.sum(mesh.directions[f][1:-1], axis=0)/2.0 # pick a point in the middle
                    etods = prh.trace(point, [1,0], f, tracer, 6, -1)
                    self.assertEquals(len(etods), 2*n) # one strip contains 2n triangles
                    self.assertEquals(sum([len(ds) for ds in etods.values()]), 2*n*6+1) # each triangle should be painted 6 times, +1 for final reflection
                    
                    etods = prh.trace(point, [math.sqrt(2),1], f, tracer, -1, 100)
                    self.assertEquals(sum([len(ds) for ds in etods.values()]), 100) # we should manage to have painted 100 elements
                    break
            
            
class TestBoundary(unittest.TestCase):
    
    def testImpedance(self):
        mesh = tum.regularsquaremesh(5, "BDY")
        mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(10))
        for k in [1,10,100]:
            direction = np.array([1,1])/math.sqrt(2.0)
            g = pcb.PlaneWaves(direction, k)
            impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)
            
            bnddata={"BDY":impbd}
            ftodirs = prb.initialrt(mesh, bnddata, k, mqs, 5)
            
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()