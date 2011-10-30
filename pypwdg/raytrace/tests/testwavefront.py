'''
Created on Oct 25, 2011

@author: joel
'''
import unittest
import numpy as np
import pypwdg.raytrace.wavefront as prw
import pypwdg.utils.timing as put
import pypwdg.test.utils.mesh as ptum

class Test(unittest.TestCase):


    def testFillIn(self): 
        x = np.arange(10.0)[:,np.newaxis]**2
        p = np.arange(10.0)[:,np.newaxis]*3
        xi,pi,idx = prw.fillin(x[:,np.newaxis], p[:,np.newaxis], 4.5)
        self.assertEqual(len(xi), idx[-1]+1)
        self.assertEqual(len(xi), len(pi))
        np.testing.assert_array_equal(x, xi[idx])
        np.testing.assert_array_equal(p, pi[idx])
        
    def testNSteps(self):
        N = 21
        deltat = 1.0
        c = 0.2
        x = np.vstack((np.linspace(0,1,N), np.zeros(N))).transpose()
        p = np.vstack((np.zeros(N), np.ones(N))).transpose()
        slowness = lambda x: np.ones(len(x))/c
        gradslowness = lambda x: np.zeros_like(x)
        xk,pk = prw.onestep(x, p, slowness, gradslowness, deltat)
        np.testing.assert_almost_equal(xk, x + p * c)
        np.testing.assert_almost_equal(pk, p / c)
        
    def testRaytraceMesh(self):
        c = 0.5
        for (Nmesh, Nrt) in [(10,10), (20,4), (4, 20)]:
            mesh = ptum.regularsquaremesh(Nmesh, "BDY")
            x0 = np.vstack((np.linspace(-0.2,1.2,Nrt), np.ones(Nrt)*(-0.2))).transpose()
            p0 = np.vstack((np.zeros(Nrt), np.ones(Nrt))).transpose()
            slowness = lambda x: np.ones(len(x))/c
            gradslowness = lambda x: np.zeros_like(x)
            
            wfs, fidxs = prw.wavefront(x0, p0, slowness, gradslowness, 1/(c * Nrt), 1.4 / c, 1/(c * Nrt))
            phases = prw.nodesToPhases(wfs, fidxs, mesh, ["BDY"])
            self.assertEqual(len(phases), (Nmesh+1)**2)
            for vp in phases:
                self.assertGreaterEqual(len(vp), 1)
                for p in vp:
                    np.testing.assert_array_almost_equal(p, [0,1/c])
            
        
        
                       

class TestUtils(unittest.TestCase):
    def testGradient(self):
        N = 10
        f = lambda x: x[:,0]**2 + x[:,1]**2
        x = np.random.rand(N,2)
        g = prw.gradient(f, 1E-7)
        gx = g(x)
        self.assertEquals(gx.shape, (N,2))
        np.testing.assert_almost_equal(gx, x * 2)

class TestPointTest(unittest.TestCase):
    def testPointTest(self):
        N = 100 # test N quadrilaterals
        M = 1000 # M points
        # The 'wavefront' is [0,1] x [0,1], divided into vertical rectangles of width 1/N
        x0 = np.vstack((np.linspace(0,1,N+1), np.zeros(N+1))).transpose()
        x1 = x0 + [0,1.0]
        wpt = prw.WavefrontQuadsPointTest(x0, x1)
        p = np.random.rand(M,2)
        qs = put.print_timing(wpt.test)(p)
        np.testing.assert_array_equal(qs, np.int32(p[:,0]* N) == np.arange(N).reshape(N,1))
        
    def testPointUnique(self):
        N = 100 # test N quadrilaterals
        M = 1000 # M points
        x0 = np.vstack((np.linspace(0,1,N+1), np.zeros(N+1))).transpose()
        x1 = x0 + [0,1.0]
        wpt = prw.WavefrontQuadsPointTest(x0, x1)
        p = np.random.rand(M,2)*2
#        p = x0 + [0.01,0.99]
        ptoqs = put.print_timing(wpt.unique)(p)
        pquad = np.int32(p[:,0] * N)
        pquad[(pquad >= N) | (p[:,1] > 1)] = -1
        np.testing.assert_array_equal(ptoqs, pquad)
        
    def testInterpolation(self):
        N = 10
        M = 100
        x0 = np.vstack((np.linspace(0,1,N+1), np.zeros(N+1))).transpose()
        x1 = x0 + [0,1.0]
        for pdim in [1,2]:
            a,b,c = np.random.rand(3,pdim)
            f = lambda x: a + b * x[:,0].reshape(-1,1) + c*x[:,1].reshape(-1,1)
            p0 = f(x0)
            p1 = f(x1)
            wpt = prw.WavefrontInterpolate(x0,x1,p0,p1)
            v = np.random.rand(M,2) * 1.5
            vfound, phases = wpt.interpolate(v)
            np.testing.assert_array_equal(vfound, np.all(v < 1.0, axis=1))
            np.testing.assert_almost_equal(phases, f(v[vfound]))
    
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFillIn']
    unittest.main()