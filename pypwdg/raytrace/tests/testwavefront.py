'''
Created on Oct 25, 2011

@author: joel
'''
import unittest
import numpy as np
import pypwdg.raytrace.wavefront as prw

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
        
                       

class TestUtils(unittest.TestCase):
    def testGradient(self):
        N = 10
        f = lambda x: x[:,0]**2 + x[:,1]**2
        x = np.random.rand(N,2)
        g = prw.gradient(f, 1E-7)
        gx = g(x)
        self.assertEquals(gx.shape, (N,2))
        np.testing.assert_almost_equal(gx, x * 2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testFillIn']
    unittest.main()