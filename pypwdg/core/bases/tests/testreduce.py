'''
Created on Jun 8, 2011

@author: joel
'''
import unittest
import pypwdg.core.bases.definitions as pcbd
import pypwdg.core.bases.utilities as pcbu
import numpy as np

class Test(unittest.TestCase):

    def testBasisReduce(self):
        k = 1
        n = 10
        NP = 30
        points = np.random.random(NP * 2).reshape(-1,2)
        basis = pcbd.PlaneWaves(pcbu.circleDirections(n), k)
        x = np.random.random(n)
        rb = pcbd.BasisReduce(basis, x)
        self.assertEquals(rb.values(points).shape, (NP,1))
        self.assertEquals(rb.derivs(points, [1,0]).shape, (NP,1))
        self.assertEquals(rb.derivs(points).shape, (NP,2,1))

        self.assertEquals(rb.values(points[0]).shape, (1,1))
        
        N = 3
        m = np.random.random(n*N).reshape(N,n)
        rb = pcbd.BasisReduce(basis, m)
        self.assertEquals(rb.values(points).shape, (NP,N))
        self.assertEquals(rb.derivs(points, [1,0]).shape, (NP,N))
        self.assertEquals(rb.derivs(points).shape, (NP,2,N))
        
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBasisReduce']
    unittest.main()