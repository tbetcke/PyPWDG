'''
Created on Apr 4, 2011

@author: joel
'''
import unittest

import pypwdg.utils.mappings as pum
import pypwdg.core.bases.reference as pcbr

import numpy.random as nr
import numpy as np        

class TestReference(unittest.TestCase):
    
    def testReference(self):
        # Tests that we get the correct directional derivatives on the reference element
        N = 5
        h = 1E-6
        for k in range(6):  # we'll go up to order 6
            ref = pcbr.Dubiner(k) # and use Dubiner bases (because all we have right now)
            for _ in range(10):  # 10 random elements              
                offset = nr.random(2)
                linear = nr.random((2,2))
#                if nl.cond(linear) < 200:
                map = pum.Affine(offset, linear)
                b = pcbr.Reference(map, ref)
                # generate some random points and normals inside the element
                p = map.apply(nr.random((N, 2))/2)
                nn = nr.random((N,2))
                n = nn / np.sqrt(np.sum(nn**2, axis=1)).reshape(-1,1)
                vhd = (b.values(p + n * h) - b.values(p - n * h)) / (2*h) # compute the derivatives in mesh coordinates using finite differences
                vd = b.derivs(p, n) # now ask the basis to compute the differences
                
                scale = np.max(vd, axis=1).reshape(-1,1)
                scale[scale==0.0] = 1.0
                np.testing.assert_array_almost_equal(vhd / scale, vd / scale, decimal = 4)
            
            
                
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()