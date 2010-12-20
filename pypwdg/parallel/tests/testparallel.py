'''
Created on Nov 12, 2010

@author: joel
'''
import unittest
from pypwdg.parallel.mpiload import *
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.messaging as ppm
import numpy as np
import numpy.testing as nt
import cPickle
import pickle
import cStringIO

N = 200
if mpiloaded:    
    def testarrays():
        return [np.arange(n, dtype=complex) for n in range(N)] + [np.arange(n, dtype=float) for n in range(N)]        
    
    @ppd.parallel(None, None)
    def getArrays():
        return testarrays()
    
    # This comes last, so that the worker processes can see the relevant methods and classes
    import pypwdg.parallel.main
    if mpiloaded and comm.rank==0:
        
        class TestUnderMpi(unittest.TestCase):
            
            def testArray(self):
                las = getArrays()
                self.assertEqual(len(las), comm.size-1)
                tla = testarrays()
                for la in las:
                    for a, ta in zip(la, tla):
                            nt.assert_array_equal(a, ta)
        
        
    
        class Test(unittest.TestCase):    
            def testArrayHandler(self):
                h = ppm.ArrayHandler("Complex", np.complex, 10, None)
                    
                cso = cStringIO.StringIO()
                p = cPickle.Pickler(cso, protocol=2)
                p.persistent_id = h.process
                
                la = [np.arange(n, dtype=complex) for n in range(15)]            
                p.dump(la)
                self.assertEqual(len(h.shapes), 5)
                
                csi = cStringIO.StringIO(cso.getvalue())
                up = cPickle.Unpickler(csi)
                up.persistent_load = h.lookup
                la2 = up.load()
                self.assertEqual(len(la), len(la2))
                for a1,a2 in zip(la, la2):
                    nt.assert_array_equal(a1, a2)
                
    
    if __name__ == "__main__":
        unittest.main()
