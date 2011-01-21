'''
Created on Jan 21, 2011

@author: joel
'''
import unittest
from pypwdg.parallel.mpiload import *
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd

import numpy as np

N = 4

class MockDictInfo(object):
    def getOwnedKeys(self):
        return np.arange((comm.rank-1)*N,comm.rank*N)
    
    def getUnownedKeys(self):
        return np.arange(comm.rank*N,(comm.rank+1)*N) % (comm.size-1)

@ppd.parallel(None, None)        
def initialisedict(ddict, a):
    for k in np.arange((comm.rank-1)*N,comm.rank*N):
        ddict[k] = k+a
    
@ppd.parallel(None, None)        
def checkdict(ddict, a):
    keys = np.arange((comm.rank -1)*N, (comm.rank+1)*N) % (comm.size-1)
    data = [ddict[k] for k in keys]
    return (data,keys + a)    

import pypwdg.parallel.main

print ppd.parallelmethods

if mpiloaded and comm.rank == 0:
    class TestDistributedDict(unittest.TestCase):
        
        def testDict(self):
            info = MockDictInfo()
            ddm = ppdd.ddictmanager(info)
            ddict = ddm.getDict()
            a = 0
                                       
            
            initialisedict(ddict, a)
            ddict.sync()
            results = checkdict(ddict, a)
            for (a1, a2) in results:
                np.testing.assert_array_equal(a1,a2)


    if __name__ == "__main__":
        unittest.main()