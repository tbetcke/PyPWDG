'''
Created on Jan 21, 2011

@author: joel
'''
import unittest
from pypwdg.parallel.mpiload import *

if mpiloaded and comm.rank == 0:
    class TestDistributedDict(unittest.TestCase):
    
    
        def testName(self):
            pass


    if __name__ == "__main__":
        unittest.main()