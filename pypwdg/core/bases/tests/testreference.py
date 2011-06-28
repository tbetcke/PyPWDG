'''
Created on Apr 4, 2011

@author: joel
'''
import unittest

import pypwdg.core.bases.reference as pcbr
import pypwdg.utils.geometry as pug
import pypwdg.core.bases.tests.testbases as pcbtt

import test.utils.mesh as tum

import numpy as np        

class TestReference(unittest.TestCase):
    
    def testReference(self):        
        mesh = tum.regularsquaremesh(2)
        structuredpoints = pug.StructuredPoints([[0.01,0.01],[0.99,0.99]], [20,30])
        rules = [pcbr.ReferenceBasisRule(pcbr.Dubiner(p)) for p in range(6)]
        pcbtt.basisDerivatives(rules, [mesh], structuredpoints, k=1.0)

    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()