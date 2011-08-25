'''
Created on Apr 4, 2011

@author: joel
'''
import unittest

import pypwdg.core.bases.reference as pcbr
import pypwdg.utils.geometry as pug
import pypwdg.core.bases.tests.testbases as pcbtt

import pypwdg.test.utils.mesh as tum

import numpy as np        

class TestReference(unittest.TestCase):
    
    def testReference(self):        
        meshes = [tum.regularsquaremesh(2)]
        meshes = tum.examplemeshes2d()
        structuredpoints = pug.StructuredPoints([[0.01,0.01],[0.99,0.99]], [20,30])
        rules = [pcbr.ReferenceBasisRule(pcbr.Dubiner(p)) for p in range(3)]
        pcbtt.basisDerivatives(rules, meshes, structuredpoints, k=1.0)

    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()