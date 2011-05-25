'''
Created on Apr 18, 2011

@author: joel
'''
import unittest

import pypwdg.utils.quadrature as puq
import pypwdg.core.bases as pcb
import pypwdg.adaptivity.planewave as prp
import test.utils.mesh as tum
import pypwdg.mesh.meshutils as pmmu
import pypwdg.core.boundary_data as pcbd
import numpy as np

import math

class Test(unittest.TestCase):


    def testL2Prod(self):
        N = 20
        k = 10
        qxw = puq.squarequadrature(N)
        D = 1
        g = pcb.PlaneWaves(pcb.circleDirections(40)[15], k)
        t1 = prp.findpw(prp.L2Prod(g.values, qxw, k), D, maxtheta = 1)
        self.assertAlmostEqual(t1, (2 * math.pi * 15) / 40)
        
        
    def testEdge(self):
        N = 10
        k = 10
        D = 1
        x,w = puq.legendrequadrature(3*N)
        x = np.hstack((x, np.zeros_like(x)))
        
        g = pcb.PlaneWaves(pcb.circleDirections(40)[15], k)
        bc = pcbd.generic_boundary_data([1j*k,1],[1j*k,1],g)
#        t1 = prp.findpw(prp.ImpedanceProd(bc, (x,w), [0,1], k), D, maxtheta = 1)

        t1 = prp.findpw(prp.L2Prod(bc.values, (x,w), k), D, maxtheta = 1)
        self.assertAlmostEqual(t1, (2 * math.pi * 15) / 40)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testL2Prod']
    unittest.main()