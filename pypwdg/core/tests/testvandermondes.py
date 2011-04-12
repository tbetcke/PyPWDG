'''
Created on Dec 19, 2010

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.utils.quadrature as puq
import pypwdg.core.vandermonde as pcv
import pypwdg.mesh.meshutils as pmm

import unittest
import test.utils.mesh as tum
        
import numpy
        
class TestVandermondes(unittest.TestCase):
    
    def testLocalVandermondes(self):
        # This test is more about exercising the code than testing any results.  We need a know simple mesh to do that
        dirs = numpy.array([[1,0]])
        k = 3
        pw = pcb.PlaneWaves(dirs, k)
        faceid = 0
        numquads = 3
        meshes = tum.examplemeshes2d()
        for mesh in meshes:
            mq = pmm.MeshQuadratures(mesh, puq.legendrequadrature(numquads))
            e2b = pcb.constructBasis(mesh, pcb.UniformBasisRule([pw]))
            LV = pcv.LocalVandermondes(mesh, e2b, mq)
            for faceid in range(mesh.nfaces):
                v = LV.getValues(faceid)
                d = LV.getDerivs(faceid)
                self.assertEqual(v.shape, (numquads, 1))
                self.assertEqual(d.shape, (numquads, 1))
                self.assertEqual(LV.numbases[faceid], 1)        
        