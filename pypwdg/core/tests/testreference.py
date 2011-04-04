'''
Created on Apr 4, 2011

@author: joel
'''
import unittest
#import pypwdg.core.reference as pcr
#import pypwdg.core.bases as pcb
#import pypwdg.utils.quadrature as puq
#import pypwdg.mesh.meshutils as pmmu
#import pypwdg.core.vandermonde as pcv
#import pypwdg.utils.sparse as pus
#import pypwdg.core.assembly as pca
#import pypwdg.mesh.structure as pms
#import test.utils.mesh as tum
#import pypwdg.core.evaluation as pce
#import pypwdg.utils.geometry as pug
#import scipy.sparse.linalg as ssl
#import scipy.sparse as ss

import pypwdg.utils.mappings as pum
import pypwdg.core.reference as pcr

import numpy.random as nr
import numpy as np
import numpy.linalg as nl
import math

class SquareBubble(object):
    
    def values(self, p):
        return -1.0/4
        

class TestReference(unittest.TestCase):
    
    def testReference(self):
        # Tests that we get the correct directional derivatives on the reference element
        N = 5
        h = 1E-6
        for k in range(4):
            ref = pcr.Dubiner(k)
            for _ in range(10):                
                offset = nr.random(2)
                linear = nr.random((2,2))
#                if nl.cond(linear) < 200:
                map = pum.Affine(offset, linear)
                b = pcr.Reference(map, ref)
                # generate some random points and normals inside the element
                p = map.apply(nr.random((N, 2))/2)
                nn = nr.random((N,2))
                n = nn / np.sqrt(np.sum(nn**2, axis=1)).reshape(-1,1)
                vhd = (b.values(p + n * h) - b.values(p - n * h)) / (2*h)
                vd = b.derivs(p, n)
                
                scale = np.max(vd, axis=1).reshape(-1,1)
                scale[scale==0.0] = 1.0
                np.testing.assert_array_almost_equal(vhd / scale, vd / scale, decimal = 4)
            
            
                
    
    
#    def testBubble(self):
#        f = SquareBubble()
#        k = 3
#        nq = 6
#        sigma = 1E-6
#
#        mesh = tum.regularsquaremesh(3)        
#        I = ss.eye(mesh.nelements, mesh.nelements)        
#        mqs = pmmu.MeshQuadratures(mesh, puq.legendrequadrature(nq))
#        emqs = pmmu.MeshElementQuadratures(mesh, puq.trianglequadrature(nq))
#        bounds = np.array([[0,0],[1,1]])
#
#        etob = pcb.constructBasis(mesh, pcr.ReferenceBases(pcr.Dubiner(k)))
#        ev = pcv.ElementVandermondes(mesh, etob, emqs)
#        H1 = pcv.LocalInnerProducts(ev.getDerivs, ev.getDerivs, emqs.quadweights, ((0,2),(0,2)))
#        H1P = pus.createvbsr(I, H1.product, etob.getSizes(), etob.getSizes())
#        
#        lv = pcv.LocalVandermondes(mesh, etob, mqs)
#        faceassembly = pca.Assembly(lv, lv, mqs.quadweights) 
#        AJ = pms.AveragesAndJumps(mesh)    
#        SI = faceassembly.assemble(np.array([[sigma * AJ.JD.T * AJ.JD,   AJ.AN.T * AJ.JD], 
#                                             [AJ.JD.T * AJ.AN,AJ.Z]]))
#        M = pms.sumfaces(mesh,H1P - SI)
#        
#        fetob = pcb.constructBasis(mesh, pcb.UniformBases([f]))
#        fv = pcv.ElementVandermondes(mesh, fetob, mqs)
#        fL2 = pcv.LocalInnerProducts(ev.getDerivs, fv, emqs.quadweights)
#        fP = pus.createvbsr(I, fL2.product, etob.getSizes(), fetob.getSizes())
#        F = pms.sumrhs(mesh, fP)
#        
#        u = ssl.linsolve(M, F)
#        p = pug.StructuredPoints(bounds, 10)
#        uestimated = pce.StructuredPointsEvaluator(mesh, etob, lambda x:x, u)
#        x,y = p.getPoints(bounds).T
#        uactual = x*(1-x) * y*(1-y)
  #      print uestimated
  #      print uactual



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()