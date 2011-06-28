'''
Created on Jun 8, 2011

@author: joel
'''
import unittest
import pypwdg.core.bases.definitions as pcbd
import pypwdg.core.bases.utilities as pcbu
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.bases.reduced as pcbred
import pypwdg.setup.problem as psp
import pypwdg.utils.quadrature as puq
import pypwdg.core.evaluation as pce
import pypwdg.utils.geometry as pug

import numpy as np

import test.utils.mesh as tum

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
        self.assertEquals(rb.derivs(points).shape, (NP,1,2))

        self.assertEquals(rb.values(points[0]).shape, (1,1))
        
        N = 3
        m = np.random.random(n*N).reshape(N,n)
        rb = pcbd.BasisReduce(basis, m)
        self.assertEquals(rb.values(points).shape, (NP,N))
        self.assertEquals(rb.derivs(points, [1,0]).shape, (NP,N))
        self.assertEquals(rb.derivs(points).shape, (NP,N,2))

def testEquivalentBases(b1, b2, mesh, structuredpoints):
    ''' Test that 2 bases are equivalent (at a set of points) 
        
        Let V1, V2 be the vandermondes for the bases.  Then we want V1 = V2 M for some M.
        Taking (reduced) QR decompositions Vi = QiRi, we get M = R2^{-1}Q2^H Q1 R1
        Hence Q1 R1 = Q2 R2 R2^{-1}Q2^H Q1 R1 = Q2 Q2^H Q1 R1
        So the condition is that Q1 Q1^H = Q2 Q2 ^H
    '''    
    eval1 = pce.StructuredPointsEvaluator(mesh, b1, lambda x : x, np.identity(b1.getIndices()[-1]))
    eval2 = pce.StructuredPointsEvaluator(mesh, b2, lambda x : x, np.identity(b2.getIndices()[-1]))
    
    # Want to test
    v1 = eval1.evaluate(structuredpoints)[0]
    v2 = eval2.evaluate(structuredpoints)[0]
    q1 = np.linalg.qr(v1)[0]
    q2 = np.linalg.qr(v2)[0]
    np.testing.assert_almost_equal(np.dot(q1, q1.T.conjugate()), np.dot(q2, q2.T.conjugate()))
    
        
class TestSVN(unittest.TestCase):
    
    def testDubiner(self):
        # The Dubiner basis is L^2-orthogonal, so we can test that the SVN reduction doesn't do much to it
        # ... On reflection, this doesn't test very much.  But it at least exercises the code, so leaving it in
        k = 10
        N = 3
        bounds=np.array([[0.1,0.1],[0.9,0.9]],dtype='d')
        npoints=np.array([20,20])
        sp = pug.StructuredPoints(bounds, npoints)
        
        for n in range(1,4):
            mesh = tum.regularsquaremesh(n)
            dubrule = pcbr.ReferenceBasisRule(pcbr.Dubiner(N))
            problem = psp.Problem(mesh, k, {})
            dubbasis = psp.constructBasis(problem, dubrule)     
            refquad = puq.trianglequadrature(N+1)
            svnrule = pcbred.SVDBasisReduceRule(refquad, dubrule)
            svnbasis = psp.constructBasis(problem, svnrule)
            # The bases should be equivalent:
            testEquivalentBases(dubbasis, svnbasis, mesh, sp)
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBasisReduce']
    unittest.main()