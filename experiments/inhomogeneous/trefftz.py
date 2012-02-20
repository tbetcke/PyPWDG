'''
Created on Feb 14, 2012

@author: joel
'''
import pypwdg.utils.quadrature as puq
import pypwdg.core.bases.reference as pcbr
import pypwdg.utils.mappings as pum
import numpy as np

class ObjectiveInnerProducts(object):
    
    def __init__(self, volumequad, u, v):
        self.vq = volumequad
        self.u = u
        self.v = v
        
    def uv(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        vvals = self.v.values(volp)
        return np.dot(vvals.T.conj(),  uvals * volw)
    
    def v1(self):
        volp, volw = self.vq
        vvals = self.v.values(volp).conj()
        return np.sum(vvals * volw, axis=0).reshape(-1,1)

class ConstraintInnerProducts(object):
    def __init__(self, volumequad, boundaryquad, u, l, dphi):
        self.vq = volumequad
        self.bq = boundaryquad
        self.u = u
        self.l = l
        self.dphi = dphi
    
    def gradugradlambda(self):
        volp, volw = self.vq
        uvals = self.u.derivs(volp)
        lvals = self.l.derivs(volp).conj()
        return np.tensordot(uvals * volw.reshape(-1,1,1), lvals, ((1,2),(1,2)))
    
    def gradugradphilambda(self):
        volp, volw = self.vq
        udvals = self.u.derivs(volp)
        dphivals = self.dphi(volp)
        lvals = self.l.values(volp)
        return np.dot(lvals.T.conj(), np.sum(udvals * dphivals, axis=2) * volw)
    
    def ugradlambdagradphi(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        dphivals = self.dphi(volp)
        lvals = self.l.derivs(volp)
        return np.dot(np.sum(lvals.conj() * dphivals, axis=2).T , uvals* volw)
    
    def ulambda(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        lvals = self.l.values(volp)
        return np.dot(lvals.T.conj(), volw * uvals)
    
    def ulambdadphi2(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        lvals = self.l.values(volp)
        dphivals = self.dphi(volp)
        dphi2 = np.sum(dphivals*dphivals, axis=1)
        return np.dot(lvals.T.conj()  * dphi2, uvals* volw)
    
    def dudnlambda(self):
        bp, bw, bn = self.bq
        uvals = self.u.derivs(bp, bn)
        lvals = self.l.values(bp)
        return np.dot(lvals.T.conj() , uvals* bw)
    
    def ulambdadphidn(self):
        bp, bw, bn = self.bq
        uvals = self.u.values(bp)
        lvals = self.l.values(bp)
        dphidn = np.sum(self.dphi(bp) * bn, axis = 1)
        return np.dot(lvals.T.conj()  * dphidn, uvals* bw)
    

c = 1.0
omega = 30    
k = omega / c
dphi = lambda p: np.array([[1,0]])

nq = 20
vq = puq.trianglequadrature(nq)
bq = puq.triangleboundary(nq)

NP = 5
idmap = pum.Affine(np.array([0,0]), np.identity(2))
polys = pcbr.Reference(idmap, pcbr.Dubiner(NP))

oip = ObjectiveInnerProducts(vq, polys, polys)
print oip.uv()
print oip.v1()  
cip = ConstraintInnerProducts(vq, bq, polys, polys, dphi)
print cip.gradugradlambda()
print cip.gradugradphilambda()
print cip.ugradlambdagradphi()
print cip.ulambda()
print cip.ulambdadphi2()

print cip.dudnlambda()
print cip.ulambdadphidn()

Ar = oip.uv()
A = np.bmat([[Ar, 0], [0, -Ar]]) 
        
        
    
    
        
