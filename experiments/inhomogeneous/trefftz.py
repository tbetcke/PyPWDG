'''
Created on Feb 14, 2012

@author: joel
'''
import pypwdg.utils.quadrature as puq
import pypwdg.core.bases.reference as pcbr
import pypwdg.utils.mappings as pum
import numpy as np

class ObjectiveInnerProducts(object):
    
    def __init__(self, volumequad, boundaryquad, u, v):
        self.vq = volumequad
        self.u = u
        self.v = v
        
    def uv(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        vvals = self.v.values(volp)
        return np.dot(vvals.H * volw,  uvals)
    
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
        return np.dot(lvals.H * volw, np.sum(udvals * dphivals, axis=2))
    
    def ugradlambdagradphi(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        dphivals = self.dphi(volp)
        lvals = self.l.values(volp)
        return np.dot(np.sum(lvals.H * dphivals, axis=2) * volw, uvals)
    
    def ulambda(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        lvals = self.l.values(volp)
        return np.dot(lvals.H, volw * uvals)
    
    def ulambdadphi2(self):
        volp, volw = self.vq
        uvals = self.u.values(volp)
        lvals = self.l.values(volp)
        dphivals = self.dphi(volp)
        dphi2 = np.sum(dphivals*dphivals, axis=1)
        return np.dot(lvals.H * volw * dphi2, uvals)
    
    def dudnlambda(self):
        bp, bw, bn = self.bq
        uvals = self.u.derivs(bp, bn)
        lvals = self.l.values(bp)
        return np.dot(lvals.H * bw, uvals)
    
    def ulambdadphidn(self):
        bp, bw, bn = self.bq
        uvals = self.u.values(bp)
        lvals = self.l.values(bp)
        dphidn = np.sum(self.dphi(bp) * bn, axis = 1)
        return np.dot(lvals.H * bw * dphidn, uvals)
    

c = 1.0
omega = 30    
k = omega / c

nq = 20
vp, vw = puq.trianglequadrature(nq)
bp, bw, bn = puq.triangleboundary(nq)

NP = 5
idmap = pum.Affine(np.array([0,0]), np.identity(2))
polys = pcbr.Reference(idmap, pcbr.Dubiner(NP), volume, boundary)  
        
        
        
        
        
    
    
        
