'''
Classes used to implement finite elements based on a reference element

@author: joel
'''

import pypwdg.core.bases.definitions as pcbd
import pypwdg.utils.polynomial as pup
import pypwdg.utils.quadrature as puq

import numpy as np
 
class ReferenceBasisRule(object):
    """ A basis that uses a (affine) transformation to a reference element (old skool)"""
    def __init__(self, reference):
        self.reference = reference
        
    def populate(self, e):
        return [Reference(e.refmap, self.reference, e.volume, e.boundary)]

        
def laplacian(values, derivs, volume, boundary):
    ''' Calculate the laplacian matrix for a Polynomial basis.
    
        Returns L where \nabla^2 u = u . L for any vandermonde matrix u        
    '''
    vx, vw = volume
    bx, bw, bn = boundary
    vv = values(vx) 
    vd = derivs(vx) 
    mass = np.tensordot(vv * vw, vv, (0,0))
    bv = values(bx)
    bdn = derivs(bx, bn)
    
    stiffness = -np.tensordot(vd * vw[...,np.newaxis], vd, ((0,2),(0,2))) + np.tensordot(bv * bw[...,np.newaxis], bdn, (0,0))
        
    return np.linalg.solve(mass, stiffness)

class Reference(pcbd.Basis):
    
    def __init__(self, map, reference, volume, boundary):
        self.mapi = map.inverse
        self.reference = reference
        self.n = reference.n
        self.L = laplacian(self.values, self.derivs, volume(reference.volume), boundary(reference.face))
                
    def values(self, x):
        return self.reference.values(self.mapi.apply(x))
    
    def derivs(self, x, n = None):   
        derivs = np.dot(self.reference.derivs(self.mapi.apply(x)),self.mapi.linear.transpose())
        return derivs if n is None else np.sum(derivs.transpose([0,2,1]) * n[..., np.newaxis], axis = 1)  
    
    def laplacian(self, x):
        return np.dot(self.values(x), self.L)
    
class Dubiner(object):
    """ At some point, this should probably try to (pre-)cache results.  """
    
    def __init__(self, k):
        self.k = k
        self.n = ((k+1) * (k+2)) / 2
        self.volume = puq.trianglequadrature(k+1)
        self.face = puq.legendrequadrature(k+1)
        
    def values(self, x):        
        return pup.DubinerTriangle(self.k, x).values()
    
    def derivs(self, x):
        return pup.DubinerTriangle(self.k, x).derivs().transpose([1,2,0])
    
    
        
    
class Legendre1D(object):
    """ 1D basis from Legendre Polynomials"""
    
    def __init__(self,n):
        self.n = n+1
        self.volume = puq.legendrequadrature(n+1)
        self.face = puq.pointquadrature()
        
    def values(self, x):
        return pup.jacobidnorm(self.n-1,0,0,0,x.ravel())
    
    def derivs(self, x):
        vals=pup.jacobidnorm(self.n-1,0,0,1,x.ravel())
        return vals[...,np.newaxis]
    
    def laplacian(self, x):
        return pup.jacobidnorm(self.n-1, 0,0,2, x.ravel())

    
    