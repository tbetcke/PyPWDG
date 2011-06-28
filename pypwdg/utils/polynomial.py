'''
Generation of polynomials.  Stolen from pypyramid.

Created on Mar 31, 2011

@author: joel
'''
import numpy as np
import scipy.special.orthogonal as sso 
import pypwdg.utils.quadrature as puq

def jacobidnorm(N,a,b,d,x):
    """ Return the dth derivative of the [d,N] shifted Jacobi(a,b) polynomials at x
    
        The polynomials are normalised to have weighted L^2 norm 1. (although, obviously
        not the derivatives)
    """
    n = np.arange(N+1, dtype=float)
    norms = np.sqrt((1 /(2*n+a+b+1)) * (sso.poch(n + 1, a) / sso.poch(n + b + 1, a)))
    return jacobid(N,a,b,d,x) / norms

def jacobid(N,a,b,d,x):
    """ Return the dth derivative of the [d,N] shifted Jacobi(a,b) polynomials at x  """
    v = getJacobi(a+d,b+d)(N-d, x)   
    fv = sso.poch(np.arange(a+b+d+1, a+b+N+2), d)[:,np.newaxis] * v

    return np.vstack((np.zeros((d, len(x))), fv )).T 

jacobicache = {}
def getJacobi(a,b):
    """ Caches the Jacobi polynomial objects"""
    j = jacobicache.get((a,b))
    if j is None:
        j = Jacobi(a,b)
        jacobicache[(a,b)] = j
    return j


class Jacobi(object):
    """ Calculate shifted Jacobi polynomials"""
    maxN = 200
    def __init__(self, a,b):
        n = np.arange(2,self.maxN + 1, dtype=float)
        self.A = np.zeros(self.maxN + 1)
        self.B = np.zeros(self.maxN + 1)
        self.C = np.zeros(self.maxN + 1)
        self.A[1] = 2.0/(a+b+2)
        self.B[1] = -(a-b)/(a+b+2.0)
        self.C[1] = 0
        self.A[2:] = 2*n*(n+a+b)/((2*n+a+b)*(2*n+a+b-1))
        self.B[2:] =-(a*a-b*b)/((2*n+a+b)*(2*n+a+b-2))
        self.C[2:] =2*(n+a-1)*(n+b-1)/((2*n+a+b-1)*(2*n+a+b-2))
    
    def __call__(self,N,x):
        v = [np.zeros(len(x))]
        x = 2*x - 1
        if N >= 0:
            v.append(np.ones(len(x)))
            for n in range(1, N+1):
                v.append((v[-1] * (x-self.B[n]) - self.C[n]*v[-2])/self.A[n])
        return np.vstack(v)[1:,:]

class DubinerTriangle(object):
    """ Represents the evaluation of a Dubiner basis of order k on a triangle at a set of points, p
        The reference triangle has vertices at (0,0), (0,1) and (1,0).
        As things stand, evaluation will probably fail at (0,1).    
    """
    def __init__(self,k, p):
        x = p[:,0]
        y = p[:,1]
        y[y==1]=0.999999 # Okay Timo.  I'm ready. fire me now.  
        self.eta2 = y
        self.eta1 = x / (1-y)  
        self.k = k
        self.P1 = jacobidnorm(k,0,0,0,self.eta1)
        self.P2 = [jacobidnorm(k - i, 2*i +1, 0, 0, self.eta2) for i in range(k+1)]
        n = np.arange(k+1)[np.newaxis,:]
        w = (1-self.eta2[:,np.newaxis])**n
        self.P1w = self.P1 * w 
        wD = np.hstack((np.zeros((len(p),1)), - w[:,:-1] * n[:,1:])) 
        self.P1wD1 = jacobidnorm(self.k,0,0,1,self.eta1) * w
        self.P1wD2 = self.P1 *wD 
        self.P2D2 = [jacobidnorm(self.k - i, 2*i +1, 0, 1, self.eta2) for i in range(self.k+1)]
        self.J = np.array([[1/(1-y), np.zeros(len(p))],[x/(1-y)**2, np.ones(len(p))]])
        
    def values(self):
        return np.hstack([self.P1w[:,[i]]  * self.P2[i] for i in range(self.k+1)])
    
    def derivs(self):
        Deta1 = np.hstack([self.P1wD1[:,[i]]  * self.P2[i] for i in range(self.k+1)])
        Deta2 = np.hstack([self.P1w[:,[i]]  * self.P2D2[i] + self.P1wD2[:,[i]] * self.P2[i] for i in range(self.k+1)])
        etagrad = np.array([Deta1, Deta2])
        return np.sum(self.J[...,np.newaxis] * etagrad[np.newaxis,...], axis=1)
        
def laplacian(k):
    ''' Calculate the laplacian matrix for the Dubiner basis.
    
        Returns L where \nabla^2 u = u . L for any vandermonde matrix u
        
        N.B. would be easy to generalise this to any polynomial basis (indeed, the calculation
        of the mass matrix is currently superfluous as the Dubiner basis is orthonomal)
        
        THIS IS NOT USED - see p.c.b.reference.py for the alternative implemenation
    '''
    tp, tw = puq.trianglequadrature(k+1)
    dt = DubinerTriangle(k,tp)
    vtp = dt.values() 
    dtp = dt.derivs() 
    mass = np.tensordot(vtp * tw, vtp, (0,0))
    stiffness = -np.tensordot(dtp * tw, dtp, ((0,1),(0,1)))
    ex, ew = puq.legendrequadrature(k+1)
    # Now do the boundary integrals
    ew = ew.reshape(-1,1)
    ep1 = np.hstack((ex, np.zeros_like(ex)))
    ep2 = np.hstack((np.zeros_like(ex),ex))
    ep3 = np.hstack((ex,1-ex))
    for p, n in ((ep1,[0,-1]),(ep2,[-1,0]), (ep3, [1,1])):
        dt = DubinerTriangle(k, p)
        vn = np.tensordot(dt.derivs(), np.array(n), (0,0)) 
        stiffness+=np.tensordot(dt.values() * ew, vn, (0,0))
        
    return np.linalg.solve(mass, stiffness)
    
          
