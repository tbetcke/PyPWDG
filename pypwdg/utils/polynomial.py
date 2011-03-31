'''
Created on Mar 31, 2011

@author: joel
'''
import numpy as np

def jacobid(N,a,b,d,x):
    """ Return the dth derivative of the [d,N] shifted Jacobi(a,b) polynomials at x"""
    from scipy.special.orthogonal import poch
    v = getJacobi(a+d,b+d)(N-d, x)   
    fv = poch(np.arange(a+b+d+1, a+b+N+2), d)[:,np.newaxis] * v
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
        As things stand, things will probably fail at (0,1).    
    """
    def __init__(self,k, p):
        self.eta2 = p[:,1]
        self.eta1 = p[:,0] / (1-self.eta2)  
        self.k = k
        self.P1 = jacobid(k,0,0,0,self.eta1)
        self.P2 = [jacobid(k - i, 2*i +1, 0, 0, self.eta2) for i in range(k+1)] 
        
        
    def values(self):
        return np.hstack([(self.P1[:,i] * (1-self.eta2)**i).reshape(-1,1) * self.P2[i] for i in range(self.k+1)])
    
    def derivs(self):
        self.P1D = jacobid(self.k,0,0,1,self.eta1)
        self.P2D = [jacobid(self.k - i, 2*i +1, 0, 1, self.eta2) for i in range(self.k+1)]
        pass
