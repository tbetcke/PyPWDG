'''
Created on Oct 24, 2011

@author: joel
'''
import numpy as np
from collections import namedtuple

def norm(a):
    print a.shape
    return np.sqrt(np.sum(a**2,axis=1))

def onestep(x, p, slowness, gradslowness, deltat):
    s = slowness(x)[:,np.newaxis]
    pnorm = norm(p)[:,np.newaxis]
    pp = p * s / pnorm    
    gs = gradslowness(x)        
    xk = x + deltat * pp / s**2
    pk = pp + deltat * gs / s
    
    return xk,pk

def ninterp(x,p,tol):
    dxt = np.int32(norm(x[1:] - x[:-1]) / tol)
    dpt = np.int32(norm(p[1:] - p[:-1]) / tol)
    return np.max((dxt,dpt),axis=0)
    
def fillin(x,p,tol):
    ni = ninterp(x,p,tol)
    cs = np.cumsum(ni)
    nni = cs[-1]
    dim = x.shape[1]
    if nni > 0:
        n = nni + len(x)
        xi = np.empty((n,dim))
        pi = np.empty((n,dim))
        for k, (i, c, xk,xk1,pk,pk1) in enumerate(zip(ni,cs, x[:-1],x[1:],p[:-1],p[1:])):
            w = np.linspace(0,1,i+1,False)[:,np.newaxis]
            
            xi[k+c-i:k+1 + c] = xk * (1-w) + xk1 * w
            pi[k+c-i:k+1 + c] = pk * (1-w) + pk1 * w
        xi[-1] = x[-1]
        pi[-1] = p[-1]
        forwardidx = np.arange(len(x))
        forwardidx[1:]+=cs
        return xi,pi, forwardidx
    else: return x,p,None

def wavefront(x0,p0,slowness,gradslowness,deltat,n,tol):
    x,p = x0,p0
    wavefronts = []
    forwardidxs = []
    for _ in range(n):
        xi,pi,fwdidx = fillin(x,p,tol)
        wavefronts.append((x,p))
        forwardidxs.append(fwdidx)
        x,p = onestep(xi,pi,slowness,gradslowness,deltat)
    return wavefront, forwardidxs
    

        
        
    