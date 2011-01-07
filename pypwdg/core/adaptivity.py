'''
Created on 1 Nov 2010

@author: joel
'''

import pypwdg.core.bases as pcb
import pypwdg.setup as ps

import numpy as np

def augmentparams(params, npw, dim):
    if params==None: params = np.zeros((0,dim))
    if len(params) < npw:
        params = np.vstack((params, pcb.circleDirections(len(params) - npw)))
    return params

def normalise(params, npw):
    p = params.reshape(npw, -1)
    return p / np.sqrt(np.sum(p**2, axis=1)).reshape(npw,1)        

class PWFBCreator(object):
    def __init__(self, k, origin, npw, nfb, params = None):        
        self.k = k
        self.origin = origin
        self.npw = npw
        self.nfb = nfb
        self.n = npw + nfb        
        self.params = augmentparams(params, npw, len(origin))
    
    def pwbasis(self, params):
        return pcb.PlaneWaves(normalise(params, self.npw), self.k)
    
    def newparams(self, params):
        return PWFBCreator(self.k,self.origin,self.npw,self.nfb,params)
     
    def getbasis(self):
        pwbasis = self.pwbasis(self.params)
        fbbasis = pcb.FourierBessel(self.origin, np.arange(0,self.nfb) - self.nfb/2)
        return pcb.BasisCombine([pwbasis, fbbasis])

    def nearbybases(self, x):
        nearby = [self]
        k,origin,npw,nfb, params = self.k,self.origin,self.npw,self.nfb, self.params
        nearby.append(PWFBCreator(k,origin,npw,nfb+1,params))
        if nfb > 3: nearby.append(PWFBCreator(k,origin,npw,nfb-1,params))
        nearby.append(PWFBCreator(k,origin,npw+1,nfb,params))
        if npw > 0: nearby.append(PWFBCreator(k,origin,npw,nfb,params[np.abs(x) != np.min(np.abs(x))]))
        return nearby

def origin(mesh, e):
    return np.average(mesh.nodes[mesh.elements[e]], axis = 0)

class AdaptiveElementToBasis(object):
    
    def __init__(self, mesh, k, npw, nfb, etopwfbc = None):
        if etopwfbc is None: 
            self.etopwfbc = dict([(e, PWFBCreator(k, origin(mesh, e), npw, nfb)) for e in range(mesh.nelements)])
        else: self.etopwfbc = etopwfbc
    
    def getValues(self, eid, points, normal=None):
        """ Return the values of the basis for element eid at points"""
        return self.etopwfbbc[eid].getbasis().values(points, normal)
    
    def getDerivs(self, eid, points, normal):
        return self.etopwfbbc[eid].getbasis().derivs(points, normal)

    def 
    
    
origin=np.average(self.mesh.nodes[self.mesh.elements[e]], axis = 0)
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, initialpw, initialfb):
        self.problem = problem
        self.basis = initialbasis
    
    
    def step(self):
        comp = ps.Computation(self.problem, self.elttobasis, False)
        solution = comp.solve()
        
        
        
        return solution
        