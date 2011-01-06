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
        

class AdaptivePWBasisCreator(object):
    def __init__(self, k, origin, npw, nfb, params = None):
        self.k = k
        self.origin = origin
        self.npw = npw
        self.nfb = nfb
        self.params = augmentparams(params, npw, len(origin))
        self.n = npw + nfb
    
    def normalise(self, params):
        p = params.reshape(self.npw, -1)
        return p / np.sqrt(np.sum(p**2, axis=1)).reshape(self.npw,1)        
        
    def testbasis(self, params):
        return pcb.PlaneWaves(self.normalise(params), self.k)
        
    def fullbasis(self, params):
        pwbasis = self.testbasis(params)
        fbbasis = pcb.FourierBessel(self.origin, np.arange(0,self.nfb) - self.nfb/2)
        return pcb.BasisCombine([pwbasis, fbbasis])
     
    def finalise(self, params):
        return AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb, self.normalise(params))
    
    def nearbybases(self, x):
        nearby = [self]
        nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb+1, self.params))
        if self.nfb > 3: nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb-1, self.params))
        nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw+1, self.nfb, self.params))
        if self.npw > 0: nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw-1, self.nfb, self.params[np.abs(x) != np.min(np.abs(x))]))
        return nearby

def origin(mesh, e):
    return np.average(mesh.nodes[mesh.elements[e]], axis = 0)

        
@distribute
class AdaptivePWBasisManager(object):
    
    def __init__(self, mesh, k, npw, nfb):
        etopwbc = dict([(e, AdaptivePWBasisCreator(k, origin(mesh, e), npw, nfb, None)) for e in mesh.partition])
    
    def buildEtoB(self):
    
    
origin=np.average(self.mesh.nodes[self.mesh.elements[e]], axis = 0)
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, initialpw, initialfb):
        self.problem = problem
        self.basis = initialbasis
    
    
    def step(self):
        comp = ps.Computation(self.problem, self.elttobasis, False)
        solution = comp.solve()
        
        
        
        return solution
        