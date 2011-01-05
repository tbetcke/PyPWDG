'''
Created on 1 Nov 2010

@author: joel
'''

import pypwdg.core.bases as pcb
import pypwdg.setup as ps

import numpy as np




class AdaptivePWBasisCreator(object):
    def __init__(self, k, origin, npw, nfb, params = None):
        self.k = k
        self.origin = origin
        self.npw = npw
        self.nfb = nfb
        self.params =  pcb.circleDirections(self.npw) if params is None else params
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
        return AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb, params)
    
    def nearbybases(self, x):
        nearby = [self]
        nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb+1, self.params))
        if self.nfb > 3: nearby.append(AdaptivePWBasisCreator(self.k, self.origin, self.npw, self.nfb-1, self.params))
         
        
        

origin=np.average(self.mesh.nodes[self.mesh.elements[e]], axis = 0)
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, initialbasis, ):
        self.problem = problem
        self.basis = initialbasis
    
    
    def step(self):
        comp = ps.Computation(self.problem, self.elttobasis, False)
        solution = comp.solve()
        
        
        
        return solution
        