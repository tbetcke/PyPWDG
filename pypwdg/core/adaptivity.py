'''
Created on 1 Nov 2010

@author: joel
'''

import pypwdg.core.bases as pcb
import pypwdg.setup as ps

class AdaptivePlaneWaves(object):
    def __init__(self, mesh, initialpw, initialfb):
        self.initialpw = initialpw
        self.initialfb = initialfb
        
    def initialparams(self, e):
        return (pcb.circleDirections(self.initialpw), self.initialfb)
    
    def getbasis(self, params):
        
    

class AdaptiveComputation(object):
    
    def __init__(self, problem, initialbasis, ):
        self.problem = problem
        self.basis = initialbasis
    
    
    def step(self):
        comp = ps.Computation(self.problem, self.elttobasis, False)
        solution = comp.solve()
        
        
        
        return solution
        