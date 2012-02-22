'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np

@ppd.distribute
class SystemMultiply(object):
    
    def __init__(self, system, sysargs, syskwargs):
        S,G = system.getSystem(*sysargs, **syskwargs)
        self.M = S.tocsr()
        self.b = G.tocsr()

    @ppd.parallelmethod()
    def getRHS(self):
        return self.b.todense()
        
    @ppd.parallelmethod()
    def multiply(self, vector):
        return self.M * vector
    
class IndirectSolver(object):

    def __init__(self):
        pass

    def solve(self, system, sysargs, syskwargs):
        matvec = SystemMultiply(system, *sysargs, **syskwargs)
        
        