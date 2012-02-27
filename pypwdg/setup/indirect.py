'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl

@ppd.distribute()
class SystemMultiply(object):
    
    def __init__(self, system, sysargs, syskwargs):
        S,G = system.getSystem(*sysargs, **syskwargs)
        self.M = S.tocsr()
        self.b = G.tocsr()
        self.dtype = self.M.dtype

    @ppd.parallelmethod()
    def getRHS(self):
        return self.b.todense()
        
    @ppd.parallelmethod()
    def multiply(self, x):
        print "multiply"
#        print x
#        print x.shape
        y = self.M * x
        print y
#        print y.shape
        return y 
       
    
class IndirectSolver(object):

    def __init__(self):
        pass

    def solve(self, system, sysargs, syskwargs):
        sm = SystemMultiply(system, sysargs, syskwargs)
        b = sm.getRHS()        
        n = len(b)
        lo = ssl.LinearOperator((n,n), sm.multiply, dtype=sm.dtype)
        return ssl.bicgstab(lo, b)
        
        
        