'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl
import pypwdg.mesh.structure as pms


#@ppd.distribute
class SystemMultiply(object):
    
    def __init__(self, system, mesh, sysargs, syskwargs):
        S,G = system.getSystem(*sysargs, **syskwargs)
        
        self.M = S.tocsr()
        self.b = G.tocsr()
        self.dtype = self.M.dtype
        
        self.PLU = None
        if mesh is not None:
            self.localidxs = S.subrows(mesh.partition) 
            P = self.M[self.localidxs][:,self.localidxs]
            self.PLU = ssl.dsolve.spilu(P)

#    @ppd.parallelmethod()
    def getRHS(self):
        return self.b.todense()

    @ppd.parallelmethod()
    def precond(self, x):
        print "precond"
        if self.PLU is not None:
            PIx = np.zeros_like(x)
            PIx[self.localidxs] = self.PLU.solve(x[self.localidxs])
            return PIx
        else:  
            return x
         
#    @ppd.parallelmethod()
    def multiply(self, x):
        print "multiply"
#        print x
#        print x.shape
        y = self.M * x
#        print y
#        print y.shape
        return y 
       
    
class IndirectSolver(object):

    def __init__(self, mesh = None):
        self.mesh = mesh

    def solve(self, system, sysargs, syskwargs):
        sm = SystemMultiply(system, self.mesh, sysargs, syskwargs)
        b = sm.getRHS()        
        n = len(b)
        lo = ssl.LinearOperator((n,n), sm.multiply, dtype=sm.dtype)
        pc = ssl.LinearOperator((n,n), sm.precond, dtype=sm.dtype)
        return ssl.gmres(lo, b, M=pc)
        
        
        