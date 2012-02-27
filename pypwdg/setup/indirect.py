'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl
import pypwdg.mesh.structure as pms


@ppd.distribute()
class SystemMultiply(object):
    
    def __init__(self, system, sysargs, syskwargs, mesh=None):
        S,G = system.getSystem(*sysargs, **syskwargs)
        
        self.M = S.tocsr()
        self.b = G.tocsr()
        self.dtype = self.M.dtype
        
        self.PLU = None
        if mesh is not None:
            self.localidxs = S.subrows(mesh.partition)
            print self.localidxs 
            P = self.M[self.localidxs][:,self.localidxs]
            self.PLU = ssl.dsolve.splu(P)

    @ppd.parallelmethod()
    def getRHS(self):
        return self.b.todense()

    @ppd.parallelmethod()
    def precond(self, x):

        if self.PLU is not None:
#            print "precond"
            PIx = np.zeros_like(x)
            PIx[self.localidxs] = self.PLU.solve(x[self.localidxs])
            return PIx
        else:  
            return x
         
    @ppd.parallelmethod()
    def multiply(self, x):
#        print "multiply"
#        print x
#        print x.shape
        y = self.M * x
#       print y
#        print y.shape
        return y 
       
    
class IndirectSolver(object):

    def __init__(self, dtype, mesh = None):
        self.mesh = mesh
        self.dtype = dtype

    def solve(self, system, sysargs, syskwargs):
        sm = SystemMultiply(system, sysargs, syskwargs, self.mesh)
        b = sm.getRHS()        
        n = len(b)
        lo = ssl.LinearOperator((n,n), sm.multiply, dtype=self.dtype)
        pc = ssl.LinearOperator((n,n), sm.precond, dtype=self.dtype)
        self.n = 0
        def counter(x):
            if self.n % 20 == 0: print self.n
            self.n = self.n+1
            
        x, status = ssl.gmres(lo, b, callback = counter, M=pc, restart=200)
        print status
        return x
        
        
        