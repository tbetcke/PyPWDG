'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl
import pypwdg.setup.computation as psc

np.set_printoptions(precision=4, threshold='nan',  linewidth=1000)

@ppd.distribute()
class DefaultOperator(object):

    
    @ppd.parallelmethod()    
    def setup(self, system, sysargs, syskwargs):   
        self.S,G = system.getSystem(*sysargs, **syskwargs) 
        self.M = self.S.tocsr()
        self.b = G.tocsr()
        self.dtype = self.M.dtype
        

    @ppd.parallelmethod()
    def rhs(self):
        return self.b.todense()
         
    @ppd.parallelmethod()
    def multiply(self, x):
        y = self.M * x
        return y 

@ppd.distribute()
class BlockPrecondOperator(DefaultOperator):       
    def __init__(self, mesh):
        self.mesh = mesh
    
    @ppd.parallelmethod()
    def setup(self, *args):
        super(BlockPrecondOperator, self).setup(*args)
        self.localidxs = self.S.subrows(self.mesh.partition)
#        print self.localidxs 
        P = self.M[self.localidxs][:,self.localidxs]
        self.PLU = ssl.dsolve.splu(P)        
        
    @ppd.parallelmethod()
    def precond(self, x):
        PIx = np.zeros_like(x)
        PIx[self.localidxs] = self.PLU.solve(x[self.localidxs])
        return PIx


class BrutalSolver(object):
    def __init__(self, dtype):
        self.dtype = dtype
    
    def solve(self, operator):
        b = operator.rhs()
        n = len(b)
        M = np.hstack([operator.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
        print M.shape, b.shape
#        print "Brutal Solver", M
#        print 'b',b
#        mp.figure()
#        mp.spy(M, markersize=1)
        x = ssl.spsolve(M, b)
#        print x
#        print x
        if hasattr(operator, 'postprocess'):
            x = operator.postprocess(x)
#        print x
        return x
        

class ItCounter(object):
    def __init__(self, stride = 20):
        self.n = 0
        self.stride = 20
    
    def __call__(self, x):
        self.n +=1
        if self.n % self.stride == 0:
            print self.n
    
class GMRESSolver(object):

    def __init__(self, dtype):
        self.dtype = dtype

    def solve(self, operator):
        b = operator.rhs()        
        n = len(b)
#        print b.shape
        lo = ssl.LinearOperator((n,n), self.op.multiply, dtype=self.dtype)
        pc = ssl.LinearOperator((n,n), self.op.precond, dtype=self.dtype) if hasattr(self.op, 'precond') else None
        
#        x, status = ssl.bicgstab(lo, b, callback = ItCounter(), M=pc)
        x, status = ssl.gmres(lo, b, callback = ItCounter(), M=pc, restart=450)
        print status

        if hasattr(self.op, 'postprocess'):
            x = self.op.postprocess(x)
        return x
            
#
#class BrutalSolver(object):
#    def __init__(self, dtype, operator):
#        self.op = operator
#        self.dtype = dtype
#    
#    def solve(self, system, sysargs, syskwargs):
#        self.op.setup(system, sysargs, syskwargs)
#        b = self.op.rhs()
#        n = len(b)
#        M = np.hstack([self.op.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
##        print M.shape, b.shape
##        print "Brutal Solver", M
#        x = ssl.spsolve(M, b)
##        print x
##        print x
#        if hasattr(self.op, 'postprocess'):
#            x = self.op.postprocess(x)
##        print x
#        return x
#        
#    
#class IndirectSolver(object):
#
#    def __init__(self, dtype, operator):
#        self.op = operator
#        self.dtype = dtype
#
#    def solve(self, system, sysargs, syskwargs):
#        self.op.setup(system,sysargs,syskwargs)
#        b = self.op.rhs()        
#        n = len(b)
##        print b.shape
#        lo = ssl.LinearOperator((n,n), self.op.multiply, dtype=self.dtype)
#        pc = ssl.LinearOperator((n,n), self.op.precond, dtype=self.dtype) if hasattr(self.op, 'precond') else None
#        
##        x, status = ssl.bicgstab(lo, b, callback = ItCounter(), M=pc)
#        x, status = ssl.gmres(lo, b, callback = ItCounter(), M=pc, restart=450)
#        print status
#
#        if hasattr(self.op, 'postprocess'):
#            x = self.op.postprocess(x)
#        return x
        
        
        