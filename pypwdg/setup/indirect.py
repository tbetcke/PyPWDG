'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl
import pypwdg.setup.computation as psc
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

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

@ppd.distribute()
class DiagonalBlockOperator(object):
    
    def __init__(self, mesh):
        self.mesh = mesh
    
    @ppd.parallelmethod()    
    def setup(self, system, sysargs, syskwargs):   
        S,G = system.getSystem(*sysargs, **syskwargs)
        self.localidxs = S.subrows(self.mesh.partition)
 
        M = S.tocsr()
        D = S.diagonal().tocsr()
        self.C = (D - M)[self.localidxs, :]
        self.Dlu = ssl.splu(D[self.localidxs, :][:, self.localidxs])
        self.n = S.shape[0]

        self.b = np.zeros(self.n, dtype=np.complex)        
        self.b[self.localidxs] = self.Dlu.solve(G.tocsr().todense().A[self.localidxs].ravel()) 
        
        self.dtype = M.dtype
    
    @ppd.parallelmethod()
    def rhs(self):        
        return self.b
    
    @ppd.parallelmethod()
    def multiply(self, x):
        y = np.zeros_like(x)
        y[self.localidxs] = x[self.localidxs] - self.Dlu.solve(self.C * x)
        return y
        

class BrutalSolver(object):
    def __init__(self, dtype, store=False):
        self.dtype = dtype
        self.store = store
    
    def solve(self, operator):
        b = operator.rhs()
        n = len(b)
        M = np.hstack([operator.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
        log.debug("M = %s, b = %s", M.shape, b.shape)
        x = ssl.spsolve(M, b)
        if self.store:
            self.M = M
            self.b = b
            self.x = x
            
        if hasattr(operator, 'postprocess'):
            x = operator.postprocess(x)
        return x
        

class ItCounter(object):
    def __init__(self, stride = 20):
        self.n = 0
        self.stride = stride
    
    def __call__(self, x):
        self.n +=1
        if self.n % self.stride == 0:
            print self.n

class ItTracker(object):
    def __init__(self, fn = None, stride = 1):
        self.its = []
        self.fn = fn
        self.stride = stride
        
    def __call__(self, x):
        if len(self.its) % 50 == 0: log.debug("res=%s, n=%s"%(x, len(self.its)))
        if len(self.its) % self.stride==0: 
            self.its.append(x if self.fn is None else self.fn(x))   
    
    def reset(self):
        its = self.its
        self.its = []
        return its

class IterativeSolver(object):
    
    def __init__(self, dtype, callback=None):
        self.dtype = dtype
        self.callback = callback if callback is not None else ItCounter()

    def solve(self, operator):
        if self.dtype=='ctor':
            dtype = np.float
            operator = ComplexToRealOperator(operator)
        else:
            dtype = self.dtype
        
        b = operator.rhs()       
        n = len(b)
        log.info("IterativeSolver solving system of size %s", n)
        
        lo = ssl.LinearOperator((n,n), operator.multiply, dtype=dtype)
        pc = ssl.LinearOperator((n,n), operator.precond, dtype=dtype) if hasattr(operator, 'precond') else None
        
        x = self.solveop(lo, b, pc, n)
        print "pre-post-process",x
        if hasattr(operator, 'postprocess'):
            x = operator.postprocess(x)
        return x

class GMRESSolver(IterativeSolver):
    def solveop(self, lo,b,pc,n):
        x,status = ssl.gmres(lo, b, callback=self.callback, M=pc, restart = n)
        log.info(status)
        return x

class BICGSTABSolver(IterativeSolver):
    def solveop(self, lo,b,pc,n):
        x,status = ssl.bicgstab(lo, b, callback=self.callback, M=pc)
        log.info(status)
        return x



class ComplexToRealOperator(object):
    def __init__(self, complexop):
        self.op = complexop
        if hasattr(complexop, 'precond'):
            self.precond = lambda x: self.ctor(self.op.precond(self.rtoc(x)))
    
    def ctor(self, x):
        return np.concatenate((x.real, x.imag))
    
    def rtoc(self, x):
        return x[0:len(x)/2] + x[len(x)/2:]*1j
    
    def rhs(self):
        return self.ctor(self.op.rhs())
    
    def multiply(self, x):
        return self.ctor(self.op.multiply(self.rtoc(x)))

    def postprocess(self, x):
        xp = self.rtoc(x)
        return self.op.postprocess(xp) if hasattr(self.op, 'postprocess') else xp
        