'''
Created on Feb 20, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import numpy as np
import scipy.sparse.linalg as ssl
import pypwdg.parallel.mpiload as ppm

import time

np.set_printoptions(precision=4, threshold='nan',  linewidth=1000)


class ItCounter(object):
    def __init__(self, stride = 20):
        self.n = 0
        self.stride = 20
    
    def __call__(self, x):
        self.n +=1
        if self.n % self.stride == 0:
            print self.n

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
class DomainDecompWorker(object):
    def __init__(self, mesh):
        self.mesh = mesh
        
    @ppd.parallelmethod()
    def calcexternalidxs(self, system, sysargs, syskwargs):   
        self.S,self.G = system.getSystem(*sysargs, **syskwargs) 
        self.extidxs = self.S.subrows(self.mesh.innerbdyelts)
        return [self.extidxs]
    
    @ppd.parallelmethod()
    def setexternalidxs(self, allextidxs):
#        print self.mesh.partition
        localidxs = self.S.subrows(self.mesh.partition)
        self.internalidxs = np.array(list(set(localidxs).difference(self.extidxs)), dtype=int)

        M = self.S.tocsr()
        b = self.G.tocsr()
        self.n = b.shape[0]
        
#        print self.extidxs, allextidxs
        self.extrow = M[self.extidxs][:, allextidxs]
        
        Dint = M[self.internalidxs][:,self.internalidxs]
        self.DI = ssl.splu(Dint)
        Dext = M[self.extidxs][:, self.extidxs]
        self.DE = ssl.splu(Dext)
        self.Cextint = M[self.internalidxs][:, self.extidxs]
        self.Cintext = M[self.extidxs][:, self.internalidxs]
        time.sleep(ppm.comm.rank * 0.1)
        
        self.localfromallext = np.zeros_like(self.extidxs, dtype=int)
        li = 0
        for gi, idx in enumerate(allextidxs):
            if idx == self.extidxs[li]:
                self.localfromallext[li] = gi
                li+=1
            if li==len(self.extidxs): break
        
        self.DI.solve(np.zeros(Dint.shape[0]))
#        print type(np.zeros(Dint.shape[0]))
#        print type(b[self.internalidxs].todense().A)
        
        self.DIbint = self.DI.solve(b[self.internalidxs].todense().A.squeeze())
        rhs = b[self.extidxs].todense().A.squeeze() - self.Cintext * self.DIbint
        return [rhs]
        
        
    @ppd.parallelmethod()
    def multiplyext(self, x):
#        print self.extrow.shape, x.shape, self.Cintext.shape, self.Cextint.shape, x[self.localfromallext].shape
#        print "multiplyext", x
        y = self.extrow * x - self.Cintext * self.DI.solve(self.Cextint * x[self.localfromallext])
#        print x.shape, y.shape
#        print ppm.comm.rank, y
        return [y]  
    
    @ppd.parallelmethod()
    def precondext(self, x):
        return x
        return [self.DE.solve(x[self.localfromallext])]
    
    @ppd.parallelmethod()
    def recoverfullsoln(self, ue):
        x = np.zeros(self.n, dtype=complex)
#        print x.shape, self.internalidxs.shape, self.DIbint.shape, self.DI.solve(self.Cextint * ue[self.localfromallext]).shape
        x[self.internalidxs] = self.DIbint - self.DI.solve(self.Cextint * ue[self.localfromallext])
        x[self.extidxs] = ue[self.localfromallext]
#        print x
        return x 
         
class DomainDecompOperator(object):
    def __init__(self, mesh):
        self.workers = DomainDecompWorker(mesh)
    
    def setup(self, system, sysargs, syskwargs):
        extidxs = self.workers.calcexternalidxs(system, sysargs, syskwargs)
        self.rhsvec = np.concatenate(self.workers.setexternalidxs(np.concatenate(extidxs)))
    
    def rhs(self):
        return self.rhsvec
    
    def multiply(self, x):
        y = np.concatenate(self.workers.multiplyext(x))
#        print y.shape
        return y
    
    def precond(self, x):
        return x
#        return np.concatenate(self.workers.precondext(x))
    
    def postprocess(self, x):
        return self.workers.recoverfullsoln(x)
        

class BrutalSolver(object):
    def __init__(self, dtype, operator):
        self.op = operator
        self.dtype = dtype
    
    def solve(self, system, sysargs, syskwargs):
        self.op.setup(system, sysargs, syskwargs)
        b = self.op.rhs()
        n = len(b)
        M = np.hstack([self.op.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
#        print M.shape, b.shape
#        print "Brutal Solver", M
        x = ssl.spsolve(M, b)
#        print x
#        print x
        if hasattr(self.op, 'postprocess'):
            x = self.op.postprocess(x)
#        print x
        return x
        
    
class IndirectSolver(object):

    def __init__(self, dtype, operator):
        self.op = operator
        self.dtype = dtype

    def solve(self, system, sysargs, syskwargs):
        self.op.setup(system,sysargs,syskwargs)
        b = self.op.rhs()        
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
        
        
        