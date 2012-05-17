'''
Created on May 17, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import scipy.sparse.linalg as ssl
import pypwdg.parallel.mpiload as ppm
import time
import numpy as np
        
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
        
        self.localfromallext = np.zeros_like(self.extidxs)
        li = 0
        for gi, idx in enumerate(allextidxs):
            if idx == self.extidxs[li]:
                self.localfromallext[li] = gi
                li+=1
            if li==len(self.extidxs): break
        
#        self.DI.solve(np.zeros(Dint.shape[0]))
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
