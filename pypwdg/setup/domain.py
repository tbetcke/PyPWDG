'''
Created on May 17, 2012

@author: joel
'''
import pypwdg.parallel.decorate as ppd
import scipy.sparse.linalg as ssl
import pypwdg.parallel.mpiload as ppm
import time
import numpy as np
import logging
import scipy.linalg as sl

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

@ppd.distribute()
class SchwarzWorker(object):
    def __init__(self, mesh):
        self.mesh = mesh
        
    @ppd.parallelmethod()
    def initialise(self, system, sysargs, syskwargs):
        """ Initialise the system in the worker
        
            returns a list of the neighbouring degrees of freedom required by this process
        """
        self.S,self.G = system.getSystem(*sysargs, **syskwargs) 
        return [self.S.subrows(self.mesh.neighbourelts)]
    
    @ppd.parallelmethod()
    def setexternalidxs(self, allextidxs):
        """ Tell this worker which degrees of freedom are exterior (i.e. are accessed by one process from another) 
        
            Returns this worker's contribution to the RHS for the Schur complement system
        """
        localidxs = self.S.subrows(self.mesh.partition) # the degrees of freedom calculated by this process
                                                        # N.B. for an overlapping method these arrays will overlap between processes -
                                                        # it still all works!
        sl = set(localidxs)
        extidxs =  np.sort(np.array(list(sl.intersection(allextidxs)), dtype=int)) # the exterior degrees for this process
        intidxs = np.sort(np.array(list(sl.difference(allextidxs)), dtype=int)) # the interior degrees for this process
        self.intind = np.zeros(self.S.shape[0], dtype=bool) 
        self.intind[intidxs] = True # Create an indicator for the interior degrees
        self.localext = allextidxs.searchsorted(extidxs)

        log.debug("local %s"%localidxs)
        log.debug("external %s"%extidxs)
        log.debug("internal %s"%intidxs)
        log.info("localext %s"%self.localext)

        M = self.S.tocsr() # Get CSR representations of the system matrix ...
        b = self.G.tocsr() # ... and the load vector

        # Decompose the system matrix.  
        self.ext_allext = M[extidxs][:, allextidxs] 
        self.int_intinv = ssl.splu(M[intidxs][:,intidxs])
        self.int_allext = M[intidxs][:, allextidxs]
        self.ext_int = M[extidxs][:, intidxs]
        self.int_ext = M[intidxs][:, extidxs]

        
        self.intsolveb = self.int_intinv.solve(b[intidxs].todense().A.squeeze())
        self.rhs = b[extidxs].todense().A.squeeze() - self.ext_int * self.intsolveb
        self.JMinv = None

        return [self.rhs]
    
        
    @ppd.parallelmethod()
    def multiplyext(self, x):
        y = self.ext_allext * x - self.ext_int * self.int_intinv.solve(self.int_allext * x)
        return [y]  
    
    @ppd.parallelmethod()
    def precondext(self, x):              
        return [self.ext_extinv.solve(x[self.localext]) if hasattr(self, 'ext_extinv') else x[self.localext] ]
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def recoverinterior(self, xe):
        """ Recover all the local interior degrees from the exterior degrees
        
            returns a tuple of the contribution to the global solution vector and an indicator of what 
            degrees have been written to.  The indicator is used to average duplicate contributions for 
            overlapping methods.
        """ 
        x = np.zeros_like(self.intind, dtype=complex)
        x[self.intind] = self.intsolveb - self.int_intinv.solve(self.int_allext * xe)
        print "recover", np.nonzero(np.abs(x[self.intind])<1E-3)
        return x, self.intind*1
    
    @ppd.parallelmethod()
    def jacobimultiply(self, xe):
        if self.JMinv is None:
            IISIE = np.hstack([self.int_intinv.solve(x.A.ravel()).reshape(-1,1) for x in self.int_ext.todense().T])
            print IISIE.shape, self.ext_int.shape, self.ext_allext[:, self.localext].shape
            
            JM = self.ext_allext[:, self.localext] - self.ext_int * IISIE
            self.JMinv = sl.lu_factor(JM)
            print len(self.JMinv)
        xe[self.localext] = 0
        return [sl.lu_solve(self.JMinv, self.rhs - self.ext_allext * xe)]

class GeneralSchwarzOperator(object):
    """ Together with SchwarzWorker, the SchwarzOperator implements a linear system whose
        solution is the fixed point of a Schwarz iteration based on an algebraic decomposition
        of the system.
        
        The decomposition is determined by the partitions of the mesh.  These partitions may be
        overlapping.  
    """        
    def __init__(self, workers):
        self.workers = workers
    
    def setup(self, system, sysargs, syskwargs):
        e = np.concatenate(self.workers.initialise(system, sysargs, syskwargs))
        self.extidxs = np.unique(e)
        print "duplicate ", len(e) - len(self.extidxs)
        self.rhsvec = np.concatenate(self.workers.setexternalidxs(self.extidxs))
        log.info("Schur complement system has %s dofs"%len(self.extidxs))
    
    def rhs(self):
        return self.rhsvec
    
    def multiply(self, x):
        y = np.concatenate(self.workers.multiplyext(x))
        return y
    
    def precond(self, x):
        return np.concatenate(self.workers.precondext(x))
    
    def jacobimultiply(self, x):
        return np.concatenate(self.workers.jacobimultiply(x))
    
    def postprocess(self, xe):
        """ Given some values at the exterior dofs, recover the global solution """
        x, count = self.workers.recoverinterior(xe) # Get the workers to recover their interior dofs
#        print count 
        count[self.extidxs]+=1
#        print 'count', count
        x[self.extidxs] += xe # Now add in the exterior stuff (no point having the workers do this)
        return x / count # Average anything that got duplicated
             
class SchwarzOperator(GeneralSchwarzOperator):
    def __init__(self, mesh):
        GeneralSchwarzOperator.__init__(self, SchwarzWorker(mesh))

 