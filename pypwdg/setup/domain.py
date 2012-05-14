'''
Created on Mar 13, 2012

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import numpy as np
            
''' Domain decomposition iterative procedure:
    - Create sub problems.  
    - Subproblems get boundary data from a mortar variable
'''


class SkeletonFaceToBasis(object):
    def __init__(self, skeleelttobasis, skeletonfacemap):
        self.elttobasis = skeleelttobasis
        self.skeletonfacemap = skeletonfacemap
             
    def evaluate(self, faceid, points):        
#        print "SkeletonFaceToBasis.evaluate %s"%faceid
        skeletonelt = self.skeletonfacemap.index[faceid]
#        print skeletonelt
        if skeletonelt >=0: 
            vals = self.elttobasis.getValues(skeletonelt, points)
            derivs = vals
#         print derivs.shape
            return (vals,derivs)
        else:
            raise Exception('Bad faceid for Skeleton %s,%s'%(faceid, skeletonelt))
    
    @property
    def numbases(self):
        return self.skeletonfacemap.expand(self.elttobasis.getSizes())
    
    @property
    def indices(self):
        return np.cumsum(np.concatenate(([0], self.numbases)))
         
class MortarSystem(object):
    def __init__(self, meshinfo, basisrule, nquadpoints, systemklass, usecache = False, **kwargs):
        sd = pmsm.SkeletonisedDomain(meshinfo, 'INTERNAL')

class MortarWorker(object):
    def __init__(self, skeletontag):
        self.skeletontag = skeletontag
        
    def setup(self, system, sysargs, syskwargs):
        AA,G = system.getSystem()
        A = AA.tocsr()
        boundary = system.getBoundary(skeletontag, (pcbd.BoundaryCoefficients([-1j*k, 1], [1, 0]), skelftob))
        
class MortarOperator(object):
    def __init__(self, mesh):
        self.workers = MortarWorker(mesh)
    
    def setup(self, system, sysargs, syskwargs):
        
        
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