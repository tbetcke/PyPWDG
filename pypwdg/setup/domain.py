'''
Created on Mar 13, 2012

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import pypwdg.core.boundary_data as pcbd
import pypwdg.mesh.submesh as pmsm
import pypwdg.mesh.structure as pms
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.parallel.decorate as ppd

import scipy.sparse.linalg as ssl
import numpy as np
import copy
            
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


@ppd.distribute()
class MortarSystem(object):
    def __init__(self, compinfo, skelcompinfo, skelftob, sd, tracebc):
        self.volumeassembly = skelcompinfo.volumeAssembly()
        self.traceassembly = compinfo.faceAssembly(skelftob)
        self.sd = sd
        self.EM = pms.ElementMatrices(skelcompinfo.problem.mesh)
        self.sumleft = lambda G: pms.sumleftfaces(compinfo.problem.mesh, G)
        self.tracebc = tracebc
        skeletob = skelcompinfo.basis
        indnb = np.vstack((skeletob.indices[:-1], skeletob.numbases)).T[np.array(self.EM.I.diagonal, dtype=bool)]
        self.idxs = [np.arange(a, a+b, dtype=int) for a,b in indnb]
    
    @ppd.parallelmethod()    
    def getMass(self):
        S2S = self.EM.I * self.sd.skel2skel
        M = self.volumeassembly.assemble([[self.EM.I + S2S, self.EM.Z], [self.EM.Z], [self.EM.Z]])
        return M
    
    @ppd.parallelmethod()
    def getTrace(self):
        S2O = (self.EM.I * self.sd.skel2oppmesh).transpose()
        Z = S2O * 0
        S = self.traceassembly.assemble([[S2O*self.tracebc[0],Z],[S2O * self.tracebc[1],Z]])
        return self.sumleft(S)
        
         
class MortarComputation(object):
    def __init__(self, problem, basisrule, mortarrule, nquadpoints, systemklass, tracebc, usecache = False, **kwargs):
        skeletontag = 'INTERNAL'
        sd = pmsm.SkeletonisedDomain(problem.mesh, skeletontag)
        problem2 = copy.copy(problem)
        problem2.mesh = sd.mesh
        self.compinfo = psc.ComputationInfo(problem2, basisrule, nquadpoints)

        skeleproblem = psp.BasisAllocator(sd.skeletonmesh)
        skelecompinfo = psc.ComputationInfo(skeleproblem, mortarrule)
        
        skeletob = skelecompinfo.basis 
        skelftob = SkeletonFaceToBasis(skeletob, sd)
        
        self.system = systemklass(self.compinfo, **kwargs)
        mortarbcs = pcbd.BoundaryCoefficients([-1j*problem.k, 1], [1, 0])
        mortarinfo = (mortarbcs, skelftob)
        self.boundary = self.system.getBoundary(skeletontag, mortarinfo)
        
        self.mortarsystem = MortarSystem(self.compinfo, skelecompinfo, skelftob, sd, tracebc)
        
    def solution(self, solve, *args, **kwargs):
        worker = MortarWorker(self.system, self.boundary, self.mortarsystem, args, kwargs)
        x = solve(self.system, args, kwargs)
        return psc.Solution(self.compinfo, x)        

@ppd.distribute()
class MortarWorker(object):
    def __init__(self, system, boundary, mortarsystem, sysargs, syskwargs):
        AA,G = system.getSystem(*sysargs, **syskwargs)
        A = AA.tocsr()
        B = boundary.load(False).tocsr()
        idxs = mortarsystem.idxs
        self.M = mortarsystem.getMass().transpose()
        self.Ainv = ssl.splu(A[idxs, :][:, idxs])
        self.Brow = B[idxs, :]
        self.Scol = mortarsystem.getTrace()[:, idxs]
        self.G = G[idxs]
    
    @ppd.parallelmethod()
    def rhs(self):
        return self.Scol * self.Ainv.solve(self.G)
    
    @ppd.parallelmethod()
    def multiply(self, x):
        return self.Scol * self.Ainv.solve(self.Brow * x) + self.M * x
    
    @ppd.parallelmethod()
    def postprocess(self, x):
        return self.Ainv.solve(self.G - self.Brow * x)      

class BrutalSolver(object):
    def __init__(self, dtype):
        self.dtype = dtype
    
    def solve(self, operator):
        b = self.operator.rhs()
        n = len(b)
        M = np.hstack([operator.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
#        print M.shape, b.shape
#        print "Brutal Solver", M
        x = ssl.spsolve(M, b)
#        print x
#        print x
        if hasattr(operator, 'postprocess'):
            x = operator.postprocess(x)
#        print x
        return x
        
    
class IndirectSolver(object):

    def __init__(self, dtype):
        self.dtype = dtype

    def solve(self, operator, sysargs, syskwargs):
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
    

class ItCounter(object):
    def __init__(self, stride = 20):
        self.n = 0
        self.stride = 20
    
    def __call__(self, x):
        self.n +=1
        if self.n % self.stride == 0:
            print self.n    
            
            