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
    def __init__(self, compinfo, skelecompinfo, skelftob, sd):
        self.volumeassembly = skelecompinfo.volumeAssembly()
        self.mesh = skelecompinfo.problem.mesh
        self.sd = sd
        self.traceassembly = compinfo.faceAssembly(skelftob)
    
    @ppd.parallelmethod()    
    def getMass(self):
        EM = pms.ElementMatrices(self.mesh)
        I = EM.I
        S2S = I * self.sd.skel2skel
        M = self.volumeassembly.assemble([[I + S2S, EM.Z], [EM.Z], [EM.Z]])
        return M
    
    @ppd.parallelmethod()
    def getTrace(self):
        
        
    
         
class MortarComputation(object):
    def __init__(self, problem, basisrule, mortarrule, nquadpoints, systemklass, usecache = False, **kwargs):
        skeletontag = 'INTERNAL'
        sd = pmsm.SkeletonisedDomain(problem.mesh, skeletontag)
        problem2 = psp.Problem(sd.mesh, problem.k, problem.bnddata)
        self.compinfo = psc.ComputationInfo(problem2, basisrule, nquadpoints)

        skeleproblem = psp.BasisAllocator(sd.skeletonmesh)
        skelecompinfo = psc.ComputationInfo(skeleproblem, mortarrule)
        
        skeletob = skelecompinfo.basis 
        skelftob = SkeletonFaceToBasis(skeletob, sd)
        
        self.system = systemklass(self.compinfo, **kwargs)
        mortarbcs = pcbd.BoundaryCoefficients([-1j*problem.k, 1], [1, 0])
        mortarinfo = (mortarbcs, skelftob)
        boundary = self.system.getBoundary(skeletontag, mortarinfo)
        tracebcs = 
        
        self.mortarsystem = MortarSystem(self.compinfo, skelecompinfo, skelftob, sd)
        
        

    def solution(self, solve, *args, **kwargs):
        x = solve(self.system, args, kwargs)
        return psc.Solution(self.compinfo, x)        

@ppd.distribute()
class MortarWorker(object):
    def __init__(self, skeletontag, idxs, system, boundary, mortarsystem, sysargs, syskwargs):
        
        
        AA,G = system.getSystem(*sysargs, **syskwargs)
        A = AA.tocsr()
        B = boundary.load(False).tocsr()
        M = mortarsystem.getMass()
        self.Mcol = M[:, idxs]
        self.Ainv = ssl.splu(A[idxs, :][:, idxs])
        self.Brow = B[idxs, :]
        self.Scol = ...
        
    def multiply(self, x):
        pass


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