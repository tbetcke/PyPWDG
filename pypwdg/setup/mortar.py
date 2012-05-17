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
import pypwdg.utils.sparseutils as pusu

import scipy.sparse.linalg as ssl
import numpy as np
import copy
import matplotlib.pyplot as mp

class SkeletonFaceToBasis(object):
    ''' Take an elementToBasis defined on a skeleton mesh and return a FaceToBasis on the underlying mesh.
        Non-skeleton faces have an empty basis on them
    '''
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
    ''' The system information to build a domain decomposition using mortar spaces '''
    def __init__(self, compinfo, skelcompinfo, skelftob, sd, tracebc):
        self.volumeassembly = skelcompinfo.volumeAssembly()
        self.traceassembly = compinfo.faceAssembly(compinfo.faceVandermondes(skelftob))
        self.sd = sd
        self.EM = pms.ElementMatrices(skelcompinfo.problem.mesh)
        self.sumleft = lambda G: pms.sumleftfaces(compinfo.problem.mesh, G)
        self.tracebc = tracebc
#        skeletob = skelcompinfo.basis
#        indicesandsizes = np.vstack((skeletob.getIndices()[:-1], skeletob.getSizes())).T
#        localindicesandsizes = indicesandsizes[np.array(self.EM.I.diagonal(), dtype=bool)]
        etob = compinfo.basis
        indicesandsizes = np.vstack((etob.getIndices()[:-1], etob.getSizes())).T
        localindicesandsizes = indicesandsizes[compinfo.problem.mesh.partition]
        
        self.idxs = np.concatenate([np.arange(i, i+s, dtype=int) for (i,s) in localindicesandsizes])
        print 'idxs',self.idxs
    
    @ppd.parallelmethod()    
    def getMass(self):
        ''' Returns a skeleton element x skeleton element mass matrix (as a vbsrmatrix) '''
#        print "getMass"
        S2S = self.EM.I * self.sd.skel2skel
#        print "S2S", S2S
        M = self.volumeassembly.assemble([[self.EM.I + S2S, self.EM.Z], [self.EM.Z, self.EM.Z]])
        return M
    
    @ppd.parallelmethod()
    def getOppositeTrace(self):
        ''' This returns the product of the traces on the mesh faces with the opposite skeleton element'''
        S2O = (self.sd.skel2mesh.transpose() * self.EM.I * self.sd.skel2oppmesh)
        Z = S2O * 0
#        print 'S20',S2O
        S = self.traceassembly.assemble([[S2O*self.tracebc[0],Z],[S2O * self.tracebc[1],Z]])
        return self.sumleft(S)
        
         
class MortarComputation(object):
    ''' A Mortar Computation.  This probably ought to be integrated with the basic Computation class - just don't know how yet
    
        Parameters:
            problem: the problem to be solved
            basisrule: how to create a basis on the mesh
            mortarrule: what basis to use for the mortar space
            systemklass: Defines the physics
            boundaryklass: Defines the physics on the boundary (can't use getBoundary on the systemklass because of limited parallelisation functionality)
            tracebc: Defines the S map
    '''
    def __init__(self, problem, basisrule, mortarrule, nquadpoints, systemklass, boundaryklass, tracebc, usecache = False, **kwargs):
        skeletontag = 'INTERNAL'
        sd = pmsm.SkeletonisedDomain(problem.mesh, skeletontag)
        problem2 = copy.copy(problem)
        problem2.mesh = sd.mesh
        self.compinfo = psc.ComputationInfo(problem2, basisrule, nquadpoints)

        skeleproblem = psp.BasisAllocator(sd.skeletonmesh)
        skelecompinfo = psc.ComputationInfo(skeleproblem, mortarrule, nquadpoints)
        
        skeletob = skelecompinfo.basis 
        skelftob = SkeletonFaceToBasis(skeletob, sd)
        
        self.system = systemklass(self.compinfo, **kwargs)
        mortarbcs = pcbd.BoundaryCoefficients([-1j*problem.k, 1], [1, 0])
        mortarinfo = (mortarbcs, skelftob)
        self.boundary = boundaryklass(self.compinfo, skeletontag, mortarinfo)
        
        self.mortarsystem = MortarSystem(self.compinfo, skelecompinfo, skelftob, sd, tracebc)
        
    def solution(self, solver, *args, **kwargs):
        ''' Calculate a solution.  The solve method should accept an operator'''
        operator = MortarOperator(self.system, self.boundary, self.mortarsystem, args, kwargs)
#        print "scol", worker.getScol().shape
        mp.spy(operator.getScol(), markersize=1)
        mp.figure()
        mp.spy(operator.getM(), markersize=1)
        x = solver.solve(operator)
        return psc.Solution(self.compinfo, x)        

@ppd.distribute()
class MortarOperator(object):
    ''' A (distributed) operator that implements a Mortar domain decomposition
    
            [A_1                 B_1           ] [u_1]   [g_1]
            [       A_2               B_2      ] [u_2]   [g_2]
            [                A_3           B_3 ] [u_3] = [g_3]
            [      -S_21 -S_31   M_11 M_12 M_13] [l_1]   [ 0 ]
            [-S_12       -S_32   M_21 M_22 M_23] [l_2]   [ 0 ]
            [-S_13 -S_23         M_31 M_32 M_33] [l_3]   [ 0 ]
        
        using the Schur complement 
            
        Note that the parallelisation is effectively done column-wise, so, for example, process 1 implements the mat-vec
        
         [ 0  ] A_1^{-1} B_1 l_1        [M_11] l_1
         [S_12]                    +    [M_12]
         [S_13]                         [M_13]
         
         and contributes to the RHS:

         [ 0  ] A_1^{-1} g_1    
         [S_12]                 
         [S_13]                 
         
    '''
    def __init__(self, system, boundary, mortarsystem, sysargs, syskwargs):
        AA,G = system.getSystem(*sysargs, **syskwargs)
        A = AA.tocsr()
        A.eliminate_zeros()
        B = boundary.load(False).tocsr()
        idxs = mortarsystem.idxs
        self.M = mortarsystem.getMass().tocsr().transpose()
        self.Ainv = ssl.splu(A[idxs, :][:, idxs])
        self.Brow = B[idxs, :]
        T = mortarsystem.getOppositeTrace().tocsr().transpose()
        self.Scol = T[:, idxs]
        self.G = G.tocsr().todense()[idxs].A.flatten()
        self.localtoglobal = pusu.sparseindex(idxs, np.arange(len(idxs)), A.shape[0], len(idxs))
        print 'nnz', B.nnz, self.Brow.nnz, A.nnz, A[idxs, :][:, idxs].nnz, T.nnz, T[:,idxs].nnz
#        print "scol", mortarsystem.getTrace().tocsr().transpose()
    
    @ppd.parallelmethod()
    def getM(self):
        return self.M
    
    @ppd.parallelmethod()
    def getScol(self):
        return self.Scol * self.localtoglobal.transpose()
    
    @ppd.parallelmethod()
    def rhs(self):
        ''' Return the RHS used for the global solve'''
#        print 'rhs', self.G, self.Ainv.solve(self.G)
        return self.Scol * self.Ainv.solve(self.G)
    
    @ppd.parallelmethod()
    def multiply(self, x):
        ''' Mat-vec multiplication used for the global solve'''
        return self.Scol * self.Ainv.solve(self.Brow * x) + self.M * x
    
    @ppd.parallelmethod()
    def postprocess(self, x):
        ''' Post-process the global solution (the lambdas) to obtain u'''
        u = self.Ainv.solve(self.G - self.Brow * x)  
#        print 'u',u
        return self.localtoglobal * u      
#
#class BrutalSolver(object):
#    def __init__(self, dtype):
#        self.dtype = dtype
#    
#    def solve(self, operator):
#        b = operator.rhs()
#        n = len(b)
#        M = np.hstack([operator.multiply(x).reshape(-1,1) for x in np.eye(n, dtype=self.dtype)])
#        print M.shape, b.shape
##        print "Brutal Solver", M
##        print 'b',b
#        mp.figure()
#        mp.spy(M, markersize=1)
#        x = ssl.spsolve(M, b)
##        print x
##        print x
#        if hasattr(operator, 'postprocess'):
#            x = operator.postprocess(x)
##        print x
#        return x
#        
#    
#class IndirectSolver(object):
#
#    def __init__(self, dtype):
#        self.dtype = dtype
#
#    def solve(self, operator, sysargs, syskwargs):
#        b = operator.rhs()        
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
#    
#
#class ItCounter(object):
#    def __init__(self, stride = 20):
#        self.n = 0
#        self.stride = 20
#    
#    def __call__(self, x):
#        self.n +=1
#        if self.n % self.stride == 0:
#            print self.n    
#            
            