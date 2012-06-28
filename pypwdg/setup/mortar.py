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

import pypwdg.core.assembly as pca

import scipy.sparse as ss
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
        skeletonelt = self.skeletonfacemap.index[faceid]
#        print "SkeletonFaceToBasis.evaluate", faceid, skeletonelt
        if skeletonelt >=0: 
            vals = self.elttobasis.getValues(skeletonelt, points)
            derivs = vals * 15E20
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
#        print 'idxs',self.idxs
    
    @ppd.parallelmethod()    
    def getMass(self, doopposite=True):
        ''' Returns a skeleton element x skeleton element mass matrix (as a vbsrmatrix) '''
#        print "getMass"
        S2S = self.EM.I * self.sd.skel2skel if doopposite else self.EM.Z
#        print "S2S", S2S
        M = self.volumeassembly.assemble([[self.EM.I + S2S, self.EM.Z], [self.EM.Z, self.EM.Z]])
        return M
    
    @ppd.parallelmethod()
    def getOppositeTrace(self):
        ''' This returns the product of the traces on the mesh faces with the opposite skeleton element'''
        S2O = (self.sd.skel2mesh.transpose() * self.EM.I * self.sd.skel2oppmesh)
        Z = S2O * 0
#        print 'S20',S2O
#        S = self.traceassembly.assemble([[S2O*self.tracebc[0],Z],[S2O * self.tracebc[1],Z]])
        S = self.traceassembly.assemble([[S2O*self.tracebc[0],Z],[Z,Z]])
        return self.sumleft(S)

@ppd.distribute()
class MortarProjection(object):
    def __init__(self, compinfo, skelftob, fnftob, sd, coeffs):
        fnv = compinfo.faceVandermondes(fnftob)
        lambdav = compinfo.faceVandermondes(skelftob)  
        self.loadassembly = pca.Assembly(lambdav, fnv, compinfo.facequads.quadweights)
        self.I = compinfo.problem.mesh.entityfaces['INTERNAL']
        self.Z = pms.AveragesAndJumps(compinfo.problem.mesh).Z
        self.coeffs = coeffs
        self.sd = sd
    
    @ppd.parallelmethod()
    def product(self):
        c1, c2 = self.coeffs
        lblock = self.loadassembly.assemble([[c1 * self.I, c2 * self.I], [self.Z, self.Z]])
#        print 'lblock', lblock.tocsr()
        lblockcol = lblock.__rmul__(self.sd.skel2mesh) * ss.csr_matrix(np.ones((self.sd.mesh.nfaces, 1)))
        return lblockcol
         
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
    def __init__(self, problem, basisrule, mortarrule, nquadpoints, systemklass, boundaryklass, s, usecache = False, **kwargs):
        skeletontag = 'INTERNAL'
        tracebc = [2*s,0]
        self.sd = pmsm.SkeletonisedDomain(problem.mesh, skeletontag)
        problem2 = copy.copy(problem)
        problem2.mesh = self.sd.mesh
        self.compinfo = psc.ComputationInfo(problem2, basisrule, nquadpoints)

        skeleproblem = psp.BasisAllocator(self.sd.skeletonmesh)
        skelecompinfo = psc.ComputationInfo(skeleproblem, mortarrule, nquadpoints)
        
        skeletob = skelecompinfo.basis 
        self.skelftob = SkeletonFaceToBasis(skeletob, self.sd)
        
        self.system = systemklass(self.compinfo, **kwargs)
        mortarbcs = pcbd.BoundaryCoefficients([s, 1], [1, 0])
        mortarinfo = (mortarbcs, self.skelftob)
        self.boundary = boundaryklass(self.compinfo, skeletontag, mortarinfo)
        
        self.mortarsystem = MortarSystem(self.compinfo, skelecompinfo, self.skelftob, self.sd, tracebc)
        
    def solution(self, solver, *args, **kwargs):
        ''' Calculate a solution.  The solve method should accept an operator'''
        operator = MortarOperator(self.system, self.boundary, self.mortarsystem, self.sd.skeletonmesh, args, kwargs)
        S = operator.getScol()
#        print 'S',S   
#        mp.spy(S, markersize=1)
#        mp.figure()
#        print operator.getScol()
#        mp.spy(operator.getM(), markersize=1)
#        print operator.getM().todense()
        x = solver.solve(operator)
        return psc.Solution(self.compinfo, x)
    
    def fakesolution(self, truesoln, coeffs, *args, **kwargs):
        operator = MortarOperator(self.system, self.boundary, self.mortarsystem, args, kwargs)
        trueftob = pcbu.UniformFaceToBases(truesoln, self.compinfo.problem.mesh)
        Ml = MortarProjection(self.compinfo, self.skelftob, trueftob, self.sd, coeffs).product().tocsr().toarray()
        M = self.mortarsystem.getMass(False).tocsr() * (1+0j)
        l = ssl.spsolve(M, Ml)
#        print 'l',l
#        print np.vstack((operator.multiply(l), operator.rhs())).T
        x = operator.postprocess(l)
#        
#        f,g = operator.fullmultiply(x, l)
#        rhs = operator.fullrhs()
#        print f.shape, g.shape, rhs.shape
#        print "comparing full solution"
#        print np.vstack((f, rhs)).T
#        print g
#        print operator.getScol() * x                
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
    def __init__(self, system, boundary, mortarsystem, skelmesh, sysargs, syskwargs):
        AA,G = system.getSystem(*sysargs, **syskwargs)
        BS = boundary.stiffness()
        #print 'BS',BS.tocsr()
        A = (AA+BS).tocsr()
        A.eliminate_zeros()
        BL = boundary.load(False).tocsr()
        MM = mortarsystem.getMass()
        self.M = MM.tocsr().transpose()
        idxs = mortarsystem.idxs
        self.skelidxs = MM.subrows(skelmesh.partition)
        print self.M.shape
        self.Ainv = ssl.splu(A[idxs, :][:, idxs])
        self.Minv = ssl.splu(self.M[self.skelidxs, :][:, self.skelidxs])
        self.Brow = -BL[idxs, :]
        T = mortarsystem.getOppositeTrace().tocsr().transpose()
        self.Scol = -T[:, idxs].conj() # Why?
        self.G = G.tocsr().todense()[idxs].A.flatten()
        self.localtoglobal = pusu.sparseindex(idxs, np.arange(len(idxs)), A.shape[0], len(idxs))
#        print 'localtoglobal', self.localtoglobal
        print 'nnz', BL.nnz, self.Brow.nnz, A.nnz, A[idxs, :][:, idxs].nnz, T.nnz, T[:,idxs].nnz
#        print 'T',T
#        print 'M', self.M
        
#        self.A = A
#        self.idxs = idxs
        
#        print 'A', A
#        print 'BL', BL
#        print 'Brow', self.Brow
        
#        print "scol", mortarsystem.getTrace().tocsr().transpose()
    
#    @ppd.parallelmethod(None, ppd.tuplesum)
#    def fullmultiply(self, u, l):
#        g = self.A * u + self.localtoglobal * self.Brow * l
#        f = - self.Scol * u[self.idxs] + self.M * l
#        return (g,f)
#
#    @ppd.parallelmethod()
#    def fullrhs(self):
#        return self.localtoglobal * self.G
        
    @ppd.parallelmethod()
    def getM(self):
        return self.M
    
    @ppd.parallelmethod()
    def getScol(self):
        return self.Scol * self.localtoglobal.transpose()
    
    @ppd.parallelmethod()
    def rhs(self):
        ''' Return the RHS used for the global solve'''
#        print 'rhs', self.Ainv.solve(self.G)
        return self.Scol * self.Ainv.solve(self.G)
    
    @ppd.parallelmethod()
    def multiply(self, l):
        ''' Mat-vec multiplication used for the global solve'''
#        print 'multiply', l, self.Scol * self.Ainv.solve(self.Brow * l), self.M * l
        return self.Scol * self.Ainv.solve(self.Brow * l) + self.M * l
    
    @ppd.parallelmethod()
    def postprocess(self, l):
        ''' Post-process the global solution (the lambdas) to obtain u'''
#        print "postprocess ",x
#        l = np.zeros_like(l)
#        print 'postprocess', self.G - self.Brow * l
        u = self.Ainv.solve(self.G - self.Brow * l)
#        print "u", u  
        return self.localtoglobal * u
#
    @ppd.parallelmethod()
    def precond(self, l):
        y = np.zeros_like(l)
        y[self.skelidxs] = self.Minv.solve(l[self.skelidxs])
        return y
    
    
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
            