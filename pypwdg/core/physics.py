'''
Created on Aug 11, 2010

@author: joel
'''
import pypwdg.utils.quadrature as puq
import pypwdg.mesh.meshutils as pmmu
import pypwdg.core.vandermonde as pcv
import pypwdg.core.assembly as pca
import pypwdg.mesh.structure as pms
import pypwdg.parallel.decorate as ppd
import pypwdg.core.bases.utilities as pcbu    
import numpy as np

class HelmholtzBoundary(object):
    def __init__(self, mesh, entity, bdyinfo, delta, facequads, facev, scaledv, entityton = None):
        bdycoeffs, bdyetob = bdyinfo
        bdyvandermondes = pcv.LocalVandermondes(mesh, bdyetob, facequads) if entityton is None else pcv.ScaledVandermondes(entityton, mesh, bdyetob, facequads)  
        rc0,rc1 = bdycoeffs.r_coeffs
        rqw0 = pmmu.ScaledQuadweights(facequads, rc0)     
        rqw1 = pmmu.ScaledQuadweights(facequads, rc1)     
        self.loadassembly = pca.Assembly(facev, bdyvandermondes, [[rqw0,rqw1],[rqw0,rqw1]])
        lc0,lc1 = bdycoeffs.l_coeffs
        lqw0 = pmmu.ScaledQuadweights(facequads, lc0)
        lqw1 = pmmu.ScaledQuadweights(facequads, lc1)
        self.weightedbdyassembly = pca.Assembly(facev, scaledv, [[lqw0,lqw1],[lqw0,lqw1]])
        self.B = mesh.entityfaces[entity]
        self.delta = delta
        self.mesh = mesh
        
    def load(self):
        ''' The load vector (due to the boundary conditions)'''            
        delta = self.delta
        B = self.B
        # todo - check the cross terms.  Works okay with delta = 1/2.  
        GB = self.loadassembly.assemble([[(1-delta) * B, (1-delta) *B], 
                                    [-delta* B,     -delta *B]])
                
        return pms.sumrhs(self.mesh,GB)
        
    def stiffness(self):
        delta = self.delta
        B = self.B
        
        # The contribution of the boundary conditions
        SB = self.weightedbdyassembly.assemble([[(1-delta) * B, (1-delta)*B],
                              [-delta * B, -delta *B]])
        return pms.sumfaces(self.mesh,SB)     

            

@ppd.distribute()    
class HelmholtzSystem(object):
    ''' Assemble a system to solve a Helmholtz problem using the DG formulation of Gittelson et al.
        It's parallelised so will only assemble the relevant bits of the system for the partition managed by
        this process.
    '''
    def __init__(self, problem, basis, nquadpoints, alpha=0.5,beta=0.5,delta=0.5, usecache=True, entityton = None):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.problem = problem
        self.entityton = entityton
        fquad, equad = puq.quadrules(problem.mesh.dim, nquadpoints)
        facequads = pmmu.MeshQuadratures(problem.mesh, fquad)
        elementquads = pmmu.MeshElementQuadratures(problem.mesh, equad)
        self.facequads = facequads
        
        self.basis = basis
        self.facevandermondes = pcv.LocalVandermondes(pcv.FaceToBasis(problem.mesh, basis), facequads, usecache=usecache)
        self.scaledvandermondes = self.facevandermondes if entityton is None else pcv.LocalVandermondes(pcv.FaceToScaledBasis(entityton, problem.mesh, basis), facequads, usecache=usecache)
        self.internalassembly = pca.Assembly(self.facevandermondes, self.facevandermondes, facequads.quadweights)
         
        self.loadassemblies = {}
        self.weightedbdyassemblies = {}
        self.boundaries = [self.getBoundary(entity, bdyinfo) for (entity, bdyinfo) in problem.bdyinfo.items()]
        
        ev = pcv.ElementVandermondes(problem.mesh, self.basis, elementquads)
        self.volumeassembly = pca.Assembly(ev, ev, elementquads.quadweights)
        def kweights(e): 
            return elementquads.quadweights(e).squeeze()* (problem.elementinfo.kp(e)(elementquads.quadpoints(e))**2)
        self.weightedassembly = pca.Assembly(ev, ev, kweights)

    @ppd.parallelmethod(None, ppd.tuplesum)        
    def getSystem(self, dovolumes = False):
        ''' Returns the stiffness matrix and load vector'''
        SI = self.internalStiffness() 
        SB = sum([b.stiffness() for b in self.boundaries])
        S = SI + SB
        G = sum([b.load() for b in self.boundaries])
        if dovolumes: 
            S+=self.volumeStiffness()
        return S,G

#    @ppd.parallelmethod()        
    def internalStiffness(self):
        ''' The contribution of the internal faces to the stiffness matrix'''
        jk = 1j * self.problem.k
        AJ = pms.AveragesAndJumps(self.problem.mesh)   
        B = self.problem.mesh.boundary # This is the integration by parts term that generally gets folded into the boundary data, but is more appropriate here
        SI = self.internalassembly.assemble([[jk * self.alpha * AJ.JD,   -AJ.AN - B], 
                                            [AJ.AD + B,                -(self.beta / jk) * AJ.JN]])    
        SFSI = pms.sumfaces(self.problem.mesh,SI)
                
        return SFSI
    
    def getBoundary(self, entity, bdyinfo):
        return HelmholtzBoundary(self.problem.mesh, entity, bdyinfo, self.delta, self.facequads, self.facevandermondes, self.scaledvandermondes, self.entityton)

#    @ppd.parallelmethod()
    def volumeStiffness(self):
        ''' The contribution of the volume terms to the stiffness matrix (should be zero if using Trefftz basis functions)'''
        E = pms.ElementMatrices(self.problem.mesh)
        L2K = self.weightedassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
        H1 = self.volumeassembly.assemble([[E.Z,E.Z],[E.Z, E.I]])
        
        AJ = pms.AveragesAndJumps(self.problem.mesh)
        B = pms.sumfaces(self.problem.mesh, self.internalassembly.assemble([[AJ.Z,AJ.Z],[AJ.I,AJ.Z]]))
        
        return H1 - L2K - B   
