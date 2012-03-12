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
    def __init__(self, mesh, bdycond, facequads, entityton):
        bdyetob = pcbu.UniformElementToBases(bdycond, mesh)
        bdyvandermondes = pcv.LocalVandermondes(mesh, bdyetob, facequads) if entityton is None else pcv.ScaledVandermondes(entityton, mesh, bdyetob, facequads)  
        rc0,rc1 = bdycond.r_coeffs
        rqw0 = pmmu.ScaledQuadweights(facequads, rc0)     
        rqw1 = pmmu.ScaledQuadweights(facequads, rc1)     
        self.loadassemblies = pca.Assembly(self.facevandermondes, bdyvandermondes, [[rqw0,rqw1],[rqw0,rqw1]])
        lc0,lc1 = bdycond.l_coeffs
        lqw0 = pmmu.ScaledQuadweights(facequads, lc0)
        lqw1 = pmmu.ScaledQuadweights(facequads, lc1)
        self.weightedbdyassemblies = pca.Assembly(self.facevandermondes, self.scaledvandermondes, [[lqw0,lqw1],[lqw0,lqw1]])
    
            

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
        fquad, equad = puq.quadrules(problem.mesh.dim, nquadpoints)
        facequads = pmmu.MeshQuadratures(problem.mesh, fquad)
        elementquads = pmmu.MeshElementQuadratures(problem.mesh, equad)

        self.basis = basis
        self.facevandermondes = pcv.LocalVandermondes(problem.mesh, basis, facequads, usecache=usecache)
        self.scaledvandermondes = self.facevandermondes if entityton is None else pcv.ScaledVandermondes(entityton, problem.mesh, basis, facequads, usecache=usecache)
        self.internalassembly = pca.Assembly(self.facevandermondes, self.facevandermondes, facequads.quadweights)
         
        self.loadassemblies = {}
        self.weightedbdyassemblies = {}
        for entity, (bdycoeffs, bdyetob) in problem.bdyinfo.items():
            bdyvandermondes = pcv.LocalVandermondes(problem.mesh, bdyetob, facequads) if entityton is None else pcv.ScaledVandermondes(entityton, problem.mesh, bdyetob, facequads)  
            rc0,rc1 = bdycoeffs.r_coeffs
            rqw0 = pmmu.ScaledQuadweights(facequads, rc0)     
            rqw1 = pmmu.ScaledQuadweights(facequads, rc1)     
            self.loadassemblies[entity] = pca.Assembly(self.facevandermondes, bdyvandermondes, [[rqw0,rqw1],[rqw0,rqw1]])
            lc0,lc1 = bdycoeffs.l_coeffs
            lqw0 = pmmu.ScaledQuadweights(facequads, lc0)
            lqw1 = pmmu.ScaledQuadweights(facequads, lc1)
            self.weightedbdyassemblies[entity] = pca.Assembly(self.facevandermondes, self.scaledvandermondes, [[lqw0,lqw1],[lqw0,lqw1]])
        
        ev = pcv.ElementVandermondes(problem.mesh, self.basis, elementquads)
        self.volumeassembly = pca.Assembly(ev, ev, elementquads.quadweights)
        def kweights(e): 
            return elementquads.quadweights(e).squeeze()* (problem.elementinfo.kp(e)(elementquads.quadpoints(e))**2)
        self.weightedassembly = pca.Assembly(ev, ev, kweights)

    @ppd.parallelmethod(None, ppd.tuplesum)        
    def getSystem(self, dovolumes = False):
        ''' Returns the stiffness matrix and load vector'''
        SI = self.internalStiffness() 
        SB = sum(self.boundaryStiffnesses())
        S = SI + SB
        G = sum([self.boundaryLoad(entity) for entity in self.loadassemblies.keys()])
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
    
#    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryStiffnesses(self):
        ''' The contribution of the boundary faces to the stiffness matrix'''
        SBs = []
        for i, bdya in self.weightedbdyassemblies.items():
            B = self.problem.mesh.entityfaces[i]
            delta = self.delta
            
            # The contribution of the boundary conditions
            SB = bdya.assemble([[(1-delta) * B, (1-delta)*B],
                                  [-delta * B, -delta *B]])
            SBs.append(pms.sumfaces(self.problem.mesh,SB))     
        return SBs
    
    
#    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryLoad(self, entity): 
        ''' The load vector (due to the boundary conditions)'''
        loadassembly = self.loadassemblies[entity]
        B = self.problem.mesh.entityfaces[entity]
            
        delta = self.delta
        # todo - check the cross terms.  Works okay with delta = 1/2.  
        GB = loadassembly.assemble([[(1-delta) * B, (1-delta) *B], 
                                    [-delta* B,     -delta *B]])
                
        return pms.sumrhs(self.problem.mesh,GB)
    
#    @ppd.parallelmethod()
    def volumeStiffness(self):
        ''' The contribution of the volume terms to the stiffness matrix (should be zero if using Trefftz basis functions)'''
        E = pms.ElementMatrices(self.problem.mesh)
        L2K = self.weightedassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
        H1 = self.volumeassembly.assemble([[E.Z,E.Z],[E.Z, E.I]])
        
        AJ = pms.AveragesAndJumps(self.problem.mesh)
        B = pms.sumfaces(self.problem.mesh, self.internalassembly.assemble([[AJ.Z,AJ.Z],[AJ.I,AJ.Z]]))
        
        return H1 - L2K - B   
