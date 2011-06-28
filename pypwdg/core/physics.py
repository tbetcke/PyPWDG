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
    
@ppd.distribute()    
class HelmholtzSystem(object):
    ''' Assemble a system to solve a Helmholtz problem using the DG formulation of Gittelson et al.
        It's parallelised so will only assemble the relevant bits of the system for the partition managed by
        this process.
    '''
    def __init__(self, problem, basis, nquadpoints, alpha=0.5,beta=0.5,delta=0.5, usecache=True):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.problem = problem
        fquad, equad = puq.quadrules(problem.mesh.dim, nquadpoints)
        facequads = pmmu.MeshQuadratures(problem.mesh, fquad)
        elementquads = pmmu.MeshElementQuadratures(problem.mesh, equad)

        self.basis = basis
        self.facevandermondes = pcv.LocalVandermondes(problem.mesh, basis, facequads, usecache=usecache)
        self.internalassembly = pca.Assembly(self.facevandermondes, self.facevandermondes, facequads.quadweights)
         
        self.bdyvandermondes = []
        self.loadassemblies = []
        for data in problem.bnddata.values():
            bdyetob = pcbu.UniformElementToBases(data, problem.mesh)
            bdyvandermondes = pcv.LocalVandermondes(problem.mesh, bdyetob, facequads)        
            self.bdyvandermondes.append(bdyvandermondes)
            self.loadassemblies.append(pca.Assembly(self.facevandermondes, bdyvandermondes, facequads.quadweights))
        
        ev = pcv.ElementVandermondes(problem.mesh, self.basis, elementquads)
        self.volumeassembly = pca.Assembly(ev, ev, elementquads.quadweights)
        def kweights(e): 
            return elementquads.quadweights(e).squeeze()* (problem.elementinfo.kp(e)(elementquads.quadpoints(e))**2)
        self.weightedassembly = pca.Assembly(ev, ev, kweights)

    @ppd.parallelmethod(None, ppd.tuplesum)        
    def getSystem(self, dovolumes = False):
        ''' Returns the stiffness matrix and load vector'''
        S = self.internalStiffness() + sum(self.boundaryStiffnesses())
        G = sum(self.boundaryLoads())
        if dovolumes: 
            S+=self.volumeStiffness()
        return S,G

    @ppd.parallelmethod()        
    def internalStiffness(self):
        ''' The contribution of the internal faces to the stiffness matrix'''
        jk = 1j * self.problem.k
        AJ = pms.AveragesAndJumps(self.problem.mesh)    
        SI = self.internalassembly.assemble([[jk * self.alpha * AJ.JD,   -AJ.AN], 
                                            [AJ.AD,                -(self.beta / jk) * AJ.JN]])        
        return pms.sumfaces(self.problem.mesh,SI)
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryStiffnesses(self):
        ''' The contribution of the boundary faces to the stiffness matrix'''
        SBs = []
        for (id, bdycondition) in self.problem.bnddata.items():
            B = self.problem.mesh.entityfaces[id]
            
            lv, ld = bdycondition.l_coeffs
            delta = self.delta
            
            SB = self.internalassembly.assemble([[lv*(1-delta) * B, (-1+(1-delta)*ld)*B],
                                              [(1-delta*lv) * B, -delta * ld*B]])
            SBs.append(pms.sumfaces(self.problem.mesh,SB))     
        return SBs
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryLoads(self): 
        ''' The load vector (due to the boundary conditions)'''
        GBs = []
        for (id, bdycondition), loadassembly in zip(self.problem.bnddata.items(), self.loadassemblies):
            B = self.problem.mesh.entityfaces[id]
            
            rv, rd = bdycondition.r_coeffs
            delta = self.delta
            # todo - check the cross terms.  Works okay with delta = 1/2.  
            GB = loadassembly.assemble([[(1-delta) *rv* B, (1-delta) * rd*B], 
                                        [-delta*rv* B,     -delta * rd*B]])
                
            GBs.append(pms.sumrhs(self.problem.mesh,GB))
        return GBs
    
    @ppd.parallelmethod()
    def volumeStiffness(self):
        ''' The contribution of the volume terms to the stiffness matrix (should be zero if using Trefftz basis functions)'''
        E = pms.ElementMatrices(self.problem.mesh)
        L2K = self.weightedassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
        H1 = self.volumeassembly.assemble([[E.Z,E.Z],[E.Z, E.I]])
        
        AJ = pms.AveragesAndJumps(self.problem.mesh)
        B = pms.sumfaces(self.problem.mesh, self.internalassembly.assemble([[AJ.Z,AJ.Z],[AJ.I,AJ.Z]]))
        
        return H1 - L2K - B   
