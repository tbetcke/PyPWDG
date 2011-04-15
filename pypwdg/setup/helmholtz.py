'''
Created on Apr 14, 2011

@author: joel
'''
import pypwdg.core.bases.utilities as pcbu
import pypwdg.parallel.decorate as ppd
import pypwdg.parallel.distributeddict as ppdd
import pypwdg.utils.quadrature as puq
import pypwdg.mesh.meshutils as pmmu
import pypwdg.core.vandermonde as pcv
import pypwdg.core.assembly as pca
import pypwdg.mesh.structure as pms
import pypwdg.core.evaluation as pce

import numpy as np

class Problem(object):
    def __init__(self, mesh, k, bnddata):
        self.mesh = mesh
        self.k = k
        self.bnddata = bnddata
        self.elementinfo = pcbu.ElementInfo(mesh, k) 
    
    def populateBasis(self, etob, basisrule):
        ''' Helper function to initialise the element to basis map in each partition'''  
        for e in self.mesh.partition:
            etob[e] = basisrule.populate(self.elementinfo.info(e))
    

@ppd.parallel(None, None)
def localPopulateBasis(etob, basisrule, problem):
    problem.populateBasis(etob, basisrule)

def constructBasis(problem, basisrule):
    ''' Build an element to basis (distributed) map based on a basisrule'''
    manager = ppdd.ddictmanager(ppdd.elementddictinfo(problem.mesh), True)
    etob = manager.getDict()
    localPopulateBasis(etob, basisrule, problem)
    manager.sync()   
    return pcbu.ElementToBases(etob, problem.mesh)

class Computation(object):
    def __init__(self, problem, basisrule, systemklass, *args, **kwargs):
        self.problem = problem
        self.basis = constructBasis(problem, basisrule)        
        self.system = systemklass(problem, self.basis, *args, **kwargs)
                
    def solution(self, solver, *args, **kwargs):
        S,G = self.system.getSystem(*args, **kwargs)                
        x = solver(S.tocsr(), G.todense().squeeze())
        return Solution(self.problem, self.basis, x)        

    
@ppd.distribute()    
class HelmholtzSystem(object):
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
        kweights = lambda e: elementquads.quadweights(e) * (problem.elementinfo.kp(e)(elementquads.quadpoints(e))**2)
        self.weightedassembly = pca.Assembly(ev, ev, kweights)

    @ppd.parallelmethod(None, ppd.tuplesum)        
    def getSystem(self, dovolumes = False):
        S = self.internalStiffness() + sum(self.boundaryStiffnesses())
        G = sum(self.boundaryLoads())
        if dovolumes: 
            S+=self.volumeStiffness()
        return S,G

    @ppd.parallelmethod()        
    def internalStiffness(self):
    
        jk = 1j * self.problem.k
        AJ = pms.AveragesAndJumps(self.problem.mesh)    
        SI = self.internalassembly.assemble([[jk * self.alpha * AJ.JD,   -AJ.AN], 
                                            [AJ.AD,                -(self.beta / jk) * AJ.JN]])        
        return pms.sumfaces(self.problem.mesh,SI)
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryStiffnesses(self):
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
        E = pms.ElementMatrices(self.problem.mesh)
        L2K = self.weightedassembly.assemble([[E.I, E.Z],[E.Z, E.Z]])
        H1 = self.volumeassembly.assemble([[E.Z,E.Z],[E.Z, E.I]])
        
        AJ = pms.AveragesAndJumps(self.problem.mesh)
        B = pms.sumfaces(self.problem.mesh, self.internalassembly.assemble([[AJ.Z,AJ.Z],[AJ.I,AJ.Z]]))
        
        return H1 - L2K - B   

def noop(x): return x

class Solution(object):
    """ The solution to a Problem """
    def __init__(self, problem, basis, x):  
        self.mesh = problem.mesh
        self.elttobasis = basis
        self.x = x
#        self.lv = lvs
#        self.bndvs = bndvs
#        self.error_dirichlet2=None
#        self.error_neumann2=None
#        self.error_boundary2=None
#        self.error_combined=None
        
              
    def writeSolution(self, bounds, npoints, realdata=True, fname='solution.vti'):
        from pypwdg.output.vtk_output import VTKStructuredPoints

        print "Evaluate Solution and Write to File"
        
        bounds=np.array(bounds,dtype='d')
        filter=np.real if realdata else np.imag

        vtk_structure=VTKStructuredPoints(pce.StructuredPointsEvaluator(self.mesh, self.elttobasis, filter, self.x))
        vtk_structure.create_vtk_structured_points(bounds,npoints)
        vtk_structure.write_to_file(fname)
   
    def evaluate(self, structuredpoints):
        spe = pce.StructuredPointsEvaluator(self.problem.mesh, self.elttobasis, noop, self.x)
        vals, count = spe.evaluate(structuredpoints)
        count[count==0] = 1
        return vals / count
   
    def evalJumpErrors(self):
        print "Evaluate Jumps"
        (self.error_dirichlet2, self.error_neumann2, self.error_boundary2) = pce.EvalElementError3(self.problem.mesh, self.problem.mqs, self.lv, self.problem.bnddata, self.bndvs).evaluate(self.x)

    def combinedError(self):        
        if self.error_dirichlet2 is None: self.evalJumpErrors()
        error_combined2 = self.problem.k ** 2 * self.error_dirichlet2 + self.error_neumann2 + self.error_boundary2
        self.error_combined = np.sqrt(error_combined2)
        return self.error_combined    
          