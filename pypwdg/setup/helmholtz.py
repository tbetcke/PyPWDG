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
def localConstructBasis(etob, basisrule, problem):
    problem.populate(etob, basisrule)
    
def constructBasis(problem, basisrule):
    ''' Build an element to basis (distributed) map based on a basisrule'''
    manager = ppdd.ddictmanager(ppdd.elementddictinfo(problem.mesh), True)
    etob = manager.getDict()
    localConstructBasis(etob, basisrule, problem)
    manager.sync()   
    return pcbu.ElementToBases(etob, problem.mesh)    

def computation(klazz, problem, basisrule, *args, **kwargs):
    basis = constructBasis(problem, basisrule)
    return klazz(problem, basis, *args, **kwargs)
    
@ppd.distribute()    
class Computation(object):
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
            bdyetob = constructBasis(problem.mesh, pcbu.UniformBasisRule([data]))
            bdyvandermondes = pcv.LocalVandermondes(problem.mesh, bdyetob, facequads)        
            self.bdyvandermondes.append(bdyvandermondes)
            self.loadassemblies.append(pca.Assembly(self.facevandermondes, bdyvandermondes, facequads.quadweights))
        
        ev = pcv.ElementVandermondes(problem.mesh, self.basis, elementquads)
        l2weights = lambda e: elementquads.quadweights(e) * problem.elementinfo.kp(e)(elementquads.quadpoints(e))
        self.L2 = pcv.LocalInnerProducts(ev.getValues, ev.getValues, l2weights)
        self.H1 = pcv.LocalInnerProducts(ev.getDerivs, ev.getDerivs, elementquads.quadweights, ((0,2),(0,2)))

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
            
            SB = self.stiffassembly.assemble([[lv*(1-delta) * B, (-1+(1-delta)*ld)*B],
                                              [(1-delta*lv) * B, -delta * ld*B]])
            SBs.append(pms.sumfaces(self.problem.mesh,SB))     
        return SBs
    
    @ppd.parallelmethod(None, ppd.tuplesum)
    def boundaryLoad(self): 
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
        AJ = pms.AveragesAndJumps(self.problem.mesh)    
        L2P = pus.createvbsr(csrelts, L2.product, elttobasis.getSizes(), elttobasis.getSizes())
        H1P = pus.createvbsr(csrelts, H1.product, elttobasis.getSizes(), elttobasis.getSizes())
        
        Zero = ss.csr_matrix((mesh.nfaces, mesh.nfaces))
        B = pms.sumfaces(mesh, stiffassembly.assemble([[Zero,Zero],[mesh.facepartition,Zero]]))
        
        return H1P - k**2 * L2P - B
        