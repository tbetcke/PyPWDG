'''
Created on Apr 15, 2011

@author: joel
'''
from pypwdg.parallel.decorate import parallelmethod, distribute, tuplesum

import pypwdg.mesh.structure as pms
import pypwdg.utils.sparse as pus
import numpy as np
import pypwdg.core.vandermonde as pcv
import pypwdg.mesh.meshutils as pmmu
import pypwdg.core.bases.utilities as pcbu
import pypwdg.utils.quadrature as puq

@distribute()
class EvalElementError(object):
    def __init__(self, mesh, quadpoints, basis, bnddata):
        self.mesh = mesh 
        fquad, _ = puq.quadrules(mesh.dim, quadpoints)
        facequads = pmmu.MeshQuadratures(mesh, fquad)

        self.vs = pcv.LocalVandermondes(mesh, basis, facequads)
        self.weights = facequads.quadweights
        self.bnddata = bnddata
        self.bndvs = []
        for data in bnddata.values():
            bdyetob = pcbu.UniformElementToBases(data, mesh)
            bdyvandermondes = pcv.LocalVandermondes(mesh, bdyetob, facequads)        
            self.bndvs.append(bdyvandermondes)

            
    @parallelmethod(reduceop = tuplesum)
    def evaluate(self, x):
        facemap = self.mesh.connectivity*np.arange(self.mesh.nfaces)
        
        nf = len(self.vs.indices) # number of faces        
        xf = lambda f: x[self.vs.indices[f]:self.vs.indices[f]+self.vs.numbases[f]] # coefficients for each (2-sided) face                        
        vx = lambda f: np.dot(self.vs.getValues(f),xf(f)) # dirichlet values of solution at the quadrature points on each (2-sided) face        
        evx = lambda f: vx(f) - vx(facemap[f]) # error on each face        
        DD = pcv.LocalInnerProducts(evx, evx, self.weights) # L2 inner product of solns for pairs of 2-sided faces (only makes sense when faces are from same pair)
        ipD = pus.createvbsr(self.mesh.internal, DD.product, np.ones(nf), np.ones(nf))
        elem_error_dirichlet = pms.sumrhs(self.mesh, ipD.tocsr()).todense().A.squeeze()
        
        nx = lambda f: np.dot(self.vs.getDerivs(f), xf(f))
        enx = lambda f: nx(f) + nx(facemap[f])
        NN = pcv.LocalInnerProducts(enx,enx,self.weights)
        ipN = pus.createvbsr(self.mesh.internal, NN.product, np.ones(nf), np.ones(nf))
        elem_error_neumann = pms.sumrhs(self.mesh, ipN.tocsr()).todense().A.squeeze()
        
        elem_error_bnd=np.zeros(self.mesh.nelements, dtype=complex)
        
        for (id, bdycondition), bndv in zip(self.bnddata.items(), self.bndvs):
            lc=bdycondition.l_coeffs
            rc=bdycondition.r_coeffs
            # boundary error
            be = lambda f: lc[0] * vx(f) + lc[1] * nx(f) - (rc[0] * bndv.getValues(f) + rc[1] * bndv.getDerivs(f)).squeeze()
            # l2norm of boundary error on faces
            BB = pcv.LocalInnerProducts(be,be,self.weights)
            ipB = pus.createvbsr(self.mesh.entityfaces[id], BB.product, np.ones(nf), np.ones(nf))
            elem_error_bnd += pms.sumrhs(self.mesh, ipB.tocsr()).todense().A.squeeze()
        
        return elem_error_dirichlet.real, elem_error_neumann.real, elem_error_bnd.real
    
def combinedError(problem, solution, quadpoints, x):
    
    (error_dirichlet2, error_neumann2, error_boundary2) = EvalElementError(problem.mesh, quadpoints, solution.basis, problem.bnddata).evaluate(x)
    error_combined2 = error_dirichlet2 + (error_neumann2 + error_boundary2)/(problem.k ** 2)
    return map(np.sqrt, (error_combined2,error_dirichlet2, error_neumann2, error_boundary2))     

    