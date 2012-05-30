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
class EvalElementResiduals(object):
    def __init__(self, problem, quadpoints, basis):
        _, self.equad = puq.quadrules(problem.mesh.dim, quadpoints)
        self.basis = basis
        self.problem = problem
        
    @parallelmethod()
    def evaluate(self, x):
        ei = pcbu.ElementInfo(self.problem.mesh, self.problem.k)
        idxs = self.basis.getIndices()
        elem_error=np.zeros(self.problem.mesh.nelements)        
        for e in self.problem.mesh.partition:
            xe = x[idxs[e]:idxs[e+1]]
            info = ei.info(e)
            p,w = info.volume(self.equad)
            v = np.dot(self.basis.getValues(e, p), xe)
            l = np.dot(self.basis.getLaplacian(e, p), xe)
            k = info.kp(p)
            pres = l + k**2 * v
            elem_error[e] = np.vdot(pres, pres * w.ravel())
        return elem_error
    
def volumeerrors(solution, quadpoints):
    return np.sqrt(EvalElementResiduals(solution.computation.problem, quadpoints, solution.computation.basis).evaluate(solution.x))


@distribute()
class EvalElementError(object):
    def __init__(self, computation):
        self.mesh = computation.problem.mesh 
        self.vs = computation.facevandermondes
        self.scaledvandermondes = self.vs # if entityton is None else pcv.ScaledVandermondes(entityton, mesh, basis, facequads)
        self.computation = computation
        self.weights = computation.facequads.quadweights
            
    @parallelmethod(reduceop = tuplesum)
    def evaluate(self, x):
        facemap = self.mesh.connectivity*np.arange(self.mesh.nfaces)
        
        nf = len(self.vs.indices) # number of faces        
        xf = lambda f: x[self.vs.indices[f]:self.vs.indices[f]+self.vs.numbases[f]] # coefficients for each (2-sided) face                        
        vx = lambda f: np.dot(self.vs.getValues(f),xf(f)) # dirichlet values of solution at the quadrature points on each (2-sided) face        
        svx = lambda f: np.dot(self.scaledvandermondes.getValues(f),xf(f)) # dirichlet values of solution at the quadrature points on each (2-sided) face        
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
        
        for entity, (coeffs, bdyftob) in self.computation.problem.bdyinfo.items():
            lc=coeffs.l_coeffs
            rc=coeffs.r_coeffs
            bndv = self.computation.faceVandermondes(bdyftob)
            # boundary error
            be = lambda f: lc[0] * svx(f) + lc[1] * nx(f) - (rc[0] * bndv.getValues(f) + rc[1] * bndv.getDerivs(f)).squeeze()
            # l2norm of boundary error on faces
            BB = pcv.LocalInnerProducts(be,be,self.weights)
            ipB = pus.createvbsr(self.mesh.entityfaces[entity], BB.product, np.ones(nf), np.ones(nf))
            elem_error_bnd += pms.sumrhs(self.mesh, ipB.tocsr()).todense().A.squeeze()
        
        return elem_error_dirichlet.real, elem_error_neumann.real, elem_error_bnd.real
    
def combinedError(solution):
    
    (error_dirichlet2, error_neumann2, error_boundary2) = EvalElementError(solution.computation).evaluate(solution.x)
    error_combined2 = error_dirichlet2 + error_boundary2 + error_neumann2/(solution.computation.problem.k ** 2)
    return map(np.sqrt, (error_combined2,error_dirichlet2, error_neumann2, error_boundary2))     
    

    