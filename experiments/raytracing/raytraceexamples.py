'''
Created on Apr 20, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.core.bases.reduced as pcbred
import pypwdg.utils.quadrature as puq
import pypwdg.core.bases.utilities as pcbu
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.basis as pob
import pypwdg.raytrace.control as prc
import pypwdg.raytrace.basisrules as prb
import pypwdg.output.mploutput as pom
import pypwdg.output.solution as pos
import pypwdg.parallel.main
import pypwdg.utils.file as puf
import pypwdg.core.bases.definitions as pcbb

from numpy import array,sqrt
import numpy as np
import math
from pypwdg.raytrace.basisrules import RaytracedShadowRule

class SourceBasisRule(object):
    def populate(self, einfo):
        cornerbases = []
        x,y = einfo.origin
        pos = 0.5
        lim = 0.4
        if x <=-lim or y >= lim:
            cornerbases.append(pcb.FourierHankel([-pos,pos], [0], einfo.k))
        if x <=-lim or y <= -lim:
            cornerbases.append(pcb.FourierHankel([-pos,-pos], [0], einfo.k))
        if x >=lim or y <= -lim:
            cornerbases.append(pcb.FourierHankel([pos,-pos], [0], einfo.k))
        if x >=lim or y >= lim:
            cornerbases.append(pcb.FourierHankel([pos, pos], [0], einfo.k))
        return cornerbases

class CrudeSourceBasisRule(object):
    def populate(self, einfo):
        return [pcb.FourierHankel(c, [0], einfo.k) for c in [[-0.5,0.5],[-0.5,-0.5], [0.5,-0.5], [0.5,0.5]]]

class CornerMultiplexBasisRule(object):
    def __init__(self, corner, elsewhere, radius):
        self.corner = corner
        self.elsewhere = elsewhere
        self.radius = radius
        self.corners = [[-0.5,0.5], [-0.5,-0.5], [0.5,-0.5], [0.5,0.5]]
        
    def populate(self, einfo):
        if np.any(np.sum((einfo.origin - self.corners)**2, axis=1) < self.radius **2):
            return self.corner.populate(einfo)
        else:
            return self.elsewhere.populate(einfo)

class FourierBesselRule(object):
        
    def populate(self, einfo):
        return [pcbb.FourierBessel(einfo.origin, [0], einfo.k)] 
        
def createproblem(k, direction=array([[1,1]])/sqrt(2)):
    g = pcb.PlaneWaves(direction, k)    
    bnddata={11:pcbd.zero_dirichlet(),
             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}
    with(puf.pushd("../../examples/2D")):
        mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    return psp.Problem(mesh, k, bnddata)
    
def traceproblem(k, direction=array([[1,1]])/sqrt(2)):
    problem = createproblem(k, direction)
    etods = prc.tracemesh(problem, {10:lambda x:direction})
    return problem, etods

def raytracesoln(problem, etods, pdeg = 2, npw = 15, radius = 0.5):
    quadpoints = 15
    k = problem.k
    rtpw = prb.OldRaytracedBasisRule(etods)
    
    fb = FourierBesselRule()
    shadow = RaytracedShadowRule(etods, fb)
    
    
    crt = SourceBasisRule()
        
    
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))

#   unonpw = pcbu.UnionBasisRule([shadow, crt])                         
#    polynonpw = pcb.ProductBasisRule(poly, unonpw)
#    basisrule = pcbu.UnionBasisRule([polynonpw,rtpw])
    
#    unionall = pcbu.UnionBasisRule([shadow, rtpw, crt])
#    polyall = pcb.ProductBasisRule(poly, unionall)
#    basisrule = polyall
    
    rt = pcbu.UnionBasisRule([rtpw, shadow])
    polyrt = pcb.ProductBasisRule(poly, rt)
    basisrule = polyrt
    
    
#    pw = pcb.planeWaveBases(2, k, nplanewaves=npw)
#    basisrule = CornerMultiplexBasisRule(pw, basisrule, radius)    
    
#    polyrt2 = pcbu.UnionBasisRule([rt,pcb.ProductBasisRule(poly, crt)])
    
#    basisrule = polyrt2
    basisrule = pcbred.SVDBasisReduceRule(puq.trianglequadrature(quadpoints), basisrule, threshold=1E-5)
    
    p = 1 if pdeg < 1 else pdeg
    h = 4.0/40
    alpha = ((p*1.0)**2 )  / (k*h)
    beta = np.min([(k*h) / (p * 1.0),1]) 
    computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints, alpha = alpha, beta = beta)
    
    solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
    bounds=array([[-2,2],[-2,2]],dtype='d')
    npoints=array([200,200])
    pos.standardoutput(computation, solution, 20, bounds, npoints, mploutput = True)
    

def polysoln(problem, pdeg):
    k = problem.k
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))                    
    h = 4.0/40
    p = 1 if pdeg < 1 else pdeg
    alpha = ((p*1.0)**2 )  / (k*h)
    beta = np.min([(k*h) / (p * 1.0),1]) 
    computation = psc.Computation(problem, poly, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
    
    solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
    bounds=array([[-2,2],[-2,2]],dtype='d')
    npoints=array([200,200])
    pos.standardoutput(computation, solution, 20, bounds, npoints, mploutput = True)


def planewavesoln(problem, npw):
    k = problem.k
    pw = pcb.planeWaveBases(2, k, nplanewaves=npw)
    h = 4.0/40
    alpha = 1 / (k*h)
    beta = np.min([(k*h), 1])  
    computation = psc.Computation(problem, pw, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
    
    solution = computation.solution(psc.DirectSolver().solve, dovolumes=False)
    bounds=array([[-2,2],[-2,2]],dtype='d')
    npoints=array([200,200])
    pos.standardoutput(computation, solution, 20, bounds, npoints, mploutput = True)


    
def showdirs():
    k = 10
    #direction=array([[1.0,1.0]])/sqrt(2)
    direction=array([[1,1]])/sqrt(2)
#    direction = array([[1,0]])
    mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    problem = psp.Problem(mesh, k, None)
    etods = prc.tracemesh(problem, {10:lambda x:direction})
    
    pom.showdirections2(problem.mesh, etods)
    pom.showmesh(problem.mesh)

#
#if __name__ == "__main__":
#    problem, etods = traceproblem(k=30)
#    raytracesoln(problem, etods, pdeg = 3)
#    pom.mp.show()
