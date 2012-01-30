'''
Created on Apr 20, 2011

@author: joel
'''
import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
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

from numpy import array,sqrt

class CornerBasisRule(object):
    def populate(self, einfo):
        cornerbases = []
        x,y = einfo.origin
        if x <=-0.5 or y >= 0.5:
            cornerbases.append(pcb.FourierHankel([-0.5,0.5], [0], einfo.k))
        if x <=-0.5 or y <= -0.5:
            cornerbases.append(pcb.FourierHankel([-0.5,-0.5], [0], einfo.k))
        if x >=0.5 or y <= -0.5:
            cornerbases.append(pcb.FourierHankel([0.5,-0.5], [0], einfo.k))
        return cornerbases

def solve(problem, pdeg, rt, solve, dovolumes = True):
    p = 1 if pdeg < 1 else pdeg
    h = 4.0/40
    alpha = ((p*1.0)**2 )  / (k*h)
    beta = (k*h) / (p * 1.0) 
    poly = pcbr.ReferenceBasisRule(pcbr.Dubiner(pdeg))                    
    polyrt = pcb.ProductBasisRule(poly, rt)

    computation = psc.Computation(problem, polyrt, pcp.HelmholtzSystem, 15, alpha = alpha, beta = beta)
    solution = computation.solution(solve, dovolumes=dovolumes)
    bounds=array([[-2,2],[-2,2]],dtype='d')
    npoints=array([200,200])
    pos.standardoutput(computation, solution, 20, bounds, npoints, mploutput = True)

def setup(k, direction):
    g = pcb.PlaneWaves(direction, k)    
    bnddata={11:pcbd.zero_dirichlet(),
             10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

    mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    problem = psp.Problem(mesh, k, bnddata)
    etods = prc.tracemesh(problem, {10:lambda x:direction})
    rt = prb.OldRaytracedBasisRule(etods)
    c = CornerBasisRule()
    return problem, pcbu.UnionBasisRule([rt,c])
    
def showdirs():
    k = 10
    #direction=array([[1.0,1.0]])/sqrt(2)
    direction=array([[3,4]])/5.0
#    direction = array([[1,0]])
    mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
    problem = psp.Problem(mesh, k, None)
    etods = prc.tracemesh(problem, {10:lambda x:direction})
    
    pom.showdirections2(problem.mesh, etods)
    pom.showmesh(problem.mesh)

def dosolution():    
    k = 10
    #direction=array([[1.0,1.0]])/sqrt(2)
    direction=array([[3,4]])/5.0
    problem,rule, rt = setup(k, direction)
    solution = solve(problem, 2, rt, psc.DirectSolver().solve)
    
    pom.mp.show()