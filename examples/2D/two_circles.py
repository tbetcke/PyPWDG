import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.core.bases.reference as pcbr
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.output.solution as pos
import pypwdg.core.bases.variable as pcbv
import pypwdg.core.physics as pcp
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import numpy as np

class QuadBubble(object):
    ''' An example of variable N.  Quadratic in r, equal to 1 on R and a at 0.     
    
        sadly, because of pickling, this can't be a lambda function and has to be before pypwdg.parallel.main 
    '''
    def __init__(self, R, a):
        self.R = R
        self.a = a
    
    def __call__(self, p):
        r2 = np.sum(p**2, axis=1)
        return self.a - r2 * (self.a-1.0) / self.R**2
    
import pypwdg.parallel.main

from numpy import array

k = 30
direction=array([[1.0,0]])
g = pcb.PlaneWaves(direction, k)

bnddata={15:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)}

bounds=array([[-4,4],[-4,4]],dtype='d')
npoints=array([500,500])

mesh = pmm.gmshMesh('two_circles.msh',dim=2)

quadpoints = 20

def elementwiseconstant():
    npw = 12
    basisrule = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw))  
    entityton = {11:1.0,12:1.5}
    
    problem = psp.VariableNProblem(entityton, mesh, k, bnddata)
    computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints)
    solution = computation.solution(psc.DirectSolver().solve)
    
    pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'twocirclesEWC')
    
def fullyvariable():
    npw = 12
    basisrule = pcb.ProductBasisRule(pcbv.PlaneWaveVariableN(pcb.circleDirections(npw)), pcbr.ReferenceBasisRule(pcbr.Dubiner(1)))
#    basisrule = pcbv.PlaneWaveVariableN(pcb.circleDirections(npw*3))  
    
    entityton = {11:1.0, 12:QuadBubble(1.0, 2.0)}
    problem = psp.VariableNProblem(entityton, mesh, k, bnddata)
    computation = psc.Computation(problem, basisrule, pcp.HelmholtzSystem, quadpoints)
    solution = computation.solution(psc.DirectSolver().solve, dovolumes=True)
    
    pos.standardoutput(computation, solution, quadpoints, bounds, npoints, 'twocirclesFV')

#elementwiseconstant()    
fullyvariable()
    
