import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
import pypwdg.core.bases.reference as pcbr

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

def expBubble(x):
    
    return np.exp(-(1-x**2))
    
entityton = {1:expBubble}

import pypwdg.parallel.main

from numpy import array,sqrt


k = 50
direction=array([[1.0]])
g = pcb.PlaneWaves(direction, k)

bnddata={10:pcbd.dirichlet(g),
         11:pcbd.zero_impedance(k)}

bounds=array([[-1,1]],dtype='d')
npoints=array([200])

mesh = pmm.lineMesh(points=[-1,1],nelems=[1000],physIds=[1])
problem = psp.VariableNProblem(entityton,mesh, k, bnddata)
#computation = psc.Computation(problem, pcb.planeWaveBases(1,k), pcp.HelmholtzSystem, 15)
computation = psc.Computation(problem, pcbr.ReferenceBasisRule(pcbr.Legendre1D(2)), pcp.HelmholtzSystem, 15)
solution = computation.solution(psc.DirectSolver().solve,dovolumes=True)
pos.standardoutput(computation, solution, 20, bounds, npoints, 'soundsoft')
