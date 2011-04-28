import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
import pypwdg.parallel.main

from numpy import array,sqrt

k = 40
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={13:pcbd.zero_dirichlet(),
         12:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('circscatt.msh',dim=2)
problem = psp.Problem(mesh, k, bnddata)

quadpoints = 15

computation = psc.Computation(problem, pcb.planeWaveBases(2,k,9), pcp.HelmholtzSystem, quadpoints)
#computation = psc.Computation(problem, pcb.FourierBesselBasisRule(range(-4,5)), pcp.HelmholtzSystem, quadpoints)
solution = computation.solution(psc.DirectSolver().solve)
pos.standardoutput(computation, solution, 20, bounds, npoints, 'circle')