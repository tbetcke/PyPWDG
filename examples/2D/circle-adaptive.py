import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.adaptivity.adaptivity2 as paa
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
import pypwdg.parallel.main

from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={13:pcbd.zero_dirichlet(),
         12:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('circscatt.msh',dim=2)

quadpoints = 10

problem=psp.Problem(mesh,k, bnddata)
etods = prc.tracemesh(problem, {12:lambda x:direction})
controller = paa.BasisController(mesh, quadpoints, etods, k, nfb=5)
computation = paa.AdaptiveComputation(problem, controller, pcp.HelmholtzSystem, quadpoints, 1)
computation.solve(psc.DirectSolver().solve, 6, pos.AdaptiveOutput1(computation, quadpoints, bounds, npoints, "circscatt").output)


#problem=psp.Problem(mesh,k, bnddata)
#ibc = paa.InitialPWFBCreator(mesh,k,3,7)
#computation = paa.AdaptiveComputation(problem, ibc, pcp.HelmholtzSystem, quadpoints, 1)
#computation.solve(psc.DirectSolver().solve, 6, pos.AdaptiveOutput1(computation, quadpoints, bounds, npoints, "squarescatt").output)

