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

k = 30
direction=array([[1.0]])
g = pcb.PlaneWaves(direction, k)

bnddata={10:pcbd.dirichlet(g),
         11:pcbd.zero_impedance(k)}

bounds=array([[0,1]],dtype='d')
npoints=array([200])

mesh = pmm.lineMesh(nelems=[2])
problem = psp.Problem(mesh, k, bnddata)
computation = psc.Computation(problem, pcb.planeWaveBases(1,k), pcp.HelmholtzSystem, 15)
solution = computation.solution(psc.DirectSolver().solve)
pos.standardoutput(computation, solution, 20, bounds, npoints, 'soundsoft')