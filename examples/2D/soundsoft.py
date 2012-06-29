import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.setup.problem as psp
import pypwdg.setup.computation as psc
import pypwdg.core.physics as pcp
import pypwdg.output.solution as pos
import pypwdg.raytrace.control as prc
import pypwdg.parallel.main
import pypwdg.output.mploutput as pom


from numpy import array,sqrt

k = 50
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)
nq = 11

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([500,500])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
problem = psp.Problem(mesh, k, bnddata)
computation = psc.DirectComputation(problem, pcb.planeWaveBases(2,k,25), nq, pcp.HelmholtzSystem, alpha=0.5, beta=0.0, delta=0.0)
solution = computation.solution()
pos.standardoutput(solution, 20, bounds, npoints, 'soundsoft')
pom.outputMeshPartition(bounds, npoints, mesh, 4)
pom.output2dsoln(bounds, solution, npoints, plotmesh=False)

