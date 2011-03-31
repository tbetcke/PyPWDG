import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)

bnddata={11:pcbd.zero_dirichlet(),
         10:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g)}

bounds=array([[-2,2],[-2,2]],dtype='d')
npoints=array([200,200])

mesh = pmm.gmshMesh('squarescatt.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=15)

problem=ps.Problem(mesh,k,20, bnddata)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond='block_diag')
solution.writeSolution(bounds,npoints,fname='soundsoft.vti')
problem.writeMesh(fname='soundsoft.vtu',scalars=solution.combinedError())

