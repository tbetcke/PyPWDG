import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array,sqrt

k = 80
direction=array([[1.0,1.0]])/sqrt(2)
g = pcb.PlaneWaves(direction, k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impbd, 
#         8:impbd}
bnddata={7:pcbd.dirichlet(g), 
         8:pcbd.dirichlet(g)}

bounds=array([[0,1],[0,1]],dtype='d')
npoints=array([100,100])

mesh = pmm.gmshMesh('square.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=16)

problem=ps.Problem(mesh, k, 20, bnddata)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond=None, part=2)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="diag", part=2)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="block_diag", part=2)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="block_diag", part=4)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="block_diag", part=8)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="block_diag", part=16)
solution = ps.Computation(problem, bases).solve(solver='gmres', precond="block_diag", part="elms")

solution.writeSolution(bounds, npoints, fname='square.vti')
problem.writeMesh(fname='square.vtu',scalars=solution.combinedError())


