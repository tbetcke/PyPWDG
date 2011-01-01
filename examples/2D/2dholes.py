import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array

k = 10
direction=array([[1.0,0]])
g = pcb.PlaneWaves(direction, k)

bnddata={21:pcbd.zero_dirichlet(),
         18:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g),
         20:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g),
         19:pcbd.zero_dirichlet()}

bounds=array([[0,5],[0,1]],dtype='d')
npoints=array([501,101])

mesh = pmm.gmshMesh('2dhole.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=15)


problem=ps.Problem(mesh,k,20, bnddata)
solution = ps.Computation(problem, bases).solve()
solution.writeSolution(bounds,npoints,fname='2dhole.vti')
problem.writeMesh(fname='2dhole.vtu',scalars=solution.combinedError())
