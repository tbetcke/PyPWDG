import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array

k = 10
direction=array([[1.0,0]])
g = pcb.PlaneWaves(direction, k)

bnddata={15:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)}

bounds=array([[-4,4],[-4,4]],dtype='d')
npoints=array([500,500])

mesh = pmm.gmshMesh('two_circles.msh',dim=2)
bases = pcb.planeWaveBases(mesh,k,nplanewaves=20)
bases.setRefractive({11:1.0,12:1.2})

problem=ps.Problem(mesh,k,20, bnddata)
solution = ps.Computation(problem, bases).solve()
solution.writeSolution(bounds,npoints,fname='two_circles.vti')
problem.writeMesh(fname='two_circles.vtu',scalars=solution.combinedError())
