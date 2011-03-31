import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.parallel.main
import pypwdg.utils.geometry as pug

import numpy as np

k = 80
direction=np.array([[1.0,1.0]])/np.sqrt(2)
#g = pcb.PlaneWaves(direction, k)
g = pcb.FourierHankel([-2,-2], [0], k)
impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impbd, 
#         8:impbd}
bnddata={7:pcbd.dirichlet(g), 
         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([500,500])

sp = pug.StructuredPoints(bounds.transpose(), npoints)
idx, points = sp.getPoints(bounds.transpose())
gvals = np.zeros(sp.length, dtype=complex)
gvals[idx] = g.values(points)
l2g = np.sqrt(np.vdot(gvals, gvals) / sp.length)

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


