import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.bases.reference as pcbr
import pypwdg.parallel.main
import pypwdg.utils.geometry as pug

import numpy as np

k = 15
direction=np.array([[1.0,1.0]])/np.sqrt(2)
g = pcb.PlaneWaves(direction, k)
#g = pcb.FourierHankel([-2,-2], [0], k)
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
problem=ps.Problem(mesh,k,30, bnddata)
h = 0.1
problem.setParams(alpha = 1.0/(k * h), beta = k * h)

print mesh.nelements

npw = 12
# Original basis:
bases = problem.constructBasis(pcb.planeWaveBases(2,k,npw))
# Polynomials only:
#bases = problem.constructBasis(pcbr.ReferenceBasisRule(pcbr.Dubiner(5)))
# Product basis:
bases = problem.constructBasis(pcb.ProductBasisRule(pcb.planeWaveBases(2,k,npw), pcbr.ReferenceBasisRule(pcbr.Dubiner(1))))

solution = ps.Computation(problem, bases, True, True).solve()
solution.writeSolution(bounds,npoints,fname='square.vti')
problem.writeMesh(fname='square.vtu',scalars=solution.combinedError())
err = solution.combinedError()
perr = solution.evaluate(sp) - gvals
print npw, np.sqrt(np.vdot(perr,perr) / sp.length) / l2g

