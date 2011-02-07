import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.adaptivity as pca
import pypwdg.parallel.main
import pypwdg.output.basis as pob
import pypwdg.utils.geometry as pug

import numpy as np

k = 80
direction=np.array([[1.0,1.0]])/np.sqrt(2)
#g = pcb.PlaneWaves(direction, k)
g = pcb.FourierHankel([-1,-1], [0], k)

impbd = pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g)

#bnddata={7:impbd, 
#         8:impbd}
bnddata={7:pcbd.dirichlet(g), 
         8:pcbd.dirichlet(g)}

bounds=np.array([[0,1],[0,1]],dtype='d')
npoints=np.array([250,250])

mesh = pmm.gmshMesh('square.msh',dim=2)

sp = pug.StructuredPoints(bounds.transpose(), npoints)
idx, points = sp.getPoints(bounds.transpose())
gvals = np.zeros(sp.length, dtype=complex)
gvals[idx] = g.values(points)
l2g = np.sqrt(np.vdot(gvals, gvals) / sp.length)

problem=ps.Problem(mesh,k,16, bnddata)
comp = pca.AdaptiveComputation(problem, pca.InitialPWFBCreator(mesh, k, 3,9))
solution = comp.solve()
solution.writeSolution(bounds,npoints,fname='squareadaptive0.vti')
err = solution.combinedError()
print np.sqrt(sum(err**2))
perr = solution.evaluate(sp) - gvals
print np.sqrt(np.vdot(perr,perr) / sp.length) / l2g
pob.vtkbasis(mesh, comp.etob, "adaptivedirs0.vtu", solution.x)
for i in range(1,20):
    comp.adapt()
    sol = comp.solve()
    err = sol.combinedError()
    print np.sqrt(sum(err**2))
    perr = sol.evaluate(sp) - gvals
    print i, np.sqrt(np.vdot(perr,perr) / sp.length) / l2g
    sol.writeSolution(bounds,npoints,fname='squareadaptive%s.vti'%(i))
    pob.vtkbasis(mesh, comp.etob, "adaptivedirs%s.vtu"%(i), sol.x)
