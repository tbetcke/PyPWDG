import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd
import pypwdg.core.adaptivity as pca

from numpy import array,sqrt

k = 20
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

problem=ps.Problem(mesh,k,6, bnddata)
comp = pca.AdaptiveComputation(problem, 3, 7)
solution = comp.solve()
#solution.writeSolution(bounds,npoints,fname='adaptive.vti')
err = solution.combinedError()
#problem.writeMesh(fname='adaptive.vtu',scalars=err)
print sqrt(sum(err**2))
for _ in range(5):
    comp.adapt()
    sol = comp.solve()
    err = sol.combinedError()
    print sqrt(sum(err**2))
