import pypwdg.setup as ps
import pypwdg.core.bases as pcb
import pypwdg.mesh.mesh as pmm
import pypwdg.core.boundary_data as pcbd

from numpy import array

k = 20 
direction=array([[1.0,0,0]])
def g(x):
    return pcb.PlaneWaves(direction, k).values(x)
def gn(x,n):
    return pcb.PlaneWaves(direction, k).derivs(x,n)



bnddata={59:pcbd.zero_dirichlet(),
         58:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         60:pcbd.generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         61:pcbd.zero_dirichlet()}

bounds=array([[0,4],[0,1],[0,.3]],dtype='d')
npoints=array([100,20,20])

comp=ps.setup(pmm.gmshMesh('hole_extrusion.msh',dim=3),k=k,nquadpoints=10,nplanewaves=4,bnddata=bnddata,usecache=False)
comp.solve()
comp.writeSolution(bounds,npoints,fname='hole_extrusion.vti')
comp.writeMesh(fname='hole_extrusion.vtu',scalars=comp.combinedError())
