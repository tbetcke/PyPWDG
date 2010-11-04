from pypwdg import setup,runParallel,gmshMesh, PlaneWaves
from pypwdg import zero_dirichlet, generic_boundary_data
from numpy import array

k = 20 
direction=array([[1.0,0,0]])
def g(x):
    return PlaneWaves(direction, k).values(x)
def gn(x,n):
    return PlaneWaves(direction, k).derivs(x,n)



bnddata={59:zero_dirichlet(),
         58:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         60:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         61:zero_dirichlet()}

runParallel()

bounds=array([[0,4],[0,1],[0,.3]],dtype='d')
npoints=array([100,20,20])

comp=setup(gmshMesh('hole_extrusion.msh',dim=3),k=k,nquadpoints=10,nplanewaves=4,bnddata=bnddata)
comp.solve()
comp.writeSolution(bounds,npoints,fname='hole_extrusion.vti')
comp.writeMesh(fname='hole_extrusion.vtu',scalars=comp.combinedError())
