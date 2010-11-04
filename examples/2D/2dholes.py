from pypwdg import setup,runParallel,gmshMesh, PlaneWaves
from pypwdg import zero_dirichlet, generic_boundary_data
from numpy import array

k = 40
direction=array([[1.0,0]])
def g(x):
    return PlaneWaves(direction, k).values(x)
def gn(x,n):
    return PlaneWaves(direction, k).derivs(x,n)



bnddata={21:zero_dirichlet(),
         18:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         20:generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn),
         19:zero_dirichlet()}

runParallel()

bounds=array([[0,5],[0,1],[0,0]],dtype='d')
npoints=array([500,100,1])

comp=setup(gmshMesh('2dhole.msh',dim=2),k=k,nquadpoints=30,nplanewaves=20,bnddata=bnddata)
comp.solve()
comp.writeSolution(bounds,npoints,fname='2dhole.vti')
comp.writeMesh(fname='2dhole.vtu',scalars=comp.combinedError())