from pypwdg import PlaneWaves
from pypwdg import setup,runParallel,gmshMesh
from pypwdg import generic_boundary_data, dirichlet
from numpy import array,sqrt

k = 15
direction=array([[1.0,1.0]])/sqrt(2)
def g(x):
    return PlaneWaves(direction, k).values(x)
def gn(x,n):
    return PlaneWaves(direction, k).derivs(x,n)

impbd = generic_boundary_data([-1j*k,1],[-1j*k,1],g=g,dg=gn)

#bnddata={7:impd), 
#         8:impd)}
bnddata={7:dirichlet(g), 
         8:dirichlet(g)}

runParallel()

bounds=array([[0,1],[0,1],[0,0]],dtype='d')
npoints=array([100,100,1])

comp=setup(gmshMesh('square.msh',dim=2),k=k,nquadpoints=20,nplanewaves=15,bnddata=bnddata)
comp.writeSolution(bounds,npoints,fname='square.vti')
comp.writeMesh(fname='square.vtu',scalars=comp.combinedError())

